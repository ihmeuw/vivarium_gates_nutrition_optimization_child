"""Collect and format the outputs of the LBWSG PAF simulation (``lbwsg_paf.yaml``).

Nothing here imports vivarium_inputs, so the simulation environment can use this
module as well as the artifact environment.
"""

import argparse
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from vivarium.artifact import Artifact

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    metadata,
    paths,
)
from vivarium_gates_nutrition_optimization_child.utilities import sanitize_location

LBWSG_PAF_MEASURE = "calculated_lbwsg_paf_on_cause.diarrheal_diseases.excess_mortality_rate"

# The artifact's age bins, which fill_subnational_paf_rows joins on exactly.
NEONATAL_AGE_BOUNDS = {
    "early_neonatal": (0.0, 0.01917808),
    "late_neonatal": (0.01917808, 0.07671233),
}
POST_NEONATAL_AGE_BOUNDS = [(0.07671233, 1.0), (1.0, 5.0)]
SEXES = ["Female", "Male"]
STRATA_COLUMNS = ["age_group", "sex", "input_draw"]
PAF_INDEX_COLUMNS = ["sex", "age_start", "age_end", "year_start", "year_end"]


def get_output_path(location: str) -> Path:
    """Return the tracked LBWSG PAF simulation output for a location."""
    return paths.LBWSG_PAF_OUTPUT_DIR / f"{sanitize_location(location)}.parquet"


def collect_paf_results(results_dir: Union[str, Path]) -> pd.DataFrame:
    """Read every shard psimulate wrote for the PAF measure, sorted by stratum and draw.

    Parameters
    ----------
    results_dir
        The ``results`` directory of a psimulate run of ``lbwsg_paf.yaml``.

    Returns
    -------
        The long-format results of all the run's jobs.
    """
    measure_dir = Path(results_dir) / LBWSG_PAF_MEASURE
    if not measure_dir.is_dir():
        raise FileNotFoundError(
            f"No {LBWSG_PAF_MEASURE} results under {results_dir}. Pass the 'results' "
            "directory that psimulate reports when it finishes."
        )
    results = pd.read_parquet(measure_dir)
    results = results.assign(
        age_group=results["age_group"].astype(str), sex=results["sex"].astype(str)
    )
    return results.sort_values(STRATA_COLUMNS).reset_index(drop=True)


def format_lbwsg_paf(results: pd.DataFrame) -> pd.DataFrame:
    """Format LBWSG PAF simulation results as the artifact's LBWSG PAF key.

    Artifact column ``draw_k`` holds the result for input draw k.

    Parameters
    ----------
    results
        Long-format results with ``age_group``, ``sex``, ``input_draw`` and
        ``value`` columns and at most one ``random_seed``.

    Returns
    -------
        The PAF indexed by sex, age and year, with one column per artifact draw
        and zero rows for the post-neonatal age groups.

    Raises
    ------
    ValueError
        If the results are incomplete, duplicated or outside (0, 1).
    """
    missing_columns = {"age_group", "sex", "input_draw", "value"} - set(results.columns)
    if missing_columns:
        raise ValueError(f"LBWSG PAF results are missing columns {sorted(missing_columns)}.")
    if "random_seed" in results and results["random_seed"].nunique() > 1:
        raise ValueError(
            "LBWSG PAF results hold more than one random seed: "
            f"{sorted(results['random_seed'].unique())}."
        )

    results = results.assign(
        age_group=results["age_group"].astype(str), sex=results["sex"].astype(str)
    )
    is_neonatal = results["age_group"].isin(NEONATAL_AGE_BOUNDS)
    if (results.loc[~is_neonatal, "value"] != 0).any():
        raise ValueError("LBWSG PAF results are non-zero outside the neonatal age groups.")
    results = results.loc[is_neonatal]
    if results.duplicated(STRATA_COLUMNS).any():
        raise ValueError("LBWSG PAF results hold more than one value for a stratum and draw.")

    paf = results.pivot(index=["sex", "age_group"], columns="input_draw", values="value")
    paf.columns = [f"draw_{draw}" for draw in paf.columns]
    expected_strata = pd.MultiIndex.from_product(
        [SEXES, list(NEONATAL_AGE_BOUNDS)], names=["sex", "age_group"]
    )
    missing_strata = expected_strata.difference(paf.index)
    missing_draws = metadata.ARTIFACT_COLUMNS.difference(paf.columns)
    extra_draws = pd.Index(paf.columns).difference(metadata.ARTIFACT_COLUMNS)
    if len(missing_strata) or len(missing_draws) or len(extra_draws):
        raise ValueError(
            "LBWSG PAF results do not cover exactly the artifact's strata and draws. "
            f"Missing strata: {list(missing_strata)}; missing draws: {list(missing_draws)}; "
            f"unexpected draws: {list(extra_draws)}."
        )
    paf = paf.reindex(index=expected_strata, columns=metadata.ARTIFACT_COLUMNS)
    if not ((paf > 0) & (paf < 1)).all().all():
        raise ValueError("LBWSG PAF results hold values outside (0, 1).")

    paf = paf.reset_index()
    paf["age_start"] = paf["age_group"].map(lambda group: NEONATAL_AGE_BOUNDS[group][0])
    paf["age_end"] = paf["age_group"].map(lambda group: NEONATAL_AGE_BOUNDS[group][1])
    post_neonatal = pd.DataFrame(
        [
            {"sex": sex, "age_start": age_start, "age_end": age_end}
            for sex in SEXES
            for age_start, age_end in POST_NEONATAL_AGE_BOUNDS
        ]
    )
    post_neonatal = pd.concat(
        [
            post_neonatal,
            pd.DataFrame(0.0, index=post_neonatal.index, columns=metadata.ARTIFACT_COLUMNS),
        ],
        axis=1,
    )
    paf = pd.concat([paf.drop(columns="age_group"), post_neonatal], ignore_index=True)
    paf["year_start"] = metadata.GBD_EXTRACT_YEAR
    paf["year_end"] = metadata.GBD_EXTRACT_YEAR + 1
    return paf.set_index(PAF_INDEX_COLUMNS)[metadata.ARTIFACT_COLUMNS].sort_index()


def get_run_location(results_dir: Union[str, Path]) -> str:
    """Return the location of the artifact a psimulate run used."""
    with open(Path(results_dir).parent / "model_specification.yaml") as file:
        specification = yaml.safe_load(file)
    artifact_path = specification["configuration"]["input_data"]["artifact_path"]
    return Artifact(artifact_path).load(data_keys.POPULATION.LOCATION)


def main() -> None:
    """Write a psimulate run's PAF results to its location's tracked output file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("results_dir", help="the 'results' directory of the psimulate run")
    args = parser.parse_args()

    results = collect_paf_results(args.results_dir)
    format_lbwsg_paf(results)
    output_path = get_output_path(get_run_location(args.results_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)
    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
