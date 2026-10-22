"""Format and check the tracked outputs of the LBWSG PAF simulation.

Nothing here imports vivarium_inputs, so the simulation environment can use this
module as well as the artifact environment.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vivarium_gates_nutrition_optimization_child.constants import metadata, paths
from vivarium_gates_nutrition_optimization_child.utilities import sanitize_location

LBWSG_PAF_TARGET = "cause.affected_unmodeled.cause_specific_mortality_rate"
LBWSG_PAF_MEASURE = f"calculated_lbwsg_paf_on_{LBWSG_PAF_TARGET}"

EARLY_NEONATAL = "early_neonatal"
LATE_NEONATAL = "late_neonatal"
# The artifact's age bins, which fill_subnational_paf_rows joins on exactly.
NEONATAL_AGE_BOUNDS = {
    EARLY_NEONATAL: (0.0, 0.01917808),
    LATE_NEONATAL: (0.01917808, 0.07671233),
}
POST_NEONATAL_AGE_BOUNDS = [(0.07671233, 1.0), (1.0, 5.0)]
SEXES = ["Female", "Male"]
STRATA_COLUMNS = ["age_group", "sex", "input_draw"]
PAF_INDEX_COLUMNS = ["sex", "age_start", "age_end", "year_start", "year_end"]


def get_output_path(location: str) -> Path:
    """Return the tracked LBWSG PAF output file for a location."""
    return (
        paths.LBWSG_PAF_OUTPUT_DIR
        / sanitize_location(location)
        / f"{LBWSG_PAF_MEASURE}.parquet"
    )


def sort_paf_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort PAF simulation results so a regenerated file diffs cleanly."""
    results = results.assign(
        age_group=results["age_group"].astype(str), sex=results["sex"].astype(str)
    )
    return results.sort_values(STRATA_COLUMNS).reset_index(drop=True)


def format_lbwsg_paf(
    results: pd.DataFrame, require_late_neonatal: bool = True
) -> pd.DataFrame:
    """Format LBWSG PAF simulation results as the artifact's LBWSG PAF key.

    Parameters
    ----------
    results
        Long-format results with ``age_group``, ``sex``, ``input_draw`` and
        ``value`` columns and at most one ``random_seed`` and ``scenario``.
    require_late_neonatal
        Whether late neonatal values must be present. The early neonatal pass
        observes no late neonatal simulants, so its late neonatal values are 0.

    Returns
    -------
        The PAF indexed by sex, age and year, with one column per artifact draw.

    Raises
    ------
    ValueError
        If the results are incomplete, duplicated or outside (0, 1).
    """
    missing_columns = {"age_group", "sex", "input_draw", "value"} - set(results.columns)
    if missing_columns:
        raise ValueError(f"LBWSG PAF results are missing columns {sorted(missing_columns)}.")
    for column in ["random_seed", "scenario"]:
        if column in results and results[column].nunique() > 1:
            raise ValueError(
                f"LBWSG PAF results hold more than one {column}: "
                f"{sorted(results[column].unique())}."
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
    if paf.isna().any().any():
        raise ValueError("LBWSG PAF results are missing a value for some stratum and draw.")

    late_neonatal = paf.index.get_level_values("age_group") == LATE_NEONATAL
    if not require_late_neonatal:
        paf.loc[late_neonatal] = 0.0
    checked = paf if require_late_neonatal else paf.loc[~late_neonatal]
    if not ((checked > 0) & (checked < 1)).all().all():
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


def check_early_neonatal_unchanged(enn_pass: pd.DataFrame, lnn_pass: pd.DataFrame) -> None:
    """Raise unless both PAF simulation passes observed the same early neonatal PAFs."""

    def early_neonatal(results: pd.DataFrame) -> pd.Series:
        results = sort_paf_results(results)
        results = results.loc[results["age_group"] == EARLY_NEONATAL]
        return results.set_index(["sex", "input_draw"])["value"]

    enn_values, lnn_values = early_neonatal(enn_pass), early_neonatal(lnn_pass)
    if not enn_values.index.equals(lnn_values.index):
        raise ValueError(
            "The two PAF passes observed different early neonatal strata or draws."
        )
    if not np.array_equal(enn_values.to_numpy(), lnn_values.to_numpy()):
        difference = (enn_values - lnn_values).abs().max()
        raise ValueError(
            "The late neonatal pass observed different early neonatal PAFs than the early "
            f"neonatal pass (largest difference {difference}). Both are observed at birth "
            "on the same grid, so they should be identical."
        )
