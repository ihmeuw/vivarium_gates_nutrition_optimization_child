"""Check LBWSG PAFs against calculations made directly from the artifact.

The early neonatal PAF is deterministic: it is the birth-prevalence-weighted
mean of the relative risk over each category's grid, so it can be recomputed
exactly. The late neonatal PAF depends on simulated deaths, so it is compared
with its expected value given the same hazard, within Monte Carlo error.

Nothing here imports vivarium_inputs, so either environment can run the checks.
"""

import math
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from vivarium.artifact import Artifact

from vivarium_gates_nutrition_optimization_child.constants import data_keys, paths
from vivarium_gates_nutrition_optimization_child.data import lbwsg_paf_results

# VPH's LBWSGRiskEffect gives a relative risk of exactly 1 in this region
# instead of evaluating the interpolator there.
TMREL_GESTATIONAL_AGE = 38.0
TMREL_BIRTH_WEIGHT = 3500.0
EARLY_NEONATAL_AGE_START = lbwsg_paf_results.NEONATAL_AGE_BOUNDS["early_neonatal"][0]
LATE_NEONATAL_AGE_START = lbwsg_paf_results.NEONATAL_AGE_BOUNDS["late_neonatal"][0]
EARLY_NEONATAL_AGE_END = lbwsg_paf_results.NEONATAL_AGE_BOUNDS["early_neonatal"][1]
DAYS_PER_YEAR = 365.25
# The engine converts rates to per-step probabilities over step_size / 365 years.
RATE_CONVERSION_DAYS_PER_YEAR = 365.0
EXACT_TOLERANCE = 1e-10
LATE_NEONATAL_Z_FLAG = 5.0

Interpolator = Callable[..., np.ndarray]


def category_boxes(categories: Dict[str, str]) -> Dict[str, Tuple[float, ...]]:
    """Parse each category's (GA lower, GA upper, BW lower, BW upper) box."""
    # The pregnancy model parses the descriptions this way to sample its births.
    boxes = {}
    for category, description in categories.items():
        values = tuple(float(value) for value in re.findall(r"(\d+)", description))
        if len(values) != 4:
            raise ValueError(f"Could not parse LBWSG category {category}: '{description}'")
        boxes[category] = values
    return boxes


def grid_points(
    boxes: Dict[str, Tuple[float, ...]], num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the GA and BW of each category's midpoint grid, category by category."""
    gestational_ages, birth_weights = [], []
    cell = (np.arange(num_points) + 0.5) / num_points
    for ga_lower, ga_upper, bw_lower, bw_upper in boxes.values():
        ga, bw = np.meshgrid(
            ga_lower + cell * (ga_upper - ga_lower), bw_lower + cell * (bw_upper - bw_lower)
        )
        gestational_ages.append(ga.ravel())
        birth_weights.append(bw.ravel())
    return np.concatenate(gestational_ages), np.concatenate(birth_weights)


def relative_risks(
    interpolator: Interpolator, gestational_age: np.ndarray, birth_weight: np.ndarray
) -> np.ndarray:
    """Evaluate relative risks as VPH's LBWSGRiskEffect does."""
    relative_risk = np.exp(interpolator(gestational_age, birth_weight, grid=False))
    is_tmrel = (gestational_age >= TMREL_GESTATIONAL_AGE) & (
        birth_weight >= TMREL_BIRTH_WEIGHT
    )
    relative_risk[is_tmrel] = 1.0
    return relative_risk


def expected_early_neonatal_paf(weights: np.ndarray, relative_risk: np.ndarray) -> float:
    """Return 1 - 1/E[RR] with categories (rows of relative_risk) weighted by weights."""
    mean_relative_risk = np.average(relative_risk.mean(axis=1), weights=weights)
    return (mean_relative_risk - 1) / mean_relative_risk


def expected_late_neonatal_paf(
    weights: np.ndarray,
    early_relative_risk: np.ndarray,
    late_relative_risk: np.ndarray,
    all_cause_mortality: float,
    unmodeled_mortality: float,
    early_paf: float,
    step_days: float,
    death_steps: int,
) -> Tuple[float, float]:
    """Return the expected late neonatal PAF and its Monte Carlo standard error.

    Each simulant survives each early neonatal step with probability
    1 - min(1, h * step / 365), where h = ACMR - U + U * (1 - PAF) * RR is the
    hazard VPH's Mortality applies. Rows of the relative risk arrays are categories.
    """
    hazard = (
        all_cause_mortality
        - unmodeled_mortality
        + unmodeled_mortality * (1 - early_paf) * early_relative_risk
    )
    death_probability = np.clip(hazard * step_days / RATE_CONVERSION_DAYS_PER_YEAR, 0.0, 1.0)
    survival = (1 - death_probability) ** death_steps

    denominator = np.sum(weights * survival.mean(axis=1))
    mean_relative_risk = np.sum(weights * (survival * late_relative_risk).mean(axis=1)) / (
        denominator
    )
    num_points = survival.shape[1]
    variance = np.sum(
        (weights / denominator) ** 2
        * (survival * (1 - survival) * (late_relative_risk - mean_relative_risk) ** 2).sum(
            axis=1
        )
        / num_points**2
    )
    paf = (mean_relative_risk - 1) / mean_relative_risk
    return paf, math.sqrt(variance) / mean_relative_risk**2


def get_simulation_settings(specification: Optional[Path] = None) -> Dict[str, Any]:
    """Return the settings of the late neonatal pass that its expectation depends on."""
    specification = specification or paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf.yaml"
    configuration = yaml.safe_load(specification.read_text())["configuration"]
    population = configuration["population"]
    step_days = configuration["time"]["step_size"]
    step_years = step_days / DAYS_PER_YEAR
    # Every simulant must leave early neonatal after the same number of steps.
    youngest = population["initialization_age_max"] / population["population_size"]
    oldest = population["initialization_age_max"]
    death_steps = {
        math.ceil((EARLY_NEONATAL_AGE_END - age) / step_years) for age in (youngest, oldest)
    }
    if len(death_steps) != 1:
        raise ValueError(
            "The PAF cohort does not leave early neonatal on a single time step: "
            f"it would take {sorted(death_steps)} steps."
        )
    return {
        "population_size": population["population_size"],
        "step_days": step_days,
        "death_steps": death_steps.pop(),
        "start_year": configuration["time"]["start"]["year"],
        "unmodeled_causes": configuration["mortality"]["unmodeled_causes"],
    }


def check_location(
    artifact_file: Path,
    location: str,
    enn_results_file: Optional[Path] = None,
    pregnancy_artifact_file: Optional[Path] = None,
    paf_output_file: Optional[Path] = None,
    grid_check_draws: int = 3,
) -> pd.DataFrame:
    """Run the LBWSG PAF checks for one location and raise if a hard check fails.

    Parameters
    ----------
    artifact_file
        An artifact holding the location's LBWSG keys and the LBWSG PAF to check.
        The late neonatal check also needs national all-cause mortality and
        affected unmodeled CSMRs, as in the PAF simulation's own artifact.
    location
        The location the artifact is for.
    enn_results_file
        The early neonatal pass's results, to check against the artifact's PAF.
    pregnancy_artifact_file
        The pregnancy model's artifact, whose LBWSG exposure the birth exposure
        key must equal.
    paf_output_file
        The PAF output the artifact's PAF must equal. Defaults to the location's
        tracked output.
    grid_check_draws
        How many draws to recompute on a much finer grid.

    Returns
    -------
        One row per check with its statistic and whether it passed.
    """
    artifact = Artifact(artifact_file)
    settings = get_simulation_settings()
    boxes = category_boxes(artifact.load(data_keys.LBWSG.CATEGORIES))
    num_categories = len(boxes)
    num_points = math.isqrt(settings["population_size"] // (2 * num_categories))
    gestational_age, birth_weight = grid_points(boxes, num_points)

    paf = _wide(artifact.load(data_keys.LBWSG.PAF))
    draws = [column for column in paf.columns if column.startswith("draw_")]
    keyed_paf = paf.set_index(lbwsg_paf_results.PAF_INDEX_COLUMNS)[draws].sort_index()
    birth_exposure = _wide(artifact.load(data_keys.LBWSG.BIRTH_EXPOSURE))
    interpolators = _wide(artifact.load(data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR))
    mortality = [_wide(artifact.load(data_keys.POPULATION.ACMR))] + [
        _wide(artifact.load(f"cause.{cause}.cause_specific_mortality_rate"))
        for cause in settings["unmodeled_causes"]
    ]
    national_mortality = all(_is_national(data) for data in mortality)

    def relative_risk_grid(sex: str, age_start: float, draw: str) -> np.ndarray:
        interpolator = _interpolator(interpolators, sex, age_start, draw)
        rr = relative_risks(interpolator, gestational_age, birth_weight)
        return rr.reshape(num_categories, -1)

    early_differences, late_z_scores = [], []
    for draw in draws:
        for sex in lbwsg_paf_results.SEXES:
            weights = _category_values(birth_exposure, boxes, sex, draw)
            early_rr = relative_risk_grid(sex, EARLY_NEONATAL_AGE_START, draw)
            early_paf = _paf_value(paf, sex, EARLY_NEONATAL_AGE_START, draw)
            early_differences.append(
                abs(early_paf - expected_early_neonatal_paf(weights, early_rr))
            )
            late_paf = _paf_value(paf, sex, LATE_NEONATAL_AGE_START, draw)
            if national_mortality and late_paf != 0:
                year = settings["start_year"]
                expected_late, standard_error = expected_late_neonatal_paf(
                    weights,
                    early_rr,
                    relative_risk_grid(sex, LATE_NEONATAL_AGE_START, draw),
                    _mortality_value(mortality[0], sex, draw, year),
                    sum(_mortality_value(data, sex, draw, year) for data in mortality[1:]),
                    early_paf,
                    settings["step_days"],
                    settings["death_steps"],
                )
                late_z_scores.append((late_paf - expected_late) / standard_error)

    rows = []
    max_early_difference = max(early_differences)
    rows.append(
        _row(
            "V1",
            "ENN PAF vs 1 - 1/E[RR] from birth exposure and interpolator, max |diff|",
            max_early_difference,
            max_early_difference <= EXACT_TOLERANCE,
            hard=True,
        )
    )

    fine_points = 301
    fine_ga, fine_bw = grid_points(boxes, fine_points)
    grid_differences = []
    for draw in draws[:grid_check_draws]:
        for sex in lbwsg_paf_results.SEXES:
            weights = _category_values(birth_exposure, boxes, sex, draw)
            interpolator = _interpolator(interpolators, sex, EARLY_NEONATAL_AGE_START, draw)
            fine = relative_risks(interpolator, fine_ga, fine_bw).reshape(num_categories, -1)
            coarse = relative_risk_grid(sex, EARLY_NEONATAL_AGE_START, draw)
            grid_differences.append(
                abs(
                    expected_early_neonatal_paf(weights, coarse)
                    - expected_early_neonatal_paf(weights, fine)
                )
            )
    rows.append(
        _row(
            "V2",
            f"ENN PAF on a {num_points}^2 grid vs a {fine_points}^2 grid, max |diff| "
            f"over {grid_check_draws} draws",
            max(grid_differences),
            None,
        )
    )

    if late_z_scores:
        z = np.array(late_z_scores)
        rows.append(
            _row(
                "V3",
                f"LNN PAF vs its expectation, z-scores: mean {z.mean():.3f}, "
                f"sd {z.std():.3f}, max |z|",
                np.abs(z).max(),
                np.abs(z).max() < LATE_NEONATAL_Z_FLAG,
            )
        )
    else:
        reason = (
            "no late neonatal PAF to check"
            if national_mortality
            else "mortality is subnational; run against the PAF simulation artifact"
        )
        rows.append(_not_run("V3", "LNN PAF vs its expectation", reason))

    if enn_results_file is not None:
        enn = lbwsg_paf_results.format_lbwsg_paf(
            pd.read_parquet(enn_results_file), require_late_neonatal=False
        )
        difference = (enn - keyed_paf).abs()
        difference = difference.loc[
            difference.index.get_level_values("age_start") == EARLY_NEONATAL_AGE_START
        ]
        rows.append(
            _row(
                "V4",
                "ENN PAF in the artifact vs the early neonatal pass, max |diff|",
                difference.max().max(),
                difference.notna().all().all() and difference.max().max() == 0,
                hard=True,
            )
        )
    else:
        rows.append(_not_run("V4", "ENN PAF vs the early neonatal pass", "no ENN results"))

    if pregnancy_artifact_file is not None:
        pregnancy = _wide(Artifact(pregnancy_artifact_file).load(data_keys.LBWSG.EXPOSURE))
        index = ["sex", "parameter"]
        difference = (
            birth_exposure.set_index(index)[draws] - pregnancy.set_index(index)[draws]
        ).abs()
        rows.append(
            _row(
                "V5",
                "Birth exposure vs the pregnancy artifact's LBWSG exposure, max |diff|",
                difference.max().max(),
                difference.notna().all().all() and difference.max().max() <= 1e-12,
                hard=True,
            )
        )
    else:
        rows.append(
            _not_run("V5", "Birth exposure vs the pregnancy model", "no pregnancy artifact")
        )

    paf_output_file = paf_output_file or lbwsg_paf_results.get_output_path(location)
    if paf_output_file.exists():
        output = pd.read_parquet(paf_output_file)
        formatted = lbwsg_paf_results.format_lbwsg_paf(output)
        mismatches = _draw_keying_mismatches(output, keyed_paf)
        rows.append(
            _row(
                "V6",
                "Draws where artifact draw_k does not hold the output's input_draw k",
                mismatches,
                mismatches == 0,
                hard=True,
            )
        )
        rows.append(
            _row(
                "V7",
                f"Artifact LBWSG PAF equals {paf_output_file.name}, max |diff|",
                (formatted - keyed_paf).abs().max().max(),
                formatted.equals(keyed_paf),
                hard=True,
            )
        )
    else:
        reason = f"{paf_output_file} does not exist"
        rows.append(_not_run("V6", "Draw keying against the PAF output", reason))
        rows.append(_not_run("V7", "Artifact LBWSG PAF equals the PAF output", reason))

    rows.append(
        _exposure_weighted_diagnostic(artifact, boxes, paf, draws, relative_risk_grid)
    )

    summary = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(f"\nLBWSG PAF checks for {location} ({artifact_file})")
        print(summary.to_string(index=False))
    failed = summary.loc[summary["hard"] & (summary["passed"] == False), "check"]
    if len(failed):
        raise ValueError(f"LBWSG PAF checks failed for {location}: {', '.join(failed)}")
    return summary


def _exposure_weighted_diagnostic(
    artifact: Artifact,
    boxes: Dict[str, Tuple[float, ...]],
    paf: pd.DataFrame,
    draws: List[str],
    relative_risk_grid: Callable[[str, float, str], np.ndarray],
) -> Dict[str, Any]:
    """Compare the PAF with one weighted by LBWSG.EXPOSURE age-group rows instead."""
    description = "PAF minus the PAF weighted by LBWSG.EXPOSURE age-group rows, mean"
    try:
        exposure = _wide(artifact.load(data_keys.LBWSG.EXPOSURE))
        differences = []
        for draw in draws:
            for sex in lbwsg_paf_results.SEXES:
                for age_start in [EARLY_NEONATAL_AGE_START, LATE_NEONATAL_AGE_START]:
                    weights = _category_values(exposure, boxes, sex, draw, age_start)
                    rr = relative_risk_grid(sex, age_start, draw)
                    differences.append(
                        _paf_value(paf, sex, age_start, draw)
                        - expected_early_neonatal_paf(weights, rr)
                    )
    except (KeyError, ValueError) as error:
        return _not_run("V8", description, f"failed: {error}")
    return _row("V8", f"{description} (diagnostic only)", np.mean(differences), None)


def _is_national(data: pd.DataFrame) -> bool:
    """Return whether artifact data holds one location and no subnational level."""
    return "subnational" not in data and data.get("location", pd.Series()).nunique() <= 1


def _wide(data: pd.DataFrame) -> pd.DataFrame:
    """Return artifact data with its index as columns."""
    return data.reset_index()


def _category_values(
    data: pd.DataFrame,
    boxes: Dict[str, Tuple[float, ...]],
    sex: str,
    draw: str,
    age_start: Optional[float] = None,
) -> np.ndarray:
    """Return one draw of category-level data for one sex, in the order of boxes."""
    rows = data.loc[data["sex"] == sex]
    if age_start is not None:
        rows = rows.loc[np.isclose(rows["age_start"], age_start)]
    values = rows.set_index("parameter")[draw]
    if values.index.duplicated().any() or set(values.index) != set(boxes):
        raise ValueError(
            f"Category data for {sex} does not cover exactly the LBWSG categories once."
        )
    return values.loc[list(boxes)].to_numpy()


def _interpolator(
    interpolators: pd.DataFrame, sex: str, age_start: float, draw: str
) -> Interpolator:
    """Return one draw's relative risk interpolator for a sex and age group."""
    rows = interpolators.loc[
        (interpolators["sex"] == sex) & np.isclose(interpolators["age_start"], age_start)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one {sex} interpolator at age {age_start}; got {len(rows)}."
        )
    return pickle.loads(bytes.fromhex(rows[draw].iloc[0]))


def _paf_value(paf: pd.DataFrame, sex: str, age_start: float, draw: str) -> float:
    """Return one draw of the PAF for a sex and age group."""
    rows = paf.loc[(paf["sex"] == sex) & np.isclose(paf["age_start"], age_start)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {sex} PAF row at age {age_start}; got {len(rows)}.")
    return float(rows[draw].iloc[0])


def _mortality_value(data: pd.DataFrame, sex: str, draw: str, year: int) -> float:
    """Return one draw of an early neonatal rate as a simulation in *year* reads it."""
    rows = data.loc[
        (data["sex"] == sex) & np.isclose(data["age_start"], EARLY_NEONATAL_AGE_START)
    ]
    containing = rows.loc[(rows["year_start"] <= year) & (year < rows["year_end"])]
    if containing.empty and (rows["year_end"] <= year).all():
        # Past the data, so the order-0 extrapolation reads the latest year.
        containing = rows.loc[rows["year_start"] == rows["year_start"].max()]
    if len(containing) != 1:
        raise ValueError(
            f"Expected one early neonatal {sex} rate to use for {year}; "
            f"got {len(containing)}."
        )
    return float(containing[draw].iloc[0])


def _draw_keying_mismatches(output: pd.DataFrame, paf: pd.DataFrame) -> int:
    """Count draws where the artifact column does not hold that input draw's value."""
    output = lbwsg_paf_results.sort_paf_results(output)
    mismatches = 0
    for _, row in output.iterrows():
        bounds = lbwsg_paf_results.NEONATAL_AGE_BOUNDS.get(row["age_group"])
        if bounds is None:
            continue
        value = paf.xs((row["sex"], *bounds), level=["sex", "age_start", "age_end"])[
            f"draw_{row['input_draw']}"
        ]
        mismatches += int(not (value == row["value"]).all())
    return mismatches


def _row(
    check: str,
    description: str,
    statistic: float,
    passed: Optional[bool],
    hard: bool = False,
) -> Dict[str, Any]:
    """Return one row of the check summary."""
    return {
        "check": check,
        "description": description,
        "statistic": statistic,
        "passed": passed,
        "hard": hard,
    }


def _not_run(check: str, description: str, reason: str) -> Dict[str, Any]:
    """Return a summary row for a check that could not run."""
    return _row(check, f"{description}: not run ({reason})", np.nan, None)
