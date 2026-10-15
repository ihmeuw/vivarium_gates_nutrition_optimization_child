"""Verify the per-draw worse-MAM fraction replaces the constant 0.33 split (MIC-7263).

Covers ``get_worse_mam_fraction``, the MAM-incidence and SAM-remission splits in
``load_wasting_transition_rates``, and the cat2/cat2.5 split in
``load_wasting_birth_prevalence``, all against small synthetic inputs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
)

NATIONAL_ID = 179
OTHER_NATIONAL_ID = 165
UNKNOWN_LOCATION_ID = 999
NATIONAL_NAME = "Ethiopia"
SUBNATIONALS = ["Afar", "Amhara"]
SEXES = ["Female", "Male"]
YEAR_START, YEAR_END = 2023, 2024
EARLY_NEONATAL = (0.0, 0.01917808)
LATE_NEONATAL = (0.01917808, 0.07671233)
OLDER_AGE_BINS = [(0.07671233, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
TRANSITIONS = ["inc_rate_mam", "ux_rem_rate_sam", "inc_rate_sam", "rem_rate_mam"]
# Wider than DRAW_COUNT so the loader must restrict, not just reorder.
FRACTION_CSV_DRAW_COUNT = metadata.DRAW_COUNT + 10
DRAW_NUMBERS = np.arange(FRACTION_CSV_DRAW_COUNT)
# Varies by draw so a scalar fraction (old 0.33, or any single value) fails.
FRACTION = pd.Series(
    0.2 + 0.001 * DRAW_NUMBERS[: metadata.DRAW_COUNT], index=metadata.ARTIFACT_COLUMNS
)
OTHER_FRACTION = pd.Series(
    0.5 + 0.001 * DRAW_NUMBERS[: metadata.DRAW_COUNT], index=metadata.ARTIFACT_COLUMNS
)
RELATIVE_RISK = pd.Series(
    1.5 + 0.002 * DRAW_NUMBERS[: metadata.DRAW_COUNT], index=metadata.ARTIFACT_COLUMNS
)
TOLERANCE = dict(rtol=1e-12, atol=1e-15)


@pytest.fixture
def loader() -> ModuleType:
    """Import the loader, skipping when vivarium_inputs is not installed."""
    pytest.importorskip("vivarium_inputs", exc_type=ImportError)
    from vivarium_gates_nutrition_optimization_child.data import loader

    return loader


def _lexicographic_draws(count: int) -> List[str]:
    """Return draw names in the string-sorted order the raw CSVs use."""
    return sorted(f"draw_{i}" for i in range(count))


@pytest.fixture
def fraction_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: ModuleType) -> Path:
    """Write a long-format worse-MAM fraction CSV for two locations and point the loader at it."""
    rows = []
    for location_id, start in [(NATIONAL_ID, 0.2), (OTHER_NATIONAL_ID, 0.5)]:
        for draw in _lexicographic_draws(FRACTION_CSV_DRAW_COUNT):
            draw_number = int(draw.split("_")[1])
            rows.append((location_id, draw, start + 0.001 * draw_number))
    path = tmp_path / "worse_exp_frac_only_loc.csv"
    pd.DataFrame(rows, columns=["location_id", "draw", "exposure"]).to_csv(path, index=True)
    monkeypatch.setattr(loader.paths, "PROBABILITIES_OF_WORSE_MAM_EXPOSURE", path)
    return path


@pytest.fixture
def national_location(monkeypatch: pytest.MonkeyPatch, loader: ModuleType) -> None:
    """Resolve any location to NATIONAL_ID."""
    monkeypatch.setattr(loader, "get_national_location_id", lambda location: NATIONAL_ID)


def _demography() -> pd.DataFrame:
    """Build an empty frame indexed like population.demographic_dimensions."""
    tuples = [
        (NATIONAL_NAME, sex, age_start, age_end, YEAR_START, YEAR_END)
        for sex in SEXES
        for age_start, age_end in [EARLY_NEONATAL, LATE_NEONATAL] + OLDER_AGE_BINS
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=metadata.ARTIFACT_INDEX_COLUMNS)
    return pd.DataFrame(index=index)


def _transition_rates() -> pd.DataFrame:
    """Build raw transition rates indexed by demography and transition, DRAW_COUNT draws wide."""
    rng = np.random.default_rng(7263)
    tuples = [
        (location, sex, age_start, age_end, YEAR_START, YEAR_END, transition)
        for location in SUBNATIONALS
        for sex in SEXES
        for age_start, age_end in OLDER_AGE_BINS
        for transition in TRANSITIONS
    ]
    index = pd.MultiIndex.from_tuples(
        tuples, names=metadata.ARTIFACT_INDEX_COLUMNS + ["transition"]
    )
    values = rng.uniform(0.05, 1.5, size=(len(index), metadata.DRAW_COUNT))
    return pd.DataFrame(values, index=index, columns=metadata.ARTIFACT_COLUMNS)


def _write_transitions_csv(rates: pd.DataFrame, directory: Path) -> None:
    """Write rates in the raw CSV layout: unnamed index, sex, ages, location, draws, parameter."""
    flat = rates.reset_index().rename(columns={"transition": "parameter"})
    columns = ["sex", "age_start", "age_end", "location"]
    columns += _lexicographic_draws(metadata.DRAW_COUNT) + ["parameter"]
    directory.mkdir(parents=True, exist_ok=True)
    flat[columns].to_csv(directory / f"{NATIONAL_ID}.csv", index=True)


def _expected_unsplit(rates: pd.DataFrame, transition: str) -> pd.DataFrame:
    """Return the pre-split rows for a transition: CSV rows plus zeros for the neonatal ages."""
    csv_rows = rates.xs(transition, level="transition")
    neonatal = _demography().query("age_start < @OLDER_AGE_BINS[0][0]").index
    zeros = pd.DataFrame(0.0, index=neonatal, columns=metadata.ARTIFACT_COLUMNS)
    return pd.concat([zeros, csv_rows]).sort_index()


def _aligned(frame: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Reorder index levels and sort so frame can be compared against like."""
    return frame.reorder_levels(like.index.names).sort_index()[like.columns]


def _assert_split(worse: pd.DataFrame, better: pd.DataFrame, original: pd.DataFrame) -> None:
    """Assert worse and better partition original by FRACTION per draw and keep its index."""
    worse = _aligned(worse, original)
    better = _aligned(better, original)
    assert worse.index.equals(original.index)
    assert better.index.equals(original.index)
    assert list(worse.columns) == list(metadata.ARTIFACT_COLUMNS)
    pd.testing.assert_frame_equal(worse + better, original, check_dtype=False, **TOLERANCE)
    pd.testing.assert_frame_equal(
        worse, original.mul(FRACTION, axis=1), check_dtype=False, **TOLERANCE
    )
    pd.testing.assert_frame_equal(
        better, original.mul(1 - FRACTION, axis=1), check_dtype=False, **TOLERANCE
    )
    nonzero = (original != 0).all(axis=1)
    ratio = worse[nonzero] / (worse[nonzero] + better[nonzero])
    assert np.allclose(ratio, np.broadcast_to(FRACTION.to_numpy(), ratio.shape), **TOLERANCE)


# ---------------------------------------------------------------------------
# get_worse_mam_fraction
# ---------------------------------------------------------------------------


def test_get_worse_mam_fraction_returns_csv_values_in_artifact_column_order(
    loader: ModuleType, fraction_csv: Path
) -> None:
    result = loader.get_worse_mam_fraction(NATIONAL_ID)

    assert isinstance(result, pd.Series)
    assert list(result.index) == list(metadata.ARTIFACT_COLUMNS)
    pd.testing.assert_series_equal(result, FRACTION, check_names=False, **TOLERANCE)


def test_get_worse_mam_fraction_selects_requested_location(
    loader: ModuleType, fraction_csv: Path
) -> None:
    result = loader.get_worse_mam_fraction(OTHER_NATIONAL_ID)

    assert list(result.index) == list(metadata.ARTIFACT_COLUMNS)
    pd.testing.assert_series_equal(result, OTHER_FRACTION, check_names=False, **TOLERANCE)


def test_get_worse_mam_fraction_raises_for_unknown_location(
    loader: ModuleType, fraction_csv: Path
) -> None:
    with pytest.raises(ValueError):
        loader.get_worse_mam_fraction(UNKNOWN_LOCATION_ID)


# ---------------------------------------------------------------------------
# load_wasting_transition_rates
# ---------------------------------------------------------------------------


@pytest.fixture
def transition_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: ModuleType,
    fraction_csv: Path,
    national_location: None,
) -> pd.DataFrame:
    """Install synthetic demography and a transitions CSV; return the raw rates."""
    rates = _transition_rates()
    transitions_dir = tmp_path / "wasting_transition_rates"
    _write_transitions_csv(rates, transitions_dir)
    monkeypatch.setattr(loader.paths, "WASTING_TRANSITIONS_DATA_DIR", transitions_dir)

    def _get_data(key: str, location: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert str(key) == str(data_keys.POPULATION.DEMOGRAPHY)
        assert location == NATIONAL_ID
        return _demography()

    monkeypatch.setattr(loader, "get_data", _get_data)
    return rates


@pytest.fixture
def transition_rates(loader: ModuleType, transition_inputs: pd.DataFrame) -> pd.DataFrame:
    """Run the loader on the synthetic inputs."""
    return loader.load_wasting_transition_rates(
        data_keys.WASTING.TRANSITION_RATES, NATIONAL_NAME
    )


@pytest.mark.parametrize(
    "source, worse, better",
    [
        ("inc_rate_mam", "inc_rate_worse_mam", "inc_rate_better_mam"),
        ("ux_rem_rate_sam", "sam_to_worse_mam", "sam_to_better_mam"),
    ],
)
def test_transition_rates_split_by_per_draw_fraction(
    transition_inputs: pd.DataFrame,
    transition_rates: pd.DataFrame,
    source: str,
    worse: str,
    better: str,
) -> None:
    original = _expected_unsplit(transition_inputs, source)
    transitions = transition_rates.index.get_level_values("transition")

    assert source not in transitions
    _assert_split(
        transition_rates.xs(worse, level="transition"),
        transition_rates.xs(better, level="transition"),
        original,
    )


@pytest.mark.parametrize("transition", ["inc_rate_sam", "rem_rate_mam"])
def test_transition_rates_leave_other_transitions_unchanged(
    transition_inputs: pd.DataFrame, transition_rates: pd.DataFrame, transition: str
) -> None:
    original = _expected_unsplit(transition_inputs, transition)
    result = _aligned(transition_rates.xs(transition, level="transition"), original)

    pd.testing.assert_frame_equal(result, original, check_dtype=False, **TOLERANCE)


# ---------------------------------------------------------------------------
# load_wasting_birth_prevalence
# ---------------------------------------------------------------------------

WASTING_BASE: Dict[str, Dict[str, float]] = {
    "Female": {"cat1": 0.05, "cat2": 0.06, "cat2.5": 0.04, "cat3": 0.15, "cat4": 0.70},
    "Male": {"cat1": 0.06, "cat2": 0.07, "cat2.5": 0.05, "cat3": 0.17, "cat4": 0.65},
}
LBW_CATEGORIES = data_values.LBWSG.LOW_BIRTH_WEIGHT_CATEGORIES[:3]
NON_LBW_CATEGORY = "cat56"
LBW_BASE: Dict[str, List[float]] = {"Female": [0.05, 0.04, 0.03], "Male": [0.06, 0.05, 0.02]}


def _wasting_prevalence(location: str, sex: str, parameter: str) -> np.ndarray:
    """Return per-draw wasting prevalence; cat1 and cat4 vary by draw so draws differ."""
    draws = DRAW_NUMBERS[: metadata.DRAW_COUNT]
    shift = 0.01 if location == SUBNATIONALS[1] else 0.0
    value = WASTING_BASE[sex][parameter] + np.zeros(metadata.DRAW_COUNT)
    if parameter == "cat1":
        value = value + 0.0001 * draws
    if parameter == "cat2":
        value = value + shift
    if parameter == "cat4":
        value = value - 0.0001 * draws - shift
    return value


def _wasting_exposure() -> pd.DataFrame:
    """Build subnational wasting exposure with a 1-5 month row and a distractor row."""
    frames = []
    for location in SUBNATIONALS:
        for sex in SEXES:
            for age_start, age_end in [OLDER_AGE_BINS[0], OLDER_AGE_BINS[1]]:
                for parameter in WASTING_BASE[sex]:
                    index = (
                        location,
                        sex,
                        age_start,
                        age_end,
                        YEAR_START,
                        YEAR_END,
                        parameter,
                    )
                    frames.append((index, _wasting_prevalence(location, sex, parameter)))
    index = pd.MultiIndex.from_tuples(
        [key for key, _ in frames], names=metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
    )
    return pd.DataFrame(
        np.vstack([values for _, values in frames]),
        index=index,
        columns=metadata.ARTIFACT_COLUMNS,
    )


def _lbw_prevalence(sex: str, category_number: int) -> np.ndarray:
    """Return per-draw exposure for one low birth weight category."""
    draws = DRAW_NUMBERS[: metadata.DRAW_COUNT]
    return LBW_BASE[sex][category_number] + 0.00005 * draws * (category_number == 0)


def _lbwsg_exposure() -> pd.DataFrame:
    """Build national LBWSG exposure with neonatal rows and one non-LBW category."""
    frames = []
    for sex in SEXES:
        for age_start, age_end in [EARLY_NEONATAL, LATE_NEONATAL]:
            total = np.zeros(metadata.DRAW_COUNT)
            for number, category in enumerate(LBW_CATEGORIES):
                value = _lbw_prevalence(sex, number)
                if (age_start, age_end) == EARLY_NEONATAL:
                    value = value * 2  # distractor age group must be filtered out
                total = total + value
                index = (
                    NATIONAL_NAME,
                    sex,
                    age_start,
                    age_end,
                    YEAR_START,
                    YEAR_END,
                    category,
                )
                frames.append((index, value))
            index = (
                NATIONAL_NAME,
                sex,
                age_start,
                age_end,
                YEAR_START,
                YEAR_END,
                NON_LBW_CATEGORY,
            )
            frames.append((index, 1 - total))
    index = pd.MultiIndex.from_tuples(
        [key for key, _ in frames], names=metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
    )
    return pd.DataFrame(
        np.vstack([values for _, values in frames]),
        index=index,
        columns=metadata.ARTIFACT_COLUMNS,
    )


def _expected_unsplit_cat2_birth_prevalence() -> pd.DataFrame:
    """Compute the MAM birth prevalence before the worse/better split from the synthetic inputs."""
    rows = []
    for location in SUBNATIONALS:
        for sex in SEXES:
            mam = _wasting_prevalence(location, sex, "cat2") + _wasting_prevalence(
                location, sex, "cat2.5"
            )
            lbw = sum(_lbw_prevalence(sex, n) for n in range(len(LBW_CATEGORIES)))
            rr = RELATIVE_RISK.to_numpy()
            adequate = mam / (rr * lbw + (1 - lbw))
            rows.append(
                ((location, sex, YEAR_START, YEAR_END, "adequate_birth_weight"), adequate)
            )
            rows.append(
                ((location, sex, YEAR_START, YEAR_END, "low_birth_weight"), adequate * rr)
            )
    index = pd.MultiIndex.from_tuples(
        [key for key, _ in rows],
        names=["location", "sex", "year_start", "year_end", "birth_weight_status"],
    )
    return pd.DataFrame(
        np.vstack([values for _, values in rows]),
        index=index,
        columns=metadata.ARTIFACT_COLUMNS,
    ).sort_index()


@pytest.fixture
def birth_prevalence(
    monkeypatch: pytest.MonkeyPatch,
    loader: ModuleType,
    fraction_csv: Path,
    national_location: None,
) -> pd.DataFrame:
    """Run load_wasting_birth_prevalence on synthetic exposure inputs."""
    data = {
        str(data_keys.WASTING.EXPOSURE): _wasting_exposure(),
        str(data_keys.LBWSG.EXPOSURE): _lbwsg_exposure(),
    }

    def _get_data(key: str, location: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return data[str(key)].copy()

    def _draws(columns: pd.Index, seed: str, distribution: Any) -> pd.Series:
        assert list(columns) == list(metadata.ARTIFACT_COLUMNS)
        return RELATIVE_RISK.copy()

    monkeypatch.setattr(loader, "get_data", _get_data)
    monkeypatch.setattr(loader, "get_random_variable_draws", _draws)
    return loader.load_wasting_birth_prevalence(
        data_keys.WASTING.BIRTH_PREVALENCE, NATIONAL_NAME
    )


def test_birth_prevalence_splits_mam_by_per_draw_fraction(
    birth_prevalence: pd.DataFrame,
) -> None:
    original = _expected_unsplit_cat2_birth_prevalence()

    _assert_split(
        birth_prevalence.xs("cat2", level="parameter"),
        birth_prevalence.xs("cat2.5", level="parameter"),
        original,
    )


def test_birth_prevalence_categories_still_sum_to_one(
    birth_prevalence: pd.DataFrame,
) -> None:
    parameters = set(birth_prevalence.index.get_level_values("parameter"))
    group_levels = [name for name in birth_prevalence.index.names if name != "parameter"]
    totals = birth_prevalence.groupby(level=group_levels).sum()

    assert parameters == {"cat1", "cat2", "cat2.5", "cat3", "cat4"}
    assert np.allclose(totals, 1.0, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Removed constant
# ---------------------------------------------------------------------------


def test_probability_of_cat2_constant_is_removed() -> None:
    assert not hasattr(data_values.WASTING, "PROBABILITY_OF_CAT2")
