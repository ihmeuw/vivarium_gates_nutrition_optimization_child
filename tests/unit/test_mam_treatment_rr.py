"""Verify the rate-form MAM treatment relative risk (MIC-7265).

Covers the ``get_mam_remission_rates`` helper contract and the relative risk that
``load_mam_treatment_rr`` builds from treated and untreated remission rates, against
small synthetic demography and treatment-parameter inputs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
)

LOCATION = "Ethiopia"
SEXES = ["Female", "Male"]
YEAR_START, YEAR_END = 2023, 2024
AGE_BINS = [
    (0.0, 0.01917808),
    (0.01917808, 0.07671233),
    (0.07671233, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
]
SIX_MONTHS = 0.5
PARAMETERS = ["cat1", "cat2", "cat3"]
AFFECTED_ENTITIES = [
    "better_moderate_acute_malnutrition_to_mild_child_wasting",
    "worse_moderate_acute_malnutrition_to_mild_child_wasting",
]
AFFECTED_MEASURE = "transition_rate"
EFFICACY_DRAW_COUNT = 1000
DRAWS = np.arange(metadata.DRAW_COUNT)
# Varies by draw so a scalar efficacy fails.
BASELINE_EFFICACY = pd.Series(0.6 + 0.0002 * DRAWS, index=metadata.ARTIFACT_COLUMNS)
ALTERNATIVE_EFFICACY = data_values.WASTING.MAM_TX_ALTERNATIVE_EFFICACY
# The row whose rates are zeroed in the both-rates-zero test; must be six months or older.
ZERO_RATE_ROW = (LOCATION, "Male", 1.0, 2.0, YEAR_START, YEAR_END)
TOLERANCE = dict(rtol=1e-12, atol=1e-15)

RateFrames = Tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture
def loader() -> ModuleType:
    """Import the loader, skipping when vivarium_inputs is not installed."""
    pytest.importorskip("vivarium_inputs", exc_type=ImportError)
    from vivarium_gates_nutrition_optimization_child.data import loader

    return loader


def _demography() -> pd.DataFrame:
    """Build an empty frame indexed like population.demographic_dimensions."""
    tuples = [
        (LOCATION, sex, age_start, age_end, YEAR_START, YEAR_END)
        for sex in SEXES
        for age_start, age_end in AGE_BINS
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=metadata.ARTIFACT_INDEX_COLUMNS)
    return pd.DataFrame(index=index)


def _efficacy_index() -> pd.MultiIndex:
    """Build the demography-by-parameter index that get_treatment_efficacy produces."""
    tuples = [row + (parameter,) for row in _demography().index for parameter in PARAMETERS]
    return pd.MultiIndex.from_tuples(
        tuples, names=metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
    )


def _age_start(index: pd.Index) -> np.ndarray:
    """Return the age_start level of an index as an array."""
    return np.asarray(index.get_level_values("age_start"), dtype=float)


@pytest.fixture
def treatment_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: ModuleType
) -> Path:
    """Write a treatment-parameter CSV for LOCATION and point utilities at it."""
    columns = [f"draw_{i}" for i in range(EFFICACY_DRAW_COUNT)]
    e_mam = 0.6 + 0.0002 * np.arange(EFFICACY_DRAW_COUNT)
    frame = pd.DataFrame(
        {
            "c_sam": np.full(EFFICACY_DRAW_COUNT, 0.5),
            "e_sam": np.full(EFFICACY_DRAW_COUNT, 0.5),
            "c_mam": np.full(EFFICACY_DRAW_COUNT, 0.5),
            "e_mam": e_mam,
        },
        index=pd.Index(columns),
    ).T
    frame.index.name = "parameter"
    frame.to_csv(tmp_path / f"{LOCATION.lower()}.csv")
    monkeypatch.setattr(loader.utilities.paths, "WASTING_TREATMENT_PARAMETERS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def demography(monkeypatch: pytest.MonkeyPatch, loader: ModuleType) -> pd.DataFrame:
    """Serve synthetic demography from get_data."""
    frame = _demography()

    def _get_data(key: str, location: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert str(key) == str(data_keys.POPULATION.DEMOGRAPHY)
        assert location == LOCATION
        return frame.copy()

    monkeypatch.setattr(loader, "get_data", _get_data)
    return frame


def _synthetic_rates() -> RateFrames:
    """Build treated and untreated rates that vary by demography row and by draw."""
    index = _demography().index
    rows = np.arange(len(index))[:, None]
    draws = DRAWS[None, :]
    tx = pd.DataFrame(
        0.05 + 0.001 * rows + 0.0001 * draws, index=index, columns=metadata.ARTIFACT_COLUMNS
    )
    ux = pd.DataFrame(
        0.005 + 0.0005 * rows + 0.00002 * draws,
        index=index,
        columns=metadata.ARTIFACT_COLUMNS,
    )
    return tx, ux


def _broadcast(rates: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    """Repeat demography-level rates onto an index that also carries a parameter level."""
    frame = rates.loc[index.droplevel("parameter")].copy()
    frame.index = index
    return frame


@pytest.fixture
def patched_rates(monkeypatch: pytest.MonkeyPatch, loader: ModuleType) -> Dict[str, Any]:
    """Replace get_mam_remission_rates with known rates; record the index it receives."""
    tx, ux = _synthetic_rates()
    received: Dict[str, Any] = {"tx": tx, "ux": ux}

    def _rates(index: pd.Index, location: str) -> RateFrames:
        received["index"] = index
        received["location"] = location
        return _broadcast(tx, index), _broadcast(ux, index)

    monkeypatch.setattr(loader, "get_mam_remission_rates", _rates)
    return received


@pytest.fixture
def relative_risk(
    loader: ModuleType,
    treatment_parameters: Path,
    demography: pd.DataFrame,
    patched_rates: Dict[str, Any],
) -> pd.DataFrame:
    """Run the loader against the patched rates."""
    return loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.RELATIVE_RISK, LOCATION)


def _efficacy(parameter: str) -> pd.Series:
    """Return the per-draw treatment efficacy for a treatment category."""
    return {
        "cat1": pd.Series(0.0, index=metadata.ARTIFACT_COLUMNS),
        "cat2": BASELINE_EFFICACY,
        "cat3": pd.Series(ALTERNATIVE_EFFICACY, index=metadata.ARTIFACT_COLUMNS),
    }[parameter]


def _expected_rr(result: pd.DataFrame, tx: pd.DataFrame, ux: pd.DataFrame) -> pd.DataFrame:
    """Compute the rate-form RR row by row for the result's index from demography rates."""
    rows = []
    for key in result.index:
        demographic = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)]
        parameter = key[result.index.names.index("parameter")]
        efficacy = _efficacy(parameter).to_numpy()
        tmrel = BASELINE_EFFICACY.to_numpy()
        tx_row = tx.loc[demographic].to_numpy()
        ux_row = ux.loc[demographic].to_numpy()
        numerator = efficacy * tx_row + (1 - efficacy) * ux_row
        denominator = tmrel * tx_row + (1 - tmrel) * ux_row
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(denominator == 0, 1.0, numerator / denominator)
        if key[result.index.names.index("age_start")] < SIX_MONTHS:
            rr = np.ones_like(rr)
        rows.append(rr)
    return pd.DataFrame(
        np.vstack(rows), index=result.index, columns=metadata.ARTIFACT_COLUMNS
    )


def _select(frame: pd.DataFrame, **levels: Any) -> pd.DataFrame:
    """Select rows whose index levels match the given values."""
    mask = np.ones(len(frame), dtype=bool)
    for level, value in levels.items():
        mask &= np.asarray(frame.index.get_level_values(level) == value)
    return frame[mask]


# ---------------------------------------------------------------------------
# get_mam_remission_rates
# ---------------------------------------------------------------------------


@pytest.fixture
def remission_rates(loader: ModuleType) -> RateFrames:
    """Call the helper on the efficacy-shaped index."""
    return loader.get_mam_remission_rates(_efficacy_index(), LOCATION)


def test_remission_rates_keep_input_index_and_artifact_columns(
    remission_rates: RateFrames,
) -> None:
    tx, ux = remission_rates
    index = _efficacy_index()

    for frame in (tx, ux):
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.equals(index)
        assert list(frame.index.names) == list(index.names)
        assert list(frame.columns) == list(metadata.ARTIFACT_COLUMNS)
        assert not frame.isna().any().any()


def test_untreated_remission_rate_is_constant_over_six_month_recovery(
    remission_rates: RateFrames,
) -> None:
    _, ux = remission_rates
    expected = 1 / data_values.WASTING.MAM_UX_RECOVERY_TIME_OVER_6MO

    assert np.allclose(ux.to_numpy(), expected, **TOLERANCE)


def test_treated_remission_rate_under_six_months_is_constant(
    remission_rates: RateFrames,
) -> None:
    tx, _ = remission_rates
    under = tx[_age_start(tx.index) < SIX_MONTHS]
    expected = 1 / data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO

    assert len(under) == 2 * 3 * 3
    assert np.allclose(under.to_numpy(), expected, **TOLERANCE)


def test_treated_remission_rate_over_six_months_is_inverse_seeded_duration(
    loader: ModuleType, remission_rates: RateFrames
) -> None:
    tx, _ = remission_rates
    subset = _efficacy_index()[_age_start(_efficacy_index()) >= SIX_MONTHS]
    durations = loader.get_random_variable_draws(
        subset, *data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO
    )
    expected = pd.DataFrame(
        np.repeat((1 / durations.to_numpy())[:, None], metadata.DRAW_COUNT, axis=1),
        index=subset,
        columns=metadata.ARTIFACT_COLUMNS,
    )

    assert durations.nunique() > 1
    pd.testing.assert_frame_equal(
        tx.loc[subset], expected, check_dtype=False, check_names=False, **TOLERANCE
    )


# ---------------------------------------------------------------------------
# load_mam_treatment_rr with the helper patched
# ---------------------------------------------------------------------------


def test_loader_requests_rates_on_the_efficacy_index(
    relative_risk: pd.DataFrame, patched_rates: Dict[str, Any]
) -> None:
    received = patched_rates["index"]

    assert patched_rates["location"] == LOCATION
    assert list(received.names) == metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
    assert set(received.get_level_values("parameter")) == set(PARAMETERS)
    assert received.droplevel("parameter").drop_duplicates().equals(_demography().index)


def test_relative_risk_index_levels_and_columns(relative_risk: pd.DataFrame) -> None:
    expected_levels = metadata.ARTIFACT_INDEX_COLUMNS + [
        "affected_entity",
        "affected_measure",
        "parameter",
    ]

    assert list(relative_risk.index.names) == expected_levels
    assert list(relative_risk.columns) == list(metadata.ARTIFACT_COLUMNS)
    assert set(relative_risk.index.get_level_values("affected_entity")) == set(
        AFFECTED_ENTITIES
    )
    assert set(relative_risk.index.get_level_values("affected_measure")) == {AFFECTED_MEASURE}
    assert set(relative_risk.index.get_level_values("parameter")) == set(PARAMETERS)
    assert len(relative_risk) == len(_efficacy_index()) * len(AFFECTED_ENTITIES)
    assert not relative_risk.index.has_duplicates
    assert not relative_risk.isna().any().any()


def test_relative_risk_matches_rate_form_formula(
    relative_risk: pd.DataFrame, patched_rates: Dict[str, Any]
) -> None:
    expected = _expected_rr(relative_risk, patched_rates["tx"], patched_rates["ux"])

    pd.testing.assert_frame_equal(relative_risk, expected, check_dtype=False, **TOLERANCE)


def test_uncovered_relative_risk_is_untreated_over_mixed_rate(
    relative_risk: pd.DataFrame, patched_rates: Dict[str, Any]
) -> None:
    cat1 = _select(relative_risk, parameter="cat1")
    older = cat1[_age_start(cat1.index) >= SIX_MONTHS]
    e2 = BASELINE_EFFICACY.to_numpy()

    assert len(older) == 2 * 3 * len(AFFECTED_ENTITIES)
    for key, row in older.iterrows():
        demographic = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)]
        tx = patched_rates["tx"].loc[demographic].to_numpy()
        ux = patched_rates["ux"].loc[demographic].to_numpy()
        expected = ux / (tx * e2 + ux * (1 - e2))
        assert np.allclose(row.to_numpy(), expected, **TOLERANCE)
        assert (row.to_numpy() < 1).all()


def test_baseline_relative_risk_is_exactly_one(relative_risk: pd.DataFrame) -> None:
    cat2 = _select(relative_risk, parameter="cat2")

    assert (cat2.to_numpy() == 1.0).all()


def test_alternative_relative_risk_mixes_rates_by_alternative_efficacy(
    relative_risk: pd.DataFrame, patched_rates: Dict[str, Any]
) -> None:
    cat3 = _select(relative_risk, parameter="cat3")
    older = cat3[_age_start(cat3.index) >= SIX_MONTHS]
    e2 = BASELINE_EFFICACY.to_numpy()
    e3 = ALTERNATIVE_EFFICACY

    assert len(older) == 2 * 3 * len(AFFECTED_ENTITIES)
    for key, row in older.iterrows():
        demographic = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)]
        tx = patched_rates["tx"].loc[demographic].to_numpy()
        ux = patched_rates["ux"].loc[demographic].to_numpy()
        expected = (e3 * tx + (1 - e3) * ux) / (e2 * tx + (1 - e2) * ux)
        assert np.allclose(row.to_numpy(), expected, **TOLERANCE)
        assert (row.to_numpy() > 1).all()


def test_both_affected_entities_carry_identical_values(relative_risk: pd.DataFrame) -> None:
    better = _select(relative_risk, affected_entity=AFFECTED_ENTITIES[0]).droplevel(
        "affected_entity"
    )
    worse = _select(relative_risk, affected_entity=AFFECTED_ENTITIES[1]).droplevel(
        "affected_entity"
    )

    assert len(better) == len(_efficacy_index())
    pd.testing.assert_frame_equal(better, worse, **TOLERANCE)


def test_relative_risk_is_one_under_six_months_regardless_of_rates(
    relative_risk: pd.DataFrame, patched_rates: Dict[str, Any]
) -> None:
    under = relative_risk[_age_start(relative_risk.index) < SIX_MONTHS]
    tx_under = patched_rates["tx"][_age_start(patched_rates["tx"].index) < SIX_MONTHS]

    assert len(under) == 2 * 3 * 3 * len(AFFECTED_ENTITIES)
    assert (tx_under.to_numpy() > 0).all()
    assert (under.to_numpy() == 1.0).all()


def test_rows_with_both_rates_zero_get_relative_risk_one(
    monkeypatch: pytest.MonkeyPatch,
    loader: ModuleType,
    treatment_parameters: Path,
    demography: pd.DataFrame,
) -> None:
    tx, ux = _synthetic_rates()
    tx.loc[ZERO_RATE_ROW] = 0.0
    ux.loc[ZERO_RATE_ROW] = 0.0

    def _rates(index: pd.Index, location: str) -> RateFrames:
        return _broadcast(tx, index), _broadcast(ux, index)

    monkeypatch.setattr(loader, "get_mam_remission_rates", _rates)
    result = loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.RELATIVE_RISK, LOCATION)
    zeroed = _select(result, sex=ZERO_RATE_ROW[1], age_start=ZERO_RATE_ROW[2])

    assert len(zeroed) == 3 * len(AFFECTED_ENTITIES)
    assert not result.isna().any().any()
    assert np.isfinite(result.to_numpy()).all()
    assert (zeroed.to_numpy() == 1.0).all()
    pd.testing.assert_frame_equal(
        result, _expected_rr(result, tx, ux), check_dtype=False, **TOLERANCE
    )


# ---------------------------------------------------------------------------
# load_mam_treatment_rr without patching the helper
# ---------------------------------------------------------------------------


def test_relative_risk_equals_duration_form_from_constants(
    loader: ModuleType, treatment_parameters: Path, demography: pd.DataFrame
) -> None:
    """Pin the rate form to the duration form for this iteration only.

    MIC-7264 replaces the recovery-time constants with location-specific rates; this
    test goes with them.
    """
    index = _efficacy_index()
    older = _age_start(index) >= SIX_MONTHS
    ux_duration = data_values.WASTING.MAM_UX_RECOVERY_TIME_OVER_6MO
    tx_duration = pd.Series(
        data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO, index=index, dtype=float
    )
    tx_duration[older] = loader.get_random_variable_draws(
        index[older], *data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO
    ).to_numpy()

    result = loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.RELATIVE_RISK, LOCATION)

    assert len(result) == len(index) * len(AFFECTED_ENTITIES)
    for key, row in result.iterrows():
        efficacy_key = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)] + (key[-1],)
        e = _efficacy(key[-1]).to_numpy()
        e_tmrel = BASELINE_EFFICACY.to_numpy()
        tx_d = tx_duration[efficacy_key]
        expected = (e * ux_duration + (1 - e) * tx_d) / (
            e_tmrel * ux_duration + (1 - e_tmrel) * tx_d
        )
        if key[metadata.ARTIFACT_INDEX_COLUMNS.index("age_start")] < SIX_MONTHS:
            expected = np.ones_like(expected)
        assert np.allclose(row.to_numpy(), expected, **TOLERANCE), key


def test_efficacy_index_matches_treatment_efficacy_helper(
    loader: ModuleType, treatment_parameters: Path, demography: pd.DataFrame
) -> None:
    efficacy, tmrel = loader.utilities.get_treatment_efficacy(
        demography.reset_index(), data_keys.WASTING.CAT2, LOCATION
    )

    assert efficacy.index.equals(_efficacy_index())
    assert tmrel.index.equals(_demography().index)
    cat2 = efficacy[efficacy.index.get_level_values("parameter") == "cat2"]
    assert np.allclose(
        cat2[metadata.ARTIFACT_COLUMNS].to_numpy(), BASELINE_EFFICACY.to_numpy()
    )


def test_rejects_other_keys(loader: ModuleType) -> None:
    with pytest.raises(ValueError):
        loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.EXPOSURE, LOCATION)
