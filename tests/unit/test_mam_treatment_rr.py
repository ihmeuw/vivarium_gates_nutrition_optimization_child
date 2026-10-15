"""Verify the MAM treatment relative risk built from data-backed remission rates.

Covers ``get_mam_remission_rates`` reading treated and untreated rates from the national
wasting-transition file, and the rate-form relative risk ``load_mam_treatment_rr`` builds
from them, against small synthetic demography, rate, and treatment-parameter inputs.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Tuple

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
# The national file carries no neonatal rows; its years differ from the artifact's.
FILE_AGE_BINS = AGE_BINS[2:]
FILE_YEAR_START, FILE_YEAR_END = 2019, 2020
FILE_DRAW_COUNT = 1000
TX_PARAMETER = "tx_rem_rate_mam"
UX_PARAMETER = "ux_rem_rate_mam"
# Parameters the loader must ignore; they carry deliberately different values.
DECOY_PARAMETERS = ["tx_rem_rate_sam", "ux_rem_rate_sam", "inc_rate_mam"]
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
AgeKey = Tuple[str, float, float]


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


def _age_key(key: Tuple[Any, ...]) -> AgeKey:
    """Return the (sex, age_start, age_end) part of an artifact index key."""
    names = metadata.ARTIFACT_INDEX_COLUMNS
    return (
        key[names.index("sex")],
        key[names.index("age_start")],
        key[names.index("age_end")],
    )


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


# ---------------------------------------------------------------------------
# Synthetic national wasting-transition file
# ---------------------------------------------------------------------------


def _file_rates(parameter: str) -> Dict[AgeKey, np.ndarray]:
    """Build 1000-draw rates per (sex, age) that vary by sex, age, draw and parameter."""
    draws = np.arange(FILE_DRAW_COUNT)
    if parameter == TX_PARAMETER:
        base, age_step, sex_step, draw_step = 0.3, 0.1, 0.05, 0.0001
    elif parameter == UX_PARAMETER:
        base, age_step, sex_step, draw_step = 0.02, 0.01, 0.005, 0.00001
    else:
        offset = 10 * (DECOY_PARAMETERS.index(parameter) + 1)
        base, age_step, sex_step, draw_step = offset, 1.0, 0.5, 0.001
    return {
        (sex, age_start, age_end): base + age_step * i + sex_step * j + draw_step * draws
        for j, sex in enumerate(SEXES)
        for i, (age_start, age_end) in enumerate(FILE_AGE_BINS)
    }


def _expected_file_rates(parameter: str) -> Dict[AgeKey, np.ndarray]:
    """Return the first DRAW_COUNT draws of the file rates per (sex, age)."""
    return {
        key: values[: metadata.DRAW_COUNT] for key, values in _file_rates(parameter).items()
    }


def _write_national_file(
    directory: Path,
    parameters: Iterable[str],
    drop_row: Optional[Tuple[str, AgeKey]] = None,
    nan_cell: Optional[Tuple[str, AgeKey, int]] = None,
) -> Path:
    """Write a national transition-rate CSV laid out like the shipped data file.

    Mirror the real file: an unnamed integer index, draw columns in lexical order, the
    parameter column last, and rows in no particular order. ``drop_row`` omits one
    (parameter, (sex, age_start, age_end)) row; ``nan_cell`` blanks one draw of one row.
    """
    draw_columns = sorted(f"draw_{i}" for i in range(FILE_DRAW_COUNT))
    records = []
    for parameter in parameters:
        for (sex, age_start, age_end), values in _file_rates(parameter).items():
            if drop_row == (parameter, (sex, age_start, age_end)):
                continue
            record = {
                "sex": sex,
                "age_start": age_start,
                "age_end": age_end,
                "year_start": FILE_YEAR_START,
                "year_end": FILE_YEAR_END,
            }
            record.update({f"draw_{i}": values[i] for i in range(FILE_DRAW_COUNT)})
            if nan_cell is not None and nan_cell[:2] == (
                parameter,
                (sex, age_start, age_end),
            ):
                record[f"draw_{nan_cell[2]}"] = np.nan
            record["parameter"] = parameter
            records.append(record)
    frame = pd.DataFrame(records)
    frame = frame[
        ["sex", "age_start", "age_end", "year_start", "year_end"]
        + draw_columns
        + ["parameter"]
    ]
    frame = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{LOCATION.lower()}.csv"
    frame.to_csv(path)
    return path


@pytest.fixture
def national_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: ModuleType
) -> Path:
    """Point the loader's national transition-rate directory at a temp directory."""
    directory = tmp_path / "wasting_transition_rates"
    monkeypatch.setattr(loader.paths, "WASTING_TRANSITIONS_NATIONAL_DATA_DIR", directory)
    return directory


@pytest.fixture
def national_rates(national_directory: Path) -> Dict[str, Any]:
    """Write the full national file; return the demography-level rates it implies."""
    _write_national_file(national_directory, DECOY_PARAMETERS + [TX_PARAMETER, UX_PARAMETER])
    return {
        "tx": _demography_rates(_expected_file_rates(TX_PARAMETER)),
        "ux": _demography_rates(_expected_file_rates(UX_PARAMETER)),
    }


def _demography_rates(file_rates: Dict[AgeKey, np.ndarray]) -> pd.DataFrame:
    """Lay file rates onto the demography index, zero where the file has no row."""
    index = _demography().index
    rows = [file_rates.get(_age_key(key), np.zeros(metadata.DRAW_COUNT)) for key in index]
    return pd.DataFrame(np.vstack(rows), index=index, columns=metadata.ARTIFACT_COLUMNS)


# ---------------------------------------------------------------------------
# Shared expectations
# ---------------------------------------------------------------------------


@pytest.fixture(params=["patched_helper", "national_file"])
def rates(request: pytest.FixtureRequest) -> Dict[str, Any]:
    """Provide demography-level rates from either the patched helper or the data file."""
    return request.getfixturevalue(
        {"patched_helper": "patched_rates", "national_file": "national_rates"}[request.param]
    )


@pytest.fixture
def relative_risk(
    loader: ModuleType,
    treatment_parameters: Path,
    demography: pd.DataFrame,
    rates: Dict[str, Any],
) -> pd.DataFrame:
    """Run the loader against the active rate source."""
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
# constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "MAM_UX_RECOVERY_TIME_OVER_6MO",
        "MAM_TX_RECOVERY_TIME_OVER_6MO",
        "MAM_TX_RECOVERY_TIME_UNDER_6MO",
    ],
)
def test_mam_recovery_time_constants_are_gone(name: str) -> None:
    assert not hasattr(data_values.WASTING, name)


@pytest.mark.parametrize(
    "name", ["SAM_TX_ALTERNATIVE_EFFICACY", "MAM_TX_ALTERNATIVE_EFFICACY"]
)
def test_alternative_treatment_efficacy_is_one(name: str) -> None:
    assert getattr(data_values.WASTING, name) == 1.0


# ---------------------------------------------------------------------------
# get_mam_remission_rates
# ---------------------------------------------------------------------------


@pytest.fixture
def remission_rates(loader: ModuleType, national_rates: Dict[str, Any]) -> RateFrames:
    """Call the helper on the efficacy-shaped index against the national file."""
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


@pytest.mark.parametrize("which, parameter", [(0, TX_PARAMETER), (1, UX_PARAMETER)])
def test_remission_rates_match_file_by_sex_and_age(
    remission_rates: RateFrames, which: int, parameter: str
) -> None:
    frame = remission_rates[which]
    expected = _expected_file_rates(parameter)
    covered = frame[_age_start(frame.index) >= FILE_AGE_BINS[0][0]]

    assert len(covered) == 2 * len(FILE_AGE_BINS) * len(PARAMETERS)
    for key, row in covered.iterrows():
        assert np.allclose(row.to_numpy(), expected[_age_key(key)], **TOLERANCE), key


def test_remission_rates_vary_by_sex_age_and_draw(remission_rates: RateFrames) -> None:
    for frame in remission_rates:
        covered = frame[_age_start(frame.index) >= FILE_AGE_BINS[0][0]]
        by_cell = covered.droplevel("parameter")

        assert by_cell.index.nunique() == 2 * len(FILE_AGE_BINS)
        assert by_cell.drop_duplicates().shape[0] == 2 * len(FILE_AGE_BINS)
        assert (covered.nunique(axis=1) == metadata.DRAW_COUNT).all()


def test_remission_rates_are_zero_for_neonatal_rows(remission_rates: RateFrames) -> None:
    for frame in remission_rates:
        neonatal = frame[_age_start(frame.index) < FILE_AGE_BINS[0][0]]

        assert sorted(set(neonatal.index.get_level_values("age_start"))) == [
            AGE_BINS[0][0],
            AGE_BINS[1][0],
        ]
        assert len(neonatal) == 2 * 2 * len(PARAMETERS)
        assert (neonatal.to_numpy() == 0.0).all()


def test_remission_rates_missing_parameter_raises(
    loader: ModuleType, national_directory: Path
) -> None:
    _write_national_file(national_directory, DECOY_PARAMETERS + [TX_PARAMETER])

    with pytest.raises(ValueError, match=UX_PARAMETER):
        loader.get_mam_remission_rates(_efficacy_index(), LOCATION)


@pytest.mark.parametrize(
    "parameter, other", [(TX_PARAMETER, UX_PARAMETER), (UX_PARAMETER, TX_PARAMETER)]
)
def test_remission_rates_missing_age_row_raises_naming_only_that_parameter(
    loader: ModuleType, national_directory: Path, parameter: str, other: str
) -> None:
    _write_national_file(
        national_directory,
        DECOY_PARAMETERS + [TX_PARAMETER, UX_PARAMETER],
        drop_row=(parameter, ("Male", 1.0, 2.0)),
    )

    with pytest.raises(ValueError, match=parameter) as excinfo:
        loader.get_mam_remission_rates(_efficacy_index(), LOCATION)
    assert "Male" in str(excinfo.value)
    assert other not in str(excinfo.value)


@pytest.mark.parametrize("parameter", [TX_PARAMETER, UX_PARAMETER])
@pytest.mark.parametrize("age_key", [("Female", 0.07671233, 0.5), ("Male", 1.0, 2.0)])
def test_remission_rates_nan_draw_in_present_row_raises(
    loader: ModuleType, national_directory: Path, parameter: str, age_key: AgeKey
) -> None:
    # Draw 17 is inside the kept DRAW_COUNT; a NaN there must not be masked or filled.
    path = _write_national_file(
        national_directory,
        DECOY_PARAMETERS + [TX_PARAMETER, UX_PARAMETER],
        nan_cell=(parameter, age_key, 17),
    )

    with pytest.raises(ValueError, match=parameter) as excinfo:
        loader.get_mam_remission_rates(_efficacy_index(), LOCATION)
    message = str(excinfo.value)
    assert path.name in message
    assert age_key[0] in message
    assert str(age_key[1]) in message
    assert str(age_key[2]) in message


# ---------------------------------------------------------------------------
# load_mam_treatment_rr
# ---------------------------------------------------------------------------


def test_loader_requests_rates_on_the_efficacy_index(
    loader: ModuleType,
    treatment_parameters: Path,
    demography: pd.DataFrame,
    patched_rates: Dict[str, Any],
) -> None:
    loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.RELATIVE_RISK, LOCATION)
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
    relative_risk: pd.DataFrame, rates: Dict[str, Any]
) -> None:
    expected = _expected_rr(relative_risk, rates["tx"], rates["ux"])

    pd.testing.assert_frame_equal(relative_risk, expected, check_dtype=False, **TOLERANCE)


def test_uncovered_relative_risk_is_untreated_over_mixed_rate(
    relative_risk: pd.DataFrame, rates: Dict[str, Any]
) -> None:
    cat1 = _select(relative_risk, parameter="cat1")
    older = cat1[_age_start(cat1.index) >= SIX_MONTHS]
    e2 = BASELINE_EFFICACY.to_numpy()

    assert len(older) == 2 * 3 * len(AFFECTED_ENTITIES)
    for key, row in older.iterrows():
        demographic = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)]
        tx = rates["tx"].loc[demographic].to_numpy()
        ux = rates["ux"].loc[demographic].to_numpy()
        expected = ux / (tx * e2 + ux * (1 - e2))
        assert np.allclose(row.to_numpy(), expected, **TOLERANCE)
        assert (row.to_numpy() < 1).all()


def test_baseline_relative_risk_is_exactly_one(relative_risk: pd.DataFrame) -> None:
    cat2 = _select(relative_risk, parameter="cat2")

    assert (cat2.to_numpy() == 1.0).all()


def test_alternative_relative_risk_is_treated_over_mixed_rate(
    relative_risk: pd.DataFrame, rates: Dict[str, Any]
) -> None:
    cat3 = _select(relative_risk, parameter="cat3")
    older = cat3[_age_start(cat3.index) >= SIX_MONTHS]
    e2 = BASELINE_EFFICACY.to_numpy()

    assert len(older) == 2 * 3 * len(AFFECTED_ENTITIES)
    for key, row in older.iterrows():
        demographic = key[: len(metadata.ARTIFACT_INDEX_COLUMNS)]
        tx = rates["tx"].loc[demographic].to_numpy()
        ux = rates["ux"].loc[demographic].to_numpy()
        expected = tx / (e2 * tx + (1 - e2) * ux)
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


def test_relative_risk_is_one_under_six_months(
    relative_risk: pd.DataFrame, rates: Dict[str, Any]
) -> None:
    under = relative_risk[_age_start(relative_risk.index) < SIX_MONTHS]
    tx_under = rates["tx"][_age_start(rates["tx"].index) < SIX_MONTHS]

    assert sorted(set(under.index.get_level_values("age_start"))) == [
        age_start for age_start, _ in AGE_BINS[:3]
    ]
    assert len(under) == 2 * 3 * 3 * len(AFFECTED_ENTITIES)
    # The 1-5 month bin has non-zero rates in every source, so the floor is not a fluke.
    assert (tx_under[_age_start(tx_under.index) == AGE_BINS[2][0]].to_numpy() > 0).all()
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
