"""Test the MAM treatment relative risk built from the national wasting transition file.

A synthetic file in the shipped layout (unnamed index, lexical draw columns, parameter column
last, shuffled rows, a decoy parameter) drives ``load_mam_treatment_rr`` end to end: every
row's relative risk must follow the research formula from that file's treated and untreated
rates, and a malformed file must raise.
"""
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple

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
AGE_BINS = [
    (0.0, 0.01917808),
    (0.01917808, 0.07671233),
    (0.07671233, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
]
FILE_AGE_BINS = AGE_BINS[2:]  # the file carries no neonatal rows
TX, UX, DECOY = "tx_rem_rate_mam", "ux_rem_rate_mam", "inc_rate_mam"
FILE_DRAWS = 1000
AgeKey = Tuple[str, float, float]


def _rate(parameter: str, sex: str, age_start: float) -> np.ndarray:
    """Return a rate per draw that differs by parameter, sex, age bin and draw."""
    base = {TX: 0.3, UX: 0.02, DECOY: 10.0}[parameter]
    i = [a for a, _ in FILE_AGE_BINS].index(age_start)
    return base + 0.1 * i + 0.05 * SEXES.index(sex) + 0.0001 * np.arange(FILE_DRAWS)


def _write_national_file(
    directory: Path,
    drop_parameter: Optional[str] = None,
    duplicate: Optional[str] = None,
    drop_row: Optional[Tuple[str, AgeKey]] = None,
    nan_cell: Optional[Tuple[str, AgeKey, int]] = None,
    zero_row: Optional[AgeKey] = None,
) -> None:
    """Write the national file in the shipped layout, with one optional defect."""
    records = []
    for parameter in [DECOY, TX, UX]:
        if parameter == drop_parameter:
            continue
        for sex in SEXES:
            for age_start, age_end in FILE_AGE_BINS:
                key = (sex, age_start, age_end)
                if drop_row == (parameter, key):
                    continue
                values = _rate(parameter, sex, age_start).copy()
                if zero_row == key and parameter != DECOY:
                    values[:] = 0.0
                if nan_cell is not None and nan_cell[:2] == (parameter, key):
                    values[nan_cell[2]] = np.nan
                draws = {f"draw_{i}": values[i] for i in range(FILE_DRAWS)}
                for year in [2019, 2020] if parameter == duplicate else [2019]:
                    records.append(
                        dict(
                            sex=sex,
                            age_start=age_start,
                            age_end=age_end,
                            year_start=year,
                            year_end=year + 1,
                            **draws,
                            parameter=parameter,
                        )
                    )
    frame = pd.DataFrame(records).sample(frac=1, random_state=7).reset_index(drop=True)
    frame = frame[
        ["sex", "age_start", "age_end", "year_start", "year_end"]
        + sorted(draws)
        + ["parameter"]
    ]
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{LOCATION.lower()}.csv")


@pytest.fixture
def run_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., pd.DataFrame]:
    """Point the loader at synthetic demography, treatment parameters and a national file."""
    pytest.importorskip("vivarium_inputs", exc_type=ImportError)
    from vivarium_gates_nutrition_optimization_child.data import loader

    demography = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(LOCATION, sex, a, b, 2023, 2024) for sex in SEXES for a, b in AGE_BINS],
            names=metadata.ARTIFACT_INDEX_COLUMNS,
        )
    )
    monkeypatch.setattr(loader, "get_data", lambda key, *_, **__: demography.copy())
    e_mam = pd.DataFrame(
        {f"draw_{i}": [0.5, 0.5, 0.5, 0.6 + 0.0002 * i] for i in range(FILE_DRAWS)},
        index=pd.Index(["c_sam", "e_sam", "c_mam", "e_mam"], name="parameter"),
    )
    e_mam.to_csv(tmp_path / f"{LOCATION.lower()}.csv")
    monkeypatch.setattr(loader.utilities.paths, "WASTING_TREATMENT_PARAMETERS_DIR", tmp_path)
    national = tmp_path / "national"
    monkeypatch.setattr(loader.paths, "WASTING_TRANSITIONS_NATIONAL_DATA_DIR", national)

    def _run(**defect: Any) -> pd.DataFrame:
        _write_national_file(national, **defect)
        return loader.load_mam_treatment_rr(data_keys.MAM_TREATMENT.RELATIVE_RISK, LOCATION)

    return _run


def _expected_rr(index: pd.MultiIndex, zero_row: Optional[AgeKey] = None) -> np.ndarray:
    """Compute the research-table relative risk for every row of the result from the file rates."""
    e_by_category = {
        "cat1": 0.0,
        "cat2": 0.6 + 0.0002 * np.arange(metadata.DRAW_COUNT),
        "cat3": data_values.WASTING.MAM_TX_ALTERNATIVE_EFFICACY,
    }
    rows = []
    for key in index:
        row = dict(zip(index.names, key))
        if row["age_start"] < 0.5:
            rows.append(np.ones(metadata.DRAW_COUNT))
            continue
        tx = _rate(TX, row["sex"], row["age_start"])[: metadata.DRAW_COUNT]
        ux = _rate(UX, row["sex"], row["age_start"])[: metadata.DRAW_COUNT]
        if zero_row == (row["sex"], row["age_start"], row["age_end"]):
            rows.append(np.ones(metadata.DRAW_COUNT))
            continue
        e, e_baseline = e_by_category[row["parameter"]], e_by_category["cat2"]
        rows.append((e * tx + (1 - e) * ux) / (e_baseline * tx + (1 - e_baseline) * ux))
    return np.vstack(rows)


def test_relative_risk_follows_the_file_rates(
    run_loader: Callable[..., pd.DataFrame]
) -> None:
    result = run_loader()

    assert result.index.names == metadata.ARTIFACT_INDEX_COLUMNS + [
        "affected_entity",
        "affected_measure",
        "parameter",
    ]
    assert list(result.columns) == list(metadata.ARTIFACT_COLUMNS)
    assert set(result.index.get_level_values("affected_entity")) == {
        "better_moderate_acute_malnutrition_to_mild_child_wasting",
        "worse_moderate_acute_malnutrition_to_mild_child_wasting",
    }
    assert len(result) == 2 * len(SEXES) * len(AGE_BINS) * 3
    np.testing.assert_allclose(
        result.to_numpy(), _expected_rr(result.index), rtol=1e-12, atol=0
    )


def test_rows_with_zero_rates_get_relative_risk_one(
    run_loader: Callable[..., pd.DataFrame]
) -> None:
    zero_row = ("Male", 1.0, 2.0)
    result = run_loader(zero_row=zero_row)

    np.testing.assert_allclose(
        result.to_numpy(), _expected_rr(result.index, zero_row), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "defect, message",
    [
        ({"drop_parameter": UX}, UX),
        ({"duplicate": TX}, "more than one"),
        ({"drop_row": (TX, ("Male", 1.0, 2.0))}, "no MAM remission rates"),
        ({"nan_cell": (UX, ("Female", 0.07671233, 0.5), 17)}, "missing draws"),
    ],
    ids=["missing_parameter", "duplicate_rows", "missing_age_row", "blank_draw"],
)
def test_malformed_file_raises(
    run_loader: Callable[..., pd.DataFrame], defect: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_loader(**defect)
