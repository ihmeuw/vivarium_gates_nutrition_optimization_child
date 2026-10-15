"""Guard the LBWSG relative-risk category alignment in the interpolation loader.

``load_lbwsg_interpolated_rr`` pivots the RR ``parameter`` column into wide columns, aligns
the birth-weight and gestational-age midpoints from the artifact ``categories`` dict to
those columns, and raises when the two category sets differ. Because the midpoints are
aligned to the columns, the result does not depend on the row or column order of the RR
frame or on the order of the categories dict.
"""
import pickle
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

gbd_mapping = pytest.importorskip("vivarium.gbd_mapping")

from vivarium_gates_nutrition_optimization_child.constants import data_keys, metadata

GUARD_MESSAGE = "LBWSG relative risk categories do not match the category metadata"


def _lbwsg_category_descriptions() -> Dict[str, str]:
    """Return the LBWSG category -> description dict in gbd_mapping order."""
    risk = gbd_mapping.risk_factors.low_birth_weight_and_short_gestation
    return dict(risk.categories.to_dict())


def _lbwsg_categories() -> List[str]:
    """Return the LBWSG category names in gbd_mapping order."""
    return list(_lbwsg_category_descriptions())


def _category_number(category: str) -> int:
    """Return the integer suffix of a category name such as ``cat124``."""
    return int(category.removeprefix("cat"))


def _expected_log_rr(category: str) -> float:
    """Return the log RR the synthetic frame encodes for a category."""
    return float(np.log(1 + _category_number(category)))


def _synthetic_rr(categories: List[str], draw_count: int = 1) -> pd.DataFrame:
    """Build a long-format RR frame with one row per sex and category and RR = 1 + number."""
    rows: List[Dict[str, Any]] = []
    for sex in ["Female", "Male"]:
        for category in categories:
            draws = {f"draw_{i}": 1.0 + _category_number(category) for i in range(draw_count)}
            rows.append(
                {
                    "location": "Ethiopia",
                    "sex": sex,
                    "age_start": 0.0,
                    "age_end": 0.01917808,
                    "year_start": 2021,
                    "year_end": 2022,
                    "parameter": category,
                    **draws,
                }
            )
    draw_columns = [f"draw_{i}" for i in range(draw_count)]
    return pd.DataFrame(rows)[metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"] + draw_columns]


def _permute_rows(rr: pd.DataFrame, row_order: str) -> pd.DataFrame:
    """Return the frame with its rows reversed or with odd rows placed before even rows."""
    if row_order == "reversed":
        positions = list(range(len(rr)))[::-1]
    elif row_order == "interleaved":
        positions = list(range(1, len(rr), 2)) + list(range(0, len(rr), 2))
    else:
        raise ValueError(row_order)
    permuted = rr.iloc[positions]
    assert not permuted.index.equals(rr.index)
    return permuted


@pytest.fixture
def loader() -> ModuleType:
    """Import the data loader, skipping when its GBD dependencies are absent."""
    pytest.importorskip("vivarium_inputs", exc_type=ImportError)
    from vivarium_gates_nutrition_optimization_child.data import loader

    return loader


@pytest.fixture
def install_get_data(
    monkeypatch: pytest.MonkeyPatch, loader: ModuleType
) -> Callable[[Dict[str, str], Optional[str]], None]:
    """Patch loader.get_data to serve a synthetic RR frame and a chosen categories dict."""
    rr = _synthetic_rr(_lbwsg_categories(), metadata.DRAW_COUNT).set_index(
        metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
    )

    def _install(
        category_descriptions: Dict[str, str], row_order: Optional[str] = None
    ) -> None:
        served = rr if row_order is None else _permute_rows(rr, row_order)

        def _get_data(key: Any, *args: Any, **kwargs: Any) -> Any:
            if str(key) == str(data_keys.LBWSG.RELATIVE_RISK):
                return served.copy()
            if str(key) == f"risk_factor.{data_keys.LBWSG.name}.categories":
                return dict(category_descriptions)
            raise KeyError(key)

        monkeypatch.setattr(loader, "get_data", _get_data)

    return _install


def _load_interpolator(loader: ModuleType) -> Any:
    """Run the loader and unpickle the first sex's draw_0 interpolator."""
    result = loader.load_lbwsg_interpolated_rr(
        data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR, "Ethiopia"
    )
    assert len(result) == 2
    assert len(result.columns) == metadata.DRAW_COUNT
    return pickle.loads(bytes.fromhex(result.iloc[0, 0]))


def _assert_interpolator_matches_categories(
    interpolator: Any, descriptions: Dict[str, str], categories: List[str]
) -> None:
    """Evaluate the interpolator at each category's own midpoint and compare to its log RR."""
    from vivarium_gates_nutrition_optimization_child.data.utilities import (
        get_intervals_from_categories,
    )

    gestational_age = get_intervals_from_categories("short_gestation", descriptions)
    birth_weight = get_intervals_from_categories("low_birth_weight", descriptions)
    for category in categories:
        value = interpolator(gestational_age[category].mid, birth_weight[category].mid).item()
        assert np.isclose(value, _expected_log_rr(category), rtol=0.0, atol=1e-9), category


def _probe_categories() -> List[str]:
    """Return two categories with different numbers, the first and last in gbd_mapping order."""
    categories = _lbwsg_categories()
    probes = [categories[0], categories[-1]]
    assert _category_number(probes[0]) != _category_number(probes[1])
    return probes


def test_load_lbwsg_interpolated_rr_accepts_consistent_categories(
    loader: ModuleType, install_get_data: Callable[[Dict[str, str], Optional[str]], None]
) -> None:
    descriptions = _lbwsg_category_descriptions()
    install_get_data(descriptions, None)

    interpolator = _load_interpolator(loader)

    _assert_interpolator_matches_categories(interpolator, descriptions, _probe_categories())


@pytest.mark.parametrize("ordering", ["sorted", "reversed"])
def test_load_lbwsg_interpolated_rr_aligns_reordered_categories(
    ordering: str,
    loader: ModuleType,
    install_get_data: Callable[[Dict[str, str], Optional[str]], None],
) -> None:
    descriptions = _lbwsg_category_descriptions()
    if ordering == "sorted":
        reordered = dict(sorted(descriptions.items()))
    else:
        reordered = dict(reversed(list(descriptions.items())))
    assert list(reordered) != list(descriptions)
    install_get_data(reordered, None)

    interpolator = _load_interpolator(loader)

    _assert_interpolator_matches_categories(interpolator, descriptions, _probe_categories())


@pytest.mark.parametrize("row_order", ["reversed", "interleaved"])
def test_load_lbwsg_interpolated_rr_ignores_rr_row_order(
    row_order: str,
    loader: ModuleType,
    install_get_data: Callable[[Dict[str, str], Optional[str]], None],
) -> None:
    descriptions = _lbwsg_category_descriptions()
    install_get_data(descriptions, row_order)

    interpolator = _load_interpolator(loader)

    _assert_interpolator_matches_categories(interpolator, descriptions, list(descriptions))


@pytest.mark.parametrize("mismatch", ["missing", "extra"])
def test_load_lbwsg_interpolated_rr_rejects_mismatched_categories(
    mismatch: str,
    loader: ModuleType,
    install_get_data: Callable[[Dict[str, str], Optional[str]], None],
) -> None:
    descriptions = _lbwsg_category_descriptions()
    if mismatch == "missing":
        descriptions.pop(next(iter(descriptions)))
    else:
        descriptions["cat999"] = "Birth prevalence - [0, 24) wks, [0, 500) g"
    install_get_data(descriptions, None)

    with pytest.raises(ValueError, match=f"^{GUARD_MESSAGE}"):
        loader.load_lbwsg_interpolated_rr(
            data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR, "Ethiopia"
        )
