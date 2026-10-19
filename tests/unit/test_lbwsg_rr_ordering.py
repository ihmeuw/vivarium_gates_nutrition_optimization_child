"""Guard the LBWSG relative-risk category alignment in the interpolation loader.

``load_lbwsg_interpolated_rr`` aligns the birth-weight and gestational-age midpoints from the
``categories`` dict to the pivoted RR columns and raises when the two category sets differ, so
the result must not depend on the order of the RR rows or of the categories dict.
"""
import pickle
from types import ModuleType
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest

gbd_mapping = pytest.importorskip("vivarium.gbd_mapping")

from vivarium_gates_nutrition_optimization_child.constants import data_keys, metadata

DESCRIPTIONS: Dict[str, str] = dict(
    gbd_mapping.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
)
GUARD_MESSAGE = "LBWSG relative risk categories do not match the category metadata"


def _log_rr(category: str) -> float:
    """Return the log RR the synthetic frame encodes: log(1 + category number)."""
    return float(np.log(1 + int(category.removeprefix("cat"))))


def _synthetic_rr() -> pd.DataFrame:
    """Build a long RR frame, one row per sex and category, two draws, RR = 1 + number."""
    index = pd.MultiIndex.from_product(
        [["Ethiopia"], ["Female", "Male"], [0.0], [0.01917808], [2021], [2022], DESCRIPTIONS],
        names=metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"],
    )
    values = [np.exp(_log_rr(category)) for category in index.get_level_values("parameter")]
    return pd.DataFrame({"draw_0": values, "draw_1": values}, index=index)


@pytest.fixture
def run_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Dict[str, str], pd.DataFrame], Any]:
    """Run the interpolator loader against a categories dict and RR frame; return the result."""
    pytest.importorskip("vivarium_inputs", exc_type=ImportError)
    from vivarium_gates_nutrition_optimization_child.data import loader

    def _run(descriptions: Dict[str, str], rr: pd.DataFrame) -> Any:
        served = {
            str(data_keys.LBWSG.RELATIVE_RISK): rr,
            f"risk_factor.{data_keys.LBWSG.name}.categories": descriptions,
        }
        monkeypatch.setattr(loader, "get_data", lambda key, *_, **__: served[str(key)])
        return loader.load_lbwsg_interpolated_rr(
            data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR, "Ethiopia"
        )

    return _run


PERMUTATIONS = {
    "in_order": (lambda d: d, lambda rr: rr),
    "dict_sorted": (lambda d: dict(sorted(d.items())), lambda rr: rr),
    "dict_reversed": (lambda d: dict(reversed(d.items())), lambda rr: rr),
    "rows_reversed": (lambda d: d, lambda rr: rr.iloc[::-1]),
    "rows_shuffled": (lambda d: d, lambda rr: rr.sample(frac=1, random_state=0)),
}


@pytest.mark.parametrize("permutation", PERMUTATIONS)
def test_every_category_gets_its_own_log_rr(permutation: str, run_loader: Callable) -> None:
    from vivarium_gates_nutrition_optimization_child.data.utilities import (
        get_intervals_from_categories,
    )

    permute_dict, permute_rows = PERMUTATIONS[permutation]
    result = run_loader(permute_dict(DESCRIPTIONS), permute_rows(_synthetic_rr()))

    assert result.shape == (2, 2)
    interpolator = pickle.loads(bytes.fromhex(result.iloc[0, 0]))
    gestational_age = get_intervals_from_categories("short_gestation", DESCRIPTIONS)
    birth_weight = get_intervals_from_categories("low_birth_weight", DESCRIPTIONS)
    for category in DESCRIPTIONS:
        value = interpolator(gestational_age[category].mid, birth_weight[category].mid).item()
        assert np.isclose(value, _log_rr(category), rtol=0.0, atol=1e-9), category


@pytest.mark.parametrize("mismatch", ["missing", "extra"])
def test_mismatched_categories_raise(mismatch: str, run_loader: Callable) -> None:
    descriptions = dict(DESCRIPTIONS)
    if mismatch == "missing":
        descriptions.pop(next(iter(descriptions)))
    else:
        descriptions["cat999"] = "Birth prevalence - [0, 24) wks, [0, 500) g"

    with pytest.raises(ValueError, match=f"^{GUARD_MESSAGE}"):
        run_loader(descriptions, _synthetic_rr())
