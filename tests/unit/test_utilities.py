import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.utilities import expand_draws


def _draws(count: int) -> pd.Series:
    """Return draws whose value is their draw number."""
    return pd.Series(np.arange(count, dtype=float), index=[f"draw_{i}" for i in range(count)])


def test_expand_draws_repeats_draws_in_order() -> None:
    expanded = expand_draws(_draws(100), 250)

    assert list(expanded.index) == [f"draw_{i}" for i in range(250)]
    np.testing.assert_array_equal(expanded.to_numpy(), np.arange(250) % 100)


def test_expand_draws_maps_by_name_not_position() -> None:
    draws = _draws(100).sample(frac=1, random_state=0)

    expanded = expand_draws(draws, 250)

    np.testing.assert_array_equal(expanded.to_numpy(), np.arange(250) % 100)


def test_expand_draws_keeps_the_first_draws_when_shrinking() -> None:
    expanded = expand_draws(_draws(500), 250)

    np.testing.assert_array_equal(expanded.to_numpy(), np.arange(250))


def test_expand_draws_rejects_a_gap_in_the_draws() -> None:
    draws = _draws(100).drop("draw_7")

    with pytest.raises(KeyError):
        expand_draws(draws, 250)
