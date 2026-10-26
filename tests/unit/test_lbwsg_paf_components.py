from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.components.lbwsg import (
    LBWSGPAFCalculationExposure,
    LBWSGPAFObserver,
)

CATEGORIES = {
    "cat2": "Birth prevalence - [34, 36) wks, [2000, 2500) g",
    "cat56": "Birth prevalence - [40, 42+] wks, [3500, 4000) g",
}
AGE_BINS = pd.DataFrame(
    {
        "age_start": [0.0, 0.01917808, 0.07671233, 0.5],
        "age_end": [0.01917808, 0.07671233, 0.5, 1.0],
    }
)
FOURTEEN_DAYS = 0.03835616


class _PopulationView:
    """Stand in for a population view backed by one DataFrame."""

    def __init__(self, population: pd.DataFrame):
        self.population = population

    def get(self, index: pd.Index, columns: List[str]) -> pd.DataFrame:
        return self.population.loc[index, columns]

    def initialize(self, data: pd.DataFrame) -> None:
        for column in data:
            self.population.loc[data.index, column] = data[column]


def _evenly_distributed_population(size: int) -> pd.DataFrame:
    """Return the ages and sexes EvenlyDistributedPopulation gives a cohort."""
    ages = np.linspace(0, FOURTEEN_DAYS, num=size + 1, endpoint=False)[1:]
    sexes = np.where(np.arange(size) % 2 == 1, "Male", "Female")
    return pd.DataFrame({"age": ages, "sex": sexes})


def test_exposure_places_each_age_bin_sex_and_category_on_an_interior_grid() -> None:
    # 2 sexes x 2 age bins x 2 categories x 3**2 grid points
    population = _evenly_distributed_population(72)
    exposure = LBWSGPAFCalculationExposure()
    exposure.lbwsg_categories = CATEGORIES
    exposure.age_bins = AGE_BINS
    exposure._population_view = _PopulationView(population)

    exposure.initialize_paf_categories(SimpleNamespace(index=population.index))
    birth_exposure = exposure._get_birth_exposure_source(population.index)

    grids = {
        "cat2": {(ga, bw) for ga in [34.5, 35.0, 35.5] for bw in [2125.0, 2250.0, 2375.0]},
        "cat56": {(ga, bw) for ga in [40.5, 41.0, 41.5] for bw in [3625.0, 3750.0, 3875.0]},
    }
    population = population.join(birth_exposure)
    groups = population.groupby(["age_bin", "sex", "lbwsg_category"], observed=True)
    assert groups.ngroups == 8
    for (_, _, category), group in groups:
        points = set(zip(group["gestational_age"], group["birth_weight"]))
        assert len(group) == 9
        assert points == grids[category]


ENN, LNN = (0.0, 0.01917808), (0.01917808, 0.07671233)
# Each stratum's share of cat2; the rest is cat56.
CAT2_SHARE = {
    ("Female", ENN): 0.2,
    ("Male", ENN): 0.4,
    ("Female", LNN): 0.1,
    ("Male", LNN): 0.25,
}


def _observer() -> LBWSGPAFObserver:
    """Return an observer whose exposure differs in every sex and neonatal age group."""
    observer = LBWSGPAFObserver("cause.diarrheal_diseases.excess_mortality_rate")
    observer.risk_effect = SimpleNamespace(relative_risk_name="relative_risk")
    observer.lbwsg_exposure = pd.DataFrame(
        [
            {
                "sex": sex,
                "age_start": ages[0],
                "age_end": ages[1],
                "parameter": category,
                "value": value,
            }
            for (sex, ages), share in CAT2_SHARE.items()
            for category, value in [("cat2", share), ("cat56", 1 - share)]
        ]
    )
    return observer


def _simulants(sexes: List[str], ages: List[float]) -> pd.DataFrame:
    """Return four simulants: two in cat2 with relative risks 2 and 4, two in cat56 with 1."""
    return pd.DataFrame(
        {
            "sex": sexes,
            "age": ages,
            "lbwsg_category": ["cat2", "cat2", "cat56", "cat56"],
            "relative_risk": [2.0, 4.0, 1.0, 1.0],
        }
    )


@pytest.mark.parametrize("sex, ages", list(CAT2_SHARE))
def test_observer_weights_categories_by_the_strata_own_exposure(
    sex: str, ages: tuple
) -> None:
    simulant_ages = [ages[0] + 0.001, ages[0] + 0.002, ages[0] + 0.003, ages[0] + 0.004]

    paf = _observer().calculate_paf(_simulants([sex] * 4, simulant_ages))

    # mean relative risk is 3 in cat2 and 1 in cat56
    mean_rr = 3 * CAT2_SHARE[(sex, ages)] + 1 * (1 - CAT2_SHARE[(sex, ages)])
    assert np.isclose(paf, (mean_rr - 1) / mean_rr)


@pytest.mark.parametrize(
    "sexes, ages",
    [
        (["Female", "Female", "Male", "Male"], [0.005] * 4),
        (["Female"] * 4, [0.005, 0.005, 0.03, 0.03]),
    ],
    ids=["two sexes", "two age groups"],
)
def test_observer_refuses_a_stratum_spanning_exposure_groups(
    sexes: List[str], ages: List[float]
) -> None:
    with pytest.raises(ValueError, match="one sex and one LBWSG exposure age group"):
        _observer().calculate_paf(_simulants(sexes, ages))


def test_observer_returns_zero_for_an_empty_stratum() -> None:
    observer = LBWSGPAFObserver("cause.diarrheal_diseases.excess_mortality_rate")
    observer.risk_effect = SimpleNamespace(relative_risk_name="relative_risk")

    empty = pd.DataFrame(columns=["sex", "age", "lbwsg_category", "relative_risk"])

    assert observer.calculate_paf(empty) == 0.0
