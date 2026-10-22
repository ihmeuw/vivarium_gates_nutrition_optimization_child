"""Check how ``LBWSGPAFCalculationExposure`` places the PAF cohort on its grid."""

import numpy as np
import pandas as pd
import pytest
from vivarium.gbd_mapping import risk_factors

from vivarium_gates_nutrition_optimization_child.components.lbwsg import (
    LBWSGPAFCalculationExposure,
    _cell_midpoints,
    _parse_category_box,
)

CATEGORIES = {
    "cat2": "Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g",
    "cat31": "Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g",
    "cat38": "Neonatal preterm and LBWSG (estimation years) - [34, 36) wks, [4000, 9999] g",
}
AGE_BINS = pd.DataFrame({"age_start": [0.0, 0.01917808, 0.07671233]})


class RecordingPopulationView:
    """A population view that serves a frame and records what is initialized."""

    def __init__(self, population: pd.DataFrame):
        self.population = population
        self.initialized = None

    def get(self, index, attributes, **kwargs):
        return self.population.loc[index, list(attributes)]

    def initialize(self, data: pd.DataFrame) -> None:
        self.initialized = data


def make_exposure(population: pd.DataFrame) -> LBWSGPAFCalculationExposure:
    exposure = LBWSGPAFCalculationExposure()
    exposure._population_view = RecordingPopulationView(population)
    exposure.lbwsg_categories = CATEGORIES
    exposure.age_bins = AGE_BINS
    return exposure


def make_cohort(points_per_side: int, max_age: float = 0.005) -> pd.DataFrame:
    size = 2 * len(CATEGORIES) * points_per_side**2
    return pd.DataFrame(
        {
            "age": np.linspace(0, max_age, size + 1)[1:],
            "sex": ["Female", "Male"] * (size // 2),
        }
    )


def test_every_category_description_parses_to_its_box():
    for (
        category,
        description,
    ) in risk_factors.low_birth_weight_and_short_gestation.categories.to_dict().items():
        gestational_age, birth_weight = _parse_category_box(description)
        assert gestational_age[0] < gestational_age[1], category
        assert birth_weight[0] < birth_weight[1], category
    assert _parse_category_box(CATEGORIES["cat31"]) == ((40.0, 42.0), (2000.0, 2500.0))
    assert _parse_category_box(CATEGORIES["cat38"]) == ((34.0, 36.0), (4000.0, 9999.0))


def test_cell_midpoints_center_equal_cells():
    np.testing.assert_allclose(_cell_midpoints(0.0, 4.0, 4), [0.5, 1.5, 2.5, 3.5])


def test_each_sex_and_category_gets_the_same_square_number_of_simulants():
    cohort = make_cohort(points_per_side=3)
    exposure = make_exposure(cohort)

    exposure.initialize_category_assignments(pd.Series(index=cohort.index).to_frame())
    assigned = exposure.population_view.initialized.join(cohort)

    counts = assigned.groupby(["sex", "lbwsg_category"]).size()
    assert set(counts) == {9}
    assert len(counts) == 2 * len(CATEGORIES)


def test_population_size_must_give_a_square_grid():
    cohort = make_cohort(points_per_side=3).iloc[: 2 * len(CATEGORIES) * 8]
    exposure = make_exposure(cohort)

    with pytest.raises(ValueError, match="k\\*\\*2"):
        exposure.initialize_category_assignments(pd.Series(index=cohort.index).to_frame())


def test_cohort_must_start_in_one_age_bin():
    cohort = make_cohort(points_per_side=3, max_age=0.03)
    exposure = make_exposure(cohort)

    with pytest.raises(ValueError, match="single age bin"):
        exposure.initialize_category_assignments(pd.Series(index=cohort.index).to_frame())


def test_birth_exposures_sit_on_the_midpoint_grid_of_each_box():
    cohort = make_cohort(points_per_side=3)
    exposure = make_exposure(cohort)
    exposure.initialize_category_assignments(pd.Series(index=cohort.index).to_frame())
    population = cohort.join(exposure.population_view.initialized)
    exposure._population_view = RecordingPopulationView(population)

    values = exposure._get_birth_exposure_source(population.index).join(population)

    for (sex, category), group in values.groupby(["sex", "lbwsg_category"]):
        (ga_lower, ga_upper), (bw_lower, bw_upper) = _parse_category_box(CATEGORIES[category])
        expected = {
            (ga, bw)
            for ga in _cell_midpoints(ga_lower, ga_upper, 3)
            for bw in _cell_midpoints(bw_lower, bw_upper, 3)
        }
        assert set(zip(group["gestational_age"], group["birth_weight"])) == expected
