"""Check the calculations the LBWSG PAF outputs are verified against."""

import numpy as np
import pytest
import yaml
from vivarium.gbd_mapping import risk_factors

from vivarium_gates_nutrition_optimization_child.constants import paths
from vivarium_gates_nutrition_optimization_child.data import lbwsg_paf_checks as checks


class ConstantLogRelativeRisk:
    """Stands in for a pickled interpolator with a constant log relative risk."""

    def __init__(self, log_relative_risk: float):
        self.log_relative_risk = log_relative_risk

    def __call__(self, gestational_age, birth_weight, grid=False):
        return np.full(len(gestational_age), self.log_relative_risk)


def test_category_boxes_parse_every_lbwsg_category():
    categories = risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()

    boxes = checks.category_boxes(categories)

    assert len(boxes) == len(categories)
    assert all(lower < upper for box in boxes.values() for lower, upper in [box[:2], box[2:]])
    with pytest.raises(ValueError, match="Could not parse"):
        checks.category_boxes({"cat1": "no numbers here"})


def test_relative_risk_is_one_in_the_tmrel_region():
    gestational_age = np.array([37.9, 38.0, 41.0, 38.0])
    birth_weight = np.array([3600.0, 3499.0, 3500.0, 4000.0])

    relative_risk = checks.relative_risks(
        ConstantLogRelativeRisk(np.log(2.0)), gestational_age, birth_weight
    )

    np.testing.assert_allclose(relative_risk, [2.0, 2.0, 1.0, 1.0])


def test_grid_points_cover_each_box_at_cell_midpoints():
    gestational_age, birth_weight = checks.grid_points(
        {"cat": (30.0, 32.0, 1000.0, 1500.0)}, 2
    )

    assert sorted(zip(gestational_age, birth_weight)) == [
        (30.5, 1125.0),
        (30.5, 1375.0),
        (31.5, 1125.0),
        (31.5, 1375.0),
    ]


def test_early_neonatal_paf_is_one_minus_the_inverse_mean_relative_risk():
    weights = np.array([0.25, 0.75])
    relative_risk = np.array([[3.0, 5.0], [1.0, 1.0]])

    paf = checks.expected_early_neonatal_paf(weights, relative_risk)

    mean_relative_risk = 0.25 * 4.0 + 0.75 * 1.0
    assert paf == pytest.approx(1 - 1 / mean_relative_risk)


def test_late_neonatal_paf_without_deaths_uses_birth_weights():
    weights = np.array([0.25, 0.75])
    early = np.array([[10.0, 10.0], [1.0, 1.0]])
    late = np.array([[3.0, 5.0], [1.0, 1.0]])

    paf, standard_error = checks.expected_late_neonatal_paf(
        weights, early, late, 0.0, 0.0, 0.8, step_days=4, death_steps=2
    )

    assert paf == pytest.approx(checks.expected_early_neonatal_paf(weights, late))
    assert standard_error == 0.0


def test_late_neonatal_expectation_and_error_match_a_simulated_cohort():
    """The expectation and standard error describe the simulated estimator."""
    rng = np.random.default_rng(7)
    weights = np.array([0.1, 0.3, 0.6])
    early = rng.uniform(1.0, 60.0, size=(3, 400))
    late = rng.uniform(1.0, 8.0, size=(3, 400))
    # Early neonatal rates per year like GBD's, so most simulants survive.
    acmr, unmodeled, early_paf, step_days, death_steps = 1.5, 1.3, 0.85, 4, 2

    expected, standard_error = checks.expected_late_neonatal_paf(
        weights, early, late, acmr, unmodeled, early_paf, step_days, death_steps
    )

    hazard = acmr - unmodeled + unmodeled * (1 - early_paf) * early
    survival = (1 - np.clip(hazard * step_days / 365.0, 0, 1)) ** death_steps
    estimates = []
    for _ in range(2000):
        alive = rng.random(survival.shape) < survival
        proportion_alive = alive.mean(axis=1)
        mean_relative_risk = (alive * late).sum(axis=1) / alive.sum(axis=1)
        overall = np.average(mean_relative_risk, weights=weights * proportion_alive)
        estimates.append((overall - 1) / overall)
    estimates = np.array(estimates)

    assert estimates.mean() == pytest.approx(expected, abs=5 * standard_error / np.sqrt(2000))
    assert estimates.std() == pytest.approx(standard_error, rel=0.1)


def test_simulation_settings_follow_the_late_neonatal_specification():
    settings = checks.get_simulation_settings()

    main = yaml.safe_load(
        (paths.MODEL_SPEC_DIR / "nutrition_optimization_child.yaml").read_text()
    )
    assert settings == {
        "population_size": 194_996,
        "step_days": 4,
        "death_steps": 2,
        "start_year": 2025,
        "unmodeled_causes": main["configuration"]["mortality_line_list"]["unmodeled_causes"],
    }


def test_simulation_settings_refuse_a_cohort_leaving_early_neonatal_unevenly(tmp_path):
    specification = yaml.safe_load((paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf.yaml").read_text())
    specification["configuration"]["population"]["initialization_age_max"] = 0.012
    path = tmp_path / "lbwsg_paf.yaml"
    path.write_text(yaml.safe_dump(specification))

    with pytest.raises(ValueError, match="single time step"):
        checks.get_simulation_settings(path)
