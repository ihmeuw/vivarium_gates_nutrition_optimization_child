"""Keep the LBWSG PAF specifications in step with the main model and the workflow."""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from vivarium.gbd_mapping import risk_factors

from vivarium_gates_nutrition_optimization_child.constants import metadata, paths
from vivarium_gates_nutrition_optimization_child.data import lbwsg_paf_results
from vivarium_gates_nutrition_optimization_child.data.lbwsg_paf_checks import (
    get_simulation_settings,
)

EARLY_NEONATAL_END = 0.01917808
PAF_OVERRIDE = (
    "risk_effect.low_birth_weight_and_short_gestation_on_"
    "cause.affected_unmodeled.cause_specific_mortality_rate"
)


def load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


@pytest.fixture
def main_configuration() -> Dict[str, Any]:
    return load(paths.MODEL_SPEC_DIR / "nutrition_optimization_child.yaml")["configuration"]


@pytest.fixture(params=["lbwsg_paf.yaml", "lbwsg_paf_enn.yaml"])
def paf_configuration(request) -> Dict[str, Any]:
    return load(paths.LBWSG_PAF_CODE_DIR / request.param)["configuration"]


def number_of_steps(configuration: Dict[str, Any]) -> int:
    time = configuration["time"]
    days = (datetime(**time["end"]) - datetime(**time["start"])).days
    return math.ceil(days / time["step_size"])


def test_paf_cause_list_matches_the_main_model(main_configuration, paf_configuration):
    assert (
        paf_configuration["mortality"]["unmodeled_causes"]
        == main_configuration["mortality_line_list"]["unmodeled_causes"]
    )


def test_paf_step_size_matches_the_main_model(main_configuration, paf_configuration):
    assert paf_configuration["time"]["step_size"] == main_configuration["time"]["step_size"]


def test_population_fills_a_square_grid_per_sex_and_category(paf_configuration):
    num_categories = len(
        risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
    )
    per_category, remainder = divmod(
        paf_configuration["population"]["population_size"], 2 * num_categories
    )
    assert remainder == 0
    assert math.isqrt(per_category) ** 2 == per_category


def test_cohort_is_still_early_neonatal_after_one_step(paf_configuration):
    step_years = paf_configuration["time"]["step_size"] / 365.25
    assert paf_configuration["population"]["initialization_age_max"] < (
        EARLY_NEONATAL_END - step_years
    )


def test_passes_run_the_steps_the_method_needs():
    enn = load(paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf_enn.yaml")["configuration"]
    lnn = load(paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf.yaml")["configuration"]

    assert number_of_steps(enn) == 1
    assert number_of_steps(lnn) == get_simulation_settings()["death_steps"] + 1


def test_only_the_early_neonatal_pass_zeroes_the_paf():
    enn = load(paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf_enn.yaml")["configuration"]
    lnn = load(paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf.yaml")["configuration"]

    assert enn[PAF_OVERRIDE] == {"data_sources": {"population_attributable_fraction": 0}}
    assert PAF_OVERRIDE not in lnn
    enn.pop(PAF_OVERRIDE)
    enn["time"].pop("end")
    lnn["time"].pop("end")
    assert enn == lnn


def test_branches_cover_every_artifact_draw_once():
    branches = load(paths.LBWSG_PAF_CODE_DIR / "lbwsg_paf_branches.yaml")

    assert branches["input_draws"] == list(range(metadata.DRAW_COUNT))
    assert len(branches["random_seeds"]) == 1
    assert "branches" not in branches


def test_psimulate_runtime_has_the_format_psimulate_requires():
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", metadata.PAF_SIM_MAX_RUNTIME)


def test_phases_alternate_between_the_two_environments():
    environments = [environment for _, environment in metadata.PAF_PHASES]

    assert environments[0] == environments[-1] == "artifact"
    assert all(first != second for first, second in zip(environments, environments[1:]))


@pytest.mark.parametrize("specification", ["lbwsg_paf.yaml", "lbwsg_paf_enn.yaml"])
def test_observer_target_names_the_tracked_measure(specification):
    components = load(paths.LBWSG_PAF_CODE_DIR / specification)["components"]
    child_components = components["vivarium_gates_nutrition_optimization_child"]["components"]
    observers = [c for c in child_components if c.startswith("LBWSGPAFObserver(")]

    assert observers == [f"LBWSGPAFObserver('{lbwsg_paf_results.LBWSG_PAF_TARGET}')"]


def test_artifact_workflow_parses(tmp_path):
    parsing = pytest.importorskip(
        "vivarium.cluster_tools.dagger.config.parsing", exc_type=ImportError
    )
    text = (paths.MODEL_SPEC_DIR / "artifact_workflow.yaml").read_text()
    for placeholder in re.findall(r"INSERT_[A-Z_]+", text):
        text = text.replace(placeholder, str(tmp_path / placeholder.lower()))
    config_file = tmp_path / "artifact_workflow.yaml"
    config_file.write_text(text)

    parsing.load_workflow_config(config_file)
