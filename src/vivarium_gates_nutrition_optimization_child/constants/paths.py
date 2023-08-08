from pathlib import Path

import vivarium_gates_nutrition_optimization_child
from vivarium_gates_nutrition_optimization_child.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization_child.__file__).resolve().parent

ARTIFACT_ROOT = BASE_DIR / "artifacts"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RAW_DATA_DIR = BASE_DIR / "data/raw"

MATERNAL_INTERVENTION_COVERAGE_CSV = RAW_DATA_DIR / "simulation_intervention_coverage.csv"

TEMPORARY_PAF_DIR = Path("/share/costeffectiveness/auxiliary_data/GBD_2019/01_original_data/population_attributable_fraction/risk_factor/lbwsg/data/")

