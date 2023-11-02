from pathlib import Path

import vivarium_gates_nutrition_optimization_child
from vivarium_gates_nutrition_optimization_child.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization_child.__file__).resolve().parent

ARTIFACT_ROOT = BASE_DIR / "artifacts"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RAW_DATA_DIR = BASE_DIR / "data/raw"

BASELINE_IFA_COVERAGE_CSV = RAW_DATA_DIR / "baseline_ifa.csv"

IFA_GA_SHIFT_DATA_DIR = RAW_DATA_DIR / "ifa_gestational_age_shifts"
MMS_GA_SHIFT_1_DATA_DIR = RAW_DATA_DIR / "mms_gestational_age_shifts/shift1"
MMS_GA_SHIFT_2_DATA_DIR = RAW_DATA_DIR / "mms_gestational_age_shifts/shift2"
SQLNS_RISK_RATIOS = RAW_DATA_DIR / "sqlns_risk_ratios.csv"
WASTING_TRANSITIONS_DATA_DIR = RAW_DATA_DIR / "wasting_transition_rates"
WASTING_TREATMENT_PARAMETERS_DIR = RAW_DATA_DIR / "wasting_treatment_parameters"
UNDERWEIGHT_CONDITIONAL_DISTRIBUTIONS = RAW_DATA_DIR / "lookup.csv"
CGF_PAFS = RAW_DATA_DIR / "cgf_pafs.csv"

TEMPORARY_PAF_DIR = Path(
    "/mnt/team/simulation_science/costeffectiveness/auxiliary_data/GBD_2019/01_original_data/population_attributable_fraction/risk_factor/lbwsg/data/"
)
