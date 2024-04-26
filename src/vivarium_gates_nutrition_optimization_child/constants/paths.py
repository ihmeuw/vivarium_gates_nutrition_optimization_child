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
SQLNS_RISK_RATIOS = RAW_DATA_DIR / "2021_sqlns_effects_by_location.csv"
WASTING_TRANSITIONS_DATA_DIR = RAW_DATA_DIR / "wasting_transition_rates/2021"
WASTING_TREATMENT_PARAMETERS_DIR = RAW_DATA_DIR / "wasting_treatment_parameters"
WASTING_RELATIVE_RISKS = RAW_DATA_DIR / "wasting_rrs_with_subcategories_no_locations.csv"
PROBABILITIES_OF_WORSE_MAM_EXPOSURE = RAW_DATA_DIR / "worse_exp_frac_no_location.csv"
UNDERWEIGHT_CONDITIONAL_DISTRIBUTIONS = RAW_DATA_DIR / "lookup.csv"
CGF_PAFS = RAW_DATA_DIR / "cgf_pafs.csv"

TEMPORARY_PAF_DIR = Path(
    "/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/lbwsg_pafs/outputs/"
)
