from pathlib import Path

import vivarium_gates_nutrition_optimization_child
from vivarium_gates_nutrition_optimization_child.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization_child.__file__).resolve().parent
CLUSTER_BASE_DIR = Path(
    "/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/"
)

ARTIFACT_ROOT = BASE_DIR / "artifacts"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RAW_DATA_DIR = BASE_DIR / "data/raw"

BASELINE_IFA_COVERAGE_CSV = RAW_DATA_DIR / "baseline_ifa.csv"

IFA_GA_SHIFT_DATA_DIR = RAW_DATA_DIR / "ifa_gestational_age_shifts"
MMS_GA_SHIFT_1_DATA_DIR = RAW_DATA_DIR / "mms_gestational_age_shifts/shift1"
MMS_GA_SHIFT_2_DATA_DIR = RAW_DATA_DIR / "mms_gestational_age_shifts/shift2"
SQLNS_RISK_RATIOS = RAW_DATA_DIR / "modified_and_standard_subnational_sqlns_effects_v1.csv"
WASTING_TRANSITIONS_DATA_DIR = RAW_DATA_DIR / "wasting_transition_rates/subnational"
WASTING_TREATMENT_PARAMETERS_DIR = RAW_DATA_DIR / "wasting_treatment_parameters"
WASTING_RELATIVE_RISKS = (
    CLUSTER_BASE_DIR / "raw_data/wasting_rrs_with_subcategories_only_locations.csv"
)
PROBABILITIES_OF_WORSE_MAM_EXPOSURE = RAW_DATA_DIR / "worse_exp_frac_only_loc.csv"
UNDERWEIGHT_CONDITIONAL_DISTRIBUTIONS_DIR = CLUSTER_BASE_DIR / "raw_data/underweight_exp/"
CGF_PAFS = CLUSTER_BASE_DIR / "raw_data/cgf_pafs/"

TEMPORARY_PAF_DIR = CLUSTER_BASE_DIR / "lbwsg_pafs/outputs/"
SUBNATIONAL_PERCENTAGES = RAW_DATA_DIR / "subnational_percentages.csv"
