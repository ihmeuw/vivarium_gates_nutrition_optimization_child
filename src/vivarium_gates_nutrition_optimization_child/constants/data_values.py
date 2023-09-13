from typing import Dict, NamedTuple, Tuple

import pandas as pd
from scipy import stats

from vivarium_gates_nutrition_optimization_child.constants.metadata import YEAR_DURATION
from vivarium_gates_nutrition_optimization_child.utilities import (
    get_lognorm_from_quantiles,
    get_norm_from_quantiles,
    get_truncnorm_from_quantiles,
    get_truncnorm_from_sd,
    get_uniform_distribution_from_limits,
)

##########################
# Cause Model Parameters #
##########################

# diarrhea and lower respiratory infection birth prevalence
BIRTH_PREVALENCE_OF_ZERO = 0

# diarrhea duration in days
DIARRHEA_DURATION: Tuple = (
    "diarrheal_diseases_duration",
    get_norm_from_quantiles(mean=4.3, lower=4.3, upper=4.3),
)

# measles duration in days
MEASLES_DURATION: int = 10

# LRI duration in days
LRI_DURATION: Tuple = (
    "lri_duration",
    get_norm_from_quantiles(mean=7.79, lower=6.2, upper=9.64),
)

# malaria duration in days
MALARIA_DURATION: Tuple = (
    "malaria_duration",
    get_uniform_distribution_from_limits(lower_limit=14.0, upper_limit=28.0),
)

# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


##########################
# LBWSG Model Parameters #
##########################
class __LBWSG(NamedTuple):
    STUNTING_EFFECT_PER_GRAM: Tuple[str, stats.norm] = (
        "stunting_effect_per_gram",
        stats.norm(loc=1e-04, scale=3e-05),
    )
    WASTING_EFFECT_PER_GRAM: float = 5.75e-05


LBWSG = __LBWSG()


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
    # Wasting age start (in years)
    DYNAMIC_START_AGE: float = 0.5

    # Wasting treatment distribution type and categories
    DISTRIBUTION: str = "ordered_polytomous"
    CATEGORIES: Dict[str, str] = {
        "cat1": "Untreated",
        "cat2": "Baseline treatment",
        "cat3": "Alternative scenario treatment",
        "cat4": "Unexposed",
    }

    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    BASELINE_SAM_TX_COVERAGE: Tuple = (
        "sam_tx_coverage",
        get_norm_from_quantiles(mean=0.488, lower=0.374, upper=0.604),
    )
    BASELINE_MAM_TX_COVERAGE: Tuple = (
        "sam_tx_coverage",
        get_norm_from_quantiles(mean=0.15, lower=0.1, upper=0.2),
    )
    ALTERNATIVE_TX_COVERAGE: float = 0.7

    # Wasting treatment efficacy
    BASELINE_SAM_TX_EFFICACY: Tuple = (
        "sam_tx_efficacy",
        get_norm_from_quantiles(mean=0.700, lower=0.64, upper=0.76),
    )
    BASELINE_MAM_TX_EFFICACY: Tuple = (
        "mam_tx_efficacy",
        get_norm_from_quantiles(mean=0.731, lower=0.585, upper=0.877),
    )
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

    # Incidence correction factor (total exit rate)
    SAM_K: Tuple = (
        "sam_incidence_correction",
        get_lognorm_from_quantiles(median=6.7, lower=5.3, upper=8.4),
    )
    ALTERNATIVE_SAM_K: Tuple = (
        "alt_sam_incidence_correction",
        get_lognorm_from_quantiles(median=3.5, lower=3.1, upper=3.9),
    )

    # Untreated time to recovery in days
    MAM_UX_RECOVERY_TIME_OVER_6MO: float = 147.0
    DEFAULT_MILD_WASTING_UX_RECOVERY_TIME: float = 1_000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 62.3  # 48.3 + 14
    MAM_TX_RECOVERY_TIME_OVER_6MO: Tuple = (
        "mam_tx_recovery_time_over_6mo",
        get_norm_from_quantiles(mean=55.3, lower=48.4, upper=63.0),
    )
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3

    R4_UNDER_12MO: Tuple = (
        "r4_under_12mo",
        get_truncnorm_from_sd(
            mean=0.006140,
            sd=0.003015,
        ),
    )

    R4_OVER_12MO: Tuple = ("r4_over_12mo", get_truncnorm_from_sd(mean=0.005043, sd=0.002428))


WASTING = __Wasting()


class __MaternalCharacteristics(NamedTuple):
    DISTRIBUTION: str = "dichotomous"
    CATEGORIES: Dict[str, str] = {
        "cat1": "uncovered",
        "cat2": "covered",
    }

    IFA_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "ifa_birth_weight_shift",
        get_norm_from_quantiles(mean=57.73, lower=7.66, upper=107.79),
    )

    BASELINE_MMN_COVERAGE: float = 0.0
    MMN_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "mmn_birth_weight_shift",
        get_norm_from_quantiles(mean=45.16, lower=32.31, upper=58.02),
    )

    BASELINE_BEP_COVERAGE: float = 0.0
    BEP_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bep_birth_weight_shift",
        get_norm_from_quantiles(mean=66.96, lower=13.13, upper=120.78),
    )

    BASELINE_IV_IRON_COVERAGE: float = 0.0
    IV_IRON_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "iv_iron_birth_weight_shift",
        get_norm_from_quantiles(mean=50.0, lower=50.0, upper=50.0),
    )

    BMI_ANEMIA_CAT3_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bmi_anemia_cat3_birth_weight_shift",
        get_norm_from_quantiles(mean=-182.0, lower=-239.0, upper=-125.0),
    )

    BMI_ANEMIA_CAT2_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bmi_anemia_cat2_birth_weight_shift",
        get_norm_from_quantiles(mean=-94.0, lower=-142.0, upper=-46.0),
    )

    BMI_ANEMIA_CAT1_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bmi_anemia_cat1_birth_weight_shift",
        get_norm_from_quantiles(mean=-275.0, lower=-336.0, upper=-213.0),
    )


MATERNAL_CHARACTERISTICS = __MaternalCharacteristics()


class __SQLNS(NamedTuple):
    COVERAGE_START_AGE: float = 0.5
    COVERAGE_END_AGE: float = 1.5
    COVERAGE_BASELINE: float = 0.0
    COVERAGE_RAMP_UP: float = 0.7

    TMREL_TO_MILD_6_to_10_MONTHS: Tuple = (
        'tmrel_to_mild_6_to_10_months', get_lognorm_from_quantiles(median=0.8, lower=0.71, upper=0.93)
    )
    TMREL_TO_MILD_10_to_18_MONTHS: Tuple = (
        'tmrel_to_mild_10_to_18_months', get_lognorm_from_quantiles(median=0.9, lower=0.84, upper=0.96)
    )
    MILD_TO_MAM_6_to_10_MONTHS: Tuple = (
        'mild_to_mam_6_to_10_months', get_lognorm_from_quantiles(median=0.7, lower=0.57, upper=0.88)
    )
    MILD_TO_MAM_10_to_18_MONTHS: Tuple = (
        'mild_to_mam_10_to_18_months', get_lognorm_from_quantiles(median=0.9, lower=0.83, upper=0.97)
    )
    MAM_TO_SAM_6_to_10_MONTHS: Tuple = (
        'mam_to_sam_6_to_10_months', get_lognorm_from_quantiles(median=0.3, lower=0.15, upper=0.68)
    )
    MAM_TO_SAM_10_to_18_MONTHS: Tuple = (
        'mam_to_sam_10_to_18_months', get_lognorm_from_quantiles(median=0.79, lower=0.64, upper=0.895)
    )

    RISK_RATIO_STUNTING_SEVERE: Tuple = (
        'sq_lns_severe_stunting_effect',
        get_lognorm_from_quantiles(median=0.83, lower=0.78, upper=0.90)
    )
    RISK_RATIO_STUNTING_MODERATE: Tuple = (
        'sq_lns_moderate_stunting_effect',
        get_lognorm_from_quantiles(median=0.89, lower=0.86, upper=0.93)
    )


SQ_LNS = __SQLNS()


##################
# Pipeline names #
##################


class __Pipelines(NamedTuple):
    """value pipeline names"""

    STUNTING_EXPOSURE: str = "child_stunting.exposure"
    WASTING_EXPOSURE: str = "child_wasting.exposure"

    @property
    def name(self):
        return "pipelines"


PIPELINES = __Pipelines()
