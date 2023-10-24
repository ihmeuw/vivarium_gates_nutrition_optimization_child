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
    LOW_BIRTH_WEIGHT_CATEGORIES = [
        "cat10",
        "cat106",
        "cat11",
        "cat116",
        "cat117",
        "cat123",
        "cat124",
        "cat14",
        "cat15",
        "cat17",
        "cat19",
        "cat2",
        "cat20",
        "cat21",
        "cat22",
        "cat23",
        "cat24",
        "cat25",
        "cat26",
        "cat27",
        "cat28",
        "cat29",
        "cat30",
        "cat31",
        "cat32",
        "cat34",
        "cat35",
        "cat36",
        "cat8",
        "cat80",
    ]


LBWSG = __LBWSG()


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
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
    ALTERNATIVE_TX_COVERAGE: float = 0.7

    # Wasting treatment efficacy
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

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
    BEP_UNDERNOURISHED_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bep_birth_weight_shift",
        get_norm_from_quantiles(mean=66.96, lower=13.13, upper=120.78),
    )
    BEP_ADEQUATELY_NOURISHED_BIRTH_WEIGHT_SHIFT: Tuple[str, stats.norm] = (
        "bep_birth_weight_shift",
        get_norm_from_quantiles(mean=15.93, lower=-20.83, upper=52.69),
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

    PROPENSITY_COLUMN = "sq_lns_propensity"
    PROPENSITY_PIPELINE = "sq_lns.propensity"
    COVERAGE_PIPELINE = "sq_lns.coverage"


SQ_LNS = __SQLNS()


##################
# Pipeline names #
##################


class __Pipelines(NamedTuple):
    """value pipeline names"""

    STUNTING_EXPOSURE: str = "child_stunting.exposure"
    WASTING_EXPOSURE: str = "child_wasting.exposure"
    UNDERWEIGHT_EXPOSURE: str = "child_underweight.exposure"

    @property
    def name(self):
        return "pipelines"


PIPELINES = __Pipelines()
