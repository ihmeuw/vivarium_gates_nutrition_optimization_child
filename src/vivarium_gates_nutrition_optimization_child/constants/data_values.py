from typing import Dict, NamedTuple, Tuple

from scipy import stats
import pandas as pd

from vivarium_gates_nutrition_optimization_child.constants.metadata import YEAR_DURATION
from vivarium_gates_nutrition_optimization_child.utilities import (
    get_norm_from_quantiles,
    get_lognorm_from_quantiles,
    get_truncnorm_from_quantiles,
    get_truncnorm_from_sd
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


# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
    # Wasting age start (in years)
    START_AGE: float = 0.5

    # Wasting treatment distribution type and categories
    DISTRIBUTION: str = 'ordered_polytomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'Untreated',
        'cat2': 'Baseline treatment',
        'cat3': 'Alternative scenario treatment',
    }

    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    BASELINE_SAM_TX_COVERAGE: Tuple = (
        'sam_tx_coverage', get_norm_from_quantiles(mean=0.488, lower=0.374, upper=0.604)
    )
    BASELINE_MAM_TX_COVERAGE: Tuple = (
        'sam_tx_coverage', get_norm_from_quantiles(mean=0.15, lower=0.1, upper=0.2)
    )
    ALTERNATIVE_TX_COVERAGE: float = 0.7

    # Wasting treatment efficacy
    BASELINE_SAM_TX_EFFICACY: Tuple = (
        'sam_tx_efficacy', get_norm_from_quantiles(mean=0.700, lower=0.64, upper=0.76)
    )
    BASELINE_MAM_TX_EFFICACY: Tuple = (
        'mam_tx_efficacy', get_norm_from_quantiles(mean=0.731, lower=0.585, upper=0.877)
    )
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

    # Incidence correction factor (total exit rate)
    SAM_K: Tuple = (
        'sam_incidence_correction', get_lognorm_from_quantiles(median=6.7, lower=5.3, upper=8.4)
    )
    ALTERNATIVE_SAM_K: Tuple = (
        'alt_sam_incidence_correction', get_lognorm_from_quantiles(median=3.5, lower=3.1, upper=3.9)
    )

    # Untreated time to recovery in days
    MAM_UX_RECOVERY_TIME_OVER_6MO: float = 147.0
    DEFAULT_MILD_WASTING_UX_RECOVERY_TIME: float = 1_000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 62.3 # 48.3 + 14
    MAM_TX_RECOVERY_TIME_OVER_6MO: Tuple = (
        'mam_tx_recovery_time_over_6mo', get_norm_from_quantiles(
            mean=55.3, lower=48.4, upper=63.0
        )
    )
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3

    DIARRHEA_PREVALENCE_RATIO: pd.Series = pd.Series(
        [1.060416, 1.061946, 1.044849],
        index=pd.Index(['cat1', 'cat2', 'cat3'], name='wasting'),
        name='value'
    )

    DIARRHEA_DURATION_VICIOUS_CYCLE: Tuple = (
        'diarrheal_diseases_duration', get_norm_from_quantiles(mean=4.576, lower=4.515, upper=4.646)
    )

    R4_UNDER_12MO: Tuple = (
        'r4_under_12mo', get_truncnorm_from_sd(
            mean=0.006140, sd=0.003015,
        )
    )

    R4_OVER_12MO: Tuple = (
        'r4_over_12mo', get_truncnorm_from_sd(
            mean=0.005043, sd=0.002428
        )
    )


WASTING = __Wasting()



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
    START_AGE: float = 0.5

    # Wasting treatment distribution type and categories
    DISTRIBUTION: str = 'ordered_polytomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'Untreated',
        'cat2': 'Baseline treatment',
        'cat3': 'Alternative scenario treatment',
        'cat4': "Unexposed"
    }

    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    BASELINE_SAM_TX_COVERAGE: Tuple = (
        'sam_tx_coverage', get_norm_from_quantiles(mean=0.488, lower=0.374, upper=0.604)
    )
    BASELINE_MAM_TX_COVERAGE: Tuple = (
        'sam_tx_coverage', get_norm_from_quantiles(mean=0.15, lower=0.1, upper=0.2)
    )
    ALTERNATIVE_TX_COVERAGE: float = 0.7

    # Wasting treatment efficacy
    BASELINE_SAM_TX_EFFICACY: Tuple = (
        'sam_tx_efficacy', get_norm_from_quantiles(mean=0.700, lower=0.64, upper=0.76)
    )
    BASELINE_MAM_TX_EFFICACY: Tuple = (
        'mam_tx_efficacy', get_norm_from_quantiles(mean=0.731, lower=0.585, upper=0.877)
    )
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

    # Incidence correction factor (total exit rate)
    SAM_K: Tuple = (
        'sam_incidence_correction', get_lognorm_from_quantiles(median=6.7, lower=5.3, upper=8.4)
    )
    ALTERNATIVE_SAM_K: Tuple = (
        'alt_sam_incidence_correction', get_lognorm_from_quantiles(median=3.5, lower=3.1, upper=3.9)
    )

    # Untreated time to recovery in days
    MAM_UX_RECOVERY_TIME_OVER_6MO: float = 147.0
    DEFAULT_MILD_WASTING_UX_RECOVERY_TIME: float = 1_000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 62.3 # 48.3 + 14
    MAM_TX_RECOVERY_TIME_OVER_6MO: Tuple = (
        'mam_tx_recovery_time_over_6mo', get_norm_from_quantiles(
            mean=55.3, lower=48.4, upper=63.0
        )
    )
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3

    R4_UNDER_12MO: Tuple = (
        'r4_under_12mo', get_truncnorm_from_sd(
            mean=0.006140, sd=0.003015,
        )
    )

    R4_OVER_12MO: Tuple = (
        'r4_over_12mo', get_truncnorm_from_sd(
            mean=0.005043, sd=0.002428
        )
    )


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
