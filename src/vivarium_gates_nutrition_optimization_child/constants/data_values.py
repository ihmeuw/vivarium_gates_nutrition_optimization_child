from typing import Dict, NamedTuple, Tuple

from scipy import stats

from vivarium_gates_nutrition_optimization_child.utilities import (
    get_norm_from_quantiles,
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
