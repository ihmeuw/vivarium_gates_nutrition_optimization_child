from typing import List, Tuple

from vivarium_gates_nutrition_optimization_child.constants import data_keys


# noinspection PyPep8Naming
class __WastingModel:
    MODEL_NAME: str = data_keys.WASTING.name
    SUSCEPTIBLE_STATE_NAME: str = f"susceptible_to_{MODEL_NAME}"
    MILD_STATE_NAME: str = f"mild_{MODEL_NAME}"
    BETTER_MODERATE_STATE_NAME = "better_moderate_acute_malnutrition"
    WORSE_MODERATE_STATE_NAME = "worse_moderate_acute_malnutrition"
    SEVERE_STATE_NAME = "severe_acute_malnutrition"


###########################
# Disease Model variables #
###########################

WASTING = __WastingModel()


def get_risk_category(state_name: str) -> str:
    return {
        WASTING.SUSCEPTIBLE_STATE_NAME: data_keys.WASTING.CAT4,
        WASTING.MILD_STATE_NAME: data_keys.WASTING.CAT3,
        WASTING.BETTER_MODERATE_STATE_NAME: data_keys.WASTING.CAT25,
        WASTING.WORSE_MODERATE_STATE_NAME: data_keys.WASTING.CAT2,
        WASTING.SEVERE_STATE_NAME: data_keys.WASTING.CAT1,
    }[state_name]
