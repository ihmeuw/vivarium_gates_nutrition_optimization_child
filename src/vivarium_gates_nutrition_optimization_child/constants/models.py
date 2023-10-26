from typing import List, Tuple

from vivarium_gates_nutrition_optimization_child.constants import data_keys


class TransitionString(str):
    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split("_TO_")
        return obj


# noinspection PyPep8Naming
class __SISModel:
    def __init__(self, model_name: str):
        self.MODEL_NAME = model_name
        self.SUSCEPTIBLE_STATE_NAME: str = f"susceptible_to_{self.MODEL_NAME}"
        self.STATE_NAME: str = self.MODEL_NAME
        self.STATES: Tuple[str, ...] = (self.SUSCEPTIBLE_STATE_NAME, self.STATE_NAME)
        self.TRANSITIONS: Tuple[TransitionString, ...] = (
            TransitionString(f"{self.SUSCEPTIBLE_STATE_NAME}_TO_{self.STATE_NAME}"),
            TransitionString(f"{self.STATE_NAME}_TO_{self.SUSCEPTIBLE_STATE_NAME}"),
        )


# noinspection PyPep8Naming
class __WastingModel:
    MODEL_NAME: str = data_keys.WASTING.name
    SUSCEPTIBLE_STATE_NAME: str = f"susceptible_to_{MODEL_NAME}"
    MILD_STATE_NAME: str = f"mild_{MODEL_NAME}"
    BETTER_MODERATE_STATE_NAME = "better_moderate_acute_malnutrition"
    WORSE_MODERATE_STATE_NAME = "worse_moderate_acute_malnutrition"
    SEVERE_STATE_NAME = "severe_acute_malnutrition"
    STATES: Tuple[str, ...] = (
        SUSCEPTIBLE_STATE_NAME,
        MILD_STATE_NAME,
        BETTER_MODERATE_STATE_NAME,
        WORSE_MODERATE_STATE_NAME,
        SEVERE_STATE_NAME,
    )
    TRANSITIONS: Tuple[TransitionString, ...] = (
        TransitionString(f"{SUSCEPTIBLE_STATE_NAME}_TO_{MILD_STATE_NAME}"),
        TransitionString(f"{MILD_STATE_NAME}_TO_{BETTER_MODERATE_STATE_NAME}"),
        TransitionString(f"{MILD_STATE_NAME}_TO_{WORSE_MODERATE_STATE_NAME}"),
        TransitionString(f"{BETTER_MODERATE_STATE_NAME}_TO_{SEVERE_STATE_NAME}"),
        TransitionString(f"{WORSE_MODERATE_STATE_NAME}_TO_{SEVERE_STATE_NAME}"),
        TransitionString(f"{SEVERE_STATE_NAME}_TO_{BETTER_MODERATE_STATE_NAME}"),
        TransitionString(f"{SEVERE_STATE_NAME}_TO_{WORSE_MODERATE_STATE_NAME}"),
        TransitionString(f"{SEVERE_STATE_NAME}_TO_{MILD_STATE_NAME}"),
        TransitionString(f"{BETTER_MODERATE_STATE_NAME}_TO_{MILD_STATE_NAME}"),
        TransitionString(f"{WORSE_MODERATE_STATE_NAME}_TO_{MILD_STATE_NAME}"),
        TransitionString(f"{MILD_STATE_NAME}_TO_{SUSCEPTIBLE_STATE_NAME}"),
    )


###########################
# Disease Model variables #
###########################

DIARRHEA = __SISModel(data_keys.DIARRHEA.name)
LRI = __SISModel(data_keys.LRI.name)
MEASLES = __SISModel(data_keys.MEASLES.name)
MALARIA = __SISModel(data_keys.MALARIA.name)
MODERATE_PEM = __SISModel(data_keys.MODERATE_PEM.name)
SEVERE_PEM = __SISModel(data_keys.SEVERE_PEM.name)
WASTING = __WastingModel()


def get_risk_category(state_name: str) -> str:
    return {
        WASTING.SUSCEPTIBLE_STATE_NAME: data_keys.WASTING.CAT4,
        WASTING.MILD_STATE_NAME: data_keys.WASTING.CAT3,
        WASTING.BETTER_MODERATE_STATE_NAME: data_keys.WASTING.CAT25,
        WASTING.WORSE_MODERATE_STATE_NAME: data_keys.WASTING.CAT2,
        WASTING.SEVERE_STATE_NAME: data_keys.WASTING.CAT1,
    }[state_name]


CAUSE_MODELS: List[__SISModel] = [DIARRHEA, LRI, MEASLES, MALARIA, MODERATE_PEM, SEVERE_PEM]

STATES = tuple(state for model in CAUSE_MODELS for state in model.STATES)
TRANSITIONS = tuple(state for model in CAUSE_MODELS for state in model.TRANSITIONS)
