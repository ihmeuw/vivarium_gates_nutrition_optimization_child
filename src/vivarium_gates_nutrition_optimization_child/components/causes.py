"""
Component to include birth prevalence in SIS model.
"""

from vivarium.framework.state_machine import State
from vivarium_public_health.disease import DiseaseModel, DiseaseState
from vivarium_public_health.disease import (
    RiskAttributableDisease as RiskAttributableDisease_,
)
from vivarium_public_health.disease import SusceptibleState


def SIS_with_birth_prevalence(cause: str) -> DiseaseModel:
    with_condition_data_functions = {
        "birth_prevalence": lambda builder, cause: builder.data.load(
            f"cause.{cause}.birth_prevalence"
        )
    }

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, get_data_functions=with_condition_data_functions)

    healthy.allow_self_transitions()
    healthy.add_transition(infected, source_data_type="rate")
    infected.allow_self_transitions()
    infected.add_transition(healthy, source_data_type="rate")

    return DiseaseModel(cause, states=[healthy, infected])


class RiskAttributableDisease(RiskAttributableDisease_):
    """This class has the states attribute so it works with the VPH disease observer"""

    def __init__(self, cause, risk):
        super().__init__(cause, risk)
        self.states = [State(state_name) for state_name in self.state_names]
