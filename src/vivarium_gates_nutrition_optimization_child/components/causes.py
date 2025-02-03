"""
Component to include birth prevalence in SIS model.
"""

from vivarium.framework.state_machine import State
from vivarium_public_health.disease import (
    DiseaseModel,
    DiseaseState,
)
from vivarium_public_health.disease import (
    RiskAttributableDisease as RiskAttributableDisease_,
)
from vivarium_public_health.disease import (
    SusceptibleState,
)
from vivarium_public_health.disease.transition import TransitionString


def SIS_with_birth_prevalence(cause: str) -> DiseaseModel:
    with_condition_data_functions = {
        "birth_prevalence": lambda builder, cause: builder.data.load(
            f"cause.{cause}.birth_prevalence"
        )
    }

    healthy = SusceptibleState(cause)
    infected = DiseaseState(cause, get_data_functions=with_condition_data_functions)

    healthy.allow_self_transitions()
    healthy.add_rate_transition(infected)
    infected.allow_self_transitions()
    infected.add_rate_transition(healthy)

    return DiseaseModel(cause, states=[healthy, infected])


class RiskAttributableDisease(RiskAttributableDisease_):
    """This class has the states attribute so it works with the VPH disease observer
    and adds the infected to susceptible transition in the __init__ so the disease observer
    registers this transition during its setup."""

    def __init__(self, cause, risk):
        super().__init__(cause, risk)
        self.states = [State(state_name) for state_name in self.state_names]
        self._transition_names.append(
            TransitionString(f"{self.cause.name}_TO_susceptible_to_{self.cause.name}")
        )

    def adjust_state_and_transitions(self):
        # we would normally add the transition here which is no longer required
        pass
