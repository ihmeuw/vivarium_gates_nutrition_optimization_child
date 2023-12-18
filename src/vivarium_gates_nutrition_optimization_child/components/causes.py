"""
Component to include birth prevalence in SIS model.
"""
import pandas as pd
from vivarium.framework.state_machine import State
from vivarium_public_health.disease import DiseaseModel as DiseaseModel_
from vivarium_public_health.disease import DiseaseState
from vivarium_public_health.disease import (
    RiskAttributableDisease as RiskAttributableDisease_,
)
from vivarium_public_health.disease import SusceptibleState
from vivarium_public_health.disease.transition import TransitionString


class DiseaseModel(DiseaseModel_):
    @property
    def columns_required(self):
        return super.columns_required() + ["alive", "tracked"]

    def setup(self, builder):
        super().setup(builder)
        if "variable_step_sizes" in self._get_data_functions:
            for state, step_size in self._get_data_functions["variable_step_sizes"].items():
                builder.time.register_step_modifier(
                    lambda index: self.modify_step(state, step_size, index)
                )

    def modify_step(self, state, step_size, index):
        infected = self.population_view.get(
            index,
            f"{self.state_column} == '{state}' and alive == 'alive' and tracked == True",
        ).index
        return pd.Series(pd.Timedelta(days=step_size), index=infected)


def SIS(cause: str, step_size_days: str = None) -> DiseaseModel:
    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(cause, allow_self_transition=True)

    healthy.add_rate_transition(infected)
    infected.add_rate_transition(healthy)

    infected_step_size_function = (
        {"variable_step_sizes": {infected.state_id: float(step_size_days)}}
        if step_size_days
        else {}
    )

    return DiseaseModel(
        cause, states=[healthy, infected], get_data_functions=infected_step_size_function
    )


def SIS_fixed_duration(cause: str, duration: str, step_size_days: str = None) -> DiseaseModel:
    duration = pd.Timedelta(days=float(duration) // 1, hours=(float(duration) % 1) * 24.0)

    healthy = SusceptibleState(cause, allow_self_transition=True)
    infected = DiseaseState(
        cause,
        get_data_functions={"dwell_time": lambda _, __: duration},
        allow_self_transition=True,
    )

    healthy.add_rate_transition(infected)
    infected.add_dwell_time_transition(healthy)

    infected_step_size_function = (
        {"variable_step_sizes": {infected.state_id: float(step_size_days)}}
        if step_size_days
        else {}
    )

    return DiseaseModel(
        cause, states=[healthy, infected], get_data_functions=infected_step_size_function
    )


def SIS_with_birth_prevalence(cause: str, step_size_days: str = None) -> DiseaseModel:
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
    infected_step_size_function = (
        {"variable_step_sizes": {infected.state_id: float(step_size_days)}}
        if step_size_days
        else {}
    )

    return DiseaseModel(
        cause, states=[healthy, infected], get_data_functions=infected_step_size_function
    )


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
