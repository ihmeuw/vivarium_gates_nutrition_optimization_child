"""
Component to include birth prevalence in SIS model.
"""

from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState


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
