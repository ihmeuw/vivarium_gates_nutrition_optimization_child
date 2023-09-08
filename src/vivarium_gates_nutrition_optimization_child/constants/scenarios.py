from typing import NamedTuple

from vivarium_gates_nutrition_optimization_child.constants.data_values import WASTING

#############
# Scenarios #
#############


class InterventionScenario:
    def __init__(
        self,
        name: str,
        has_alternative_sam_treatment: bool = False,
        has_alternative_mam_treatment: bool = False,
    ):
        self.name = name
        self.has_alternative_sam_treatment = has_alternative_sam_treatment
        self.has_alternative_mam_treatment = has_alternative_mam_treatment


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    SAM_TREATMENT: InterventionScenario = InterventionScenario(
        "sam_treatment",
        has_alternative_sam_treatment=True,
    )
    MAM_TREATMENT: InterventionScenario = InterventionScenario(
        "mam_treatment",
        has_alternative_sam_treatment=True,
        has_alternative_mam_treatment=True,
    )

    def __get_item__(self, item):
        return self._asdict()[item]


INTERVENTION_SCENARIOS = __InterventionScenarios()
