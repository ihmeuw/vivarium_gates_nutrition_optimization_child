from typing import NamedTuple

from vivarium_gates_nutrition_optimization_child.constants.data_values import WASTING

#############
# Scenarios #
#############


class InterventionScenario:
    def __init__(
        self,
        name: str,
        sam_tx_coverage: str = "baseline",
        mam_tx_coverage: str = "baseline",
        sqlns_coverage: str = "baseline",
    ):
        self.name = name
        self.sam_tx_coverage = sam_tx_coverage
        self.mam_tx_coverage = mam_tx_coverage
        self.sqlns_coverage = sqlns_coverage


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    SCENARIO_0: InterventionScenario = InterventionScenario(
        "scenario_0_zero_coverage", "none", "none", "none"
    )
    SCENARIO_1: InterventionScenario = InterventionScenario(
        "scenario_1_sam_tx", "full", "none", "none"
    )
    SCENARIO_2: InterventionScenario = InterventionScenario(
        "scenario_2_mam_tx", "none", "full", "none"
    )
    SCENARIO_3: InterventionScenario = InterventionScenario(
        "scenario_3_sqlns", "none", "none", "full"
    )
    # no scenario 4
    SCENARIO_5: InterventionScenario = InterventionScenario(
        "scenario_5_sam_and_mam", "full", "full", "none"
    )
    SCENARIO_6: InterventionScenario = InterventionScenario(
        "scenario_6_sam_and_sqlns", "full", "none", "full"
    )
    SCENARIO_7: InterventionScenario = InterventionScenario(
        "scenario_7_mam_and_sqlns", "none", "full", "full"
    )
    SCENARIO_8: InterventionScenario = InterventionScenario(
        "scenario_8_all", "full", "full", "full"
    )
    SCENARIO_9: InterventionScenario = InterventionScenario(
        "scenario_9_targeted_sqlns", "none", "none", "targeted"
    )
    SCENARIO_10: InterventionScenario = InterventionScenario(
        "scenario_10_targeted_sqlns_sam", "full", "none", "targeted"
    )
    SCENARIO_11: InterventionScenario = InterventionScenario(
        "scenario_11_targeted_sqlns_mam", "none", "full", "targeted"
    )
    SCENARIO_12: InterventionScenario = InterventionScenario(
        "scenario_12_targeted_sqlns_sam_mam", "full", "full", "targeted"
    )
    SCENARIO_13: InterventionScenario = InterventionScenario(
        "scenario_13_targeted_mam", "none", "targeted", "none"
    )
    SCENARIO_14: InterventionScenario = InterventionScenario(
        "scenario_14_sam_and_targeted_mam", "full", "targeted", "none"
    )
    SCENARIO_15: InterventionScenario = InterventionScenario(
        "scenario_15_sqlns_and_targeted_mam", "none", "targeted", "full"
    )
    SCENARIO_16: InterventionScenario = InterventionScenario(
        "scenario_16_sqlns_and_sam_targeted_mam", "full", "targeted", "full"
    )
    SCENARIO_17: InterventionScenario = InterventionScenario(
        "scenario_17_targeted_mam_targeted_sqlns", "none", "targeted", "targeted"
    )
    SCENARIO_18: InterventionScenario = InterventionScenario(
        "scenario_18_targeted_mam_targeted_sqlns_sam", "full", "targeted", "targeted"
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
