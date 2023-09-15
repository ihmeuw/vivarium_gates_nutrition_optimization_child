from typing import NamedTuple

from vivarium_gates_nutrition_optimization_child.constants.data_values import WASTING

#############
# Scenarios #
#############


class InterventionScenario:
    def __init__(
        self,
        name: str,
        sam_tx_coverage: str = 'baseline',
        mam_tx_coverage: str = 'baseline',
        sqlns_coverage: str = 'baseline',
    ):
        self.name = name
        self.sam_tx_coverage = sam_tx_coverage
        self.mam_tx_coverage = mam_tx_coverage
        self.sqlns_coverage = sqlns_coverage


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    SCENARIO_0: InterventionScenario = InterventionScenario("scenario_0_zero_coverage", "none", "none", "none")
    SCENARIO_1: InterventionScenario = InterventionScenario("scenario_1_sam_tx", 'full', 'none', 'none')
    SCENARIO_2: InterventionScenario = InterventionScenario("scenario_2_mam_tx", 'none', 'full', 'none')
    SCENARIO_3: InterventionScenario = InterventionScenario("scenario_3_sqlns", 'none', 'none', 'full')
    # no scenario 4
    SCENARIO_5: InterventionScenario = InterventionScenario("scenario_5_sam_and_mam", "full", "full", "none")
    SCENARIO_6: InterventionScenario = InterventionScenario("scenario_6_sam_and_sqlns", "full", "none", "full")
    SCENARIO_7: InterventionScenario = InterventionScenario("scenario_7_mam_and_sqlns", "none", "full", "full")
    SCENARIO_8: InterventionScenario = InterventionScenario("scenario_8_all", "full", "full", "full")

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
