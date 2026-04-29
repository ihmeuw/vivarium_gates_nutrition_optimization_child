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
        complicated_sam_tx_type: str = "baseline",
    ):
        self.name = name
        self.sam_tx_coverage = sam_tx_coverage
        self.mam_tx_coverage = mam_tx_coverage
        self.sqlns_coverage = sqlns_coverage
        self.complicated_sam_tx_type = complicated_sam_tx_type


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    SCENARIO_0: InterventionScenario = InterventionScenario(
        "zero_coverage__", "none", "none", "none", "none"
    )
    SCENARIO_1: InterventionScenario = InterventionScenario(
        "targeted_mam_tx__", "none", "targeted", "none", "none"
    )
    SCENARIO_2: InterventionScenario = InterventionScenario(
        "targeted_sqlns__", "none", "none", "targeted", "none"
    )
    SCENARIO_3: InterventionScenario = InterventionScenario(
        "targeted_mam_tx__targeted_sqlns__", "none", "targeted", "targeted", "none"
    )
    SCENARIO_4: InterventionScenario = InterventionScenario(
        "universal_mam_tx__", "none", "full", "none", "none"
    )
    SCENARIO_5: InterventionScenario = InterventionScenario(
        "targeted_sqlns__universal_mam_tx__", "none", "full", "targeted", "none"
    )
    SCENARIO_6: InterventionScenario = InterventionScenario(
        "targeted_mam_tx__universal_sqlns__", "none", "targeted", "full", "none"
    )
    SCENARIO_7: InterventionScenario = InterventionScenario(
        "universal_sqlns__", "none", "none", "full", "none"
    )
    SCENARIO_8: InterventionScenario = InterventionScenario(
        "universal_mam_tx__universal_sqlns__", "none", "full", "full", "none"
    )
    SCENARIO_9: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__", "none", "none", "none", "stabilization"
    )
    SCENARIO_10: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__targeted_mam_tx__",
        "none", "targeted", "none", "stabilization"
    )
    SCENARIO_11: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__targeted_sqlns__",
        "none", "none", "targeted", "stabilization"
    )
    SCENARIO_12: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__targeted_mam_tx__targeted_sqlns__",
        "none", "targeted", "targeted", "stabilization"
    )
    SCENARIO_13: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__universal_mam_tx__",
        "none", "full", "none", "stabilization"
    )
    SCENARIO_14: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__targeted_sqlns__universal_mam_tx__",
        "none", "full", "targeted", "stabilization"
    )
    SCENARIO_15: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__targeted_mam_tx__universal_sqlns__",
        "none", "targeted", "full", "stabilization"
    )
    SCENARIO_16: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__universal_sqlns__",
        "none", "none", "full", "stabilization"
    )
    SCENARIO_17: InterventionScenario = InterventionScenario(
        "complicated_sam_stabilization__universal_mam_tx__universal_sqlns__",
        "none", "full", "full", "stabilization"
    )
    SCENARIO_18: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__", "none", "none", "none", "recovery"
    )
    SCENARIO_19: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__targeted_mam_tx__",
        "none", "targeted", "none", "recovery"
    )
    SCENARIO_20: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__targeted_sqlns__",
        "none", "none", "targeted", "recovery"
    )
    SCENARIO_21: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__targeted_mam_tx__targeted_sqlns__",
        "none", "targeted", "targeted", "recovery"
    )
    SCENARIO_22: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__universal_mam_tx__",
        "none", "full", "none", "recovery"
    )
    SCENARIO_23: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__targeted_sqlns__universal_mam_tx__",
        "none", "full", "targeted", "recovery"
    )
    SCENARIO_24: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__targeted_mam_tx__universal_sqlns__",
        "none", "targeted", "full", "recovery"
    )
    SCENARIO_25: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__universal_sqlns__",
        "none", "none", "full", "recovery"
    )
    SCENARIO_26: InterventionScenario = InterventionScenario(
        "complicated_sam_recovery__universal_mam_tx__universal_sqlns__",
        "none", "full", "full", "recovery"
    )
    SCENARIO_27: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__", "full", "none", "none", "none"
    )
    SCENARIO_28: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__targeted_mam_tx__",
        "full", "targeted", "none", "none"
    )
    SCENARIO_29: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__targeted_sqlns__",
        "full", "none", "targeted", "none"
    )
    SCENARIO_30: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__targeted_mam_tx__targeted_sqlns__",
        "full", "targeted", "targeted", "none"
    )
    SCENARIO_31: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__universal_mam_tx__",
        "full", "full", "none", "none"
    )
    SCENARIO_32: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__targeted_sqlns__universal_mam_tx__",
        "full", "full", "targeted", "none"
    )
    SCENARIO_33: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__targeted_mam_tx__universal_sqlns__",
        "full", "targeted", "full", "none"
    )
    SCENARIO_34: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__universal_sqlns__",
        "full", "none", "full", "none"
    )
    SCENARIO_35: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__universal_mam_tx__universal_sqlns__",
        "full", "full", "full", "none"
    )
    SCENARIO_36: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__",
        "full", "none", "none", "stabilization"
    )
    SCENARIO_37: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__targeted_mam_tx__",
        "full", "targeted", "none", "stabilization"
    )
    SCENARIO_38: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__targeted_sqlns__",
        "full", "none", "targeted", "stabilization"
    )
    SCENARIO_39: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__targeted_mam_tx__targeted_sqlns__",
        "full", "targeted", "targeted", "stabilization"
    )
    SCENARIO_40: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__universal_mam_tx__",
        "full", "full", "none", "stabilization"
    )
    SCENARIO_41: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__targeted_sqlns__universal_mam_tx__",
        "full", "full", "targeted", "stabilization"
    )
    SCENARIO_42: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__targeted_mam_tx__universal_sqlns__",
        "full", "targeted", "full", "stabilization"
    )
    SCENARIO_43: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__universal_sqlns__",
        "full", "none", "full", "stabilization"
    )
    SCENARIO_44: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_stabilization__universal_mam_tx__universal_sqlns__",
        "full", "full", "full", "stabilization"
    )
    SCENARIO_45: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__",
        "full", "none", "none", "recovery"
    )
    SCENARIO_46: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__targeted_mam_tx__",
        "full", "targeted", "none", "recovery"
    )
    SCENARIO_47: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__targeted_sqlns__",
        "full", "none", "targeted", "recovery"
    )
    SCENARIO_48: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__targeted_mam_tx__targeted_sqlns__",
        "full", "targeted", "targeted", "recovery"
    )
    SCENARIO_49: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__universal_mam_tx__",
        "full", "full", "none", "recovery"
    )
    SCENARIO_50: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__targeted_sqlns__universal_mam_tx__",
        "full", "full", "targeted", "recovery"
    )
    SCENARIO_51: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__targeted_mam_tx__universal_sqlns__",
        "full", "targeted", "full", "recovery"
    )
    SCENARIO_52: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__universal_sqlns__",
        "full", "none", "full", "recovery"
    )
    SCENARIO_53: InterventionScenario = InterventionScenario(
        "uncomplicated_sam_tx__complicated_sam_recovery__universal_mam_tx__universal_sqlns__",
        "full", "full", "full", "recovery"
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
