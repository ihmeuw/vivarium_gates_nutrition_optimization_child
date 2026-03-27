"""Prevention and treatment models"""

from typing import Dict, List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    models,
    scenarios,
)
from vivarium_gates_nutrition_optimization_child.constants.paths import (
    SQLNS_TARGETING_GHI,
)


class SQLNSTreatment(Component):
    """Manages SQ-LNS prevention"""

    def __init__(self):
        super().__init__()
        self.propensity_name = data_values.SQ_LNS.PROPENSITY_NAME
        self.coverage_name = data_values.SQ_LNS.COVERAGE_NAME
        self.sqlns_targeted_ghi = pd.read_csv(SQLNS_TARGETING_GHI)[
            ["location", "targeted_ghi"]
        ]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(f"initial_{self.name}_propensity")
        self.scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.child_scenario
        ]
        self.coverage_value = self.get_coverage_value(builder)

        builder.value.register_attribute_producer(
            self.coverage_name,
            source=self.get_current_coverage,
            required_resources=[self.propensity_name, "age", "subnational"],
        )

        builder.value.register_attribute_modifier(
            f"{models.WASTING.MILD_STATE_NAME}.incidence_rate",
            modifier=self.apply_tmrel_to_mild_wasting_treatment,
            required_resources=[self.coverage_name],
        )

        builder.value.register_attribute_modifier(
            f"{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.BETTER_MODERATE_STATE_NAME}.transition_rate",
            modifier=self.apply_mild_to_mam_wasting_treatment,
            required_resources=[self.coverage_name],
        )

        builder.value.register_attribute_modifier(
            f"{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.WORSE_MODERATE_STATE_NAME}.transition_rate",
            modifier=self.apply_mild_to_mam_wasting_treatment,
            required_resources=[self.coverage_name],
        )

        builder.value.register_attribute_modifier(
            f"{models.WASTING.BETTER_MODERATE_STATE_NAME}_to_{models.WASTING.SEVERE_STATE_NAME}.transition_rate",
            modifier=self.apply_mam_to_sam_wasting_treatment,
            required_resources=[self.coverage_name],
        )

        builder.value.register_attribute_modifier(
            f"{models.WASTING.WORSE_MODERATE_STATE_NAME}_to_{models.WASTING.SEVERE_STATE_NAME}.transition_rate",
            modifier=self.apply_mam_to_sam_wasting_treatment,
            required_resources=[self.coverage_name],
        )

        builder.value.register_attribute_modifier(
            "risk_factor.child_stunting.exposure_parameters",
            modifier=self.apply_stunting_treatment,
            required_resources=[self.coverage_name],
        )

        self.tmrel_to_mild_wasting_risk_ratio_table = self.get_risk_ratios(
            builder, "tmrel_to_mild_wasting_risk_ratio", "tmrel_to_mild_rate"
        )
        self.mild_to_mam_wasting_risk_ratio_table = self.get_risk_ratios(
            builder, "mild_to_mam_wasting_risk_ratio", "mild_to_mam_rate"
        )
        self.mam_to_sam_wasting_risk_ratio_table = self.get_risk_ratios(
            builder, "mam_to_sam_wasting_risk_ratio", "mam_to_sam_rate"
        )
        self.severe_stunting_risk_ratio_table = self.get_risk_ratios(
            builder, "severe_stunting_risk_ratio", "severe_stunting_prevalence_ratio"
        )
        self.moderate_stunting_risk_ratio_table = self.get_risk_ratios(
            builder, "moderate_stunting_risk_ratio", "moderate_stunting_prevalence_ratio"
        )

        builder.population.register_initializer(
            self.initialize_propensity,
            self.propensity_name,
            required_resources=[self.randomness],
        )

    def get_coverage_value(self, builder: Builder) -> float:
        coverage_map = {
            "baseline": data_values.SQ_LNS.COVERAGE_BASELINE,
            "targeted": 1,
            "none": 0,
            "full": 1,
        }
        return coverage_map[self.scenario.sqlns_coverage]

    def get_risk_ratios(
        self, builder: Builder, table_name: str, affected_outcome: str
    ) -> LookupTable:
        sqlns_effect_size = builder.configuration.intervention.sqlns_effect_size
        risk_ratios = builder.data.load(data_keys.SQLNS_TREATMENT.RISK_RATIOS)
        risk_ratios = risk_ratios.query("affected_outcome==@affected_outcome").drop(
            "affected_outcome", axis=1
        )
        risk_ratios = risk_ratios.query("effect_size==@sqlns_effect_size").drop(
            "effect_size", axis=1
        )
        return self.build_lookup_table(
            builder, table_name, risk_ratios, value_columns="value"
        )

    def initialize_propensity(self, pop_data):
        self.population_view.update(
            pd.Series(self.randomness.get_draw(pop_data.index), name=self.propensity_name)
        )

    def get_current_coverage(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get_attributes(index, ["age", "subnational"])
        propensity = self.population_view.get_attributes(index, self.propensity_name)
        # Targeted ghi will be yes if we want to have similuants covered for specific subnationals
        ghi = pd.Series(
            data=self.sqlns_targeted_ghi["targeted_ghi"].to_list(),
            index=self.sqlns_targeted_ghi["location"].to_list(),
        ).to_dict()
        pop["targeted_ghi"] = pop["subnational"].map(ghi)

        coverage = pd.Series("uncovered", index=index)

        covered = (
            (propensity < self.coverage_value)
            & (data_values.SQ_LNS.COVERAGE_START_AGE <= pop["age"])
            & (pop["age"] <= data_values.SQ_LNS.COVERAGE_END_AGE)
        )
        if self.scenario.sqlns_coverage == "targeted":
            covered = covered & (pop["targeted_ghi"] == "yes")
        received = (propensity < self.coverage_value) & (
            data_values.SQ_LNS.COVERAGE_END_AGE < pop["age"]
        )

        coverage[covered] = "covered"
        coverage[received] = "received"

        return coverage

    def apply_tmrel_to_mild_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.population_view.get_attributes(index, self.coverage_name) == "covered"
        target[covered] = target[covered] * self.tmrel_to_mild_wasting_risk_ratio_table(index)

        return target

    def apply_mild_to_mam_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.population_view.get_attributes(index, self.coverage_name) == "covered"
        target[covered] = target[covered] * self.mild_to_mam_wasting_risk_ratio_table(index)

        return target

    def apply_mam_to_sam_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.population_view.get_attributes(index, self.coverage_name) == "covered"
        target[covered] = target[covered] * self.mam_to_sam_wasting_risk_ratio_table(index)

        return target

    def apply_stunting_treatment(self, index: pd.Index, target: pd.DataFrame) -> pd.DataFrame:
        cat1_decrease = target.loc[:, "cat1"] * (
            1 - self.severe_stunting_risk_ratio_table(index)
        )
        cat2_decrease = target.loc[:, "cat2"] * (
            1 - self.moderate_stunting_risk_ratio_table(index)
        )

        coverages = self.population_view.get_attributes(index, self.coverage_name)
        covered = coverages != "uncovered"

        target.loc[covered, "cat1"] = target.loc[covered, "cat1"] - cat1_decrease.loc[covered]
        target.loc[covered, "cat2"] = target.loc[covered, "cat2"] - cat2_decrease.loc[covered]
        target.loc[covered, "cat4"] = (
            target.loc[covered, "cat4"]
            + cat1_decrease.loc[covered]
            + cat2_decrease.loc[covered]
        )

        return target
