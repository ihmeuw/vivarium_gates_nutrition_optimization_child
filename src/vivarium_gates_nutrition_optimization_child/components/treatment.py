"""Prevention and treatment models"""

from typing import Any, Dict, List, Optional

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


class SQLNSTreatment(Component):
    """Manages SQ-LNS prevention"""

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age"]

    @property
    def columns_created(self) -> Optional[List[str]]:
        return [self.propensity_column_name]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [self._randomness_stream_name],
        }

    def __init__(self):
        super().__init__()
        self._randomness_stream_name = f"initial_{self.name}_propensity"
        self.propensity_column_name = data_values.SQ_LNS.PROPENSITY_COLUMN
        self.propensity_pipeline_name = data_values.SQ_LNS.PROPENSITY_PIPELINE
        self.coverage_pipeline_name = data_values.SQ_LNS.COVERAGE_PIPELINE

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self._randomness_stream_name)

        self.coverage_value = self.get_coverage_value(builder)

        self.propensity = builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=lambda index: self.population_view.get(index)[self.propensity_column_name],
            requires_columns=[self.propensity_column_name],
        )

        self.coverage = builder.value.register_value_producer(
            self.coverage_pipeline_name,
            source=self.get_current_coverage,
            requires_columns=["age"],
            requires_values=[self.propensity_pipeline_name],
        )

        builder.value.register_value_modifier(
            f"{models.WASTING.MILD_STATE_NAME}.incidence_rate",
            modifier=self.apply_tmrel_to_mild_wasting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

        builder.value.register_value_modifier(
            f"{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.BETTER_MODERATE_STATE_NAME}.transition_rate",
            modifier=self.apply_mild_to_mam_wasting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

        builder.value.register_value_modifier(
            f"{models.WASTING.MILD_STATE_NAME}_to_{models.WASTING.WORSE_MODERATE_STATE_NAME}.transition_rate",
            modifier=self.apply_mild_to_mam_wasting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

        builder.value.register_value_modifier(
            f"{models.WASTING.BETTER_MODERATE_STATE_NAME}_to_{models.WASTING.SEVERE_STATE_NAME}.transition_rate",
            modifier=self.apply_mam_to_sam_wasting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

        builder.value.register_value_modifier(
            f"{models.WASTING.WORSE_MODERATE_STATE_NAME}_to_{models.WASTING.SEVERE_STATE_NAME}.transition_rate",
            modifier=self.apply_mam_to_sam_wasting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

        builder.value.register_value_modifier(
            "risk_factor.child_stunting.exposure_parameters",
            modifier=self.apply_stunting_treatment,
            requires_values=[self.coverage_pipeline_name],
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables["tmrel_to_mild_wasting_risk_ratio"] = self.get_risk_ratios(
            builder, "tmrel_to_mild_rate"
        )
        self.lookup_tables["mild_to_mam_wasting_risk_ratio"] = self.get_risk_ratios(
            builder, "mild_to_mam_rate"
        )
        self.lookup_tables["mam_to_sam_wasting_risk_ratio"] = self.get_risk_ratios(
            builder, "mam_to_sam_rate"
        )
        self.lookup_tables["severe_stunting_risk_ratio"] = self.get_risk_ratios(
            builder, "severe_stunting_prevalence_ratio"
        )
        self.lookup_tables["moderate_stunting_risk_ratio"] = self.get_risk_ratios(
            builder, "moderate_stunting_prevalence_ratio"
        )

    def get_coverage_value(self, builder: Builder) -> float:
        scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.child_scenario
        ]
        coverage_map = {
            "baseline": data_values.SQ_LNS.COVERAGE_BASELINE,
            "none": 0,
            "full": 1,
        }
        return coverage_map[scenario.sqlns_coverage]

    def get_risk_ratios(self, builder: Builder, affected_outcome: str) -> LookupTable:
        risk_ratios = builder.data.load(data_keys.SQLNS_TREATMENT.RISK_RATIOS)
        risk_ratios = risk_ratios.query("affected_outcome==@affected_outcome").drop(
            "affected_outcome", axis=1
        )
        return self.build_lookup_table(builder, risk_ratios, value_columns=["value"])

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(
            pd.Series(
                self.randomness.get_draw(pop_data.index), name=self.propensity_column_name
            )
        )

    def get_current_coverage(self, index: pd.Index) -> pd.Series:
        age = self.population_view.get(index)["age"]
        propensity = self.propensity(index)

        coverage = pd.Series("uncovered", index=index)

        covered = (
            (propensity < self.coverage_value)
            & (data_values.SQ_LNS.COVERAGE_START_AGE <= age)
            & (age <= data_values.SQ_LNS.COVERAGE_END_AGE)
        )
        received = (propensity < self.coverage_value) & (
            data_values.SQ_LNS.COVERAGE_END_AGE < age
        )

        coverage[covered] = "covered"
        coverage[received] = "received"

        return coverage

    def apply_tmrel_to_mild_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.coverage(index) == "covered"
        target[covered] = target[covered] * self.lookup_tables[
            "tmrel_to_mild_wasting_risk_ratio"
        ](index)

        return target

    def apply_mild_to_mam_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.coverage(index) == "covered"
        target[covered] = target[covered] * self.lookup_tables[
            "mild_to_mam_wasting_risk_ratio"
        ](index)

        return target

    def apply_mam_to_sam_wasting_treatment(
        self, index: pd.Index, target: pd.Series
    ) -> pd.Series:
        covered = self.coverage(index) == "covered"
        target[covered] = target[covered] * self.lookup_tables[
            "mam_to_sam_wasting_risk_ratio"
        ](index)

        return target

    def apply_stunting_treatment(self, index: pd.Index, target: pd.DataFrame) -> pd.DataFrame:
        cat1_decrease = target.loc[:, "cat1"] * (
            1 - self.lookup_tables["severe_stunting_risk_ratio"](index)
        )
        cat2_decrease = target.loc[:, "cat2"] * (
            1 - self.lookup_tables["moderate_stunting_risk_ratio"](index)
        )

        coverages = self.coverage(index)
        covered = coverages != "uncovered"

        target.loc[covered, "cat1"] = target.loc[covered, "cat1"] - cat1_decrease.loc[covered]
        target.loc[covered, "cat2"] = target.loc[covered, "cat2"] - cat2_decrease.loc[covered]
        target.loc[covered, "cat4"] = (
            target.loc[covered, "cat4"]
            + cat1_decrease.loc[covered]
            + cat2_decrease.loc[covered]
        )

        return target
