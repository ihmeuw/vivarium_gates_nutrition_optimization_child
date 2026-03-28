import itertools
from typing import Any, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.utilities import EntityString

from vivarium_gates_nutrition_optimization_child.components import (
    CGFPolytomousDistribution,
)
from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values


class ChildUnderweight(Risk):
    """Model underweight risk in children. We model underweight using probability distributions
    conditional on stunting and wasting exposure. Instead of using a standard exposure distribution,
    the expoure pipeline will determine which distribution to use separately for each joint stunting
    and wasting state."""

    def __init__(self):
        super().__init__("risk_factor.child_underweight")

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.distributions = self._get_distributions(builder)

    def get_distribution_type(self, builder: Builder) -> None:
        pass

    def get_exposure_distribution(self, builder: Builder) -> None:
        pass

    def register_exposure_pipeline(self, builder: Builder) -> Pipeline:
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=self.get_current_exposure,
            required_resources=[
                "age",
                "sex",
                self.propensity_name,
                data_values.PIPELINES.STUNTING_EXPOSURE,
                data_values.PIPELINES.WASTING_EXPOSURE,
            ],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
        )

    def _get_distributions(self, builder: Builder) -> Dict[str, CGFPolytomousDistribution]:
        """Store and setup distributions for each joint stunting and wasting state."""
        distributions = {}
        stunting_categories = [f"cat{i+1}" for i in range(4)]
        wasting_categories = [f"cat{i + 1}" for i in range(4)] + ["cat2.5"]
        all_distribution_data = builder.data.load(data_keys.UNDERWEIGHT.EXPOSURE)

        for stunting_cat, wasting_cat in itertools.product(
            stunting_categories, wasting_categories
        ):
            distribution_data = all_distribution_data[
                (all_distribution_data["stunting_parameter"] == stunting_cat)
                & (all_distribution_data["wasting_parameter"] == wasting_cat)
            ]
            distribution_data = distribution_data.drop(
                ["stunting_parameter", "wasting_parameter"], axis=1
            )

            wasting_cat = wasting_cat.replace(".", "")
            key = f"risk_factor.stunting_{stunting_cat}_wasting_{wasting_cat}_underweight"

            distributions[key] = CGFPolytomousDistribution(
                EntityString(key), distribution_data
            )
        for dist in distributions.values():
            # HACK / FIXME [MIC-6756]
            self._components._manager._current_component = dist
            dist.setup_component(builder)
            self._components._manager._current_component = self
        return distributions

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        """Calculate exposures separately for each joint stunting and wasting state and concatenate."""
        if len(index) == 0:
            # only happens on first time step when there's no simulants
            return pd.Series(index=index)

        pop = self.population_view.get(
            index,
            [
                data_values.PIPELINES.STUNTING_EXPOSURE,
                data_values.PIPELINES.WASTING_EXPOSURE,
                self.propensity_name,
            ],
        ).rename(
            columns={
                data_values.PIPELINES.STUNTING_EXPOSURE: "stunting",
                data_values.PIPELINES.WASTING_EXPOSURE: "wasting",
            }
        )

        exposures = []
        for group, group_df in pop.groupby(["stunting", "wasting"]):
            stunting_category, wasting_category = group
            # update key to not include dot
            wasting_category = "cat25" if wasting_category == "cat2.5" else wasting_category
            distribution = self.distributions[
                f"risk_factor.stunting_{stunting_category}_wasting_{wasting_category}_underweight"
            ]
            exposure = distribution.exposure_ppf(group_df.index)
            exposures.append(exposure)
        return pd.concat(exposures).sort_index()


class CGFRiskEffect(RiskEffect):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:

        sub_risk_configs = {
            risk: {
                "data_sources": {
                    "relative_risk": f"{risk}.relative_risk",
                },
                "data_source_parameters": {
                    "relative_risk": {},
                },
            }
            for risk in self.cgf_models
        }

        config = {
            self.name: {
                "sub_risks": sub_risk_configs,
                "data_sources": {
                    "population_attributable_fraction": f"{self.risk}.population_attributable_fraction",
                },
            },
        }
        return config

    def __init__(self, target: str):
        """
        Parameters
        ----------
        target :
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        super().__init__("risk_factor.child_growth_failure", target)
        self.cgf_models = [
            EntityString(f"risk_factor.{risk}")
            for risk in [
                data_keys.WASTING.name,
                data_keys.UNDERWEIGHT.name,
                data_keys.STUNTING.name,
            ]
        ]
        # This is to access to the distribution type before setup
        self._exposure_distribution_type = "ordered_polytomous"
        # Override relative risk name to include the measure to avoid collisions
        # between instances targeting the same entity with different measures
        self.relative_risk_name = (
            f"{self.risk.name}_on_{self.target.name}.{self.target.measure}.relative_risk"
        )

    def setup(self, builder: Builder) -> None:
        self.sub_exposure_names = {risk: f"{risk.name}.exposure" for risk in self.cgf_models}
        self.sub_risk_rr_tables = {}
        super().setup(builder)

    def build_rr_lookup_table(self, builder) -> None:
        for risk in self.cgf_models:
            rr_data = self.load_relative_risk(builder, self.configuration.sub_risks[risk])
            rr_value_columns = None
            if self.is_exposure_categorical:
                rr_data, rr_value_columns = self.process_categorical_data(builder, rr_data)
            self.sub_risk_rr_tables[risk] = self.build_lookup_table(
                builder, f"{risk.name}_relative_risk", rr_data, rr_value_columns
            )

    def get_distribution_type(self, builder: Builder) -> str:
        return self._exposure_distribution_type

    def register_relative_risk_pipeline(self, builder):
        builder.value.register_attribute_producer(
            self.relative_risk_name,
            source=self._relative_risk_source,
            required_resources=list(self.sub_exposure_names.values()),
        )

    def adjust_target(self, index: pd.Index, target: pd.Series) -> pd.Series:
        exposures = self.population_view.get(index, list(self.sub_exposure_names.values()))
        if index.empty:
            return target

        for risk in self.cgf_models:
            index_columns = ["index", risk.name]
            rr = self.sub_risk_rr_tables[risk](index)
            exposure = exposures[self.sub_exposure_names[risk]].reset_index()
            exposure.columns = index_columns
            exposure = exposure.set_index(index_columns)

            relative_risk = rr.stack().reset_index()
            relative_risk.columns = index_columns + ["value"]
            relative_risk = relative_risk.set_index(index_columns)

            effect = relative_risk.loc[exposure.index, "value"].droplevel(risk.name)
            target *= effect
        return target
