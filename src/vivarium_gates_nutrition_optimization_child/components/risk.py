import itertools
from typing import Any, Callable, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import RiskExposureDistribution
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
        self.stunting = builder.value.get_value(data_values.PIPELINES.STUNTING_EXPOSURE)
        self.wasting = builder.value.get_value(data_values.PIPELINES.WASTING_EXPOSURE)
        self.distributions = self._get_distributions(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        pass

    def get_distribution_type(self, builder: Builder) -> str:
        pass

    def get_exposure_distribution(self, builder: Builder) -> RiskExposureDistribution:
        pass

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[
                self.propensity_pipeline_name,
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
            dist.setup_component(builder)
        return distributions

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        """Calculate exposures separately for each joint stunting and wasting state and concatenate."""
        if len(index) == 0:
            return pd.Series(
                index=index
            )  # only happens on first time step when there's no simulants
        propensity = self.propensity(index).rename("propensity")
        wasting = self.wasting(index).rename("wasting")
        stunting = self.stunting(index).rename("stunting")
        pop = pd.concat([stunting, wasting, propensity], axis=1)

        exposures = []
        for group, group_df in pop.groupby(["stunting", "wasting"]):
            stunting_category, wasting_category = group
            # update key to not include dot
            wasting_category = "cat25" if wasting_category == "cat2.5" else wasting_category
            distribution = self.distributions[
                f"risk_factor.stunting_{stunting_category}_wasting_{wasting_category}_underweight"
            ]
            exposure = distribution.ppf(group_df["propensity"])
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

    def build_all_lookup_tables(self, builder: Builder) -> None:
        for risk in self.cgf_models:
            rr_data = self.get_relative_risk_data(builder, self.configuration.sub_risks[risk])
            rr_value_columns = None
            if self.is_exposure_categorical:
                rr_data, rr_value_columns = self.process_categorical_data(builder, rr_data)
            self.lookup_tables[f"{risk.name}_relative_risk"] = self.build_lookup_table(
                builder, rr_data, rr_value_columns
            )

        paf_data = self.get_filtered_data(
            builder, self.configuration.data_sources.population_attributable_fraction
        )
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, paf_data
        )

    def get_distribution_type(self, builder: Builder) -> str:
        return self._exposure_distribution_type

    def get_risk_exposure(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            risk: builder.value.get_value(f"{risk.name}.exposure") for risk in self.cgf_models
        }

    def get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            exposures = self.exposure
            for risk in self.cgf_models:
                index_columns = ["index", risk.name]
                rr = self.lookup_tables[f"{risk.name}_relative_risk"](index)
                exposure = exposures[risk](index).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ["value"]
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, "value"].droplevel(risk.name)
                target *= effect
            return target

        return adjust_target
