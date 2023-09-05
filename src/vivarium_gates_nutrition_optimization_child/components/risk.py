from functools import partial, update_wrapper
import itertools
from typing import Callable, Dict

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    pivot_categorical,
    get_relative_risk_data,
)
from vivarium_public_health.utilities import EntityString, TargetString
from vivarium_public_health.risks.distributions import PolytomousDistribution

from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values


class ChildUnderweight(Risk):
    """Model underweight risk in children. We model underweight using probability distributions
    conditional on stunting and wasting exposure. Instead of using a standard exposure distribution,
    the expoure pipeline will determine which distribution to use separately for each joint stunting
    and wasting state."""

    def __init__(self):
        super().__init__("risk_factor.child_underweight")
        self._sub_components = []  # no exposure distribution

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.stunting = builder.value.get_value(data_values.PIPELINES.STUNTING_EXPOSURE)
        self.wasting = builder.value.get_value(data_values.PIPELINES.WASTING_EXPOSURE)
        self.distributions = self._get_distributions(builder)

    def _get_exposure_distribution(self) -> None:
        pass

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[
                self.propensity_pipeline_name,
                data_values.PIPELINES.STUNTING_EXPOSURE,
                data_values.PIPELINES.WASTING_EXPOSURE,
            ],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
        )

    def _get_distributions(self, builder: Builder) -> Dict[str, PolytomousDistribution]:
        """Store and setup distributions for each joint stunting and wasting state."""
        distributions = {}
        categories = [f"cat{i+1}" for i in range(4)]
        all_distribution_data = builder.data.load(data_keys.UNDERWEIGHT.EXPOSURE)

        for category_1, category_2 in itertools.product(categories, categories):
            key = f"stunting_{category_1}_wasting_{category_2}"
            distribution_data = all_distribution_data.query(
                "stunting_parameter == @category_1 and " "wasting_parameter == @category_2"
            )
            distribution_data = distribution_data.drop(
                ["stunting_parameter", "wasting_parameter"], axis=1
            )
            distribution_data = pivot_categorical(distribution_data)
            distributions[key] = PolytomousDistribution(key, distribution_data)
        for dist in distributions.values():
            dist.setup(builder)
        return distributions

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
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
            distribution = self.distributions[
                f"stunting_{stunting_category}_wasting_{wasting_category}"
            ]
            exposure = distribution.ppf(group_df["propensity"])
            exposures.append(exposure)
        return pd.concat(exposures).sort_index()

class CGFRiskEffect(RiskEffect):

    def __init__(self, target: str):
        """
        Parameters
        ----------
        target :
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        self.risk = EntityString("risk_factor.child_growth_failure")
        self.cgf_models = [EntityString(f"risk_factor.{risk}") for risk in [data_keys.WASTING.name, data_keys.WASTING.name, data_keys.STUNTING.name]]
        self.target = TargetString(target)
        self.configuration_defaults = self._get_configuration_defaults()

        self.target_pipeline_name = f"{self.target.name}.{self.target.measure}"
        self.target_paf_pipeline_name = f"{self.target_pipeline_name}.paf"

        # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        for risk in self.cgf_models:
            self._register_target_modifier(builder, risk)

        self.population_attributable_fraction = (
            self._get_population_attributable_fraction_source(builder)
        )

        self._register_paf_modifier(builder)


    def _get_relative_risk_source(self, builder: Builder, risk: EntityString) -> LookupTable:
        relative_risk_data = get_relative_risk_data(builder, risk, self.target)
        return builder.lookup.build_table(
            relative_risk_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _get_target_modifier(
        self, builder: Builder, risk: EntityString
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(risk: EntityString, index: pd.Index, target: pd.Series) -> pd.Series:
            index_columns = ["index", risk.name]
            rr = self._get_relative_risk_source(builder, risk)(index)
            exposure = builder.value.get_value(f"{risk}.exposure")(index).reset_index()
            exposure.columns = index_columns
            exposure = exposure.set_index(index_columns)

            relative_risk = rr.stack().reset_index()
            relative_risk.columns = index_columns + ["value"]
            relative_risk = relative_risk.set_index(index_columns)

            effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)
            affected_rates = target * effect
            return affected_rates
        target_modifier = partial(adjust_target, risk)
        update_wrapper(target_modifier, adjust_target)
        return target_modifier

    def _register_target_modifier(self, builder: Builder, risk: EntityString) -> None:
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            modifier= self._get_target_modifier(builder, risk),
            requires_values=[f"{risk.name}.exposure"],
            requires_columns=["age", "sex"],
        )