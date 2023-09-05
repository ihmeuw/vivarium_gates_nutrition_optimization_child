import itertools
from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    pivot_categorical,
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
        self.risk = EntityString()
        self.target = TargetString(target)
        self.configuration_defaults = self._get_configuration_defaults()

        self.exposure_pipeline_name = f"{self.risk.name}.exposure"
        self.target_pipeline_name = f"{self.target.name}.{self.target.measure}"
        self.target_paf_pipeline_name = f"{self.target_pipeline_name}.paf"