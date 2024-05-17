import itertools
from typing import Any, Callable, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk, RiskEffect
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
    get_relative_risk_data,
    pivot_categorical,
)
from vivarium_public_health.utilities import EntityString, TargetString

from vivarium_gates_nutrition_optimization_child.components.distribution import (
    PolytomousDistribution,
)
from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values
from vivarium_gates_nutrition_optimization_child.components.distribution import (
    PolytomousDistribution,
)
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

    def build_all_lookup_tables(self, builder: Builder) -> None:
        pass

    def get_exposure_distribution(self) -> None:
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

    def _get_distributions(self, builder: Builder) -> Dict[str, PolytomousDistribution]:
        """Store and setup distributions for each joint stunting and wasting state."""
        distributions = {}
        stunting_categories = [f"cat{i+1}" for i in range(4)]
        wasting_categories = [f"cat{i + 1}" for i in range(4)] + ["cat2.5"]
        all_distribution_data = builder.data.load(data_keys.UNDERWEIGHT.EXPOSURE)

        for stunting_cat, wasting_cat in itertools.product(
            stunting_categories, wasting_categories
        ):
            if wasting_cat == "cat2.5":
                # this key will not be parsed properly by the distribution if it contains a dot
                key = f"risk_factor.stunting_{stunting_cat}_wasting_cat25_underweight"
            else:
                key = f"risk_factor.stunting_{stunting_cat}_wasting_{wasting_cat}_underweight"
            distribution_data = all_distribution_data.query(
                "stunting_parameter == @stunting_cat and wasting_parameter == @wasting_cat"
            )
            distribution_data = distribution_data.drop(
                ["stunting_parameter", "wasting_parameter"], axis=1
            ) 
            distribution_data = pivot_categorical(builder, self.risk, distribution_data)
            distributions[key] = PolytomousDistribution(key, distribution_data)
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
        risk_effect_config = super().configuration_defaults
        return {
            self.get_name(risk, self.target): risk_effect_config[self.name]
            for risk in [self.risk] + self.cgf_models
        }

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
        self.target = TargetString(target)
        # This is to access to the distribution type before setup
        self._distribution_type = "ordered_polytomous"

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables[
            "population_attributable_fraction"
        ] = self.get_population_attributable_fraction_source(builder)
        self.lookup_tables["relative_risk"] = self.get_relative_risk_source(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables[
            "population_attributable_fraction"
        ] = self.get_population_attributable_fraction_source(builder)
        self.lookup_tables["relative_risk"] = self.get_relative_risk_source(builder)

    def get_distribution_type(self, builder: Builder) -> str:
        return self._distribution_type

    def get_risk_exposure(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            risk: builder.value.get_value(f"{risk.name}.exposure") for risk in self.cgf_models
        }

    def get_relative_risk_source(self, builder: Builder) -> Dict[str, LookupTable]:
        # TODO: get_relative_risk data needs to take distribution arg but we don't have
        # access to it yet because we are in setup_components and not setup
        rr_data = {
            risk: get_relative_risk_data(builder, risk, self.target, self._distribution_type)
            for risk in self.cgf_models
        }
        return {
            risk: self.build_lookup_table(builder, rr_data[risk][0], rr_data[risk][1])
            for risk in self.cgf_models
        }

    def get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        paf_data = builder.data.load(f"{self.risk}.population_attributable_fraction")
        correct_target = (paf_data["affected_entity"] == self.target.name) & (
            paf_data["affected_measure"] == self.target.measure
        )
        paf_data = paf_data[correct_target].drop(
            columns=["affected_entity", "affected_measure"]
        )
        return self.build_lookup_table(builder, paf_data, ["value"])

    def get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            rrs = self.lookup_tables["relative_risk"]
            exposures = self.exposure
            for risk in self.cgf_models:
                index_columns = ["index", risk.name]
                rr = rrs[risk](index)
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
