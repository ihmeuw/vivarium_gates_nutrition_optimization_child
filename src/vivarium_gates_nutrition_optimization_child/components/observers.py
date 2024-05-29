from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.metrics.disability import (
    DisabilityObserver as DisabilityObserver_,
)
from vivarium_public_health.metrics.disease import DiseaseObserver
from vivarium_public_health.metrics.mortality import (
    MortalityObserver as MortalityObserver_,
)
from vivarium_public_health.metrics.stratification import (
    ResultsStratifier as ResultsStratifier_,
)

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    results,
)
<<<<<<< HEAD
from vivarium_gates_nutrition_optimization_child.constants.metadata import (
    SUBNATIONAL_LOCATION_DICT,
)

=======
from vivarium_gates_nutrition_optimization_child.constants.metadata import SUBNATIONAL_LOCATION_DICT
>>>>>>> 81ea417 (Subnational SQ-LNS effect data, adding subnational stratifiction, changing model specs)

class ResultsStratifier(ResultsStratifier_):
    """Centralized component for handling results stratification.
    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.
    """

    def get_age_bins(self, builder: Builder) -> pd.DataFrame:
        """Define final age groups for production runs."""
        age_bins = super().get_age_bins(builder)
        data_dict = {
            "age_start": [0.0, 0.019178, 0.076712, 0.5, 1.0, 2.0],  # [0.0, 0.5, 1.5],
            "age_end": [0.019178, 0.076712, 0.5, 1.0, 2.0, 5.0],  # [0.5, 1.5, 5],
            "age_group_name": [
                "early_neonatal",
                "late_neonatal",
                "1-5_months",
                "6-11_months",
                "12_to_23_months",
                "2_to_4",
                # "0_to_6_months",
                # "6_to_18_months",
                # "18_to_59_months",
            ],
        }

        return pd.DataFrame(data_dict)

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""
        super().register_stratifications(builder)

        builder.results.register_stratification(
            "wasting_state",
            [category.value for category in data_keys.ChildWastingCategories],
            self.child_wasting_stratification_mapper,
            is_vectorized=True,
            requires_values=["child_wasting.exposure"],
        )
        builder.results.register_stratification(
            "stunting_state",
            [category.value for category in data_keys.CGFCategories],
            self.child_growth_risk_factor_stratification_mapper,
            is_vectorized=True,
            requires_values=["child_stunting.exposure"],
        )
        builder.results.register_stratification(
            "underweight_state",
            [category.value for category in data_keys.CGFCategories],
            self.child_growth_risk_factor_stratification_mapper,
            is_vectorized=True,
            requires_values=["child_underweight.exposure"],
        )
        builder.results.register_stratification(
            "maternal_supplementation",
            results.MATERNAL_SUPPLEMENTATION_TYPES,
            is_vectorized=True,
            requires_columns=["maternal_supplementation_exposure"],
        )
        builder.results.register_stratification(
            "bmi_anemia",
            ["cat4", "cat3", "cat2", "cat1"],
            is_vectorized=True,
            requires_columns=["maternal_bmi_anemia_exposure"],
        )
        builder.results.register_stratification(
            "sam_treatment",
            ["covered", "uncovered"],
            self.map_wasting_treatment,
            is_vectorized=True,
            requires_values=[f"{data_keys.SAM_TREATMENT.name}.exposure"],
        )
        builder.results.register_stratification(
            "mam_treatment",
            ["covered", "uncovered"],
            self.map_wasting_treatment,
            is_vectorized=True,
            requires_values=[f"{data_keys.MAM_TREATMENT.name}.exposure"],
        )
        builder.results.register_stratification(
            "sqlns_coverage",
            ["covered", "uncovered", "received"],
            is_vectorized=True,
            requires_values=[data_values.SQ_LNS.COVERAGE_PIPELINE],
        )
        builder.results.register_stratification(
            "birth_weight_status",
            ["low_birth_weight", "adequate_birth_weight"],
            is_vectorized=True,
            requires_columns=["birth_weight_status"],
        )
        location = builder.data.load("population.location")
        builder.results.register_stratification(
            "subnational",
            SUBNATIONAL_LOCATION_DICT[location],
            is_vectorized=True,
            requires_columns=["subnational"],
        )

    ###########################
    # Stratifications Details #
    ###########################

    # noinspection PyMethodMayBeStatic
    def child_growth_risk_factor_stratification_mapper(self, pop: pd.DataFrame) -> pd.Series:
        # applicable to stunting and wasting
        mapper = {
            "cat4": data_keys.CGFCategories.UNEXPOSED.value,
            "cat3": data_keys.CGFCategories.MILD.value,
            "cat2": data_keys.CGFCategories.MODERATE.value,
            "cat1": data_keys.CGFCategories.SEVERE.value,
        }
        return pop.squeeze(axis=1).map(mapper)

    def child_wasting_stratification_mapper(self, pop: pd.DataFrame) -> pd.Series:
        # applicable to stunting and wasting
        mapper = {
            "cat4": data_keys.ChildWastingCategories.UNEXPOSED.value,
            "cat3": data_keys.ChildWastingCategories.MILD.value,
            "cat2.5": data_keys.ChildWastingCategories.BETTER_MODERATE.value,
            "cat2": data_keys.ChildWastingCategories.WORSE_MODERATE.value,
            "cat1": data_keys.ChildWastingCategories.SEVERE.value,
        }
        return pop.squeeze(axis=1).map(mapper)

    def map_wasting_treatment(self, pop: pd.DataFrame) -> pd.Series:
        # Both SAM and MAM treatments
        mapper = {
            "cat3": "covered",
            "cat2": "covered",
            "cat1": "uncovered",
        }
        return pop.squeeze(axis=1).map(mapper)

    def map_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        """Map age with age group name strings

        Parameters
        ----------
        pop
            A DataFrame with one column, an age to be mapped to an age group name string

        Returns
        ------
        pandas.Series
            A pd.Series with age group name string corresponding to the pop passed into the function
        """
        bins = self.age_bins["age_start"].to_list() + [self.age_bins["age_end"].iloc[-1]]
        labels = self.age_bins["age_group_name"].to_list()
        # need to include lowest to map people who are exactly 0
        age_group = pd.cut(
            pop.squeeze(axis=1), bins, labels=labels, include_lowest=True
        ).rename("age_group")
        return age_group


class BirthObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "birth": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self):
        super().__init__()
        self.birth_weight_column_name = "birth_weight_exposure"
        self.gestational_age_column_name = "gestational_age_exposure"
        self.low_birth_weight_limit = 2500  # grams

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [
            "entrance_time",
            self.birth_weight_column_name,
            self.gestational_age_column_name,
        ]

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.config = builder.configuration.stratification.birth

        builder.results.register_observation(
            name=f"live_births",
            pop_filter="alive=='alive'",
            aggregator=self.count_live_births,
            requires_columns=["entrance_time"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )
        builder.results.register_observation(
            name=f"birth_weight_sum",
            pop_filter="alive=='alive'",
            aggregator=self.sum_birth_weights,
            requires_columns=["entrance_time", self.birth_weight_column_name],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )
        builder.results.register_observation(
            name=f"gestational_age_sum",
            pop_filter="alive=='alive'",
            aggregator=self.sum_gestational_ages,
            requires_columns=["entrance_time", self.gestational_age_column_name],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )
        builder.results.register_observation(
            name=f"low_weight_births",
            pop_filter="alive=='alive'",
            aggregator=self.count_low_weight_births,
            requires_columns=["entrance_time", self.birth_weight_column_name],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )

    ########################
    # Event-driven methods #
    ########################
    def count_live_births(self, x: pd.DataFrame) -> float:
        born_this_step = x["entrance_time"] == self.clock()
        return sum(born_this_step)

    def sum_birth_weights(self, x: pd.DataFrame) -> float:
        born_this_step = x["entrance_time"] == self.clock()
        return x.loc[born_this_step, self.birth_weight_column_name].sum()

    def sum_gestational_ages(self, x: pd.DataFrame) -> float:
        born_this_step = x["entrance_time"] == self.clock()
        return x.loc[born_this_step, self.gestational_age_column_name].sum()

    def count_low_weight_births(self, x: pd.DataFrame) -> float:
        born_this_step = x["entrance_time"] == self.clock()
        has_low_birth_weight = (
            x.loc[born_this_step, self.birth_weight_column_name] < self.low_birth_weight_limit
        )
        return sum(has_low_birth_weight)


class MortalityObserver(MortalityObserver_):
    """This is a class to make component ordering work in the model spec."""

    pass


class ChildWastingObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("child_wasting")
        self.disease = self.risk = "child_wasting"
        self.exposure_pipeline_name = f"{self.risk}.exposure"

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [self.disease, "sex", "age"]

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

        disease_model = builder.components.get_component(
            f"child_wasting_model.{self.disease}"
        )

        # not needed for final runs
        for category in self.categories:
            builder.results.register_observation(
                name=f"{self.risk}_{category}_person_time",
                pop_filter=f'alive == "alive" and `{self.exposure_pipeline_name}`=="{category}" and tracked==True',
                aggregator=self.aggregate_state_person_time,
                requires_columns=["alive"],
                requires_values=[self.exposure_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

        for transition in disease_model.transition_names:
            filter_string = (
                f'{self.previous_state_column_name} == "{transition.from_state}" '
                f'and {self.disease} == "{transition.to_state}" '
                f"and tracked==True"
            )
            builder.results.register_observation(
                name=f"{transition}_event_count",
                pop_filter=filter_string,
                requires_columns=[self.previous_state_column_name, self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="collect_metrics",
            )


class MortalityHazardRateObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "mortality_hazard_rate": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self):
        super().__init__()
        self.mortality_rate_pipeline_name = "mortality_rate"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.config = builder.configuration.stratification.mortality_hazard_rate
        self.mortality_rates = builder.value.get_value(self.mortality_rate_pipeline_name)

        builder.results.register_observation(
            name=f"mortality_hazard_rate_first_moment",
            pop_filter="alive=='alive'",
            aggregator=self.calculate_mortality_hazard_rate,
            requires_columns=["alive"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="collect_metrics",
        )

    def calculate_mortality_hazard_rate(self, x: pd.DataFrame) -> float:
        # sum mortality rates across all causes
        summed_mortality_rates = self.mortality_rates(x.index).sum()
        return sum(summed_mortality_rates)
