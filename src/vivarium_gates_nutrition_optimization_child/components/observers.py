from collections import Counter
from typing import Dict

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
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

from vivarium_gates_nutrition_optimization_child.constants import data_keys, results


class ResultsStratifier(ResultsStratifier_):
    """Centralized component for handling results stratification.
    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.
    """

    def get_age_bins(self, builder: Builder) -> pd.DataFrame:
        """Define more granular age groups for SQ-LNS V&V."""
        age_bins = super().get_age_bins(builder)
        data_dict = {
            "age_start": [0.5, 10 / 12, 1, 1.5],
            "age_end": [10 / 12, 1, 1.5, 2],
            "age_group_name": [
                "6_to_10_months",
                "10_to_11_months",
                "12_to_17_months",
                "18_to_23_months",
            ],
        }
        new_age_bins = pd.DataFrame(data_dict)

        # remove duplicated ages and concat
        new_age_starts = data_dict["age_start"]
        age_bins = age_bins.query("age_start not in @new_age_starts")
        age_bins = pd.concat([age_bins, new_age_bins])

        return age_bins.sort_values(["age_start"]).reset_index(drop=True)

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""
        super().register_stratifications(builder)

        builder.results.register_stratification(
            "wasting_state",
            [category.value for category in data_keys.CGFCategories],
            self.child_growth_risk_factor_stratification_mapper,
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


class DisabilityObserver(DisabilityObserver_):
    def on_post_setup(self, event: Event) -> None:
        for cause in self._cause_components:
            if (
                cause.has_disability
                or cause.name == "disease_model.moderate_protein_energy_malnutrition"
            ):
                self.disability_pipelines[cause.state_id] = cause.disability_weight


class BirthObserver:
    configuration_defaults = {
        "stratification": {
            "birth": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "BirthObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "birth_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.config = builder.configuration.stratification.birth
        self.birth_weight_column_name = "birth_weight_exposure"
        self.gestational_age_column_name = "gestational_age_exposure"
        self.low_birth_weight_limit = 2500  # grams

        columns_required = [
            "entrance_time",
            self.birth_weight_column_name,
            self.gestational_age_column_name,
        ]
        self.population_view = builder.population.get_view(columns_required)

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


class ChildWastingObserver(DiseaseObserver):
    def __init__(self):
        self.disease = self.risk = "child_wasting"
        self.configuration_defaults = self.get_configuration_defaults()
        self.current_state_column_name = self.disease
        self.previous_state_column_name = f"previous_{self.disease}"
        self.exposure_pipeline_name = f"{self.risk}.exposure"

    def __repr__(self):
        return "ChildWastingObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "child_wasting_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

        disease_model = builder.components.get_component(f"disease_model.{self.disease}")

        builder.population.initializes_simulants(
            self.on_initialize_simulants, creates_columns=[self.previous_state_column_name]
        )

        columns_required = [self.disease, self.previous_state_column_name]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)

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
