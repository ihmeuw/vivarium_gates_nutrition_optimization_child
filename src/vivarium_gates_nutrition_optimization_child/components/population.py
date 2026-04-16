"""
==========================
Module for Base Population
==========================

This module contains a component for creating a base population from line list data.

"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.population.base_population import BasePopulation, Disability
from vivarium_public_health.population.data_transformations import (
    assign_demographic_proportions,
)
from vivarium_public_health.population.mortality import Mortality

from vivarium_gates_nutrition_optimization_child import utilities as utils
from vivarium_gates_nutrition_optimization_child.constants import data_keys
from vivarium_gates_nutrition_optimization_child.constants.paths import (
    SUBNATIONAL_PERCENTAGES,
)


class PopulationLineList(BasePopulation):
    """
    Component to produce and age simulants based on line list data.
    """

    @property
    def time_step_priority(self) -> int:
        return 8

    def __init__(self):
        """Remove AgeOutSimulants and replace Mortality in subcomponents."""
        super().__init__()
        self._sub_components = [MortalityLineList(), Disability()]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Modifies the BasePopulation set up component.

        Specifically, we need to modify the initializers. We are unable call the
        super().setup() for the default `initialize_population` initializer (which
        creates age, sex, location, and entrance/exit times) and then register a
        second initializer for the new subnational and maternal_id columns because
        we need to register the maternal_id column as a randomness key column (along
        with the age column).
        """

        # Copy/paste from BasePopulation modulo the initializer registration.
        self.config = builder.configuration.population
        self.key_columns = builder.configuration.randomness.key_columns
        if self.config.include_sex not in ["Male", "Female", "Both"]:
            raise ValueError(
                "Configuration key 'population.include_sex' must be one "
                "of ['Male', 'Female', 'Both']. "
                f"Provided value: {self.config.include_sex}."
            )

        # TODO: Remove this when we remove deprecated keys.
        # Validate configuration for deprecated keys
        self._validate_config_for_deprecated_keys()

        source_population_structure = self._load_population_structure(builder)
        self.demographic_proportions = assign_demographic_proportions(
            source_population_structure,
            include_sex=self.config.include_sex,
        )
        self.randomness = self.get_randomness_streams(builder)
        self.register_simulants = builder.randomness.register_simulants

        # Additional attributes
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.location = self._get_location(builder)
        self.subnational = builder.configuration.intervention.subnational

        self._created_columns = [
            "age",
            "sex",
            "alive",
            "subnational",
            "location",
            "entrance_time",
            "exit_time",
            "maternal_id",
        ]
        builder.population.register_initializer(
            self.initialize_population,
            columns=self._created_columns,
        )

    def initialize_population(self, pop_data: SimulantData) -> None:
        """
        Creates simulants based on their birth date from the line list data.  Their demographic characteristics are also
        determined by the input data.
        """
        new_simulants = pd.DataFrame(columns=self._created_columns, index=pop_data.index)

        if pop_data.creation_time >= self.start_time:
            new_births = pop_data.user_data["new_births"]
            new_births.index = pop_data.index

            # Create columns for state table
            new_simulants["age"] = 0.0
            new_simulants["sex"] = new_births["sex"]
            new_simulants["alive"] = new_births["alive"]
            new_simulants["location"] = self.location
            new_simulants["entrance_time"] = pop_data.creation_time
            new_simulants["exit_time"] = new_births["exit_time"]
            new_simulants["maternal_id"] = new_births["maternal_id"]

        self.register_simulants(new_simulants[self.key_columns])

        if pop_data.creation_time >= self.start_time:
            if self.subnational == "All":
                new_simulants["subnational"] = self._get_subnational_locations(
                    new_simulants.index
                )
            else:
                new_simulants["subnational"] = self.subnational

        self.population_view.initialize(new_simulants)

    def _get_location(self, builder: Builder) -> Dict[str, str]:
        return builder.data.load("population.location")

    def _get_subnational_locations(self, pop_index: pd.Index) -> pd.Series:
        subnational_percents = pd.read_csv(SUBNATIONAL_PERCENTAGES)
        subnational_percents = subnational_percents.loc[
            subnational_percents["national_location"] == self.location
        ]
        location_choices = self.randomness["general_purpose"].choice(
            index=pop_index,
            choices=subnational_percents["location"],
            p=subnational_percents["percent_in_location"],
            additional_key="subnational_location_choice",
        )
        return location_choices


class MortalityLineList(Mortality):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        # Change the name from super to self.name
        config_defaults = super().configuration_defaults
        config_defaults[self.name] = config_defaults.pop("mortality")
        return config_defaults

    def initialize_mortality(self, pop_data: SimulantData) -> None:
        """Initialize mortality include stillbirths based on the line list data."""
        pop_update = pd.DataFrame(
            index=pop_data.index,
            columns=[
                "is_alive",
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
            ],
        )
        if not pop_data.index.empty:
            pop_update["is_alive"] = True
            pop_update[self.cause_of_death_column_name] = "not_dead"
            pop_update[self.years_of_life_lost_column_name] = 0.0

            # Update stillbirths
            is_stillborn = (
                pop_data.user_data["new_births"]["pregnancy_outcome"] == "stillbirth"
            )
            pop_update.loc[is_stillborn, "is_alive"] = False
            pop_update.loc[is_stillborn, self.cause_of_death_column_name] = "stillborn"

        self.population_view.initialize(pop_update)


class EvenlyDistributedPopulation(BasePopulation):
    """
    Component for producing and aging simulants which are initialized with ages
    evenly distributed between age start and age end, and evenly split between
    male and female.
    """

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)
        self.subnational = builder.configuration.intervention.subnational

        builder.population.register_initializer(
            self.initialize_subnational,
            columns=["subnational"],
        )

    def initialize_population(self, pop_data: SimulantData) -> None:
        age_start = pop_data.user_data.get("age_start", self.config.initialization_age_min)
        age_end = pop_data.user_data.get("age_end", self.config.initialization_age_max)

        population = pd.DataFrame(index=pop_data.index)
        population["entrance_time"] = pop_data.creation_time
        population["exit_time"] = pd.NaT
        population["location"] = self.location
        population["age"] = np.linspace(
            age_start, age_end, num=len(population) + 1, endpoint=False
        )[1:]
        population["sex"] = "Female"
        population.loc[population.index % 2 == 1, "sex"] = "Male"
        self.register_simulants(population[list(self.key_columns)])
        self.population_view.initialize(population)

    def initialize_subnational(self, pop_data: SimulantData) -> None:
        if self.subnational == "All":
            subnational = self._distribute_subnational_locations(pop_data.index)
        else:
            subnational = pd.Series(
                self.subnational, index=pop_data.index, name="subnational"
            )
        self.population_view.initialize(subnational)

    def _distribute_subnational_locations(self, pop_index: pd.Index) -> pd.Series:
        subnational_locations = pd.read_csv(SUBNATIONAL_PERCENTAGES)
        subnational_locations = subnational_locations.loc[
            subnational_locations["national_location"] == self.location
        ]["location"].unique()

        # Get repeating array of subnationals then fill remaining rows if necessary
        filled_subnationals = np.repeat(
            subnational_locations, repeats=len(pop_index) / len(subnational_locations)
        )
        remainder = len(pop_index) - len(filled_subnationals)
        if remainder > 0:
            extra_fill = subnational_locations[:remainder]
            filled_subnationals = np.append(filled_subnationals, extra_fill)

        return pd.Series(filled_subnationals, index=pop_index, name="subnational")
