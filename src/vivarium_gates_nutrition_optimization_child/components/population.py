"""
==========================
Module for Base Population
==========================

This module contains a component for creating a base population from line list data.

"""

from typing import Dict, List

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.population.base_population import BasePopulation
from vivarium_public_health.population.data_transformations import (
    assign_demographic_proportions,
    load_population_structure,
)

from vivarium_gates_nutrition_optimization_child.constants import data_keys
from vivarium_gates_nutrition_optimization_child.constants.paths import (
    SUBNATIONAL_PERCENTAGES,
)


class PopulationLineList(BasePopulation):
    """
    Component to produce and age simulants based on line list data.
    """

    @property
    def columns_created(self) -> List[str]:
        return [
            "age",
            "sex",
            "alive",
            "subnational",
            "location",
            "entrance_time",
            "exit_time",
            "maternal_id",
        ]

    @property
    def time_step_priority(self) -> int:
        return 8

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.population
        self.key_columns = builder.configuration.randomness.key_columns
        if self.config.include_sex not in ["Male", "Female", "Both"]:
            raise ValueError(
                "Configuration key 'population.include_sex' must be one "
                "of ['Male', 'Female', 'Both']. "
                f"Provided value: {self.config.include_sex}."
            )

        source_population_structure = load_population_structure(builder)
        self.population_data = assign_demographic_proportions(
            source_population_structure,
            include_sex=self.config.include_sex,
        )

        self.randomness = {
            "general_purpose": builder.randomness.get_stream("population_generation"),
            "bin_selection": builder.randomness.get_stream(
                "bin_selection", initializes_crn_attributes=True
            ),
            "age_smoothing": builder.randomness.get_stream(
                "age_smoothing", initializes_crn_attributes=True
            ),
            "age_smoothing_age_bounds": builder.randomness.get_stream(
                "age_smoothing_age_bounds", initializes_crn_attributes=True
            ),
        }
        self.register_simulants = builder.randomness.register_simulants

        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.location = self._get_location(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Creates simulants based on their birth date from the line list data.  Their demographic characteristics are also
        determined by the input data.
        """
        new_simulants = pd.DataFrame(columns=self.columns_created, index=pop_data.index)

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
            new_simulants["subnational"] = self._get_subnational_locations(
                new_simulants.index
            )

        self.population_view.update(new_simulants)

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

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        age_start = pop_data.user_data.get("age_start", self.config.initialization_age_min)
        age_end = pop_data.user_data.get("age_end", self.config.initialization_age_max)

        population = pd.DataFrame(index=pop_data.index)
        population["entrance_time"] = pop_data.creation_time
        population["exit_time"] = pd.NaT
        population["alive"] = "alive"
        population["location"] = self.location
        population["age"] = np.linspace(
            age_start, age_end, num=len(population) + 1, endpoint=False
        )[1:]
        population["sex"] = "Female"
        population.loc[population.index % 2 == 1, "sex"] = "Male"
        self.register_simulants(population[list(self.key_columns)])
        population["subnational"] = self._distribute_subnational_locations(population.index)
        self.population_view.update(population)

    def _distribute_subnational_locations(self, pop_index: pd.Index) -> pd.Series:
        subnational_percents = pd.read_csv(SUBNATIONAL_PERCENTAGES)
        subnational_percents = subnational_percents.loc[
            subnational_percents["national_location"] == self.location
        ]
        # Equal weights
        location_choices = self.randomness["general_purpose"].choice(
            index=pop_index,
            choices=subnational_percents["location"],
            additional_key="subnational_location_choice",
        )
        return location_choices
