"""
==========================
Module for Base Population
==========================

This module contains a component for creating a base population from line list data.

"""
import glob
from typing import List

import pandas as pd
import numpy as np
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.population.base_population import BasePopulation
from vivarium_public_health.population.data_transformations import (
    assign_demographic_proportions,
    load_population_structure,
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
            "location",
            "entrance_time",
            "exit_time",
            "maternal_id",
        ]
        
    @property
    def columns_required(self) -> List[str]:
        return ["tracked"]
    
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
        # builder.time.register_step_modifier(self.modify_step)

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
        self.population_view.update(new_simulants)

    def _get_location(self, builder: Builder) -> str:
        return builder.data.load("population.location")
    
    def modify_step(self, index: pd.Index) -> pd.Series:
        """
        Sets simlant step size to 0.5 for neonates and 4 for 1-5 months.
        """
        neonates = self.population_view.get(index, "age < 0.5 and alive == 'alive' and tracked == True").index
        early_infants = self.population_view.get(index, "age >= 0.5 and age < 1.0 and alive == 'alive' and tracked == True").index
        step_size = pd.concat([pd.Series(pd.Timedelta(days=0.5), index=neonates),
                              pd.Series(pd.Timedelta(days=4), index=early_infants)])
        return step_size