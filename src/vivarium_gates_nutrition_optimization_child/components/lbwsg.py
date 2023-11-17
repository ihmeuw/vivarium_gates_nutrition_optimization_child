"""
====================================
Low Birth Weight and Short Gestation
====================================

This is a module to subclass the LBWSG component in Vivrium Public Health to use its functionality but to do so on
simulants who are initialized from line list data.

"""
from typing import Dict, List

import itertools
import math
import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRisk,
    LBWSGRiskEffect,
)
from vivarium_gates_nutrition_optimization_child.constants import data_keys
from typing import TYPE_CHECKING, Any, Dict, List, Optional


class LBWSGLineList(LBWSGRisk):
    """
    Component to initialize low birthweight and short gestation data for simulants based on existing line list data.
    """

    @property
    def columns_created(self) -> List[str]:
        return super().columns_created + [
            self.raw_gestational_age_exposure_column_name,
            self.birth_weight_status_column_name,
        ]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [self.randomness_stream_name],
        }

    def __init__(self):
        super().__init__()
        self.raw_gestational_age_exposure_column_name = "raw_gestational_age_exposure"
        self.birth_weight_status_column_name = "birth_weight_status"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)
        self.start_time = get_time_stamp(builder.configuration.time.start)

    def get_birth_exposure_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        def get_pipeline(axis_: str):
            return builder.value.register_value_producer(
                self.birth_exposure_pipeline_name(axis_),
                source=lambda index: self.get_birth_exposure(axis_, index),
                preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            )

        return {
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.AXES
        }

    ########################
    # Event-driven methods #
    ########################

    # noinspection PyAttributeOutsideInit
    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        if pop_data.creation_time < self.start_time:
            columns = [self.exposure_column_name(axis) for axis in self.AXES] + [
                self.raw_gestational_age_exposure_column_name,
                self.birth_weight_status_column_name,
            ]
            new_simulants = pd.DataFrame(columns=columns, index=pop_data.index)
            self.population_view.update(new_simulants)
        else:
            self.new_births = pop_data.user_data["new_births"]
            self.new_births.index = pop_data.index
            # add raw gestational age exposure to state table
            gestational_age = pop_data.user_data["new_births"]["gestational_age"].copy()
            gestational_age.name = self.raw_gestational_age_exposure_column_name
            self.population_view.update(gestational_age)

            super().on_initialize_simulants(pop_data)

            # add birth weight status to state table
            birth_weight = self.population_view.get(pop_data.index)["birth_weight_exposure"]
            birth_weight_status = np.where(
                birth_weight <= 2500, "low_birth_weight", "adequate_birth_weight"
            )
            birth_weight_status = pd.Series(
                birth_weight_status, name=self.birth_weight_status_column_name
            )
            self.population_view.update(birth_weight_status)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_birth_exposure(self, axis: str, index: pd.Index) -> pd.Series:
        return self.new_births.loc[index, axis]


class LBWSGPAFCalculationRiskEffect(LBWSGRiskEffect):
    """Risk effect component for calculating PAFs for LBWSG."""

    def get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(0)


class LBWSGPAFCalculationExposure(LBWSGRisk):
    @property
    def columns_required(self) -> Optional[List[str]]:
        return ['age', 'sex']

    @property
    def columns_created(self) -> List[str]:
        return [self.exposure_column_name(axis) for axis in self.AXES]

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.lbwsg_categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        self.age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)

    def get_birth_exposure_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        def get_pipeline(axis_: str):
            return builder.value.register_value_producer(
                self.birth_exposure_pipeline_name(axis_),
                source=lambda index: self.get_birth_exposure(axis_, index),
                requires_columns=["age", "sex"],
                preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            )

        return {
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.AXES
        }

    def get_birth_exposure(self, axis: str, index: pd.Index) -> pd.DataFrame:
        pop = self.population_view.subview(['age', 'sex']).get(index)
        pop['age_bin'] = pd.cut(pop['age'], self.age_bins['age_start'])
        pop = pop.sort_values(['sex', 'age'])

        lbwsg_categories = self.lbwsg_categories.keys()
        num_repeats, remainder = divmod(len(pop), 2*len(lbwsg_categories))
        if remainder != 0:
            raise ValueError("Population size should be multiple of double the number of LBWSG categories."
                             f"Population size is {len(pop)}, but should be a multiple of "
                             f"{2*len(lbwsg_categories)}.")

        assigned_categories = list(lbwsg_categories) * (2*num_repeats)
        pop['lbwsg_category'] = assigned_categories

        num_simulants_in_category = int( len(pop) / (len(lbwsg_categories) * 4) )
        num_points_in_interval = int( math.sqrt(num_simulants_in_category) )

        exposure_values = pd.Series(name=axis, index=pop.index, dtype=float)
        for age_bin, sex, cat in itertools.product(pop['age_bin'].unique(), ['Male', 'Female'], lbwsg_categories):
            description = self.lbwsg_categories[cat]

            birthweight_endpoints = [float(val) for val in description.split(", [")[1].split(")")[0].split(", ")]
            birthweight_interval_values = np.linspace(birthweight_endpoints[0],
                                                      birthweight_endpoints[1],
                                                      num=num_points_in_interval+2)[1:-1]

            gestational_age_endpoints = [float(val) for val in description.split("- [")[1].split(")")[0].split(", ")]
            gestational_age_interval_values = np.linspace(gestational_age_endpoints[0],
                                                          gestational_age_endpoints[1],
                                                          num=num_points_in_interval+2)[1:-1]

            birthweight_points, gestational_age_points = np.meshgrid(birthweight_interval_values,
                                                                     gestational_age_interval_values)
            lbwsg_exposures = pd.DataFrame({'birth_weight': birthweight_points.flatten(),
                                            'gestational_age': gestational_age_points.flatten()})

            subset_index = pop.query("lbwsg_category==@cat and age_bin==@age_bin and sex==@sex").index
            exposure_values.loc[subset_index] = lbwsg_exposures[axis].values

        return exposure_values
