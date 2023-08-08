"""
====================================
Low Birth Weight and Short Gestation
====================================

This is a module to subclass the LBWSG component in Vivrium Public Health to use its functionality but to do so on
simulants who are initialized from line list data.

"""
from typing import Dict

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks.data_transformations import get_exposure_post_processor
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import LBWSGRisk


class LBWSGLineList(LBWSGRisk):
    """
    Component to initialize low birthweight and short gestation data for simulants based on existing line list data.
    """
    @property
    def name(self) -> str:
        return "line_list_low_birth_weight_and_short_gestation"

    def __repr__(self):
        return "LBWSGLineList()"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        super().setup(builder)
        self.start_time = get_time_stamp(builder.configuration.time.start)

    def _get_birth_exposure_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        def get_pipeline(axis_: str):
            return builder.value.register_value_producer(
                self.birth_exposure_pipeline_name(axis_),
                source=lambda index: self._get_birth_exposure(axis_, index),
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
            columns = [self.exposure_column_name(axis) for axis in self.AXES]
            new_simulants = pd.DataFrame(columns=columns, index=pop_data.index)
            self.population_view.update(new_simulants)
        else:
            self.new_births = pop_data.user_data["new_births"]
            self.new_births.index = pop_data.index

            super().on_initialize_simulants(pop_data)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_birth_exposure(self, axis: str, index: pd.Index) -> pd.Series:
        return self.new_births.loc[index, axis]
