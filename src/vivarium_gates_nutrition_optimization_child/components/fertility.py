"""
================
Fertility Models
================

Fertility module to create simulants from existing data

"""
from pathlib import Path

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health import utilities

PREGNANCY_DURATION = pd.Timedelta(days=9 * utilities.DAYS_PER_MONTH)


class FertilityLineList:
    """
    This class will determine what simulants need to be added to the state table based on their birth data from existing
    line list data.  Simulants will be registered to the state table on the time steps in which their birth takes place.
    """

    configuration_defaults = {}

    def __repr__(self):
        return "FertilityLineList()"

    @property
    def name(self):
        return "line_list_fertility"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.clock = builder.time.clock()
        self.simulant_creator = builder.population.get_simulant_creator()

        # Requirements for input data
        self.birth_records = self._get_birth_records(builder)

        builder.event.register_listener("time_step", self.on_time_step)

    @staticmethod
    def _get_birth_records(builder: Builder) -> pd.DataFrame:
        """
        Method to load existing fertility data to use as birth records.
        """
        data_directory = Path(builder.configuration.input_data.fertility_input_data_path)
        scenario = builder.configuration.intervention.scenario
        draw = builder.configuration.input_data.input_draw_number
        seed = builder.configuration.randomness.random_seed

        file_path = data_directory / f"scenario_{scenario}_draw_{draw}_seed_{seed}.hdf"
        birth_records = pd.read_hdf(file_path)
        # Hard coding for now because input data has the wrong birth date
        # TODO: remove hardcoding and keep type casting once fertility_input_data_path
        # TODO: contains this birth date
        birth_records["birth_date"] = pd.to_datetime("2024-12-30")
        return birth_records

    def on_time_step(self, event: Event) -> None:
        """Adds new simulants every time step determined by a simulant's birth date in the line list data.
        Parameters
        ----------
        event
            The event that triggered the function call.
        """
        birth_records = self.birth_records
        born_previous_step_mask = (birth_records["birth_date"] < self.clock()) & (
            birth_records["birth_date"] > self.clock() - event.step_size
        )
        # TODO: remove resetting index when using actual child data
        born_previous_step = birth_records[born_previous_step_mask].reset_index().copy()
        born_previous_step.loc[:, "maternal_id"] = born_previous_step.index
        simulants_to_add = len(born_previous_step)

        if simulants_to_add > 0:
            self.simulant_creator(
                simulants_to_add,
                population_configuration={
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                    "new_births": born_previous_step,
                },
            )
