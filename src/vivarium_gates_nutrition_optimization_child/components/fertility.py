"""
================
Fertility Models
================

Fertility module to create simulants from existing data

"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health import utilities

PREGNANCY_DURATION = pd.Timedelta(days=9 * utilities.DAYS_PER_MONTH)


class FertilityLineList(Component):
    """
    This class will determine what simulants need to be added to the state table based on their birth data from existing
    line list data.  Simulants will be registered to the state table on the time steps in which their birth takes place.
    """

    @property
    def columns_required(self) -> List[str]:
        return ["alive", "cause_of_death"]

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.clock = builder.time.clock()
        self.simulant_creator = builder.population.get_simulant_creator()

        # Requirements for input data
        self.birth_records = self._get_birth_records(builder)

    @staticmethod
    def _get_birth_records(builder: Builder) -> pd.DataFrame:
        """
        Method to load existing fertility data to use as birth records.
        """
        data_directory = Path(builder.configuration.input_data.fertility_input_data_path)
        scenario = builder.configuration.intervention.maternal_scenario
        draw = builder.configuration.input_data.input_draw_number
        seed = builder.configuration.randomness.random_seed

        file_path = data_directory / f"scenario_{scenario}_draw_{draw}_seed_{seed}.hdf"
        birth_records = pd.read_hdf(file_path)
        birth_records["birth_date"] = pd.to_datetime(birth_records["birth_date"])
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
        born_previous_step = birth_records[born_previous_step_mask].copy()
        # everyone is currently born on the first time step so this is always empty after the first time step
        if born_previous_step.empty:
            return
        born_previous_step.loc[:, "maternal_id"] = born_previous_step.index
        # stillbirths should be initialized as dead and with exit time
        born_previous_step.loc[:, "alive"] = "alive"
        born_previous_step.loc[:, "exit_time"] = np.datetime64("NaT")

        is_stillbirth = born_previous_step["pregnancy_outcome"] == "stillbirth"
        born_previous_step.loc[is_stillbirth, "alive"] = "dead"
        born_previous_step.loc[is_stillbirth, "exit_time"] = self.clock()

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

    def on_time_step_cleanup(self, event: Event) -> None:
        # update cause_of_death on cleanup because mortality handles that column on initialization
        pop = self.population_view.get(event.index)
        is_stillborn = (pop["alive"] == "dead") & (pop["cause_of_death"] == "not_dead")
        pop.loc[is_stillborn, "cause_of_death"] = "stillborn"
        self.population_view.update(pop)
