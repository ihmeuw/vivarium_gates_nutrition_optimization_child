from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.utilities import EntityString

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
    models,
    scenarios,
)
from vivarium_gates_nutrition_optimization_child.constants.data_keys import WASTING
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class ChildWasting:
    def __init__(self):
        self.dynamic_model = DynamicChildWasting()
        self.static_model = StaticChildWasting()

    @property
    def sub_components(self):
        return [
            self.dynamic_model,
            self.static_model,
        ]

    @property
    def name(self):
        return f"child_wasting"

    def __repr__(self):
        return "ChildWasting()"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.population_view = builder.population.get_view(
            [
                "alive",
                "age",
                "sex",
                self.dynamic_model.state_column,
                self.static_model.propensity_column_name,
            ]
        )
        self.exposure = builder.value.register_value_producer(
            f"{self.name}.exposure",
            source=self.get_current_exposure,
            requires_columns=[self.dynamic_model.state_column],
            requires_values=[self.static_model.exposure_pipeline_name],
            preferred_post_processor=get_exposure_post_processor(
                builder, EntityString(f"risk_factor.{self.name}")
            ),
        )

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        exposure = pop[self.dynamic_model.state_column].apply(models.get_risk_category)
        under_six_months = (pop["age"] < data_values.WASTING.DYNAMIC_START_AGE) & (
            pop["alive"] == "alive"
        )
        if under_six_months.any():
            exposure[under_six_months] = self.static_model.exposure(
                pop[under_six_months].index
            )
        return exposure


class StaticChildWasting(Risk):
    def __init__(self):
        """
        Parameters
        ----------
        risk :
            the type and name of a risk, specified as "type.name". Type is singular.
        """
        self.risk = EntityString("risk_factor.child_wasting")
        name = "static_child_wasting"
        self.configuration_defaults = self._get_configuration_defaults()
        self.exposure_distribution = self._get_exposure_distribution()
        self._sub_components = [self.exposure_distribution]

        self._randomness_stream_name = f"initial_{name}_propensity"
        self.propensity_column_name = f"{name}_propensity"
        self.propensity_pipeline_name = f"{name}.propensity"
        self.exposure_pipeline_name = f"{name}.exposure"


class WastingTreatment(Risk):
    def __init__(self, treatment_type: str):
        super().__init__(treatment_type)

        self.previous_wasting_column = f"previous_{data_keys.WASTING.name}"
        self.wasting_column = data_keys.WASTING.name

        self.treated_state = self._get_treated_state()

    ##########################
    # Initialization methods #
    ##########################

    def _get_treated_state(self) -> str:
        return self.risk.name.split("_treatment")[0]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self._register_on_time_step_prepare_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [self.propensity_column_name, self.previous_wasting_column, self.wasting_column]
        )

    def _register_on_time_step_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_cleanup(self, event: Event):
        pop = self.population_view.get(event.index)
        propensity = pop[self.propensity_column_name]
        remitted_mask = (pop[self.previous_wasting_column] == self.treated_state) & pop[
            self.wasting_column
        ] != self.treated_state
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(propensity)


class DynamicChildWastingModel(DiseaseModel):
    def setup(self, builder):
        """Perform this component's setup."""
        super(DiseaseModel, self).setup(builder)

        self.configuration_age_start = builder.configuration.population.age_start
        self.configuration_age_end = builder.configuration.population.age_end

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(
            cause_specific_mortality_rate,
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=["age", "sex"],
        )

        self.population_view = builder.population.get_view(
            ["age", "sex", self.state_column, "static_child_wasting_propensity"]
        )
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.state_column],
            requires_columns=["age", "sex", "static_child_wasting_propensity"],
        )

        builder.event.register_listener("time_step", self.on_time_step)
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)

    def on_initialize_simulants(self, pop_data):
        population = self.population_view.subview(
            ["age", "sex", "static_child_wasting_propensity"]
        ).get(pop_data.index)

        assert self.initial_state in {s.state_id for s in self.states}

        state_names, weights_bins = self.get_state_weights(pop_data.index, "birth_prevalence")

        if state_names and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population["sex_id"] = population.sex.apply({"Male": 1, "Female": 2}.get)

            condition_column = self.assign_initial_status_to_simulants(
                population,
                state_names,
                weights_bins,
                population["static_child_wasting_propensity"],
            )

            condition_column = condition_column.rename(
                columns={"condition_state": self.state_column}
            )
        else:
            condition_column = pd.Series(
                self.initial_state, index=population.index, name=self.state_column
            )
        self.population_view.update(condition_column)

    @staticmethod
    def assign_initial_status_to_simulants(
        simulants_df, state_names, weights_bins, propensities
    ):
        simulants = simulants_df[["age", "sex", "static_child_wasting_propensity"]].copy()

        choice_index = (propensities.values[np.newaxis].T > weights_bins).sum(axis=1)
        initial_states = pd.Series(np.array(state_names)[choice_index], index=simulants.index)

        simulants.loc[:, "condition_state"] = initial_states
        return simulants


# noinspection PyPep8Naming
def DynamicChildWasting() -> DynamicChildWastingModel:
    tmrel = SusceptibleState(models.WASTING.MODEL_NAME)
    mild = DiseaseState(
        models.WASTING.MILD_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_mild_wasting_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_mild_wasting_birth_prevalence,
        },
    )
    moderate = DiseaseState(
        models.WASTING.MODERATE_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_mam_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_mam_birth_prevalence,
        },
    )
    severe = DiseaseState(
        models.WASTING.SEVERE_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_sam_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_sam_birth_prevalence,
        },
    )

    # Add transitions for tmrel
    tmrel.allow_self_transitions()
    tmrel.add_transition(
        mild,
        source_data_type="rate",
        get_data_functions={
            "incidence_rate": load_wasting_rate,
        },
    )

    # Add transitions for mild
    mild.allow_self_transitions()
    mild.add_transition(
        moderate,
        source_data_type="rate",
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    mild.add_transition(
        tmrel,
        source_data_type="rate",
        get_data_functions={
            "remission_rate": load_wasting_rate,
        },
    )

    # Add transitions for moderate
    moderate.allow_self_transitions()
    moderate.add_transition(
        severe,
        source_data_type="rate",
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    moderate.add_transition(
        mild,
        source_data_type="rate",
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    # Add transitions for severe
    severe.allow_self_transitions()
    severe.add_transition(
        moderate,
        source_data_type="rate",
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    severe.add_transition(
        mild,
        source_data_type="rate",
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    return DynamicChildWastingModel(
        models.WASTING.MODEL_NAME,
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0},
        states=[severe, moderate, mild, tmrel],
    )


# noinspection PyUnusedLocal
def load_pem_excess_mortality_rate(builder: Builder, cause: str) -> pd.DataFrame:
    return builder.data.load(data_keys.PEM.EMR)


# noinspection PyUnusedLocal
def load_mild_wasting_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT3)


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT3].reset_index()


def load_wasting_rate(builder: Builder, *args) -> pd.DataFrame:
    args_to_transition_map = {
        ("mild_child_wasting",): "inc_rate_mild",
        ("mild_child_wasting", "moderate_acute_malnutrition"): "inc_rate_mam",
        ("moderate_acute_malnutrition", "severe_acute_malnutrition"): "inc_rate_sam",
        ("child_wasting",): "rem_rate_mild",
        ("moderate_acute_malnutrition", "mild_child_wasting"): "rem_rate_mam",
        ("severe_acute_malnutrition", "mild_child_wasting"): "tx_rem_rate_sam",
        ("severe_acute_malnutrition", "moderate_acute_malnutrition"): "ux_rem_rate_sam",
    }
    transition = args_to_transition_map[args]
    data = get_transition_data(builder, transition)
    return data


def get_transition_data(builder: Builder, transition: str) -> pd.DataFrame:
    rates = builder.data.load("risk_factor.child_wasting.transition_rates").query(
        "transition==@transition"
    )
    rates = rates.drop("transition", axis=1).reset_index(drop=True)
    rates = rates.rename({"value": 0}, axis=1)
    return rates


# noinspection PyUnusedLocal
def load_mam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT2)


# noinspection PyUnusedLocal
def load_mam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT2].reset_index()


# noinspection PyUnusedLocal
def load_sam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT1].reset_index()


# Sub-loader functions
def load_child_wasting_exposures(builder: Builder) -> pd.DataFrame:
    exposures = (
        builder.data.load(WASTING.EXPOSURE)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .pivot(columns="parameter")
    )

    exposures.columns = exposures.columns.droplevel(0)
    return exposures


def load_child_wasting_birth_prevalence(
    builder: Builder, wasting_category: str
) -> pd.DataFrame:
    exposure = load_child_wasting_exposures(builder)[wasting_category]
    birth_prevalence = (
        exposure[
            exposure.index.get_level_values("age_end")
            == data_values.WASTING.DYNAMIC_START_AGE
        ]
        .droplevel(["age_start", "age_end"])
        .reset_index()
    )
    return birth_prevalence
