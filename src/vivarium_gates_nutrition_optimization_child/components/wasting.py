from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from vivarium.framework.artifact.artifact import ArtifactException
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
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


class WastingTreatment(Risk):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        base_risk_config = super().configuration_defaults
        return {self.name: base_risk_config[self.name]}

    @property
    def time_step_prepare_priority(self) -> int:
        # we want to reset propensities before updating previous state column
        return 4

    def __init__(self, treatment_type: str):
        super().__init__(treatment_type)

        self.previous_wasting_column = f"previous_{data_keys.WASTING.name}"
        self.wasting_column = data_keys.WASTING.name
        self.treated_state = self._get_treated_state()
        self.previous_treatment_column = f"previous_{self.treated_state}_treatment"

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
        self.scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.child_scenario
        ]

        builder.population.register_initializer(
            self.initialize_previous_treatment_column, self.previous_treatment_column
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_previous_treatment_column(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series("cat1", index=pop_data.index, name=self.previous_treatment_column)
        )

    def on_time_step_prepare(self, event: Event):
        """define previous_treatment and redraw propensities upon transition to new wasting state"""
        pop = self.population_view.get_attributes(
            event.index,
            [
                self.wasting_column,
                self.previous_wasting_column,
                self.propensity_name,
                self.exposure_name,
            ],
        )
        # previous treatment column (for results stratification)
        previous_treatment = pop[self.exposure_name]
        previous_treatment.name = self.previous_treatment_column
        # update propensity
        propensity = pop[self.propensity_name].copy()
        remitted_mask = (pop[self.previous_wasting_column] == self.treated_state) & pop[
            self.wasting_column
        ] != self.treated_state
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(pd.concat([previous_treatment, propensity], axis=1))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def register_exposure_pipeline(self, builder):
        builder.value.register_attribute_producer(
            self.exposure_name,
            source=self.get_current_exposure,
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
        )

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        if index.empty:
            return pd.Series(index=index)

        # Convert RangeIndex into Index if needed
        if isinstance(index, pd.RangeIndex):
            index = pd.Index(index.to_list())

        is_mam_component = self.risk.name == "moderate_acute_malnutrition_treatment"
        coverage_to_exposure_map = {"none": "cat1", "full": "cat2"}

        if is_mam_component:
            mam_coverage = self.scenario.mam_tx_coverage
            if mam_coverage == "baseline":  # return standard exposure if baseline
                return self.exposure_distribution.exposure_ppf(index)
            elif mam_coverage == "targeted":
                # initialize exposures as cat1 (untreated)
                exposures = pd.Series("cat1", index=index)
                # define relevant booleans
                pop = self.population_view.get_attributes(
                    index,
                    [
                        "age",
                        data_values.PIPELINES.WASTING_EXPOSURE,
                        data_values.PIPELINES.UNDERWEIGHT_EXPOSURE,
                    ],
                )
                age = pop["age"]
                wasting = pop[data_values.PIPELINES.WASTING_EXPOSURE]
                underweight = pop[data_values.PIPELINES.UNDERWEIGHT_EXPOSURE]

                in_mam_state = (wasting == "cat2") | (wasting == "cat2.5")
                in_worse_mam_state = wasting == "cat2"
                in_age_range = (age >= 0.5) & (age < 2)
                is_severely_underweight = underweight == "cat1"
                under_6_months = age < 0.5

                is_covered = in_mam_state & (
                    (in_age_range | is_severely_underweight) | in_worse_mam_state
                )

                exposures.loc[is_covered] = "cat2"
                exposures.loc[under_6_months] = "cat1"
                return exposures
            else:  # except for simulants under 6 months who are untreated,
                # return either all or none covered
                exposure_value = coverage_to_exposure_map[mam_coverage]
                exposure = pd.Series(exposure_value, index=index)
                age = self.population_view.get_attributes(index, "age")
                exposure.loc[age < 0.5] = "cat1"
                return exposure

        else:  # we're in the SAM treatment component
            sam_coverage = self.scenario.sam_tx_coverage
            if sam_coverage == "baseline":
                return self.exposure_distribution.exposure_ppf(index)
            else:
                exposure = coverage_to_exposure_map[sam_coverage]
                return pd.Series(exposure, index=index)


class ChildWastingModel(DiseaseModel):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        disease_config = super().configuration_defaults
        risk_config = {
            "risk_factor.child_wasting": {
                "data_sources": {
                    "exposure": "risk_factor.child_wasting.exposure",
                },
                "rebinned_exposed": [],
                "category_thresholds": [],
            }
        }
        return {**disease_config, **risk_config}

    def setup(self, builder):
        """Perform this component's setup."""

        self.initialization_weights_pipelines = [
            state.birth_prevalence_pipeline for state in self.states
        ]

        self.configuration_age_start = builder.configuration.population.initialization_age_min
        self.configuration_age_end = builder.configuration.population.initialization_age_max
        self.randomness = builder.randomness.get_stream(f"{self.state_column}_initial_states")

        builder.value.register_attribute_producer(
            f"{self.state_column}.exposure",
            source=self.get_current_exposure,
            required_resources=["age", "alive", self.state_column],
            preferred_post_processor=get_exposure_post_processor(
                builder, EntityString(f"risk_factor.{self.state_column}")
            ),
        )

        self.csmr_table = self.build_lookup_table(builder, "cause_specific_mortality_rate")
        builder.value.register_attribute_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            required_resources=["age", "sex"],
        )

        builder.population.register_initializer(
            self.initialize_wasting,
            [
                self.state_column,
                f"initial_{self.state_column}_propensity",
            ],
            required_resources=[
                self.randomness,
                *self.initialization_weights_pipelines,
            ],
        )

    def initialize_wasting(self, pop_data):
        initial_propensity = self.randomness.get_draw(pop_data.index).rename(
            f"initial_{self.state_column}_propensity"
        )

        birth_prevalence = self.population_view.get_attributes(
            pop_data.index, self.initialization_weights_pipelines
        )

        if not pop_data.index.empty:
            condition = self.assign_initial_status_to_simulants(
                pop_data.index,
                birth_prevalence,
                initial_propensity,
            )
            condition.name = self.state_column
        else:
            condition = pd.Series(
                self.residual_state.state_id, index=pop_data.index, name=self.state_column
            )
        self.population_view.update(pd.concat([condition, initial_propensity], axis=1))

    @staticmethod
    def assign_initial_status_to_simulants(
        pop_index, birth_prevalence, propensities
    ) -> pd.Series:
        state_names = birth_prevalence.columns.tolist()
        state_names = [name.replace(".birth_prevalence", "") for name in state_names]
        weights = birth_prevalence.to_numpy()
        cumulative_weights = np.cumsum(weights, axis=1)
        choice_index = (propensities.values[np.newaxis].T > cumulative_weights).sum(axis=1)
        initial_states = pd.Series(np.array(state_names)[choice_index], index=pop_index)
        return initial_states

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        return self.population_view.get_attributes(index, self.state_column).apply(
            self.get_risk_category
        )

    @staticmethod
    def get_risk_category(state_name: str) -> str:
        return {
            models.WASTING.SUSCEPTIBLE_STATE_NAME: data_keys.WASTING.CAT4,
            models.WASTING.MILD_STATE_NAME: data_keys.WASTING.CAT3,
            models.WASTING.BETTER_MODERATE_STATE_NAME: data_keys.WASTING.CAT25,
            models.WASTING.WORSE_MODERATE_STATE_NAME: data_keys.WASTING.CAT2,
            models.WASTING.SEVERE_STATE_NAME: data_keys.WASTING.CAT1,
        }[state_name]


# noinspection PyPep8Naming
def ChildWasting() -> ChildWastingModel:
    tmrel = SusceptibleState(models.WASTING.MODEL_NAME)
    mild = DiseaseState(
        models.WASTING.MILD_STATE_NAME,
        cause_type="sequela",
        prevalence=lambda builder: load_mild_wasting_exposure(builder),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
        birth_prevalence=lambda builder: load_mild_wasting_birth_prevalence(builder),
    )
    better_moderate = DiseaseState(
        models.WASTING.BETTER_MODERATE_STATE_NAME,
        cause_type="sequela",
        prevalence=lambda builder: load_better_mam_exposure(builder),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
        birth_prevalence=lambda builder: load_better_mam_birth_prevalence(builder),
    )
    worse_moderate = DiseaseState(
        models.WASTING.WORSE_MODERATE_STATE_NAME,
        cause_type="sequela",
        prevalence=lambda builder: load_worse_mam_exposure(builder),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
        birth_prevalence=lambda builder: load_worse_mam_birth_prevalence(builder),
    )
    severe = DiseaseState(
        models.WASTING.SEVERE_STATE_NAME,
        cause_type="sequela",
        prevalence=lambda builder: load_sam_exposure(builder),
        disability_weight=0.0,
        excess_mortality_rate=0.0,
        birth_prevalence=lambda builder: load_sam_birth_prevalence(builder),
    )
    # Add transitions for tmrel
    tmrel.add_rate_transition(
        mild,
        transition_rate=lambda builder: get_transition_data(builder, "inc_rate_mild"),
    )

    # Add transitions for mild
    mild.add_rate_transition(
        better_moderate,
        transition_rate=lambda builder: get_transition_data(builder, "inc_rate_better_mam"),
    )
    mild.add_rate_transition(
        worse_moderate,
        transition_rate=lambda builder: get_transition_data(builder, "inc_rate_worse_mam"),
    )
    mild.add_rate_transition(
        tmrel,
        transition_rate=lambda builder: get_transition_data(builder, "rem_rate_mild"),
    )

    # Add transitions for moderate
    better_moderate.add_rate_transition(
        severe,
        transition_rate=lambda builder: get_transition_data(builder, "inc_rate_sam"),
    )
    better_moderate.add_rate_transition(
        mild,
        transition_rate=lambda builder: get_transition_data(builder, "rem_rate_mam"),
    )

    worse_moderate.add_rate_transition(
        severe,
        transition_rate=lambda builder: get_transition_data(builder, "inc_rate_sam"),
    )
    worse_moderate.add_rate_transition(
        mild,
        transition_rate=lambda builder: get_transition_data(builder, "rem_rate_mam"),
    )

    # Add transitions for severe
    severe.add_rate_transition(
        better_moderate,
        transition_rate=lambda builder: get_transition_data(builder, "sam_to_better_mam"),
    )
    severe.add_rate_transition(
        worse_moderate,
        transition_rate=lambda builder: get_transition_data(builder, "sam_to_worse_mam"),
    )
    severe.add_rate_transition(
        mild,
        transition_rate=lambda builder: get_transition_data(builder, "tx_rem_rate_sam"),
    )

    return ChildWastingModel(
        models.WASTING.MODEL_NAME,
        cause_specific_mortality_rate=0.0,
        states=[severe, better_moderate, worse_moderate, mild, tmrel],
    )


# noinspection PyUnusedLocal
def load_pem_excess_mortality_rate(builder: Builder, cause: str) -> pd.DataFrame:
    return builder.data.load(data_keys.PEM.EMR)


# noinspection PyUnusedLocal
def load_mild_wasting_birth_prevalence(builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, data_keys.WASTING.CAT3)


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(builder: Builder) -> Union[float, pd.DataFrame]:
    exposure = load_child_wasting_exposures(builder)
    if isinstance(exposure, pd.DataFrame):
        exposure = (
            exposure[data_keys.WASTING.CAT3]
            .reset_index()
            .rename(columns={data_keys.WASTING.CAT3: "value"})
        )
    return exposure


# def load_mild_remission_rate(builder: Builder, input_state) -> pd.DataFrame:
#     return get_transition_data(builder, "rem_rate_mild")


def get_transition_data(builder: Builder, transition: str) -> pd.DataFrame:
    rates = builder.data.load("risk_factor.child_wasting.transition_rates").query(
        "transition==@transition"
    )
    rates = rates.drop("transition", axis=1).reset_index(drop=True)
    return rates


# noinspection PyUnusedLocal
def load_better_mam_birth_prevalence(builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, data_keys.WASTING.CAT25)


# noinspection PyUnusedLocal
def load_worse_mam_birth_prevalence(builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, data_keys.WASTING.CAT2)


# noinspection PyUnusedLocal
def load_better_mam_exposure(builder: Builder) -> Union[float, pd.DataFrame]:
    exposure = load_child_wasting_exposures(builder)
    if isinstance(exposure, pd.DataFrame):
        exposure = (
            exposure[data_keys.WASTING.CAT25]
            .reset_index()
            .rename(columns={data_keys.WASTING.CAT25: "value"})
        )
    return exposure


# noinspection PyUnusedLocal
def load_worse_mam_exposure(builder: Builder) -> Union[float, pd.DataFrame]:
    exposure = load_child_wasting_exposures(builder)
    if isinstance(exposure, pd.DataFrame):
        exposure = (
            exposure[data_keys.WASTING.CAT2]
            .reset_index()
            .rename(columns={data_keys.WASTING.CAT2: "value"})
        )
    return exposure


# noinspection PyUnusedLocal
def load_sam_birth_prevalence(builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, data_keys.WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(builder: Builder) -> Union[float, pd.DataFrame]:
    exposure = load_child_wasting_exposures(builder)
    if isinstance(exposure, pd.DataFrame):
        exposure = (
            exposure[data_keys.WASTING.CAT1]
            .reset_index()
            .rename(columns={data_keys.WASTING.CAT1: "value"})
        )
    return exposure


# Sub-loader functions
def load_child_wasting_exposures(builder: Builder) -> Union[float, pd.DataFrame]:

    # Get wasting exposure data
    try:
        exposures = builder.data.load(data_keys.WASTING.EXPOSURE)
    except ArtifactException:
        # No exposure if data is not in artifact
        return 0
    # Set index for either national or subnational
    try:
        exposures = exposures.set_index(metadata.ARTIFACT_INDEX_COLUMNS).pivot(
            columns="parameter"
        )
    except KeyError:
        exposures = exposures.set_index(metadata.SUBNATIONAL_INDEX_COLUMNS).pivot(
            columns="parameter"
        )
    exposures.columns = exposures.columns.droplevel(0)
    return exposures


def load_child_wasting_birth_prevalence(
    builder: Builder, wasting_category: str
) -> pd.DataFrame:
    birth_prevalence = builder.data.load(data_keys.WASTING.BIRTH_PREVALENCE)
    birth_prevalence = (
        birth_prevalence.query("parameter == @wasting_category")
        .reset_index()
        .drop(["parameter", "index"], axis=1)
        .rename(columns={wasting_category: "value"})
    )

    return birth_prevalence
