from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.utilities import EntityString, is_non_zero

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
    models,
    scenarios,
)
from vivarium_gates_nutrition_optimization_child.constants.data_keys import WASTING
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class WastingTreatment(Risk):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        base_risk_config = super().configuration_defaults
        return {self.risk: base_risk_config[self.name]}

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

    @property
    def columns_created(self) -> List[str]:
        return [self.propensity_column_name, self.previous_treatment_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", self.previous_wasting_column, self.wasting_column]

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
        self.wasting_exposure = builder.value.get_value(
            data_values.PIPELINES.WASTING_EXPOSURE
        )
        self.underweight_exposure = builder.value.get_value(
            data_values.PIPELINES.UNDERWEIGHT_EXPOSURE
        )
        self.treatment_exposure = builder.value.get_value(
            f"{self.treated_state}_treatment.exposure"
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # propensity
        self.population_view.update(
            pd.Series(
                self.randomness.get_draw(pop_data.index), name=self.propensity_column_name
            )
        )
        # previous treatment (initialize as uncovered)
        self.population_view.update(
            pd.Series("cat1", index=pop_data.index, name=self.previous_treatment_column)
        )

    def on_time_step_prepare(self, event: Event):
        """define previous_treatment and redraw propensities upon transition to new wasting state"""
        pop = self.population_view.get(event.index)
        # previous treatment column (for results stratification)
        previous_treatment = self.treatment_exposure(pop.index)
        previous_treatment.name = self.previous_treatment_column
        # update propensity
        propensity = pop[self.propensity_column_name]
        remitted_mask = (pop[self.previous_wasting_column] == self.treated_state) & pop[
            self.wasting_column
        ] != self.treated_state
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(pd.concat([previous_treatment, propensity], axis=1))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        is_mam_component = self.risk.name == "moderate_acute_malnutrition_treatment"
        coverage_to_exposure_map = {"none": "cat1", "full": "cat2"}

        if is_mam_component:
            mam_coverage = self.scenario.mam_tx_coverage
            if mam_coverage == "baseline":  # return standard exposure if baseline
                propensity = self.propensity(index)
                return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
            elif mam_coverage == "targeted":
                # initialize exposures as cat1 (untreated)
                exposures = pd.Series("cat1", index=index)

                # define relevant booleans
                wasting = self.wasting_exposure(index)
                age = self.population_view.get(index)["age"]
                underweight = self.underweight_exposure(index)

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
                age = self.population_view.get(index)["age"]
                exposure.loc[age < 0.5] = "cat1"
                return exposure

        else:  # we're in the SAM treatment component
            sam_coverage = self.scenario.sam_tx_coverage
            if sam_coverage == "baseline":
                propensity = self.propensity(index)
                return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
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

    @property
    def columns_created(self) -> List[str]:
        return [self.state_column, f"initial_{self.state_column}_propensity"]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "sex", "birth_weight_status"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["age", "sex", "birth_weight_status"],
            "requires_values": [],
            "requires_streams": [],
        }

    def setup(self, builder):
        """Perform this component's setup."""
        super(DiseaseModel, self).setup(builder)

        self.configuration_age_start = builder.configuration.population.initialization_age_min
        self.configuration_age_end = builder.configuration.population.initialization_age_max
        self.randomness = builder.randomness.get_stream(f"{self.state_column}_initial_states")

        self.exposure = builder.value.register_value_producer(
            f"{self.state_column}.exposure",
            source=self.get_current_exposure,
            requires_columns=["age", "alive", self.state_column],
            preferred_post_processor=get_exposure_post_processor(
                builder, EntityString(f"risk_factor.{self.state_column}")
            ),
        )

        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=["age", "sex"],
        )

    def on_initialize_simulants(self, pop_data):
        initial_propensity = self.randomness.get_draw(pop_data.index).rename(
            f"initial_{self.state_column}_propensity"
        )
        population = self.population_view.subview(["age", "sex"]).get(pop_data.index)

        assert self.initial_state in {s.state_id for s in self.states}

        state_names, weights_bins = self.get_state_weights(pop_data.index, "birth_prevalence")

        if state_names and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population["sex_id"] = population.sex.apply({"Male": 1, "Female": 2}.get)

            condition_column = self.assign_initial_status_to_simulants(
                population,
                state_names,
                weights_bins,
                initial_propensity,
            )

            condition_column = condition_column.rename(
                columns={"condition_state": self.state_column}
            )
        else:
            condition_column = pd.Series(
                self.initial_state, index=population.index, name=self.state_column
            )
        self.population_view.update(pd.concat([condition_column, initial_propensity], axis=1))

    @staticmethod
    def assign_initial_status_to_simulants(
        simulants_df, state_names, weights_bins, propensities
    ) -> pd.DataFrame:
        simulants = simulants_df[["age", "sex"]].copy()

        choice_index = (propensities.values[np.newaxis].T > weights_bins).sum(axis=1)
        initial_states = pd.Series(np.array(state_names)[choice_index], index=simulants.index)

        simulants.loc[:, "condition_state"] = initial_states
        return simulants

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        exposure = pop[self.state_column].apply(models.get_risk_category)
        return exposure


# noinspection PyPep8Naming
def ChildWasting() -> ChildWastingModel:
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
    better_moderate = DiseaseState(
        models.WASTING.BETTER_MODERATE_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_better_mam_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_better_mam_birth_prevalence,
        },
    )
    worse_moderate = DiseaseState(
        models.WASTING.WORSE_MODERATE_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_worse_mam_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_worse_mam_birth_prevalence,
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
    tmrel.add_rate_transition(
        mild,
        get_data_functions={
            "incidence_rate": load_wasting_rate,
        },
    )

    # Add transitions for mild
    mild.allow_self_transitions()
    mild.add_rate_transition(
        better_moderate,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    mild.add_rate_transition(
        worse_moderate,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    mild.add_rate_transition(
        tmrel,
        get_data_functions={
            "remission_rate": load_mild_remission_rate,
        },
    )

    # Add transitions for moderate
    better_moderate.allow_self_transitions()
    better_moderate.add_rate_transition(
        severe,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    better_moderate.add_rate_transition(
        mild,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    worse_moderate.allow_self_transitions()
    worse_moderate.add_rate_transition(
        severe,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    worse_moderate.add_rate_transition(
        mild,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    # Add transitions for severe
    severe.allow_self_transitions()
    severe.add_rate_transition(
        better_moderate,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    severe.add_rate_transition(
        worse_moderate,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    severe.add_rate_transition(
        mild,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    return ChildWastingModel(
        models.WASTING.MODEL_NAME,
        get_data_functions={"cause_specific_mortality_rate": lambda *_: 0},
        states=[severe, better_moderate, worse_moderate, mild, tmrel],
    )


# noinspection PyUnusedLocal
def load_pem_excess_mortality_rate(builder: Builder, cause: str) -> pd.DataFrame:
    return builder.data.load(data_keys.PEM.EMR)


# noinspection PyUnusedLocal
def load_mild_wasting_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT3)


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return (
        load_child_wasting_exposures(builder)[WASTING.CAT3]
        .reset_index()
        .rename(columns={WASTING.CAT3: "value"})
    )


def load_wasting_rate(builder: Builder, *wasting_states) -> pd.DataFrame:
    states_to_transition_map = {
        ("mild_child_wasting",): "inc_rate_mild",
        ("mild_child_wasting", "better_moderate_acute_malnutrition"): "inc_rate_better_mam",
        ("mild_child_wasting", "worse_moderate_acute_malnutrition"): "inc_rate_worse_mam",
        ("better_moderate_acute_malnutrition", "severe_acute_malnutrition"): "inc_rate_sam",
        ("worse_moderate_acute_malnutrition", "severe_acute_malnutrition"): "inc_rate_sam",
        ("susceptible_to_child_wasting",): "rem_rate_mild",
        ("better_moderate_acute_malnutrition", "mild_child_wasting"): "rem_rate_mam",
        ("worse_moderate_acute_malnutrition", "mild_child_wasting"): "rem_rate_mam",
        ("severe_acute_malnutrition", "mild_child_wasting"): "tx_rem_rate_sam",
        (
            "severe_acute_malnutrition",
            "better_moderate_acute_malnutrition",
        ): "sam_to_better_mam",
        (
            "severe_acute_malnutrition",
            "worse_moderate_acute_malnutrition",
        ): "sam_to_worse_mam",
    }
    transition = states_to_transition_map[wasting_states]
    data = get_transition_data(builder, transition)
    return data


def load_mild_remission_rate(builder: Builder, input_state) -> pd.DataFrame:
    return get_transition_data(builder, "rem_rate_mild")


def get_transition_data(builder: Builder, transition: str) -> pd.DataFrame:
    rates = builder.data.load("risk_factor.child_wasting.transition_rates").query(
        "transition==@transition"
    )
    rates = rates.drop("transition", axis=1).reset_index(drop=True)
    return rates


# noinspection PyUnusedLocal
def load_better_mam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT25)


# noinspection PyUnusedLocal
def load_worse_mam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT2)


# noinspection PyUnusedLocal
def load_better_mam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return (
        load_child_wasting_exposures(builder)[WASTING.CAT25]
        .reset_index()
        .rename(columns={WASTING.CAT25: "value"})
    )


# noinspection PyUnusedLocal
def load_worse_mam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return (
        load_child_wasting_exposures(builder)[WASTING.CAT2]
        .reset_index()
        .rename(columns={WASTING.CAT2: "value"})
    )


# noinspection PyUnusedLocal
def load_sam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return (
        load_child_wasting_exposures(builder)[WASTING.CAT1]
        .reset_index()
        .rename(columns={WASTING.CAT1: "value"})
    )


# Sub-loader functions
def load_child_wasting_exposures(builder: Builder) -> pd.DataFrame:
    exposures = (
        builder.data.load(WASTING.EXPOSURE)
        .reset_index()
        .set_index(metadata.DEMOGRAPHIC_COLUMNS + ["subnational"])
        .drop(columns="index")
        .pivot(columns="parameter")
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
