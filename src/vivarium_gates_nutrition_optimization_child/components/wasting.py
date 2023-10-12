from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.utilities import EntityString
from vivarium_public_health.utilities import is_non_zero
from vivarium.framework.values import list_combiner, union_post_processor

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
    models,
    scenarios,
)
from vivarium_gates_nutrition_optimization_child.constants.data_keys import WASTING
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class ChildWasting(Component):
    @property
    def columns_required(self) -> Optional[List[str]]:
        return [
            "alive",
            "age",
            self.dynamic_model.state_column,
        ]

    def __init__(self):
        super().__init__()
        self.dynamic_model = DynamicChildWasting()
        self.static_model = StaticChildWasting()

    @property
    def sub_components(self):
        return [
            self.dynamic_model,
            self.static_model,
        ]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.exposure = builder.value.register_value_producer(
            f"{self.name}.exposure",
            source=self.get_current_exposure,
            requires_columns=["age", "alive", self.dynamic_model.state_column],
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
        # use super's init to get exposure distribution
        # but overwrite other names
        super().__init__("risk_factor.child_wasting")

        name = "static_child_wasting"
        self._randomness_stream_name = f"initial_{name}_propensity"
        self.propensity_column_name = f"{name}_propensity"
        self.propensity_pipeline_name = f"{name}.propensity"
        self.exposure_pipeline_name = f"{name}.exposure"


class WastingTreatment(Risk):
    @property
    def time_step_prepare_priority(self) -> int:
        # we want to reset propensities before updating previous state column
        return 4

    @property
    def name(self) -> str:
        return f"wasting_treatment_{self.risk}"

    def __init__(self, treatment_type: str):
        super().__init__(treatment_type)

        self.previous_wasting_column = f"previous_{data_keys.WASTING.name}"
        self.wasting_column = data_keys.WASTING.name

        self.treated_state = self._get_treated_state()

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

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_prepare(self, event: Event):
        """'redraw propensities upon transition to new wasting state"""
        pop = self.population_view.get(event.index)
        propensity = pop[self.propensity_column_name]
        remitted_mask = (pop[self.previous_wasting_column] == self.treated_state) & pop[
            self.wasting_column
        ] != self.treated_state
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(propensity)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        is_mam_component = self.risk.name == "moderate_acute_malnutrition_treatment"
        coverage_to_exposure_map = {"none": "cat1", "full": "cat2"}

        # simulants under 6 months should not be on treatment
        if len(index) > 0:
            # all simulants are the same age so just check the first simulant
            if self.population_view.get(pd.Index([0]))['age'].squeeze() < .5:
                return pd.Series("cat1", index=index)

        if is_mam_component:
            mam_coverage = self.scenario.mam_tx_coverage
            if mam_coverage == "baseline":  # return standard exposure if baseline
                propensity = self.propensity(index)
                return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
            elif mam_coverage == "targeted":
                # initialize exposures as cat1 using index
                exposures = pd.Series("cat1", index=index)

                # define relevant booleans
                wasting = self.wasting_exposure(index)
                age = self.population_view.get(index)["age"]
                underweight = self.underweight_exposure(index)

                in_mam_state = wasting == "cat2"
                in_age_range = (age >= 0.5) & (age < 2)
                is_severely_underweight = underweight == "cat1"

                is_covered = (in_mam_state & in_age_range) | (
                    in_mam_state & is_severely_underweight
                )
                exposures.loc[is_covered] = "cat2"
                return exposures
            else:  # return either all or none covered otherwise
                exposure = coverage_to_exposure_map[mam_coverage]
                return pd.Series(exposure, index=index)

        else:  # we're in the SAM treatment component
            sam_coverage = self.scenario.sam_tx_coverage
            if sam_coverage == "baseline":
                propensity = self.propensity(index)
                return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
            else:
                exposure = coverage_to_exposure_map[sam_coverage]
                return pd.Series(exposure, index=index)


class WastingDiseaseState(DiseaseState):
    """DiseaseState where birth prevalence LookupTables is parametrized by birthweight status."""
    def setup(self, builder: Builder) -> None:
        """Identical to DiseaseState setup but use birthweight parameter when building birth prevalence lookup table."""
        super(DiseaseState, self).setup(builder)
        self.clock = builder.time.clock()

        prevalence_data = self.load_prevalence_data(builder)
        self.prevalence = builder.lookup.build_table(
            prevalence_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        self.birth_prevalence = builder.lookup.build_table(
            birth_prevalence_data, key_columns=["sex", "birth_weight_status"], parameter_columns=["year"]
        )

        dwell_time_data = self.load_dwell_time_data(builder)
        self.dwell_time = builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=builder.lookup.build_table(
                dwell_time_data, key_columns=["sex"], parameter_columns=["age", "year"]
            ),
            requires_columns=["age", "sex"],
        )

        disability_weight_data = self.load_disability_weight_data(builder)
        self.has_disability = is_non_zero(disability_weight_data)
        self.base_disability_weight = builder.lookup.build_table(
            disability_weight_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.disability_weight = builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", self.model],
        )
        builder.value.register_value_modifier(
            "disability_weight", modifier=self.disability_weight
        )

        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        self.has_excess_mortality = is_non_zero(excess_mortality_data)
        self.base_excess_mortality_rate = builder.lookup.build_table(
            excess_mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.excess_mortality_rate = builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=["age", "sex", "alive", self.model],
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

        self.randomness_prevalence = builder.randomness.get_stream(
            f"{self.state_id}_prevalent_cases"
        )


class DynamicChildWastingModel(DiseaseModel):
    @property
    def columns_created(self) -> List[str]:
        return [self.state_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "sex", "static_child_wasting_propensity", "birth_weight_status"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["age", "sex", "static_child_wasting_propensity"],
            "requires_values": [],
            "requires_streams": [],
        }

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
    mild = WastingDiseaseState(
        models.WASTING.MILD_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_mild_wasting_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_mild_wasting_birth_prevalence,
        },
    )
    moderate = WastingDiseaseState(
        models.WASTING.MODERATE_STATE_NAME,
        cause_type="sequela",
        get_data_functions={
            "prevalence": load_mam_exposure,
            "disability_weight": lambda *_: 0,
            "excess_mortality_rate": lambda *_: 0,
            "birth_prevalence": load_mam_birth_prevalence,
        },
    )
    severe = WastingDiseaseState(
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
        moderate,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    mild.add_rate_transition(
        tmrel,
        get_data_functions={
            "remission_rate": load_wasting_rate,
        },
    )

    # Add transitions for moderate
    moderate.allow_self_transitions()
    moderate.add_rate_transition(
        severe,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )
    moderate.add_rate_transition(
        mild,
        get_data_functions={
            "transition_rate": load_wasting_rate,
        },
    )

    # Add transitions for severe
    severe.allow_self_transitions()
    severe.add_rate_transition(
        moderate,
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


def load_wasting_rate(builder: Builder, *wasting_states) -> pd.DataFrame:
    states_to_transition_map = {
        ("mild_child_wasting",): "inc_rate_mild",
        ("mild_child_wasting", "moderate_acute_malnutrition"): "inc_rate_mam",
        ("moderate_acute_malnutrition", "severe_acute_malnutrition"): "inc_rate_sam",
        ("susceptible_to_child_wasting",): "rem_rate_mild",
        ("moderate_acute_malnutrition", "mild_child_wasting"): "rem_rate_mam",
        ("severe_acute_malnutrition", "mild_child_wasting"): "tx_rem_rate_sam",
        ("severe_acute_malnutrition", "moderate_acute_malnutrition"): "ux_rem_rate_sam",
    }
    transition = states_to_transition_map[wasting_states]
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
    birth_prevalence = builder.data.load(data_keys.WASTING.BIRTH_PREVALENCE)
    birth_prevalence = birth_prevalence.query("parameter == @wasting_category").reset_index().drop(['parameter', 'index'], axis=1)
    return birth_prevalence
