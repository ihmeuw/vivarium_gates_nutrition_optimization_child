from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView
from vivarium.framework.event import Event
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor
)
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.utilities import EntityString

from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values, metadata, models, scenarios
from vivarium_gates_nutrition_optimization_child.constants.data_keys import WASTING
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable

class ChildWasting:

    def __init__(self):
        self.dynamic_model = DynamicChildWasting()
        self.static_model = Risk('risk_factor.child_wasting')

    @property
    def sub_components(self):
        return [
            self.dynamic_model,
            self.static_model,
        ]

class RiskModel(DiseaseModel):

    configuration_defaults = {
        "risk": {
            "mild_child_wasting_untreated_recovery_time":
                data_values.WASTING.DEFAULT_MILD_WASTING_UX_RECOVERY_TIME,
        }
    }

    ##########################
    # Initialization methods #
    ##########################

    def __init__(self, risk, **kwargs):
        super().__init__(risk, **kwargs)
        self.configuration_defaults = self._get_configuration_defaults()

    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            self.state_column: {
                **Risk.configuration_defaults['risk'], **RiskModel.configuration_defaults['risk']
            }
        }

    # This would be a preferable name, but the generic DiseaseObserver works with no modifications
    # if we use the standard naming from DiseaseModel. Extending to DiseaseObserver to RiskObserver
    # would provide no functional gain and involve copy-pasting a bunch of code

    # @property
    # def name(self):
    #     return f"risk_model.{self.state_column}"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.configuration_age_start = builder.configuration.population.age_start
        self.configuration_age_end = builder.configuration.population.age_end

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(
            cause_specific_mortality_rate,
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        builder.value.register_value_modifier(
            'cause_specific_mortality_rate',
            self.adjust_cause_specific_mortality_rate,
            requires_columns=['age', 'sex']
        )

        self.population_view = builder.population.get_view(
            ['age', 'sex', self.state_column, f'initial_{self.state_column}_propensity']
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.state_column, f'initial_{self.state_column}_propensity'],
            requires_columns=['age', 'sex'],
            requires_streams=[f'{self.state_column}_initial_states'],
        )

        self.randomness = builder.randomness.get_stream(f'{self.state_column}_initial_states')

        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)

        self.exposure = builder.value.register_value_producer(
            f'{self.state_column}.exposure',
            source=self.get_current_exposure,
            requires_columns=['age', 'sex', self.state_column],
            preferred_post_processor=get_exposure_post_processor(
                builder, EntityString(f'risk_factor.{self.state_column}')
            )
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data):
        super().on_initialize_simulants(pop_data)
        initial_propensity = (
            self.randomness
            .get_draw(pop_data.index)
            .rename(f'initial_{self.state_column}_propensity')
        )
        self.population_view.update(initial_propensity)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_current_exposure(self, index: pd.Index) -> pd.Series:
        wasting_state = (
            self.population_view.subview([self.state_column]).get(index).squeeze(axis=1)
        )
        return wasting_state.apply(models.get_risk_category)

class WastingTreatment(Risk):

    def __init__(self, treatment_type: str):
        super().__init__(treatment_type)

        self.previous_wasting_column = f'previous_{data_keys.WASTING.name}'
        self.wasting_column = data_keys.WASTING.name

        self.treated_state = self._get_treated_state()

    ##########################
    # Initialization methods #
    ##########################

    def _get_treated_state(self) -> str:
        return self.risk.name.split('_treatment')[0]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self._register_on_time_step_prepare_listener(builder)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.propensity_column_name,
                                            self.previous_wasting_column,
                                            self.wasting_column])

    def _register_on_time_step_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_cleanup(self, event: Event):
        pop = self.population_view.get(event.index)
        propensity = pop[self.propensity_column_name]
        remitted_mask = ((pop[self.previous_wasting_column] == self.treated_state)
                         & pop[self.wasting_column] != self.treated_state)
        propensity.loc[remitted_mask] = self.randomness.get_draw(remitted_mask.index)
        self.population_view.update(propensity)

# noinspection PyPep8Naming
def DynamicChildWasting() -> RiskModel:
    tmrel = SusceptibleState(models.WASTING.MODEL_NAME)
    mild = DiseaseState(
        models.WASTING.MILD_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mild_wasting_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'birth_prevalence': load_mild_wasting_birth_prevalence,
        }
    )
    moderate = DiseaseState(
        models.WASTING.MODERATE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mam_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'birth_prevalence': load_mam_birth_prevalence,
        }
    )
    severe = DiseaseState(
        models.WASTING.SEVERE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_sam_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'birth_prevalence': load_sam_birth_prevalence,
        }
    )

    # Add transitions for tmrel
    tmrel.allow_self_transitions()
    tmrel.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'incidence_rate': load_mild_wasting_incidence_rate,
        }
    )

    # Add transitions for mild
    mild.allow_self_transitions()
    mild.add_transition(
        moderate,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_mam_incidence_rate,
        }
    )
    mild.add_transition(
        tmrel,
        source_data_type='rate',
        get_data_functions={
            'remission_rate': load_mild_wasting_remission_rate,
        }
    )

    # Add transitions for moderate
    moderate.allow_self_transitions()
    moderate.add_transition(
        severe,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_incidence_rate,
        }
    )
    moderate.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_mam_remission_rate,
        }
    )

    # Add transitions for severe
    severe.allow_self_transitions()
    severe.add_transition(
        moderate,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_untreated_remission_rate,
        }
    )
    severe.add_transition(
        mild,
        source_data_type='rate',
        get_data_functions={
            'transition_rate': load_sam_treated_remission_rate,
        }
    )

    return RiskModel(
        models.WASTING.MODEL_NAME,
        get_data_functions={'cause_specific_mortality_rate': lambda *_: 0},
        states=[severe, moderate, mild, tmrel]
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


# noinspection PyUnusedLocal, DuplicatedCode
def load_mild_wasting_incidence_rate(builder: Builder, cause: str) -> pd.DataFrame:
    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_mild_incidence_probability(
        builder, exposures, adjustment, mortality_probs
    )
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


# noinspection DuplicatedCode
def get_daily_mild_incidence_probability(
        builder: Builder,
        exposures: pd.DataFrame,
        adjustment: pd.Series,
        mortality_probs: pd.DataFrame,
) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    mild_remission_prob = get_mild_wasting_remission_probability(
        builder, adj_exposures[WASTING.CAT3].index
    )

    # i3: ap0*f4/ap4 + ap3*r4/ap4 - d4
    i3 = (
        adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT4]
        + adj_exposures[WASTING.CAT3] * mild_remission_prob / adj_exposures[WASTING.CAT4]
        - mortality_probs[WASTING.CAT4]
    )
    _reset_underage_transitions(i3)
    return i3


# noinspection PyUnusedLocal
def load_mild_wasting_remission_rate(builder: Builder, cause: str) -> pd.DataFrame:
    index = _get_index(builder)
    daily_probability = get_mild_wasting_remission_probability(builder, index)
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


def get_mild_wasting_remission_probability(builder: Builder, index: pd.Index) -> pd.Series:
    draw = builder.configuration.input_data.input_draw_number
    r4_over_12mo = get_random_variable(draw, *data_values.WASTING.R4_OVER_12MO)
    r4_under_12mo = get_random_variable(draw, *data_values.WASTING.R4_UNDER_12MO)

    r4 = pd.Series(index=index, name='mild_wasting_remission')
    r4[index.get_level_values('age_end') <= 5.0] = r4_over_12mo
    r4[index.get_level_values('age_end') <= 1.0] = r4_under_12mo

    _reset_underage_transitions(r4)
    return 1 - np.exp(-r4)


# noinspection PyUnusedLocal
def load_mam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT2)


# noinspection PyUnusedLocal
def load_mam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT2].reset_index()


# noinspection PyUnusedLocal, DuplicatedCode
def load_mam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    draw = builder.configuration.input_data.input_draw_number
    mam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT2)
    sam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT1)
    mam_tx_efficacy = get_random_variable(draw, *data_values.WASTING.BASELINE_MAM_TX_EFFICACY)
    sam_tx_efficacy = get_random_variable(draw, *data_values.WASTING.BASELINE_SAM_TX_EFFICACY)

    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_mam_incidence_probability(
        builder,
        exposures,
        adjustment,
        mortality_probs,
        mam_tx_coverage,
        sam_tx_coverage,
        mam_tx_efficacy,
        sam_tx_efficacy
    )
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


# noinspection DuplicatedCode
def get_daily_mam_incidence_probability(
        builder: Builder,
        exposures: pd.DataFrame,
        adjustment: pd.Series,
        mortality_probs: pd.DataFrame,
        mam_tx_coverage: float,
        sam_tx_coverage: float,
        mam_tx_efficacy: float,
        sam_tx_efficacy: float
) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    treated_sam_remission_prob = get_daily_sam_treated_remission_probability(
        adj_exposures.index, sam_tx_coverage, sam_tx_efficacy
    )
    mam_remission_prob = get_daily_mam_remission_probability(
        builder, adj_exposures.index, mam_tx_coverage, mam_tx_efficacy
    )

    # i2: ap0*f3/ap3 + ap0*f4/ap3 + ap1*t1/ap3 + ap2*r3/ap3 - d3 - ap4*d4/ap3
    i2 = (
        adjustment * exposures[WASTING.CAT3] / adj_exposures[WASTING.CAT3]
        + adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT3]
        + adj_exposures[WASTING.CAT1] * treated_sam_remission_prob / adj_exposures[WASTING.CAT3]
        + adj_exposures[WASTING.CAT2] * mam_remission_prob / adj_exposures[WASTING.CAT3]
        - mortality_probs[WASTING.CAT3]
        - adj_exposures[WASTING.CAT4] * mortality_probs[WASTING.CAT4] / adj_exposures[WASTING.CAT3]
    )
    _reset_underage_transitions(i2)
    return i2


# noinspection PyUnusedLocal
def load_mam_remission_rate(builder: Builder, *args) -> float:
    draw = builder.configuration.input_data.input_draw_number
    index = _get_index(builder)
    mam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT2)
    mam_tx_efficacy = get_random_variable(draw, *data_values.WASTING.BASELINE_MAM_TX_EFFICACY)

    daily_probability = get_daily_mam_remission_probability(builder, index, mam_tx_coverage, mam_tx_efficacy)
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


def get_daily_mam_remission_probability(
        builder: Builder,
        index: pd.Index,
        mam_tx_coverage: float,
        mam_tx_efficacy: float
) -> pd.Series:
    draw = builder.configuration.input_data.input_draw_number
    mam_tx_recovery_time = pd.Series(index=index, name='mam_remission')
    mam_tx_recovery_time[index.get_level_values('age_start') < 0.5] = (
        data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO
    )
    mam_tx_recovery_time[0.5 <= index.get_level_values('age_start')] = (
        get_random_variable(draw, *data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO)
    )
    mam_tx_eff_coverage = mam_tx_coverage * mam_tx_efficacy

    # r3: mam_tx_eff_coverage * 1/mam_tx_recovery_time
    #     + (1-mam_tx_eff_coverage)*(1/mam_ux_recovery_time)
    annual_remission_rate = (
        mam_tx_eff_coverage * metadata.YEAR_DURATION / mam_tx_recovery_time
        + ((1 - mam_tx_eff_coverage) * metadata.YEAR_DURATION
           / data_values.WASTING.MAM_UX_RECOVERY_TIME_OVER_6MO)
    )
    r3 = _convert_annual_rate_to_daily_probability(annual_remission_rate)
    _reset_underage_transitions(r3)
    return r3


# noinspection PyUnusedLocal
def load_sam_birth_prevalence(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(builder: Builder, cause: str) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT1].reset_index()


# noinspection PyUnusedLocal, DuplicatedCode
def load_sam_incidence_rate(builder: Builder, *args) -> pd.DataFrame:
    draw = builder.configuration.input_data.input_draw_number
    sam_k_distribution = scenarios.SAM_K_SCENARIOS[builder.configuration.sam_k].distribution

    sam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT1)
    sam_tx_efficacy = get_random_variable(draw, *data_values.WASTING.BASELINE_SAM_TX_EFFICACY)
    sam_k = get_random_variable(draw, *sam_k_distribution)

    exposures = load_child_wasting_exposures(builder)
    adjustment = load_acmr_adjustment(builder)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_sam_incidence_probability(
        exposures, adjustment, mortality_probs, sam_tx_coverage, sam_tx_efficacy, sam_k
    )
    incidence_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return incidence_rate.reset_index()


def get_daily_sam_incidence_probability(
        exposures: pd.DataFrame,
        adjustment: pd.Series,
        mortality_probs: pd.DataFrame,
        sam_tx_coverage: float,
        sam_tx_efficacy: float,
        sam_k: float
) -> pd.Series:
    adj_exposures = adjust_exposure(exposures, adjustment)
    treated_sam_remission_prob = get_daily_sam_treated_remission_probability(
        adj_exposures.index, sam_tx_coverage, sam_tx_efficacy
    )
    untreated_sam_remission_prob = get_daily_sam_untreated_remission_probability(
        mortality_probs, sam_tx_coverage, sam_tx_efficacy, sam_k
    )

    # i1: ap0*f2/ap2 + ap0*f3/ap2 + ap0*f4/ap2 + ap1*r2/ap2
    #     + ap1*t1/ap2 - d2 - ap3*d3/ap2 - ap4*d4/ap2
    i1 = (
        adjustment * exposures[WASTING.CAT2] / adj_exposures[WASTING.CAT2]
        + adjustment * exposures[WASTING.CAT3] / adj_exposures[WASTING.CAT2]
        + adjustment * exposures[WASTING.CAT4] / adj_exposures[WASTING.CAT2]
        + adj_exposures[WASTING.CAT1] * untreated_sam_remission_prob / adj_exposures[WASTING.CAT2]
        + adj_exposures[WASTING.CAT1] * treated_sam_remission_prob / adj_exposures[WASTING.CAT2]
        - mortality_probs[WASTING.CAT2]
        - adj_exposures[WASTING.CAT3] * mortality_probs[WASTING.CAT3] / adj_exposures[WASTING.CAT2]
        - adj_exposures[WASTING.CAT4] * mortality_probs[WASTING.CAT4] / adj_exposures[WASTING.CAT2]
    )
    _reset_underage_transitions(i1)
    return i1


# noinspection PyUnusedLocal
def load_sam_untreated_remission_rate(builder: Builder, *args) -> pd.Series:
    draw = builder.configuration.input_data.input_draw_number
    sam_k_distribution = scenarios.SAM_K_SCENARIOS[builder.configuration.sam_k].distribution

    sam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT1)
    sam_tx_efficacy = get_random_variable(draw, *data_values.WASTING.BASELINE_SAM_TX_EFFICACY)
    sam_k = get_random_variable(draw, *sam_k_distribution)
    mortality_probs = load_daily_mortality_probabilities(builder)

    daily_probability = get_daily_sam_untreated_remission_probability(
        mortality_probs, sam_tx_coverage, sam_tx_efficacy, sam_k
    )
    remission_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return remission_rate.reset_index()


def get_daily_sam_untreated_remission_probability(
        mortality_probs: pd.DataFrame,
        sam_tx_coverage: pd.Series,
        sam_tx_efficacy: float,
        sam_k: float
) -> pd.Series:
    treated_sam_remission_prob = get_daily_sam_treated_remission_probability(
        mortality_probs[WASTING.CAT1].index, sam_tx_coverage, sam_tx_efficacy
    )
    treated_sam_remission_rate = _convert_daily_probability_to_annual_rate(
        treated_sam_remission_prob
    )
    sam_mortality_rate = _convert_daily_probability_to_annual_rate(mortality_probs[WASTING.CAT1])

    # r2: sam_k - t1 - d1
    annual_remission_rate = sam_k - treated_sam_remission_rate - sam_mortality_rate
    r2 = _convert_annual_rate_to_daily_probability(annual_remission_rate)
    _reset_underage_transitions(r2)
    return r2


# noinspection PyUnusedLocal
def load_sam_treated_remission_rate(builder: Builder, *args) -> float:
    index = _get_index(builder)
    sam_tx_coverage = load_wasting_treatment_coverage(builder, data_keys.WASTING.CAT1)
    sam_tx_efficacy = get_random_variable(
        builder.configuration.input_data.input_draw_number,
        *data_values.WASTING.BASELINE_SAM_TX_EFFICACY
    )

    daily_probability = get_daily_sam_treated_remission_probability(
        index, sam_tx_coverage, sam_tx_efficacy
    )
    remission_rate = _convert_daily_probability_to_annual_rate(daily_probability)
    return remission_rate.reset_index()


def get_daily_sam_treated_remission_probability(
        index: pd.Index, sam_tx_coverage: pd.Series, sam_tx_efficacy: float
) -> float:
    sam_tx_recovery_time = pd.Series(
        data_values.WASTING.SAM_TX_RECOVERY_TIME_OVER_6MO, index=index, name='sam_remission'
    )

    # t1: tx_coverage * sam_tx_efficacy * (1/sam_tx_recovery_time)
    annual_remission_rate = (
        sam_tx_coverage * sam_tx_efficacy * metadata.YEAR_DURATION / sam_tx_recovery_time
    )
    t1 = _convert_annual_rate_to_daily_probability(annual_remission_rate)
    _reset_underage_transitions(t1)
    return t1


# Sub-loader functions

def load_child_wasting_exposures(builder: Builder) -> pd.DataFrame:
    exposures = (
        builder.data.load(WASTING.EXPOSURE)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .pivot(columns='parameter')
    )

    exposures.columns = exposures.columns.droplevel(0)
    return exposures


def load_wasting_treatment_coverage(builder: Builder, wasting_category: str) -> pd.Series:
    if wasting_category == data_keys.WASTING.CAT1:
        treatment_type = data_keys.SAM_TREATMENT
    elif wasting_category == data_keys.WASTING.CAT2:
        treatment_type = data_keys.MAM_TREATMENT
    else:
        raise ValueError(f'Not a treated wasting category: {wasting_category}')

    raw_data = builder.data.load(treatment_type.EXPOSURE)
    tx_coverage = (
        raw_data[raw_data.parameter == treatment_type.TMREL_CATEGORY]
        .drop(columns=['parameter'])
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .value
    )
    return tx_coverage


def load_child_wasting_birth_prevalence(builder: Builder, wasting_category: str) -> pd.DataFrame:
    exposure = load_child_wasting_exposures(builder)[wasting_category]
    birth_prevalence = (
        exposure[exposure.index.get_level_values('age_end') == data_values.WASTING.START_AGE]
        .droplevel(['age_start', 'age_end'])
        .reset_index()
    )
    return birth_prevalence


def load_acmr_adjustment(builder: Builder) -> pd.Series:
    acmr = get_data_series(builder.data.load(data_keys.POPULATION.ACMR))
    adjustment = _convert_annual_rate_to_daily_probability(acmr)
    return adjustment


def load_daily_mortality_probabilities(builder: Builder) -> pd.DataFrame:
    """"
    Returns a DataFrame with daily mortality probabilities for each wasting state

    DataFrame has the standard artifact index, and columns for each wasting state
    """

    # ---------- Load mortality rate input data ---------- #
    causes = [
        data_keys.DIARRHEA,
        data_keys.MEASLES,
        data_keys.LRI,
        data_keys.PEM,
    ]

    # acmr
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end' ]
    acmr = get_data_series(builder.data.load(data_keys.POPULATION.ACMR))

    # emr_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    emr_c = pd.concat([
        get_data_series(builder.data.load(c.EMR)).rename(c.name) for c in causes
    ], axis=1)
    emr_c.columns.name = 'affected_entity'
    emr_c = emr_c.stack()

    # csmr_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    csmr_c = pd.concat([
        get_data_series(builder.data.load(c.CSMR)).rename(c.name) for c in causes
    ], axis=1)
    csmr_c.columns.name = 'affected_entity'
    csmr_c = csmr_c.stack()

    # incidence_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    incidence_c = pd.concat(
        [
            get_data_series(builder.data.load(c.INCIDENCE_RATE)).rename(c.name)
            for c in causes if c != data_keys.PEM
        ], axis=1
    )
    incidence_c.columns.name = 'affected_entity'
    incidence_c = incidence_c.stack()

    # paf_c
    # index = [ 'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity' ]
    paf_c = (
        get_data_series(builder.data.load(WASTING.PAF))
        .droplevel('affected_measure')
    )

    # rr_ci
    # index = [
    #   'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter
    # ]
    rr_ci = (
        get_data_series(builder.data.load(WASTING.RELATIVE_RISK))
        .droplevel('affected_measure')
    )

    # duration_c
    # index = [
    #   'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter
    # ]
    diarrhea_duration = get_random_variable(
        builder.configuration.input_data.input_draw_number, *data_values.DIARRHEA_DURATION
    )
    lri_duration = get_random_variable(
        builder.configuration.input_data.input_draw_number, *data_values.LRI_DURATION
    )
    duration_c = (
        pd.Series(
            [diarrhea_duration, data_values.MEASLES_DURATION, lri_duration],
            index=pd.Index(
                [data_keys.DIARRHEA.name, data_keys.MEASLES.name, data_keys.LRI.name],
                name='affected_entity'
            )
        ).reindex(index=rr_ci.index, level='affected_entity')
    )
    duration_c.loc[duration_c.index.get_level_values('age_start') == 0.0] = (
        data_values.EARLY_NEONATAL_CAUSE_DURATION
    )
    duration_c = duration_c / metadata.YEAR_DURATION  # convert to duration in years

    # prevalence_pem_i
    # index = [
    #   'sex', 'age_start', 'age_end', 'year_start', 'year_end', 'affected_entity', 'parameter
    # ]
    prevalence_pem_i = (
        pd.DataFrame(
            {'value': [1.0, 1.0, 0.0, 0.0], 'affected_entity': [data_keys.PEM.name]},
            index=pd.Index([f'cat{i}' for i in range(1, 5)], name='parameter')
        ).reindex(
            index=rr_ci.index.droplevel('affected_entity').drop_duplicates(),
            level='parameter'
        ).set_index('affected_entity', append=True)
        .reorder_levels(rr_ci.index.names)
        .squeeze()
    )

    # ------------ Calculate mortality rates ------------ #

    # mr_i = acmr + sum_c(emr_c * prevalence_ci - csmr_c)
    # prevalence_ci = incidence_ci * duration_c
    # incidence_ci = incidence_c * (1 - paf_c) * rr_ci

    # Get wasting state incidence and prevalence for non-PEM causes
    incidence_ci = rr_ci * incidence_c * (1 - paf_c)
    prevalence_ci = incidence_ci * duration_c

    # add pem prevalence to prevalence_ci
    prevalence_ci = pd.concat([prevalence_ci, prevalence_pem_i])

    mr_i = (
        acmr + (
            (prevalence_ci * emr_c - csmr_c)
            .groupby(metadata.ARTIFACT_INDEX_COLUMNS + ['parameter'])
            .sum()
        )
    )
    mr_i = mr_i.unstack()

    # Convert annual mortality rates to daily mortality probabilities
    daily_mortality_probability = _convert_annual_rate_to_daily_probability(mr_i)
    return daily_mortality_probability

def adjust_exposure(exposures: pd.DataFrame, adjustment: pd.Series) -> pd.DataFrame:
    return exposures.div(1 + adjustment, axis='index')


def get_data_series(data: pd.DataFrame) -> pd.Series:
    return (
        data
        .set_index(data.columns[:-1].to_list())
        .squeeze()
    )


def _get_index(builder):
    return (
        builder.data.load(data_keys.POPULATION.DEMOGRAPHY)
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
        .index
    )


def _convert_annual_rate_to_daily_probability(
        rate: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    return 1 - np.exp(-rate / metadata.YEAR_DURATION)


def _convert_daily_probability_to_annual_rate(
        probability: Union[pd.Series, float]
) -> Union[pd.Series, float]:
    return -np.log(1 - probability) * metadata.YEAR_DURATION


def _reset_underage_transitions(transition_rates: pd.Series) -> None:
    transition_rates[
        transition_rates.index.get_level_values('age_end') <= data_values.WASTING.START_AGE
    ] = 0.0