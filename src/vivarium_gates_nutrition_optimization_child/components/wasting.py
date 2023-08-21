from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor
)
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.utilities import EntityString

from vivarium_gates_nutrition_optimization_child.components.risk import RiskEffect
from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values, metadata, models, scenarios
from vivarium_gates_nutrition_optimization_child.constants.data_keys import WASTING
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class RiskState(DiseaseState):

    def load_excess_mortality_rate_data(self, builder):
        if 'excess_mortality_rate' in self._get_data_functions:
            return self._get_data_functions['excess_mortality_rate'](self.cause, builder)
        else:
            return builder.data.load(f'{self.cause_type}.{self.cause}.excess_mortality_rate')


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
        self.birth_weight_effect_pipeline_name = 'birth_weight_shift_on_mild_wasting.effect_size'

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

        self.birth_weight_effect = builder.value.get_value(self.birth_weight_effect_pipeline_name)

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
            requires_values=[self.birth_weight_effect_pipeline_name],
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

    ##################
    # Helper methods #
    ##################

    def get_state_weights(self, pop_index, prevalence_type):
        states = [
            s for s in self.states
            if hasattr(s, f'{prevalence_type}') and getattr(s, f'{prevalence_type}') is not None
        ]

        if not states:
            return states, None

        state_names = [s.state_id for s in states] + [self.initial_state]

        weights = (
            pd.concat([getattr(s, f'{prevalence_type}')(pop_index) for s in states], axis=1)
            .reset_index(drop=True)
        )

        cat3_increase = self.birth_weight_effect(pop_index).reset_index(drop=True)
        weights = apply_birth_weight_effect(weights, cat3_increase)
        weights[data_keys.WASTING.CAT4] = 1 - weights.sum(axis=1)

        weights = np.array(weights)
        weights_bins = np.cumsum(weights, axis=1)

        return state_names, weights_bins


# noinspection PyPep8Naming
def ChildWasting() -> RiskModel:
    tmrel = SusceptibleState(models.WASTING.MODEL_NAME)
    mild = RiskState(
        models.WASTING.MILD_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mild_wasting_exposure,
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'birth_prevalence': load_mild_wasting_birth_prevalence,
        }
    )
    moderate = RiskState(
        models.WASTING.MODERATE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_mam_exposure,
            'excess_mortality_rate': load_pem_excess_mortality_rate,
            'birth_prevalence': load_mam_birth_prevalence,
        }
    )
    severe = RiskState(
        models.WASTING.SEVERE_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'prevalence': load_sam_exposure,
            'excess_mortality_rate': load_pem_excess_mortality_rate,
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


class DiarrheaRiskEffect(RiskEffect):
    def __init__(self, target: str):
        super(DiarrheaRiskEffect, self).__init__(f'risk_factor.{data_keys.DIARRHEA.name}', target)
        self.diarrhea_exposure_column_name = data_keys.DIARRHEA.name
        self.source_state, self.sink_state = self.get_source_and_sink()

    def get_source_and_sink(self) -> Tuple[str, str]:
        if self.target.measure == 'transition_rate':
            source_state, sink_state = [
                models.get_risk_category(state) for state in self.target.name.split('_to_')
            ]
        elif self.target.measure == 'incidence_rate':
            source_state = models.get_risk_category(models.WASTING.SUSCEPTIBLE_STATE_NAME)
            sink_state = models.get_risk_category(models.WASTING.MILD_STATE_NAME)
        else:
            raise ValueError(
                f'Unsupported target measure. Supported measures are "transition_rate" and'
                f' "incidence_rate". Provided {self.target.measure}'
            )
        return source_state, sink_state

    def _get_distribution_type(self, builder: Builder) -> str:
        return 'dichotomous'

    def _get_risk_exposure(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        population_view = builder.population.get_view(
            ['age', 'sex', self.diarrhea_exposure_column_name]
        )

        def get_exposure(index: pd.Index) -> pd.Series:
            return population_view.get(index)[self.diarrhea_exposure_column_name]

        return get_exposure

    def _get_relative_risk_source(self, builder: Builder) -> LookupTable:
        diarrhea_exposure, susceptible_exposure = load_wasting_with_diarrhea_exposure(builder)

        incidence_diarrhea = load_wasting_incidence_with_diarrhea(
            builder, self.sink_state, diarrhea_exposure, susceptible_exposure
        )
        incidence_susceptible = get_wasting_incidence_without_diarrhea(
            builder, self.source_state, incidence_diarrhea
        )

        # rr_i{x}: (i_D{x} * p_S{x-1}) / (i_S{x} * p_D{x-1})
        relative_risk = (
            (incidence_diarrhea * susceptible_exposure.xs(self.source_state, level='wasting'))
            / (incidence_susceptible * diarrhea_exposure.xs(self.source_state, level='wasting'))
        ).rename(models.DIARRHEA.STATE_NAME)
        relative_risk[relative_risk < 0.0] = 0.0
        relative_risk = relative_risk.to_frame()
        relative_risk[models.DIARRHEA.SUSCEPTIBLE_STATE_NAME] = 1.0

        return builder.lookup.build_table(
            relative_risk.reset_index(),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

    def _get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        # paf = (mean_rr - 1) / mean_rr
        # mean_rr = sum_d(exposure_d|w * rr_t|d)
        #
        # sum_d: sum over diarrhea states d
        # exposure_d|w: exposure of diarrhea given wasting state w = p_dw / (p_Dw + p_Sw)
        # rr_t|d: relative risk of transition t given diarrhea state d

        raw_diarrhea_exposure, raw_susceptible_exposure = (
            load_wasting_with_diarrhea_exposure(builder)
        )
        raw_diarrhea_exposure = raw_diarrhea_exposure.xs(self.source_state, level='wasting')
        raw_susceptible_exposure = raw_susceptible_exposure.xs(self.source_state, level='wasting')

        rr = (
            self.relative_risk._table.data
            .set_index(metadata.ARTIFACT_INDEX_COLUMNS)
            [models.DIARRHEA.STATE_NAME]
        )

        mean_rr = (
            (raw_diarrhea_exposure * rr + raw_susceptible_exposure)
            / (raw_diarrhea_exposure + raw_susceptible_exposure)
        )

        paf = (mean_rr - 1) / mean_rr
        return builder.lookup.build_table(
            paf.reset_index(),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

    def _register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            modifier=self.target_modifier,
            requires_columns=['age', 'sex', self.diarrhea_exposure_column_name]
        )


# noinspection PyUnusedLocal
def load_pem_excess_mortality_rate(cause: str, builder: Builder) -> pd.DataFrame:
    return builder.data.load(data_keys.PEM.EMR)


# noinspection PyUnusedLocal
def load_mild_wasting_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT3)


# noinspection PyUnusedLocal
def load_mild_wasting_exposure(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_exposures(builder)[WASTING.CAT3].reset_index()


# noinspection PyUnusedLocal, DuplicatedCode
def load_mild_wasting_incidence_rate(cause: str, builder: Builder) -> pd.DataFrame:
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
def load_mild_wasting_remission_rate(cause: str, builder: Builder) -> pd.DataFrame:
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
def load_mam_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT2)


# noinspection PyUnusedLocal
def load_mam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
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
def load_sam_birth_prevalence(cause: str, builder: Builder) -> pd.DataFrame:
    return load_child_wasting_birth_prevalence(builder, WASTING.CAT1)


# noinspection PyUnusedLocal
def load_sam_exposure(cause: str, builder: Builder) -> pd.DataFrame:
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


def load_wasting_incidence_with_diarrhea(
        builder: Builder,
        sink_state: str,
        diarrhea_exposure: pd.Series,
        susceptible_exposure: pd.Series
) -> pd.Series:

    entrance_rate = get_entrance_rate(builder, diarrhea_exposure)
    diarrhea_incidence = get_diarrhea_incidence(builder, susceptible_exposure)
    diarrhea_remission = get_diarrhea_remission(builder, diarrhea_exposure)
    diarrhea_mortality = load_mortality_with_diarrhea(builder, diarrhea_exposure)
    sam_tx_remission = get_sam_tx_remission_with_diarrhea(builder, diarrhea_exposure)
    sam_ux_remission = get_sam_ux_remission_with_diarrhea(builder, diarrhea_exposure)
    mam_remission = get_mam_remission_with_diarrhea(builder, diarrhea_exposure)
    mild_remission = get_mild_remission_with_diarrhea(builder, diarrhea_exposure)

    if sink_state == data_keys.WASTING.CAT1:
        # i_D1 = -b_D1 - di_1 + dr_1 + m_D1 + r_D1tx + r_D1ux
        incidence = (
            -entrance_rate.xs(data_keys.WASTING.CAT1, level='wasting')
            - diarrhea_incidence.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT1, level='wasting')
            + sam_tx_remission
            + sam_ux_remission
        )
    elif sink_state == data_keys.WASTING.CAT2:
        # i_D2 = -b_D1 - b_D2 - 2.0*di_1 + dr_1 + dr_2 + m_D1 + m_D2 + r_D1tx + r_D2
        incidence = (
            -entrance_rate.xs(data_keys.WASTING.CAT1, level='wasting')
            - entrance_rate.xs(data_keys.WASTING.CAT2, level='wasting')
            - 2.0 * diarrhea_incidence.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT2, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT2, level='wasting')
            + sam_tx_remission
            + mam_remission
        )
    else:
        # i_D3 = -b_D1 - b_D2 - b_D3 - 2.0*di_1 - di_3 + dr_1 + dr_2 + dr_3 + m_D1 + m_D2 + m_D3
        #        + r_D3
        incidence = (
            -entrance_rate.xs(data_keys.WASTING.CAT1, level='wasting')
            - entrance_rate.xs(data_keys.WASTING.CAT2, level='wasting')
            - entrance_rate.xs(data_keys.WASTING.CAT3, level='wasting')
            - 2.0 * diarrhea_incidence.xs(data_keys.WASTING.CAT1, level='wasting')
            - diarrhea_incidence.xs(data_keys.WASTING.CAT3, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT2, level='wasting')
            + diarrhea_remission.xs(data_keys.WASTING.CAT3, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT1, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT2, level='wasting')
            + diarrhea_mortality.xs(data_keys.WASTING.CAT3, level='wasting')
            + mild_remission
        )
    return incidence


def get_wasting_incidence_without_diarrhea(
        builder: Builder, source_state: str, wasting_incidence_with_diarrhea: pd.Series
) -> pd.Series:

    wasting_exposure = load_child_wasting_exposures(builder)
    if source_state == data_keys.WASTING.CAT2:
        incidence_rate = get_data_series(load_sam_incidence_rate(builder))
    elif source_state == data_keys.WASTING.CAT3:
        incidence_rate = get_data_series(load_mam_incidence_rate(builder))
    else:
        incidence_rate = get_data_series(load_mild_wasting_incidence_rate('', builder))

    incidence = incidence_rate * wasting_exposure[source_state] - wasting_incidence_with_diarrhea
    return incidence


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


def load_wasting_with_diarrhea_exposure(builder: Builder) -> Tuple[pd.Series, pd.Series]:
    prev_diarrhea = get_data_series(builder.data.load(data_keys.DIARRHEA.PREVALENCE))
    wasting_exposure = (
        load_child_wasting_exposures(builder)
        .rename_axis('wasting', axis=1)
        .stack()
    )

    prevalence_ratio = (
        data_values.WASTING.DIARRHEA_PREVALENCE_RATIO
        .reindex(wasting_exposure.index, level='wasting')
    )

    prev_wasting_diarrhea = (
        prevalence_ratio * wasting_exposure * prev_diarrhea
        / ((prevalence_ratio - 1) * prev_diarrhea + 1)
    )

    prev_wasting_diarrhea[
        prev_wasting_diarrhea.index.get_level_values('wasting') == data_keys.WASTING.CAT4
    ] = (
        prev_diarrhea - (
            prev_wasting_diarrhea
            .to_frame()
            .query(f"wasting != '{data_keys.WASTING.CAT4}'")
            .groupby(metadata.ARTIFACT_INDEX_COLUMNS)
            .sum()
            .squeeze()
        )
    )

    prev_wasting_no_diarrhea = wasting_exposure - prev_wasting_diarrhea
    return prev_wasting_diarrhea, prev_wasting_no_diarrhea


def load_mortality_with_diarrhea(
        builder: Builder, wasting_diarrhea_exposure: pd.Series
) -> pd.Series:
    causes = [data_keys.DIARRHEA, data_keys.MEASLES, data_keys.LRI, data_keys.PEM]

    acmr = get_data_series(builder.data.load(data_keys.POPULATION.ACMR))
    emr_pem = get_data_series(builder.data.load(data_keys.PEM.EMR))
    emr_diarrhea = get_data_series(builder.data.load(data_keys.DIARRHEA.EMR))

    csmr_c = pd.concat([
        get_data_series(builder.data.load(c.CSMR)).rename(c.name) for c in causes
    ], axis=1)
    csmr_c.columns.name = 'affected_entity'

    rr = (
        get_data_series(builder.data.load(data_keys.WASTING.RELATIVE_RISK))
        .droplevel('affected_measure')
        .unstack('affected_entity')
    )
    rr.index = rr.index.set_names('wasting', level='parameter')

    paf = (
        get_data_series(builder.data.load(data_keys.WASTING.PAF))
        .droplevel('affected_measure')
        .unstack('affected_entity')
    )

    mortality = wasting_diarrhea_exposure * (
        acmr
        - (
            csmr_c[data_keys.DIARRHEA.name]
            + emr_diarrhea * (1 - paf[data_keys.DIARRHEA.name]) * rr[data_keys.DIARRHEA.name]
        )
        - csmr_c[data_keys.PEM.name] + emr_pem
        - (
            csmr_c[data_keys.LRI.name] * paf[data_keys.LRI.name] * rr[data_keys.LRI.name]
        )
        - csmr_c[data_keys.MEASLES.name] * paf[data_keys.MEASLES.name] * rr[data_keys.MEASLES.name]
    )
    return mortality


def get_diarrhea_incidence(builder: Builder, wasting_no_diarrhea_exposure: pd.Series) -> pd.Series:
    incidence_rate = get_data_series(builder.data.load(data_keys.DIARRHEA.INCIDENCE_RATE))
    diarrhea_incidence = incidence_rate * wasting_no_diarrhea_exposure
    return diarrhea_incidence


def get_diarrhea_remission(builder: Builder, wasting_diarrhea_exposure: pd.Series) -> pd.Series:
    diarrhea_duration = get_random_variable(
        builder.configuration.input_data.input_draw_number,
        *data_values.WASTING.DIARRHEA_DURATION_VICIOUS_CYCLE
    ) / metadata.YEAR_DURATION

    remission_rate = 1 / diarrhea_duration
    diarrhea_remission = remission_rate * wasting_diarrhea_exposure
    return diarrhea_remission


def get_entrance_rate(builder: Builder, wasting_diarrhea_exposure: pd.Series) -> pd.Series:
    # b_Dx = ACMR * p_Dx
    acmr = get_data_series(builder.data.load(data_keys.POPULATION.ACMR))
    return acmr * wasting_diarrhea_exposure


def get_sam_tx_remission_with_diarrhea(
        builder: Builder, wasting_diarrhea_exposure: pd.Series
) -> pd.Series:
    remission = (
        get_data_series(load_sam_treated_remission_rate(builder))
        * wasting_diarrhea_exposure.xs(data_keys.WASTING.CAT1, level='wasting')
    )
    return remission


def get_sam_ux_remission_with_diarrhea(
        builder: Builder, wasting_diarrhea_exposure: pd.Series
) -> pd.Series:
    remission = (
        get_data_series(load_sam_untreated_remission_rate(builder))
        * wasting_diarrhea_exposure.xs(data_keys.WASTING.CAT1, level='wasting')
    )
    return remission


def get_mam_remission_with_diarrhea(
        builder: Builder, wasting_diarrhea_exposure: pd.Series
) -> pd.Series:
    remission = (
        get_data_series(load_mam_remission_rate(builder))
        * wasting_diarrhea_exposure.xs(data_keys.WASTING.CAT2, level='wasting')
    )
    return remission


def get_mild_remission_with_diarrhea(
        builder: Builder, wasting_diarrhea_exposure: pd.Series
) -> pd.Series:
    remission = (
        get_data_series(load_mild_wasting_remission_rate('', builder))
        * wasting_diarrhea_exposure.xs(data_keys.WASTING.CAT3, level='wasting')
    )
    return remission


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

def apply_birth_weight_effect(target: pd.DataFrame, cat3_increase: pd.Series) -> pd.DataFrame:
    sam_and_mam = target[data_keys.STUNTING.CAT1] + target[data_keys.STUNTING.CAT2]
    apply_effect = cat3_increase < sam_and_mam
    target.loc[apply_effect, data_keys.STUNTING.CAT3] = (
            target[data_keys.STUNTING.CAT3] + cat3_increase
    )
    target.loc[apply_effect, data_keys.STUNTING.CAT2] = (
            target[data_keys.STUNTING.CAT2] * (1 - cat3_increase / sam_and_mam)
    )
    target.loc[apply_effect, data_keys.STUNTING.CAT1] = (
            target[data_keys.STUNTING.CAT1] * (1 - cat3_increase / sam_and_mam)
    )
    return target