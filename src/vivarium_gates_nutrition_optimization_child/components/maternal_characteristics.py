"""
Component for maternal supplementation and risk effects
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.risks.data_transformations import (
    pivot_categorical,
    rebin_relative_risk_data,
)

from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values, paths
from vivarium_gates_nutrition_optimization_child.constants.data_keys import (
    BEP_SUPPLEMENTATION,
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    STUNTING,
    WASTING,
)
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class MaternalCharacteristics:

    configuration_defaults = {
        IFA_SUPPLEMENTATION.name: {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
        MMN_SUPPLEMENTATION.name: {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
        BEP_SUPPLEMENTATION.name: {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
        "iv_iron": {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
        "maternal_bmi_anemia": {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        },
    }

    def __init__(self):
        self.supplementation_exposure_column_name = "maternal_supplementation_exposure"
        self.maternal_bmi_anemia_exposure_column_name = "maternal_bmi_anemia_exposure"

        self.bep_exposure_pipeline_name = f"{BEP_SUPPLEMENTATION.name}.exposure"
        self.ifa_exposure_pipeline_name = f"{IFA_SUPPLEMENTATION.name}.exposure"
        self.mmn_exposure_pipeline_name = f"{MMN_SUPPLEMENTATION.name}.exposure"
        self.maternal_bmi_anemia_exposure_pipeline_name = "maternal_bmi_anemia.exposure"

    def __repr__(self):
        return "MaternalCharacteristics()"

    @property
    def name(self) -> str:
        return "maternal_characteristics"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.bep_exposure_pipeline = builder.value.register_value_producer(
            self.bep_exposure_pipeline_name,
            source=self._get_bep_exposure,
            requires_columns=[self.supplementation_exposure_column_name],
        )
        self.ifa_exposure_pipeline = builder.value.register_value_producer(
            self.ifa_exposure_pipeline_name,
            source=self._get_ifa_exposure,
            requires_columns=[self.supplementation_exposure_column_name],
        )
        self.mmn_exposure_pipeline = builder.value.register_value_producer(
            self.mmn_exposure_pipeline_name,
            source=self._get_mmn_exposure,
            requires_columns=[self.supplementation_exposure_column_name],
        )
        self.maternal_bmi_anemia_exposure_pipeline = builder.value.register_value_producer(
            self.maternal_bmi_anemia_exposure_pipeline_name,
            source=self._get_maternal_bmi_anemia_exposure,
            requires_columns=[self.maternal_bmi_anemia_exposure_column_name],
        )

        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[
                self.supplementation_exposure_column_name,
                self.maternal_bmi_anemia_exposure_column_name,
            ],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Initialize simulants from line list data. Population configuration
        contains a key "new_births" which is the line list data.
        """
        columns = [
            self.supplementation_exposure_column_name,
            self.maternal_bmi_anemia_exposure_column_name,
        ]
        new_simulants = pd.DataFrame(columns=columns, index=pop_data.index)

        if pop_data.creation_time >= self.start_time:
            new_births = pop_data.user_data["new_births"]
            new_births.index = pop_data.index

            maternal_supplementation = new_births["maternal_intervention"].copy()
            maternal_supplementation[maternal_supplementation == "invalid"] = "uncovered"
            new_simulants[
                self.supplementation_exposure_column_name
            ] = maternal_supplementation

            new_simulants[self.maternal_bmi_anemia_exposure_column_name] = new_births[
                "joint_bmi_anemia_category"
            ]

        self.population_view.update(new_simulants)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.supplementation_exposure_column_name,
                self.maternal_bmi_anemia_exposure_column_name,
            ]
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################
    def _get_bep_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        has_bep = pop[self.supplementation_exposure_column_name] == "bep"

        exposure = pd.Series(BEP_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_bep] = BEP_SUPPLEMENTATION.CAT2
        return exposure

    def _get_ifa_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        has_ifa = pop[self.supplementation_exposure_column_name].isin(["ifa", "mms", "bep"])

        exposure = pd.Series(IFA_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_ifa] = IFA_SUPPLEMENTATION.CAT2
        return exposure

    def _get_mmn_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        has_mmn = pop[self.supplementation_exposure_column_name].isin(["mms", "bep"])

        exposure = pd.Series(MMN_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_mmn] = MMN_SUPPLEMENTATION.CAT2
        return exposure

    def _get_maternal_bmi_anemia_exposure(self, index: pd.Index) -> pd.Series:
        exposure = self.population_view.get(index)[
            self.maternal_bmi_anemia_exposure_column_name
        ]
        exposure.name = self.maternal_bmi_anemia_exposure_pipeline_name
        return exposure


class AdditiveRiskEffect(RiskEffect):
    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.effect_pipeline_name = f"{self.risk.name}_on_{self.target.name}.effect"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.effect = self._get_effect_pipeline(builder)
        self.excess_shift_source = self._get_excess_shift_source(builder)
        self.risk_specific_shift_source = self._get_risk_specific_shift_source(builder)

    # NOTE: this RR will never be used. Overriding superclass to avoid error
    def _get_relative_risk_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(1)

    # NOTE: this PAF will never be used. Overriding superclass to avoid error
    def _get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return builder.lookup.build_table(0)

    def _get_effect_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.effect_pipeline_name,
            source=self.get_effect,
            requires_values=[self.exposure_pipeline_name],
        )

    def _get_excess_shift_source(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            f"{self.risk}.excess_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        excess_shift_data = rebin_relative_risk_data(builder, self.risk, excess_shift_data)
        excess_shift_data = pivot_categorical(excess_shift_data)
        return builder.lookup.build_table(
            excess_shift_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            affected_rates = target + self.effect(index)
            return affected_rates

        return adjust_target

    def _get_risk_specific_shift_source(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            f"{self.risk}.risk_specific_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        return builder.lookup.build_table(
            risk_specific_shift_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _register_paf_modifier(self, builder: Builder) -> None:
        pass

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_effect(self, index: pd.Index) -> pd.Series:
        index_columns = ["index", self.risk.name]

        excess_shift = self.excess_shift_source(index)
        exposure = self.exposure(index).reset_index()
        exposure.columns = index_columns
        exposure = exposure.set_index(index_columns)

        relative_risk = excess_shift.stack().reset_index()
        relative_risk.columns = index_columns + ["value"]
        relative_risk = relative_risk.set_index(index_columns)

        raw_effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)

        risk_specific_shift = self.risk_specific_shift_source(index)
        effect = raw_effect - risk_specific_shift
        return effect


class MMSEffectOnGestationalAge(AdditiveRiskEffect):
    '''Model effect of multiple micronutrient supplementation on gestational age.
    Unique component because the excess shift value depends on IFA-shifted gestational age.'''
    def __init__(self):
        super().__init__('risk_factor.multiple_micronutrient_supplementation', 'risk_factor.gestational_age.birth_exposure')
        self.effect_pipeline_name = f"{self.risk.name}_on_gestational_age.effect"
        self.excess_shift_pipeline_name = f'{self.risk.name}_on_{self.target.name}.excess_shift'
        self.risk_specific_shift_pipeline_name = f'{self.risk.name}_on_{self.target.name}.risk_specific_shift'
        self.raw_gestational_age_exposure_column_name = 'raw_gestational_age_exposure'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.population_view = self._get_population_view(builder)
        self.ifa_on_gestational_age = builder.components.get_component(f'risk_effect.risk_factor.iron_folic_acid_supplementation.{self.target}')
        self.mms_subpop_1_excess_shift = self._get_mms_excess_shift_data(builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1)
        self.mms_subpop_2_excess_shift = self._get_mms_excess_shift_data(builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.raw_gestational_age_exposure_column_name])

    def _get_mms_excess_shift_data(self, builder: Builder, key: str) -> Dict[str, LookupTable]:
        excess_shift_data = builder.data.load(key, affected_entity=self.target.name, affected_measure=self.target.measure)
        excess_shift_data = self.build_excess_shift_lookup_table(builder, excess_shift_data)
        return excess_shift_data

    def build_excess_shift_lookup_table(self, builder: Builder, excess_shift_data: pd.DataFrame) -> LookupTable:
        '''Reads excess shift data from artifact and returns LookupTable with that data.'''
        excess_shift_data = rebin_relative_risk_data(builder, self.risk, excess_shift_data)
        excess_shift_data = pivot_categorical(excess_shift_data)
        return builder.lookup.build_table(
            excess_shift_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################
    def _get_excess_shift_source(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.excess_shift_pipeline_name,
            source=self.get_excess_shift,
            requires_columns=[self.raw_gestational_age_exposure_column_name],
        )

    def _get_risk_specific_shift_source(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.risk_specific_shift_pipeline_name,
            source=self.get_risk_specific_shift,
        )

    # write as partial functions
    def get_excess_shift(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        raw_gestational_age = pop[self.raw_gestational_age_exposure_column_name]
        ifa_shifted_gestational_age = raw_gestational_age + self.ifa_on_gestational_age.effect(index)
        # excess shift is (mms_shift_1 + mms_shift_2) for subpop_2 and mms_shift_1 for subpop_1
        mms_shift_2 = self.mms_subpop_2_excess_shift(index)['cat2'] - self.mms_subpop_1_excess_shift(index)['cat2']
        is_subpop_1 = ifa_shifted_gestational_age < (32 - mms_shift_2)
        is_subpop_2 = ifa_shifted_gestational_age >= (32 - mms_shift_2)

        subpop_1_index = pop[is_subpop_1].index
        subpop_2_index = pop[is_subpop_2].index

        excess_shift = pd.concat([self.mms_subpop_1_excess_shift(subpop_1_index), self.mms_subpop_2_excess_shift(subpop_2_index)])

        return excess_shift

    def get_risk_specific_shift(self, index: pd.Index) -> pd.Series:
        # MMS doesn't exist in baseline so we don't need to adjust gestational age to get an "MMS-less" population
        return pd.Series(0, index=index)


class BirthWeightShiftEffect:
    def __init__(self):
        self.ifa_effect_pipeline_name = f"{IFA_SUPPLEMENTATION.name}_on_birth_weight.effect"
        self.mmn_effect_pipeline_name = f"{MMN_SUPPLEMENTATION.name}_on_birth_weight.effect"
        self.bep_effect_pipeline_name = f"{BEP_SUPPLEMENTATION.name}_on_birth_weight.effect"

        self.stunting_exposure_parameters_pipeline_name = (
            f"risk_factor.{STUNTING.name}.exposure_parameters"
        )

        self.wasting_exposure_parameters_pipeline_name = (
            f"risk_factor.{WASTING.name}.exposure_parameters"
        )

    def __repr__(self):
        return f"BirthWeightShiftEffect()"

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"birth_weight_shift_effect"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.stunting_effect_per_gram = self._get_stunting_effect_per_gram(builder)
        self.wasting_effect_per_gram = data_values.LBWSG.WASTING_EFFECT_PER_GRAM

        self.pipelines = {
            pipeline_name: builder.value.get_value(pipeline_name)
            for pipeline_name in [
                self.ifa_effect_pipeline_name,
                self.mmn_effect_pipeline_name,
                self.bep_effect_pipeline_name,
            ]
        }

        builder.value.register_value_modifier(
            self.stunting_exposure_parameters_pipeline_name,
            modifier=self._modify_stunting_exposure_parameters,
            requires_values=list(self.pipelines.keys()),
        )

        builder.value.register_value_modifier(
            self.wasting_exposure_parameters_pipeline_name,
            modifier=self._modify_wasting_exposure_parameters,
            requires_values=list(self.pipelines.keys()),
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _modify_stunting_exposure_parameters(
        self, index: pd.Index, target: pd.DataFrame
    ) -> pd.DataFrame:
        cat3_increase = (
            self._get_total_birth_weight_shift(index) * self.stunting_effect_per_gram
        )
        return self._apply_birth_weight_effect(target, cat3_increase)

    def _modify_wasting_exposure_parameters(
        self, index: pd.Index, target: pd.DataFrame
    ) -> pd.DataFrame:
        cat3_increase = (
            self._get_total_birth_weight_shift(index) * self.wasting_effect_per_gram
        )
        return self._apply_birth_weight_effect(target, cat3_increase)

    ##################
    # Helper methods #
    ##################

    def _get_total_birth_weight_shift(self, index: pd.Index) -> pd.Series:
        return pd.concat(
            [pipeline(index) for pipeline in self.pipelines.values()], axis=1
        ).sum(axis=1)

    # noinspection PyMethodMayBeStatic
    def _get_stunting_effect_per_gram(self, builder: Builder) -> pd.Series:
        return get_random_variable(
            builder.configuration.input_data.input_draw_number,
            *data_values.LBWSG.STUNTING_EFFECT_PER_GRAM,
        )

    @staticmethod
    def _apply_birth_weight_effect(
        target: pd.DataFrame, cat3_increase: pd.Series
    ) -> pd.DataFrame:
        # no changes if all probability in cat4
        if (target["cat4"] == 1).all():
            return target

        sam_and_mam = target["cat1"] + target["cat2"]
        cat3 = target["cat3"]

        # can't remove more from a category than exists in its categories
        true_cat3_increase = np.maximum(
            np.minimum(sam_and_mam, cat3_increase), np.minimum(cat3, -cat3_increase)
        )

        target["cat3"] = target["cat3"] + true_cat3_increase
        target["cat2"] = target["cat2"] * (1 - true_cat3_increase / sam_and_mam)
        target["cat1"] = target["cat1"] * (1 - true_cat3_increase / sam_and_mam)
        return target
