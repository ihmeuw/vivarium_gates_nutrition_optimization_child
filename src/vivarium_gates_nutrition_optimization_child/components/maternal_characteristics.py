"""
Component for maternal supplementation and risk effects
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import RiskEffect

from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values
from vivarium_gates_nutrition_optimization_child.constants.data_keys import (
    BEP_SUPPLEMENTATION,
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    STUNTING,
    WASTING,
)
from vivarium_gates_nutrition_optimization_child.utilities import get_random_variable


class MaternalCharacteristics(Component):
    def __init__(self):
        super().__init__()
        self.supplementation_exposure_name = "maternal_supplementation.exposure"
        self.maternal_bmi_anemia_exposure_name = "maternal_bmi_anemia.exposure"

        self.bep_exposure_name = f"{BEP_SUPPLEMENTATION.name}.exposure"
        self.ifa_exposure_name = f"{IFA_SUPPLEMENTATION.name}.exposure"
        self.mmn_exposure_name = f"{MMN_SUPPLEMENTATION.name}.exposure"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.start_time = get_time_stamp(builder.configuration.time.start)
        builder.value.register_attribute_producer(
            self.bep_exposure_name,
            source=self._get_bep_exposure,
            required_resources=[self.supplementation_exposure_name],
        )
        builder.value.register_attribute_producer(
            self.ifa_exposure_name,
            source=self._get_ifa_exposure,
            required_resources=[self.supplementation_exposure_name],
        )
        builder.value.register_attribute_producer(
            self.mmn_exposure_name,
            source=self._get_mmn_exposure,
            required_resources=[self.supplementation_exposure_name],
        )
        builder.population.register_initializer(
            self.initialize_from_line_list,
            [
                self.supplementation_exposure_name,
                self.maternal_bmi_anemia_exposure_name,
            ],
        )

    def initialize_from_line_list(self, pop_data: SimulantData) -> None:
        """
        Initialize simulants from line list data. Population configuration
        contains a key "new_births" which is the line list data.
        """
        new_simulants = pd.DataFrame(
            columns=[
                self.supplementation_exposure_name,
                self.maternal_bmi_anemia_exposure_name,
            ],
            index=pop_data.index,
        )

        if pop_data.creation_time >= self.start_time:
            new_births = pop_data.user_data["new_births"]
            new_births.index = pop_data.index

            maternal_supplementation = new_births["maternal_intervention"].copy()
            maternal_supplementation[maternal_supplementation == "invalid"] = "uncovered"
            new_simulants[self.supplementation_exposure_name] = maternal_supplementation

            new_simulants[self.maternal_bmi_anemia_exposure_name] = new_births[
                "joint_bmi_anemia_category"
            ]

        self.population_view.update(new_simulants)

    ##################################
    # Pipeline sources and modifiers #
    ##################################
    def _get_bep_exposure(self, index: pd.Index) -> pd.Series:
        has_bep = self.population_view.get_attributes(index, self.supplementation_exposure_name) == "bep"

        exposure = pd.Series(BEP_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_bep] = BEP_SUPPLEMENTATION.CAT2
        return exposure

    def _get_ifa_exposure(self, index: pd.Index) -> pd.Series:
        has_ifa = self.population_view.get_attributes(index, self.supplementation_exposure_name).isin(["ifa", "mms", "bep"])
        breakpoint() # SBACHMEI - HAVEN'T HIT YET
        exposure = pd.Series(IFA_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_ifa] = IFA_SUPPLEMENTATION.CAT2
        return exposure

    def _get_mmn_exposure(self, index: pd.Index) -> pd.Series:
        has_mmn = self.population_view.get_attributes(index, self.supplementation_exposure_name).isin(["mms", "bep"])
        breakpoint() # SBACHMEI - HAVEN'T HIT YET
        exposure = pd.Series(MMN_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_mmn] = MMN_SUPPLEMENTATION.CAT2
        return exposure


class AdditiveRiskEffect(RiskEffect):
    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.effect_name = f"{self.risk.name}_on_{self.target.name}.effect"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.excess_shift_table = self.get_excess_shift_lookup_table(builder)
        self.risk_specific_shift_table = self.get_risk_specific_shift_lookup_table(builder)
        self.excess_shift = self.get_excess_shift(builder)

        builder.value.register_attribute_producer(
            self.effect_name,
            source=self.get_effect,
            required_resources=[self.exposure_name],
        )

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        self._exposure_distribution_type = "ordered_polytomous"
        return self.build_lookup_table(builder, "relative_risk", 1)

    def build_paf_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(builder, "paf", 0)

    def get_excess_shift_lookup_table(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            f"{self.risk}.excess_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )

        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        return self.build_lookup_table(builder, "excess_shift", excess_shift_data, value_cols)

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            "low_birth_weight_and_short_gestation.birth_exposure",
            modifier=self.adjust_target,
            required_resources=[self.relative_risk_name],
        )

    def adjust_target(self, index: pd.Index, target: pd.DataFrame) -> pd.Series:
        effect = self.population_view.get_attributes(index, self.effect_name)
        target["birth_weight"] += effect
        return target

    def get_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            f"{self.risk}.risk_specific_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        return self.build_lookup_table(
            builder, "risk_specific_shift", risk_specific_shift_data, "value"
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        pass

    def get_excess_shift(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        return self.excess_shift_table

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_effect(self, index: pd.Index) -> pd.Series:
        index_columns = ["index", self.risk.name]
        excess_shift = self.excess_shift(index)
        exposure = self.population_view.get_attributes(
            index, self.exposure_name
        ).reset_index()
        exposure.columns = index_columns
        exposure = exposure.set_index(index_columns)

        relative_risk = excess_shift.stack().reset_index()
        relative_risk.columns = index_columns + ["value"]
        relative_risk = relative_risk.set_index(index_columns)

        raw_effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)

        risk_specific_shift = self.risk_specific_shift_table(index)
        effect = raw_effect - risk_specific_shift
        return effect


class MMSEffectOnGestationalAge(AdditiveRiskEffect):
    """Model effect of multiple micronutrient supplementation on gestational age.
    Unique component because the excess shift value depends on IFA-shifted gestational age."""

    def __init__(self):
        super().__init__(
            "risk_factor.multiple_micronutrient_supplementation",
            "risk_factor.gestational_age.birth_exposure",
        )
        self.excess_shift_pipeline_name = (
            f"{self.risk.name}_on_{self.target.name}.excess_shift"
        )
        self.risk_specific_shift_pipeline_name = (
            f"{self.risk.name}_on_{self.target.name}.risk_specific_shift"
        )
        self.raw_gestational_age_exposure_column_name = "raw_gestational_age_exposure"

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [self.raw_gestational_age_exposure_column_name]

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.ifa_on_gestational_age = builder.components.get_component(
            f"risk_effect.iron_folic_acid_supplementation_on_{self.target}"
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables["relative_risk"] = self.build_lookup_table(builder, 1)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, 0
        )
        self.lookup_tables["risk_specific_shift"] = self.get_risk_specific_shift_lookup_table(
            builder
        )
        self.lookup_tables["mms_subpop1_excess_shift"] = self._get_mms_excess_shift_data(
            builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1
        )
        self.lookup_tables["mms_subpop2_excess_shift"] = self._get_mms_excess_shift_data(
            builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2
        )

    def _get_mms_excess_shift_data(self, builder: Builder, key: str) -> LookupTable:
        excess_shift_data = builder.data.load(
            key, affected_entity=self.target.name, affected_measure=self.target.measure
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        return self.build_lookup_table(builder, excess_shift_data, value_cols)

    def get_excess_shift(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.excess_shift_pipeline_name,
            source=self.get_excess_shift_source,
            requires_columns=[self.raw_gestational_age_exposure_column_name],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(builder, 0)

    def get_excess_shift_source(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        raw_gestational_age = pop[self.raw_gestational_age_exposure_column_name]
        ifa_shifted_gestational_age = (
            raw_gestational_age + self.ifa_on_gestational_age.effect(index)
        )
        # excess shift is (mms_shift_1 + mms_shift_2) for subpop_2 and mms_shift_1 for subpop_1
        mms_shift_2 = (
            self.lookup_tables["mms_subpop2_excess_shift"](index)["cat2"]
            - self.lookup_tables["mms_subpop1_excess_shift"](index)["cat2"]
        )
        is_subpop_1 = ifa_shifted_gestational_age < (32 - mms_shift_2)
        is_subpop_2 = ifa_shifted_gestational_age >= (32 - mms_shift_2)

        subpop_1_index = pop[is_subpop_1].index
        subpop_2_index = pop[is_subpop_2].index

        excess_shift = pd.concat(
            [
                self.lookup_tables["mms_subpop1_excess_shift"](subpop_1_index),
                self.lookup_tables["mms_subpop2_excess_shift"](subpop_2_index),
            ]
        )

        return excess_shift


class BEPEffectOnBirthweight(AdditiveRiskEffect):
    """Model effect of BEP on birthweight. Unique component because effect of BEP depends
    on mother's BMI status."""

    def __init__(self):
        super().__init__(
            "risk_factor.balanced_energy_protein_supplementation",
            "risk_factor.birth_weight.birth_exposure",
        )

    def get_excess_shift_lookup_table(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            f"{self.risk}.excess_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        excess_shift_data.rename(columns={"maternal_bmi_anemia_exposure": "maternal_bmi_anemia.exposure"}, inplace=True)
        return self.build_lookup_table(builder, "excess_shift", excess_shift_data, value_cols)


class BirthWeightShiftEffect(Component):
    def __init__(self):
        super().__init__()
        self.ifa_effect_pipeline_name = f"{IFA_SUPPLEMENTATION.name}_on_birth_weight.effect"
        self.mmn_effect_pipeline_name = f"{MMN_SUPPLEMENTATION.name}_on_birth_weight.effect"
        self.bep_effect_pipeline_name = f"{BEP_SUPPLEMENTATION.name}_on_birth_weight.effect"

        self.stunting_exposure_parameters_pipeline_name = (
            f"risk_factor.{STUNTING.name}.exposure_parameters"
        )

        self.wasting_exposure_parameters_pipeline_name = (
            f"risk_factor.{WASTING.name}.exposure_parameters"
        )

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
