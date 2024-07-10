from typing import Any, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.disease import DiseaseState
from vivarium_public_health.results import COLUMNS
from vivarium_public_health.results.disease import DiseaseObserver
from vivarium_public_health.results.mortality import (
    MortalityObserver as MortalityObserver_,
)
from vivarium_public_health.results.stratification import (
    ResultsStratifier as ResultsStratifier_,
)

from vivarium_gates_nutrition_optimization_child.constants import data_keys, data_values
from vivarium_gates_nutrition_optimization_child.constants.metadata import (
    SUBNATIONAL_LOCATION_DICT,
)


class ResultsStratifier(ResultsStratifier_):
    """Centralized component for handling results stratification.
    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.
    """

    def get_age_bins(self, builder: Builder) -> pd.DataFrame:
        """Define final age groups for production runs."""
        age_bins = super().get_age_bins(builder)
        data_dict = {
            "age_start": [0.0, 0.019178, 0.076712, 0.5, 1.0, 2.0],  # [0.0, 0.5, 1.5],
            "age_end": [0.019178, 0.076712, 0.5, 1.0, 2.0, 5.0],  # [0.5, 1.5, 5],
            "age_group_name": [
                "early_neonatal",
                "late_neonatal",
                "1-5_months",
                "6-11_months",
                "12_to_23_months",
                "2_to_4",
                # "0_to_6_months",
                # "6_to_18_months",
                # "18_to_59_months",
            ],
        }

        return pd.DataFrame(data_dict)

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""
        super().register_stratifications(builder)

        # builder.results.register_stratification(
        #     "wasting_state",
        #     [category.value for category in data_keys.ChildWastingCategories],
        #     self.child_wasting_stratification_mapper,
        #     is_vectorized=True,
        #     requires_values=["child_wasting.exposure"],
        # )
        # builder.results.register_stratification(
        #     "stunting_state",
        #     [category.value for category in data_keys.CGFCategories],
        #     self.child_growth_risk_factor_stratification_mapper,
        #     is_vectorized=True,
        #     requires_values=["child_stunting.exposure"],
        # )
        # builder.results.register_stratification(
        #     "underweight_state",
        #     [category.value for category in data_keys.CGFCategories],
        #     self.child_growth_risk_factor_stratification_mapper,
        #     is_vectorized=True,
        #     requires_values=["child_underweight.exposure"],
        # )
        # builder.results.register_stratification(
        #     "maternal_supplementation",
        #     results.MATERNAL_SUPPLEMENTATION_TYPES,
        #     is_vectorized=True,
        #     requires_columns=["maternal_supplementation_exposure"],
        # )
        # builder.results.register_stratification(
        #     "bmi_anemia",
        #     ["cat4", "cat3", "cat2", "cat1"],
        #     is_vectorized=True,
        #     requires_columns=["maternal_bmi_anemia_exposure"],
        # )
        builder.results.register_stratification(
            "sam_treatment",
            ["covered", "uncovered"],
            self.map_wasting_treatment,
            is_vectorized=True,
            requires_values=[f"{data_keys.SAM_TREATMENT.name}.exposure"],
        )
        builder.results.register_stratification(
            "mam_treatment",
            ["covered", "uncovered"],
            self.map_wasting_treatment,
            is_vectorized=True,
            requires_values=[f"{data_keys.MAM_TREATMENT.name}.exposure"],
        )
        builder.results.register_stratification(
            "sqlns_coverage",
            ["covered", "uncovered", "received"],
            is_vectorized=True,
            requires_values=[data_values.SQ_LNS.COVERAGE_PIPELINE],
        )
        # builder.results.register_stratification(
        #     "birth_weight_status",
        #     ["low_birth_weight", "adequate_birth_weight"],
        #     is_vectorized=True,
        #     requires_columns=["birth_weight_status"],
        # )
        location = builder.data.load("population.location")
        builder.results.register_stratification(
            "subnational",
            SUBNATIONAL_LOCATION_DICT[location],
            is_vectorized=True,
            requires_columns=["subnational"],
        )

    ###########################
    # Stratifications Details #
    ###########################

    # noinspection PyMethodMayBeStatic
    # def child_growth_risk_factor_stratification_mapper(self, pop: pd.DataFrame) -> pd.Series:
    #     # applicable to stunting and wasting
    #     mapper = {
    #         "cat4": data_keys.CGFCategories.UNEXPOSED.value,
    #         "cat3": data_keys.CGFCategories.MILD.value,
    #         "cat2": data_keys.CGFCategories.MODERATE.value,
    #         "cat1": data_keys.CGFCategories.SEVERE.value,
    #     }
    #     return pop.squeeze(axis=1).map(mapper)

    # def child_wasting_stratification_mapper(self, pop: pd.DataFrame) -> pd.Series:
    #     # applicable to stunting and wasting
    #     mapper = {
    #         "cat4": data_keys.ChildWastingCategories.UNEXPOSED.value,
    #         "cat3": data_keys.ChildWastingCategories.MILD.value,
    #         "cat2.5": data_keys.ChildWastingCategories.BETTER_MODERATE.value,
    #         "cat2": data_keys.ChildWastingCategories.WORSE_MODERATE.value,
    #         "cat1": data_keys.ChildWastingCategories.SEVERE.value,
    #     }
    #     return pop.squeeze(axis=1).map(mapper)

    def map_wasting_treatment(self, pop: pd.DataFrame) -> pd.Series:
        # Both SAM and MAM treatments
        mapper = {
            "cat3": "covered",
            "cat2": "covered",
            "cat1": "uncovered",
        }
        return pop.squeeze(axis=1).map(mapper)

    def map_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        """Map age with age group name strings

        Parameters
        ----------
        pop
            A DataFrame with one column, an age to be mapped to an age group name string

        Returns
        ------
        pandas.Series
            A pd.Series with age group name string corresponding to the pop passed into the function
        """
        bins = self.age_bins["age_start"].to_list() + [self.age_bins["age_end"].iloc[-1]]
        labels = self.age_bins["age_group_name"].to_list()
        # need to include lowest to map people who are exactly 0
        pop = pop.copy()
        pop.loc[pop.age > 5, "age"] = 5
        age_group = pd.cut(
            pop.squeeze(axis=1), bins, labels=labels, include_lowest=True
        ).rename("age_group")
        return age_group


# class BirthObserver(Component):
#     CONFIGURATION_DEFAULTS = {
#         "stratification": {
#             "birth": {
#                 "exclude": [],
#                 "include": [],
#             }
#         }
#     }

#     def __init__(self):
#         super().__init__()
#         self.birth_weight_column_name = "birth_weight_exposure"
#         self.gestational_age_column_name = "gestational_age_exposure"
#         self.low_birth_weight_limit = 2500  # grams

#     @property
#     def columns_required(self) -> Optional[List[str]]:
#         return [
#             "entrance_time",
#             self.birth_weight_column_name,
#             self.gestational_age_column_name,
#         ]

#     #################
#     # Setup methods #
#     #################

#     # noinspection PyAttributeOutsideInit
#     def setup(self, builder: Builder) -> None:
#         self.clock = builder.time.clock()
#         self.config = builder.configuration.stratification.birth

#         builder.results.register_observation(
#             name=f"live_births",
#             pop_filter="alive=='alive'",
#             aggregator=self.count_live_births,
#             requires_columns=["entrance_time"],
#             additional_stratifications=self.config.include,
#             excluded_stratifications=self.config.exclude,
#             when="collect_metrics",
#         )
#         builder.results.register_observation(
#             name=f"birth_weight_sum",
#             pop_filter="alive=='alive'",
#             aggregator=self.sum_birth_weights,
#             requires_columns=["entrance_time", self.birth_weight_column_name],
#             additional_stratifications=self.config.include,
#             excluded_stratifications=self.config.exclude,
#             when="collect_metrics",
#         )
#         builder.results.register_observation(
#             name=f"gestational_age_sum",
#             pop_filter="alive=='alive'",
#             aggregator=self.sum_gestational_ages,
#             requires_columns=["entrance_time", self.gestational_age_column_name],
#             additional_stratifications=self.config.include,
#             excluded_stratifications=self.config.exclude,
#             when="collect_metrics",
#         )
#         builder.results.register_observation(
#             name=f"low_weight_births",
#             pop_filter="alive=='alive'",
#             aggregator=self.count_low_weight_births,
#             requires_columns=["entrance_time", self.birth_weight_column_name],
#             additional_stratifications=self.config.include,
#             excluded_stratifications=self.config.exclude,
#             when="collect_metrics",
#         )

#     ########################
#     # Event-driven methods #
#     ########################
#     def count_live_births(self, x: pd.DataFrame) -> float:
#         born_this_step = x["entrance_time"] == self.clock()
#         return sum(born_this_step)

#     def sum_birth_weights(self, x: pd.DataFrame) -> float:
#         born_this_step = x["entrance_time"] == self.clock()
#         return x.loc[born_this_step, self.birth_weight_column_name].sum()

#     def sum_gestational_ages(self, x: pd.DataFrame) -> float:
#         born_this_step = x["entrance_time"] == self.clock()
#         return x.loc[born_this_step, self.gestational_age_column_name].sum()

#     def count_low_weight_births(self, x: pd.DataFrame) -> float:
#         born_this_step = x["entrance_time"] == self.clock()
#         has_low_birth_weight = (
#             x.loc[born_this_step, self.birth_weight_column_name] < self.low_birth_weight_limit
#         )
#         return sum(has_low_birth_weight)


class MortalityObserver(MortalityObserver_):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        stillborn = DiseaseState("stillborn")
        stillborn.set_model("stillborn")
        self.causes_of_death += [stillborn]


class ChildWastingObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("child_wasting")
        self.exposure_pipeline_name = f"{self.disease}.exposure"

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.disease: {
                    "exclude": [],
                    "include": [],
                }
            }
        }

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]
        self.disease_model = builder.components.get_component(
            f"child_wasting_model.{self.disease}"
        )
        self.entity_type = self.disease_model.cause_type
        self.entity = self.disease_model.cause
        self.transition_stratification_name = f"transition_{self.disease}"

    # We want diseaste state to be categories instead of the state_ids
    def register_disease_state_stratification(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "wasting_categories",
            list(builder.data.load(f"risk_factor.{self.disease}.categories").keys()),
            requires_values=[self.exposure_pipeline_name],
        )

    def register_person_time_observation(self, builder: Builder, pop_filter: str) -> None:
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_{self.disease}",
            pop_filter=pop_filter,
            when="time_step__prepare",
            requires_columns=["alive"],
            additional_stratifications=self.config.include + ["wasting_categories"],
            excluded_stratifications=self.config.exclude,
            aggregator=self.aggregate_state_person_time,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        if "transition_count_" in measure:
            results = results[results[self.transition_stratification_name] != "no_transition"]
            sub_entity = self.transition_stratification_name
        if "person_time_" in measure:
            sub_entity = "wasting_categories"
        results.rename(columns={sub_entity: COLUMNS.SUB_ENTITY}, inplace=True)
        return results
