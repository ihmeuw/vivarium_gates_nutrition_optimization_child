"""
====================================
Low Birth Weight and Short Gestation
====================================

This is a module to subclass the LBWSG component in Vivrium Public Health to use its functionality but to do so on
simulants who are initialized from line list data.

"""

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from vivarium.engine.component import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.event import Event
from vivarium.engine.framework.population import SimulantData
from vivarium.engine.framework.time import get_time_stamp
from vivarium.engine.framework.values import Pipeline
from vivarium.public_health.causal_factor.utilities import get_exposure_post_processor
from vivarium.public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    AXES,
    BIRTH_WEIGHT,
    GESTATIONAL_AGE,
    LBWSGRisk,
    LBWSGRiskEffect,
    Risk,
)
from vivarium.public_health.utilities import TargetString

from vivarium_gates_nutrition_optimization_child.components.calibration_paf import (
    fill_subnational_paf_rows,
)
from vivarium_gates_nutrition_optimization_child.constants import data_keys


class LBWSGLineList(LBWSGRisk):
    """
    Component to initialize low birthweight and short gestation data for simulants based on existing line list data.
    """

    def __init__(self):
        super().__init__()
        self.raw_gestational_age_exposure_column_name = "raw_gestational_age_exposure"
        self.birth_weight_status_column_name = "birth_weight_status"

    def setup(self, builder: Builder):
        Risk.setup(self, builder)

        self.configuration_age_end = builder.configuration.population.initialization_age_max
        self.start_time = get_time_stamp(builder.configuration.time.start)

        self.register_birth_exposure_pipeline(builder)

        continuous_propensity_columns = [
            self.get_continuous_propensity_name(axis) for axis in AXES
        ]
        builder.population.register_initializer(
            initializer=self.initialize_categorical_and_continuous_propensities,
            columns=[self.categorical_propensity_name, *continuous_propensity_columns],
            required_resources=[self.randomness],
        )

        builder.population.register_initializer(
            initializer=self.initialize_exposure,
            columns=[self.get_exposure_name(axis) for axis in AXES]
            + [
                self.raw_gestational_age_exposure_column_name,
                self.birth_weight_status_column_name,
            ],
            required_resources=[self.birth_exposure_pipeline],
        )

    def register_birth_exposure_pipeline(self, builder: Builder) -> Dict[str, Pipeline]:
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=lambda index: self.get_birth_exposure(index),
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            description=(
                "The simulant's LBWSG exposure at birth, read from the maternal model's "
                "birth line list"
            ),
        )

    ########################
    # Event-driven methods #
    ########################

    # noinspection PyAttributeOutsideInit
    def initialize_exposure(self, pop_data: SimulantData) -> None:
        if pop_data.creation_time < self.start_time:
            # We initialize the simulation with no simulants so create an empty df
            columns = [self.get_exposure_name(axis) for axis in AXES] + [
                self.raw_gestational_age_exposure_column_name,
                self.birth_weight_status_column_name,
            ]
            new_simulants = pd.DataFrame(columns=columns, index=pop_data.index)
            self.population_view.initialize(new_simulants)
        else:
            self.new_births = pop_data.user_data["new_births"]
            self.new_births.index = pop_data.index
            # add raw gestational age exposure to state table
            gestational_age = pop_data.user_data["new_births"]["gestational_age"].copy()
            gestational_age.name = self.raw_gestational_age_exposure_column_name
            self.population_view.initialize(gestational_age)
            super().initialize_exposure(pop_data)

            # add birth weight status to state table
            birth_weight = self.population_view.get(pop_data.index, "birth_weight.exposure")
            birth_weight_status = np.where(
                birth_weight <= 2500, "low_birth_weight", "adequate_birth_weight"
            )
            birth_weight_status = pd.Series(
                birth_weight_status, name=self.birth_weight_status_column_name
            )
            self.population_view.initialize(birth_weight_status)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_birth_exposure(self, index: pd.Index) -> pd.Series:
        return self.new_births.loc[index, AXES]


class SubnationalLBWSGRiskEffect(LBWSGRiskEffect):
    """LBWSGRiskEffect whose calibration PAF is filled across subnational rows.

    Use in place of ``LBWSGRiskEffect`` where LBWSG shares a target rate with
    the subnational, post-neonatal CGF effect (diarrheal/LRI excess mortality)
    so vph's ``raw_union`` can align the two PAFs.
    """

    def get_calibration_constant_data(self, builder: Builder) -> pd.DataFrame:
        """Broadcast the national, neonatal-only LBWSG PAF onto the shared grid."""
        paf = super().get_calibration_constant_data(builder)
        return fill_subnational_paf_rows(builder, paf)


class LBWSGPAFCalculationExposure(LBWSGRisk):
    """Exposure component that places the PAF simulation's cohort on a grid.

    Each (sex, LBWSG category) holds k**2 simulants, placed at the cell midpoints
    of a k by k grid over the category's gestational age and birth weight box.
    """

    # The exposure columns are registered by LBWSGRisk's own initializer.
    COLUMNS_CREATED = ["lbwsg_category", "age_bin"]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.lbwsg_categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        self.age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)
        builder.population.register_initializer(
            initializer=self.initialize_category_assignments,
            columns=self.COLUMNS_CREATED,
            required_resources=["age", "sex"],
        )
        super().setup(builder)

    def register_birth_exposure_pipeline(self, builder: Builder) -> None:
        """Register the birth exposure pipeline with the columns its source reads."""
        # The base class requires only the exposure distribution, which this source
        # never samples, so the initializer order would otherwise be left to chance.
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=self._get_birth_exposure_source,
            required_resources=["sex", "lbwsg_category"],
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
            description="The simulant's LBWSG exposure at birth, on the PAF grid",
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_category_assignments(self, pop_data: SimulantData) -> None:
        """Assign every (sex, category) the same square number of simulants."""
        pop = self.population_view.get(pop_data.index, ["age", "sex"])
        pop["age_bin"] = pd.cut(pop["age"], self.age_bins["age_start"])
        if pop["age_bin"].nunique(dropna=False) != 1:
            raise ValueError(
                "The LBWSG PAF cohort must start in a single age bin, but its ages fall "
                f"in {sorted(pop['age_bin'].astype(str).unique())}."
            )
        pop = pop.sort_values(["sex", "age"])

        lbwsg_categories = list(self.lbwsg_categories.keys())
        num_repeats, remainder = divmod(len(pop), 2 * len(lbwsg_categories))
        if remainder != 0 or math.isqrt(num_repeats) ** 2 != num_repeats:
            raise ValueError(
                "Population size must be 2 * (number of LBWSG categories) * k**2 for an "
                "integer k, so that each (sex, category) fills a square grid. Population "
                f"size is {len(pop)} with {len(lbwsg_categories)} categories."
            )

        pop["lbwsg_category"] = lbwsg_categories * (2 * num_repeats)
        self.population_view.initialize(pop[self.COLUMNS_CREATED])

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_birth_exposure_source(self, index: pd.Index) -> pd.DataFrame:
        """Spread each (sex, category) evenly over the category's box."""
        pop = self.population_view.get(index, ["sex", "lbwsg_category"])
        exposure = pd.DataFrame(index=pop.index, columns=AXES, dtype=float)

        for (_, category), group in pop.groupby(["sex", "lbwsg_category"]):
            num_points = math.isqrt(len(group))
            if num_points**2 != len(group):
                raise ValueError(
                    f"LBWSG category {category} holds {len(group)} simulants of one sex, "
                    "which is not a square number."
                )
            gestational_age_endpoints, birth_weight_endpoints = _parse_category_box(
                self.lbwsg_categories[category]
            )
            birth_weight_points, gestational_age_points = np.meshgrid(
                _cell_midpoints(*birth_weight_endpoints, num_points),
                _cell_midpoints(*gestational_age_endpoints, num_points),
            )
            exposure.loc[group.index, BIRTH_WEIGHT] = birth_weight_points.ravel()
            exposure.loc[group.index, GESTATIONAL_AGE] = gestational_age_points.ravel()

        return exposure


class LBWSGPAFObserver(Component):
    """Observe the LBWSG PAF of a birth cohort in each neonatal age group.

    The cohort is born together, so at each observation every living simulant is
    in the same age group. The early neonatal PAF is observed at birth, before any
    death, and weights the categories by GBD birth prevalence. The late neonatal
    PAF is observed on entry to late neonatal and weights the categories by birth
    prevalence times the share of each category this simulation left alive. Each
    age group keeps its value from the first time step that observes it.
    """

    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "lbwsg_paf": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.birth_exposure = builder.data.load(data_keys.LBWSG.BIRTH_EXPOSURE)
        # The weights are merged onto the per-category mean relative risks, so a
        # category missing from either side would silently renormalize the rest.
        categories = set(builder.data.load(data_keys.LBWSG.CATEGORIES))
        birth_exposure_categories = set(self.birth_exposure["parameter"])
        if birth_exposure_categories != categories:
            raise ValueError(
                "LBWSG birth exposure categories do not match the LBWSG categories. "
                f"Only in birth exposure: {sorted(birth_exposure_categories - categories)}; "
                f"only in categories: {sorted(categories - birth_exposure_categories)}."
            )
        age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)
        self.age_bin_edges = sorted(age_bins["age_start"]) + [age_bins["age_end"].max()]
        self.risk_effect = builder.components.get_component(
            f"risk_effect.low_birth_weight_and_short_gestation_on_{self.target}"
        )
        self.config = builder.configuration.stratification.lbwsg_paf
        self.pop_size = builder.configuration.population.population_size

        builder.results.register_stratified_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}",
            pop_filter="is_alive == True",
            aggregator=self.calculate_paf,
            results_updater=self.results_updater,
            # Everything the aggregator reads, so that it reads it off the frame the
            # results system hands over rather than looking it up by index.
            requires_attributes=[
                "is_alive",
                "sex",
                "lbwsg_category",
                self.risk_effect.relative_risk_name,
            ],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            # Before mortality and aging, so the first step sees the cohort at birth.
            when="time_step__prepare",
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step_prepare(self, event: Event) -> None:
        """Raise if the living cohort spans more than one age group."""
        pop = self.population_view.get(event.index, ["age", "is_alive"])
        living_ages = pop.loc[pop["is_alive"], "age"]
        age_groups = pd.cut(living_ages, self.age_bin_edges, right=False).unique()
        if len(age_groups) > 1:
            raise ValueError(
                "The living LBWSG PAF cohort spans more than one age group "
                f"({sorted(str(group) for group in age_groups)}), so the PAF observed "
                "now would mix them. Check the initial ages and the step size."
            )

    ##################
    # Helper methods #
    ##################

    def results_updater(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """Keep each stratum's first non-zero PAF."""
        return old.where(old != 0.0, new.reindex(old.index, fill_value=0.0))

    def calculate_paf(self, x: pd.DataFrame) -> float:
        """Return one stratum's PAF, weighting categories by birth prevalence still alive."""
        if x.empty:
            # The cohort occupies one age group at a time, so the other age group is
            # handed over empty; the results updater ignores the zero.
            return 0.0

        relative_risk = x[self.risk_effect.relative_risk_name].rename("relative_risk")
        lbwsg_category = x["lbwsg_category"]
        sexes = x["sex"].unique()
        if len(sexes) != 1:
            raise ValueError(
                f"Stratified data contains {sorted(sexes)}, but this observer "
                "(LBWSGPAFObserver) needs sex-stratified data. Check that 'sex' is "
                "among the simulation's stratifications."
            )
        sex = sexes[0]

        lbwsg_prevalence = self.birth_exposure.rename(
            {"parameter": "lbwsg_category", "value": "prevalence"}, axis=1
        )
        lbwsg_prevalence = lbwsg_prevalence.loc[lbwsg_prevalence["sex"] == sex]

        # The share alive is 1 at birth, so the early neonatal weights are birth
        # prevalence and the late neonatal weights are depleted by this simulation.
        weights = calculate_mortality_weights(self, sex)
        lbwsg_prevalence = lbwsg_prevalence.merge(weights)
        lbwsg_prevalence["prevalence"] = (
            lbwsg_prevalence["prevalence"] * lbwsg_prevalence["proportion_alive"]
        )
        lbwsg_prevalence = lbwsg_prevalence.drop(columns=["proportion_alive"])

        mean_rrs = (
            pd.concat([lbwsg_category, relative_risk], axis=1)
            .groupby("lbwsg_category", as_index=False)
            .mean()
        )
        mean_rrs = mean_rrs.merge(lbwsg_prevalence, on="lbwsg_category")

        mean_rr = np.average(mean_rrs["relative_risk"], weights=mean_rrs["prevalence"])
        paf = (mean_rr - 1) / mean_rr

        return paf


#####################
# Utility functions #
#####################


def calculate_mortality_weights(component: Component, sex: str) -> pd.DataFrame:
    """Calculate the fraction of each LBWSG category still alive, for one sex.

    The whole cohort is read, not just the simulants under observation, so the
    denominator is everyone born into the category.
    """
    full_index = pd.Index(range(component.pop_size))
    pop_data = component.population_view.get(
        full_index, ["lbwsg_category", "is_alive", "sex"], include_untracked=True
    )
    pop_data = pop_data.loc[pop_data["sex"] == sex]
    weights = (
        pop_data.groupby(["lbwsg_category", "sex"])["is_alive"]
        .agg(proportion_alive=lambda x: x.mean())
        .reset_index()
    )
    return weights


def _parse_category_box(
    description: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the (gestational age, birth weight) endpoints in a category description."""
    # descriptions look like '... - [34, 36) wks, [2000, 2500) g' or '[40, 42+] wks'
    gestational_age = tuple(
        float(val)
        for val in description.split("- [")[1].split(")")[0].split("+")[0].split(", ")
    )
    birth_weight = tuple(
        float(val)
        for val in description.split(", [")[1].split(")")[0].split("]")[0].split(", ")
    )
    return gestational_age, birth_weight


def _cell_midpoints(lower: float, upper: float, num_points: int) -> np.ndarray:
    """Return the midpoints of num_points equal cells spanning [lower, upper]."""
    return lower + (np.arange(num_points) + 0.5) * (upper - lower) / num_points
