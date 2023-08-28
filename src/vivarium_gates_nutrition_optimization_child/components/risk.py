import pandas as pd
from vivarium_public_health.risks import Risk
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline

from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_gates_nutrition_optimization_child.constants import data_values, paths
from vivarium_public_health.risks.distributions import PolytomousDistribution
import itertools
from typing import Dict


class Underweight(Risk):
    def __init__(self):
        super().__init__('risk_factor.underweight')
        self._sub_components = []

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.stunting = builder.value.get_value(data_values.PIPELINES.STUNTING_EXPOSURE)
        self.wasting = builder.value.get_value(data_values.PIPELINES.WASTING_EXPOSURE)
        self.distributions = self._get_distributions(builder)

    def _get_exposure_distribution(self) -> None:
        pass

    def _get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self._get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[self.propensity_pipeline_name, data_values.PIPELINES.STUNTING_EXPOSURE, data_values.PIPELINES.WASTING_EXPOSURE],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk),
        )

    def _get_distributions(self, builder: Builder) -> Dict[str, PolytomousDistribution]:
        # TODO: update to read from artifact when RT data has been updated
        # TODO: for now we're using the same conditional distribution
        # TODO: for all stunting and wasting values
        distributions = {}
        df = pd.read_csv(paths.RAW_DATA_DIR / "underweight_exposure_data.csv")
        categories = [f"cat{i+1}" for i in range(4)]
        for category_1, category_2 in itertools.product(categories, categories):
            key = f"stunting_{category_1}_wasting_{category_2}"
            distributions[key] = PolytomousDistribution(key, df)
        for dist in distributions.values():
            dist.setup(builder)
        return distributions

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        if len(index) == 0:
            return pd.Series(index=index) # only happens on first time step when there's no simulants
        propensity = self.propensity(index).rename('propensity')
        wasting = self.wasting(index).rename('wasting')
        stunting = self.stunting(index).rename('stunting')
        pop = pd.concat([stunting, wasting, propensity], axis=1)

        exposures = []
        for group, group_df in pop.groupby(['stunting', 'wasting']):
            stunting, wasting = group
            distribution = self.distributions[f"stunting_{stunting}_wasting_{wasting}"]
            exposure = distribution.ppf(group_df['propensity'])
            exposures.append(exposure)
        return pd.concat(exposures).sort_index()
