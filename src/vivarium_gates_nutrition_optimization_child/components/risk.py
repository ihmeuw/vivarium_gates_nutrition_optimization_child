from typing import Callable

import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium_public_health.risks import RiskEffect as RiskEffect_
from vivarium_public_health.risks.data_transformations import get_distribution_type

class RiskEffect(RiskEffect_):

    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.exposure_pipeline_name = f'{self.risk.name}.exposure'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.exposure_distribution_type = self._get_distribution_type(builder)
        self.exposure = self._get_risk_exposure(builder)
        self.relative_risk = self._get_relative_risk_source(builder)
        self.population_attributable_fraction = self._get_population_attributable_fraction_source(
            builder
        )
        self.target_modifier = self._get_target_modifier(builder)

        self._register_target_modifier(builder)
        self._register_paf_modifier(builder)

    def _get_distribution_type(self, builder: Builder) -> str:
        return get_distribution_type(builder, self.risk)

    def _get_risk_exposure(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return builder.value.get_value(self.exposure_pipeline_name)

    def _get_target_modifier(self, builder: Builder) -> Callable[[pd.Index, pd.Series], pd.Series]:
        if self.exposure_distribution_type in ['normal', 'lognormal', 'ensemble']:
            tmred = builder.data.load(f"{self.risk}.tmred")
            tmrel = 0.5 * (tmred["min"] + tmred["max"])
            scale = builder.data.load(f"{self.risk}.relative_risk_scalar")

            def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
                rr = self.relative_risk(index)
                exposure = self.exposure(index)
                relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
                return target * relative_risk
        else:
            index_columns = ['index', self.risk.name]

            def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
                rr = self.relative_risk(index)
                exposure = self.exposure(index).reset_index()
                exposure.columns = index_columns
                exposure = exposure.set_index(index_columns)

                relative_risk = rr.stack().reset_index()
                relative_risk.columns = index_columns + ['value']
                relative_risk = relative_risk.set_index(index_columns)

                effect = relative_risk.loc[exposure.index, 'value'].droplevel(self.risk.name)
                affected_rates = target * effect
                return affected_rates

        return adjust_target
