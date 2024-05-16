from typing import List, Tuple

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks.distributions import (
    PolytomousDistribution as PolytomousDistribution_,
)


class PolytomousDistribution(PolytomousDistribution_):
    @property
    def categories(self) -> List[str]:
        return ["cat1", "cat2", "cat3", "cat4"]

    def __init__(self, risk: str, _exposure_data: Tuple[pd.DataFrame, List[str]]):
        super().__init__(risk, _exposure_data)
        self._exposure_data = _exposure_data[0]

    def get_exposure_parameters(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_parameters_pipeline_name,
            source=self.build_lookup_table(
                builder, self._exposure_data, value_columns=self.categories
            ),
        )
