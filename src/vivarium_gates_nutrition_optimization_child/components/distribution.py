from typing import Optional

import numpy as np
import pandas as pd
from vivarium.config_tree import LayeredConfigTree
from vivarium.engine.framework.engine import Builder
from vivarium.public_health.causal_factor.distributions import PolytomousDistribution
from vivarium.public_health.utilities import EntityString


class CGFPolytomousDistribution(PolytomousDistribution):
    @property
    def name(self) -> str:
        return f"cgf_polytomous_distribution.{self.causal_factor}"

    def __init__(self, risk: EntityString, exposure_data: pd.DataFrame):
        super().__init__(risk, "ordered_polytomous", exposure_data)
        self.causal_factor_propensity = "child_underweight.propensity"

    def get_configuration(self, builder: "Builder") -> Optional[LayeredConfigTree]:
        pass

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        # These sub-distributions are called directly by ChildUnderweight,
        # not through the pipeline system, so we skip registering a PPF
        # pipeline (which would require a nonexistent propensity pipeline).
        pass
