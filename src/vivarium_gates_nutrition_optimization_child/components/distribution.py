from typing import Optional

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder
from vivarium_public_health.risks.distributions import (
    MissingDataError,
    PolytomousDistribution,
)
from vivarium_public_health.utilities import EntityString


class CGFPolytomousDistribution(PolytomousDistribution):
    def __init__(self, risk: EntityString, exposure_data: pd.DataFrame):
        super().__init__(risk, "ordered_polytomous", exposure_data)
        # Overwrite the risk propensity name since all sub-distributions share the
        # parent ChildUnderweight's propensity column
        self.risk_propensity = "child_underweight.propensity"

    def get_configuration(self, builder: "Builder") -> Optional[LayeredConfigTree]:
        pass

    def register_exposure_ppf_pipeline(self, builder: Builder) -> None:
        # These sub-distributions are called directly by ChildUnderweight,
        # not through the pipeline system, so we skip registering a PPF
        # pipeline (which would require a nonexistent propensity pipeline).
        pass
