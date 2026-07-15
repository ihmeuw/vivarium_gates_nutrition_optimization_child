"""
================================
Calibration PAF subnational fill
================================

Reindex a RiskEffect's calibration-constant (PAF) table onto a shared grid.

When two RiskEffects target the same rate -- here CGF and LBWSG on the
diarrheal/LRI excess mortality rate -- vph hands both PAF tables to the
``raw_union`` combiner, which aligns them by index. CGF is subnational- and
post-neonatal-specific; LBWSG is national and neonatal-only. Left unaligned,
the union produces NaN and setup fails. Reindexing both onto
``subnational x sex x under-5 age bin x year`` with 0-fill lets them combine;
because the two risks are age-disjoint, the fill changes no values.
"""

import pandas as pd
from vivarium.engine.framework.engine import Builder

from vivarium_gates_nutrition_optimization_child.constants import data_keys, metadata

_DIMENSION_COLUMNS = [
    "subnational",
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]


def fill_subnational_paf_rows(builder: Builder, paf: pd.DataFrame) -> pd.DataFrame:
    """Reindex a PAF table onto the shared subnational x sex x under-5 age x year grid.

    Missing rows are 0-filled. When ``paf`` lacks a subnational level its values
    are broadcast across every subnational. Non-DataFrame inputs (e.g. scalars)
    are returned unchanged.
    """
    if not isinstance(paf, pd.DataFrame):
        return paf

    value_columns = [c for c in paf.columns if c not in _DIMENSION_COLUMNS]

    # Pull the canonical age bins from the artifact so boundaries match the
    # tables' own float representation exactly (avoids near-equal mismatches).
    age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)
    under_5 = (
        age_bins[age_bins["age_end"] <= 5.0][["age_start", "age_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    location = builder.data.load(data_keys.POPULATION.LOCATION)
    if isinstance(location, (list, tuple, pd.Series, pd.Index)):
        location = list(location)[0]
    subnationals = metadata.SUBNATIONAL_LOCATION_DICT[location]

    sexes = sorted(paf["sex"].unique())
    years = paf[["year_start", "year_end"]].drop_duplicates()

    grid = (
        pd.DataFrame({"subnational": subnationals})
        .merge(pd.DataFrame({"sex": sexes}), how="cross")
        .merge(under_5, how="cross")
        .merge(years, how="cross")
    )

    merge_keys = [c for c in _DIMENSION_COLUMNS if c in paf.columns]
    filled = grid.merge(paf, on=merge_keys, how="left")
    filled[value_columns] = filled[value_columns].fillna(0.0)
    return filled[_DIMENSION_COLUMNS + value_columns]
