from typing import Dict, Tuple

import pandas as pd
from vivarium.artifact import EntityKey
from vivarium.gbd_mapping import causes, covariates, risk_factors
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    paths,
)
from vivarium_gates_nutrition_optimization_child.constants.metadata import LOCATIONS


def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]


def get_intervals_from_categories(lbwsg_type: str, categories: Dict[str, str]) -> pd.Series:
    if lbwsg_type == "low_birth_weight":
        category_endpoints = pd.Series(
            {
                cat: parse_low_birth_weight_description(description)
                for cat, description in categories.items()
            },
            name=f"{lbwsg_type}.endpoints",
        )
    elif lbwsg_type == "short_gestation":
        category_endpoints = pd.Series(
            {
                cat: parse_short_gestation_description(description)
                for cat, description in categories.items()
            },
            name=f"{lbwsg_type}.endpoints",
        )
    else:
        raise ValueError(
            f"Unrecognized risk type {lbwsg_type}.  Risk type must be low_birth_weight or short_gestation"
        )
    category_endpoints.index.name = "parameter"

    return category_endpoints


def parse_low_birth_weight_description(description: str) -> pd.Interval:
    # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'

    endpoints = pd.Interval(
        *[
            float(val)
            for val in description.split(", [")[1].split(")")[0].split("]")[0].split(", ")
        ]
    )
    return endpoints


def parse_short_gestation_description(description: str) -> pd.Interval:
    # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'

    endpoints = pd.Interval(
        *[
            float(val)
            for val in description.split("- [")[1].split(")")[0].split("+")[0].split(", ")
        ]
    )
    return endpoints


def get_treatment_efficacy(
    demography: pd.DataFrame, treatment_type: str, location: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_efficacy = {
        data_keys.WASTING.CAT1: get_wasting_treatment_parameter_data("e_sam", location),
        data_keys.WASTING.CAT2: get_wasting_treatment_parameter_data("e_mam", location),
    }
    alternative_efficacy = {
        data_keys.WASTING.CAT1: data_values.WASTING.SAM_TX_ALTERNATIVE_EFFICACY,
        data_keys.WASTING.CAT2: data_values.WASTING.MAM_TX_ALTERNATIVE_EFFICACY,
    }
    idx_as_frame = demography.merge(
        pd.DataFrame({"parameter": [f"cat{i}" for i in range(1, 4)]}), how="cross"
    )
    index = idx_as_frame.set_index(list(idx_as_frame.columns)).index

    efficacy = pd.DataFrame({f"draw_{i}": 1.0 for i in range(0, 1000)}, index=index)
    efficacy[index.get_level_values("parameter") == "cat1"] *= 0.0
    efficacy[index.get_level_values("parameter") == "cat2"] *= baseline_efficacy[
        treatment_type
    ]
    efficacy[index.get_level_values("parameter") == "cat3"] *= alternative_efficacy[
        treatment_type
    ]

    tmrel_efficacy = efficacy[
        efficacy.index.get_level_values("parameter") == data_keys.MAM_TREATMENT.TMREL_CATEGORY
    ].droplevel("parameter")
    return efficacy, tmrel_efficacy


def get_wasting_treatment_parameter_data(parameter: str, location: str) -> pd.Series:
    """Get coverage or efficacy values for SAM or MAM treatment for all draws."""
    draws = pd.read_csv(paths.WASTING_TREATMENT_PARAMETERS_DIR / f"{location.lower()}.csv")
    draws = draws.query("parameter==@parameter").drop("parameter", axis=1)
    draws = draws.T.squeeze()  # transpose and convert to series
    return draws


def rename_subnational_level(data: pd.DataFrame) -> pd.DataFrame:
    if data.index.get_level_values("location")[0] not in LOCATIONS:
        data.index = data.index.rename({"location": "subnational"})
    return data
