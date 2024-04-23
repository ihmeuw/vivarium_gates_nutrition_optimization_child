"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""

import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from gbd_mapping import Cause, RiskFactor, sequelae
from scipy.interpolate import RectBivariateSpline, griddata
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import gbd
from vivarium_inputs import extract
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.globals import DEMOGRAPHIC_COLUMNS, DRAW_COLUMNS
from vivarium_inputs.mapping_extension import AlternativeRiskFactor
from vivarium_public_health.utilities import TargetString

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    data_values,
    metadata,
    paths,
)
from vivarium_gates_nutrition_optimization_child.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
)
from vivarium_gates_nutrition_optimization_child.data import utilities
from vivarium_gates_nutrition_optimization_child.utilities import (
    get_lognorm_from_quantiles,
    get_random_variable_draws,
)

NATIONAL_LEVEL_DATA_KEYS = [
    data_keys.POPULATION.LOCATION,
    data_keys.POPULATION.STRUCTURE,
    data_keys.POPULATION.AGE_BINS,
    data_keys.POPULATION.DEMOGRAPHY,
    data_keys.POPULATION.TMRLE,
    data_keys.DIARRHEA.DURATION,
    data_keys.DIARRHEA.REMISSION_RATE,
    data_keys.DIARRHEA.RESTRICTIONS,
    data_keys.MEASLES.RESTRICTIONS,
    data_keys.LRI.DURATION,
    data_keys.LRI.REMISSION_RATE,
    data_keys.LRI.RESTRICTIONS,
    data_keys.MALARIA.DURATION,
    data_keys.MALARIA.REMISSION_RATE,
    data_keys.MALARIA.RESTRICTIONS,
    data_keys.WASTING.EXPOSURE,
    data_keys.WASTING.DISTRIBUTION,
    data_keys.WASTING.ALT_DISTRIBUTION,
    data_keys.WASTING.CATEGORIES,
    data_keys.WASTING.RELATIVE_RISK,
    data_keys.WASTING.PAF,
    data_keys.STUNTING.PAF,
    data_keys.STUNTING.EXPOSURE,
    data_keys.STUNTING.DISTRIBUTION,
    data_keys.STUNTING.ALT_DISTRIBUTION,
    data_keys.STUNTING.CATEGORIES,
    data_keys.STUNTING.RELATIVE_RISK,
    data_keys.UNDERWEIGHT.RELATIVE_RISK,
    data_keys.UNDERWEIGHT.DISTRIBUTION,
    data_keys.UNDERWEIGHT.CATEGORIES,
    data_keys.PEM.RESTRICTIONS,
    data_keys.MODERATE_PEM.RESTRICTIONS,
    data_keys.SEVERE_PEM.RESTRICTIONS,
    data_keys.SAM_TREATMENT.EXPOSURE,
    data_keys.SAM_TREATMENT.DISTRIBUTION,
    data_keys.SAM_TREATMENT.CATEGORIES,
    data_keys.SAM_TREATMENT.RELATIVE_RISK,
    data_keys.SAM_TREATMENT.PAF,
    data_keys.MAM_TREATMENT.EXPOSURE,
    data_keys.MAM_TREATMENT.DISTRIBUTION,
    data_keys.MAM_TREATMENT.CATEGORIES,
    data_keys.MAM_TREATMENT.RELATIVE_RISK,
    data_keys.MAM_TREATMENT.PAF,
    data_keys.LBWSG.DISTRIBUTION,
    data_keys.LBWSG.CATEGORIES,
    data_keys.LBWSG.EXPOSURE,
    data_keys.LBWSG.RELATIVE_RISK,
    data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR,
    data_keys.LBWSG.PAF,
    data_keys.AFFECTED_UNMODELED_CAUSES.URI_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.OTITIS_MEDIA_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.MENINGITIS_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.ENCEPHALITIS_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_PRETERM_BIRTH_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_ENCEPHALOPATHY_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_SEPSIS_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_JAUNDICE_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.OTHER_NEONATAL_DISORDERS_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.SIDS_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_LRI_CSMR,
    data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_DIARRHEAL_DISEASES_CSMR,
    data_keys.IFA_SUPPLEMENTATION.DISTRIBUTION,
    data_keys.IFA_SUPPLEMENTATION.CATEGORIES,
    data_keys.IFA_SUPPLEMENTATION.EXPOSURE,
    data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT,
    data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT,
    data_keys.MMN_SUPPLEMENTATION.DISTRIBUTION,
    data_keys.MMN_SUPPLEMENTATION.CATEGORIES,
    data_keys.MMN_SUPPLEMENTATION.EXPOSURE,
    data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT,
    data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1,
    data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2,
    data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT,
    data_keys.BEP_SUPPLEMENTATION.DISTRIBUTION,
    data_keys.BEP_SUPPLEMENTATION.CATEGORIES,
    data_keys.BEP_SUPPLEMENTATION.EXPOSURE,
    data_keys.BEP_SUPPLEMENTATION.EXCESS_SHIFT,
    data_keys.BEP_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT,
    data_keys.IV_IRON.DISTRIBUTION,
    data_keys.IV_IRON.CATEGORIES,
    data_keys.IV_IRON.EXPOSURE,
    data_keys.IV_IRON.EXCESS_SHIFT,
    data_keys.IV_IRON.RISK_SPECIFIC_SHIFT,
    data_keys.MATERNAL_BMI_ANEMIA.DISTRIBUTION,
    data_keys.MATERNAL_BMI_ANEMIA.CATEGORIES,
    data_keys.MATERNAL_BMI_ANEMIA.EXPOSURE,
    data_keys.MATERNAL_BMI_ANEMIA.EXCESS_SHIFT,
    data_keys.MATERNAL_BMI_ANEMIA.RISK_SPECIFIC_SHIFT,
]


def get_data(
    lookup_key: str, location: Union[str, List[int]], fetch_subnationals: bool = False
) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.POPULATION.CRUDE_BIRTH_RATE: load_standard_data,
        data_keys.DIARRHEA.DURATION: load_duration,
        data_keys.DIARRHEA.PREVALENCE: load_prevalence_from_incidence_and_duration,
        data_keys.DIARRHEA.INCIDENCE_RATE: load_standard_data,
        data_keys.DIARRHEA.REMISSION_RATE: load_neonatal_deleted_remission_from_duration,
        data_keys.DIARRHEA.DISABILITY_WEIGHT: load_standard_data,
        data_keys.DIARRHEA.EMR: load_emr_from_csmr_and_prevalence,
        data_keys.DIARRHEA.CSMR: load_neonatal_deleted_csmr,
        data_keys.DIARRHEA.RESTRICTIONS: load_metadata,
        data_keys.DIARRHEA.BIRTH_PREVALENCE: load_post_neonatal_birth_prevalence,
        data_keys.MEASLES.PREVALENCE: load_standard_data,
        data_keys.MEASLES.INCIDENCE_RATE: load_standard_data,
        data_keys.MEASLES.DISABILITY_WEIGHT: load_standard_data,
        data_keys.MEASLES.EMR: load_standard_data,
        data_keys.MEASLES.CSMR: load_standard_data,
        data_keys.MEASLES.RESTRICTIONS: load_metadata,
        data_keys.LRI.DURATION: load_duration,
        data_keys.LRI.INCIDENCE_RATE: load_standard_data,
        data_keys.LRI.PREVALENCE: load_prevalence_from_incidence_and_duration,
        data_keys.LRI.REMISSION_RATE: load_neonatal_deleted_remission_from_duration,
        data_keys.LRI.DISABILITY_WEIGHT: load_standard_data,
        data_keys.LRI.EMR: load_emr_from_csmr_and_prevalence,
        data_keys.LRI.CSMR: load_neonatal_deleted_csmr,
        data_keys.LRI.RESTRICTIONS: load_metadata,
        data_keys.MALARIA.DURATION: load_duration,
        data_keys.MALARIA.PREVALENCE: load_prevalence_from_incidence_and_duration,
        data_keys.MALARIA.INCIDENCE_RATE: load_standard_data,
        data_keys.MALARIA.REMISSION_RATE: load_neonatal_deleted_malaria_remission_from_duration,
        data_keys.MALARIA.DISABILITY_WEIGHT: load_standard_data,
        data_keys.MALARIA.EMR: load_emr_from_csmr_and_prevalence,
        data_keys.MALARIA.CSMR: load_neonatal_deleted_csmr,
        data_keys.MALARIA.RESTRICTIONS: load_metadata,
        data_keys.MALARIA.BIRTH_PREVALENCE: load_post_neonatal_birth_prevalence,
        data_keys.WASTING.DISTRIBUTION: load_metadata,
        data_keys.WASTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.WASTING.CATEGORIES: load_metadata,
        data_keys.WASTING.EXPOSURE: load_gbd_2021_exposure,
        data_keys.WASTING.RELATIVE_RISK: load_wasting_rr,
        data_keys.WASTING.PAF: load_categorical_paf,
        data_keys.WASTING.TRANSITION_RATES: load_wasting_transition_rates,
        data_keys.WASTING.BIRTH_PREVALENCE: load_wasting_birth_prevalence,
        data_keys.STUNTING.DISTRIBUTION: load_metadata,
        data_keys.STUNTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.STUNTING.CATEGORIES: load_metadata,
        data_keys.STUNTING.EXPOSURE: load_gbd_2021_exposure,
        data_keys.STUNTING.RELATIVE_RISK: load_gbd_2021_rr,
        data_keys.STUNTING.PAF: load_categorical_paf,
        data_keys.UNDERWEIGHT.DISTRIBUTION: load_metadata,
        data_keys.UNDERWEIGHT.EXPOSURE: load_underweight_exposure,
        data_keys.UNDERWEIGHT.CATEGORIES: load_metadata,
        data_keys.UNDERWEIGHT.RELATIVE_RISK: load_gbd_2021_rr,
        data_keys.CHILD_GROWTH_FAILURE.PAF: load_cgf_paf,
        data_keys.PEM.EMR: load_pem_emr,
        data_keys.PEM.CSMR: load_pem_csmr,
        data_keys.PEM.RESTRICTIONS: load_pem_restrictions,
        data_keys.MODERATE_PEM.DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.MODERATE_PEM.EMR: load_pem_emr,
        data_keys.MODERATE_PEM.CSMR: load_pem_csmr,
        data_keys.MODERATE_PEM.RESTRICTIONS: load_pem_restrictions,
        data_keys.SEVERE_PEM.DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.SEVERE_PEM.EMR: load_pem_emr,
        data_keys.SEVERE_PEM.CSMR: load_pem_csmr,
        data_keys.SEVERE_PEM.RESTRICTIONS: load_pem_restrictions,
        data_keys.SAM_TREATMENT.EXPOSURE: load_wasting_treatment_exposure,
        data_keys.SAM_TREATMENT.DISTRIBUTION: load_wasting_treatment_distribution,
        data_keys.SAM_TREATMENT.CATEGORIES: load_wasting_treatment_categories,
        data_keys.SAM_TREATMENT.RELATIVE_RISK: load_sam_treatment_rr,
        data_keys.SAM_TREATMENT.PAF: load_categorical_paf,
        data_keys.MAM_TREATMENT.EXPOSURE: load_wasting_treatment_exposure,
        data_keys.MAM_TREATMENT.DISTRIBUTION: load_wasting_treatment_distribution,
        data_keys.MAM_TREATMENT.CATEGORIES: load_wasting_treatment_categories,
        data_keys.MAM_TREATMENT.RELATIVE_RISK: load_mam_treatment_rr,
        data_keys.MAM_TREATMENT.PAF: load_categorical_paf,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,  ## Still 2019 age bins, but doesn't have effect past NN
        data_keys.LBWSG.RELATIVE_RISK: load_lbwsg_rr,  ## Still 2019 age bins, but doesn't have effect past NN
        data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR: load_lbwsg_interpolated_rr,  ## Still 2019 age bins, but doesn't have effect past NN
        data_keys.LBWSG.PAF: load_lbwsg_paf,  ## Still 2019 age bins, but doesn't have effect past NN
        data_keys.AFFECTED_UNMODELED_CAUSES.URI_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.OTITIS_MEDIA_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.MENINGITIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.ENCEPHALITIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_PRETERM_BIRTH_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_ENCEPHALOPATHY_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_SEPSIS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_JAUNDICE_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.OTHER_NEONATAL_DISORDERS_CSMR: load_standard_data,
        data_keys.AFFECTED_UNMODELED_CAUSES.SIDS_CSMR: load_sids_csmr,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_LRI_CSMR: load_neonatal_lri_csmr,
        data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_DIARRHEAL_DISEASES_CSMR: load_neonatal_diarrhea_csmr,
        data_keys.IFA_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.IFA_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.IFA_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: load_ifa_excess_shift,
        data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        data_keys.MMN_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.MMN_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.MMN_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1: load_excess_gestational_age_shift,
        data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2: load_excess_gestational_age_shift,
        data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        data_keys.BEP_SUPPLEMENTATION.DISTRIBUTION: load_intervention_distribution,
        data_keys.BEP_SUPPLEMENTATION.CATEGORIES: load_intervention_categories,
        data_keys.BEP_SUPPLEMENTATION.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.BEP_SUPPLEMENTATION.EXCESS_SHIFT: load_bep_excess_shift,
        data_keys.BEP_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        data_keys.IV_IRON.DISTRIBUTION: load_intervention_distribution,
        data_keys.IV_IRON.CATEGORIES: load_intervention_categories,
        data_keys.IV_IRON.EXPOSURE: load_dichotomous_treatment_exposure,
        data_keys.IV_IRON.EXCESS_SHIFT: load_treatment_excess_shift,
        data_keys.IV_IRON.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        data_keys.MATERNAL_BMI_ANEMIA.DISTRIBUTION: load_maternal_bmi_anemia_distribution,
        data_keys.MATERNAL_BMI_ANEMIA.CATEGORIES: load_maternal_bmi_anemia_categories,
        data_keys.MATERNAL_BMI_ANEMIA.EXPOSURE: load_maternal_bmi_anemia_exposure,
        data_keys.MATERNAL_BMI_ANEMIA.EXCESS_SHIFT: load_maternal_bmi_anemia_excess_shift,
        data_keys.MATERNAL_BMI_ANEMIA.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        data_keys.SQLNS_TREATMENT.RISK_RATIOS: load_sqlns_risk_ratios,
    }

    if lookup_key in NATIONAL_LEVEL_DATA_KEYS or not fetch_subnationals:
        data = mapping[lookup_key](lookup_key, location)
    else:
        subnational_ids = fetch_subnational_ids(location)
        data = mapping[lookup_key](lookup_key, subnational_ids)

    return data


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(key: str, location: Union[str, List[int]]) -> pd.DataFrame:
    if location == "LMICs":
        world_bank_1 = filter_population(
            interface.get_population_structure("World Bank Low Income")
        )
        world_bank_2 = filter_population(
            interface.get_population_structure("World Bank Lower Middle Income")
        )
        population_structure = pd.concat([world_bank_1, world_bank_2])
    else:
        population_structure = filter_population(interface.get_population_structure(location))
    return population_structure


def filter_population(unfiltered: pd.DataFrame) -> pd.DataFrame:
    unfiltered = unfiltered.reset_index()
    filtered_pop = unfiltered[(unfiltered.age_end <= 5)]
    filtered_pop = filtered_pop.set_index(ARTIFACT_INDEX_COLUMNS)

    return filtered_pop


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    all_age_bins = interface.get_age_bins().reset_index()
    return (
        all_age_bins[all_age_bins.age_start < 5]
        .set_index(["age_start", "age_end", "age_group_name"])
        .sort_index()
    )


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    demographic_dimensions = interface.get_demographic_dimensions(location)
    is_under_five = demographic_dimensions.index.get_level_values("age_end") <= 5
    return demographic_dimensions[is_under_five]


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_entity(key)

    use_2019_data_keys = [
        data_keys.MEASLES.PREVALENCE,
        data_keys.MEASLES.INCIDENCE_RATE,
        data_keys.MEASLES.DISABILITY_WEIGHT,
        data_keys.MEASLES.EMR,
        data_keys.MEASLES.CSMR,
        data_keys.LRI.CSMR,
    ]

    neonatal_deleted_keys = [
        data_keys.DIARRHEA.INCIDENCE_RATE,
        data_keys.DIARRHEA.DISABILITY_WEIGHT,
        data_keys.MALARIA.INCIDENCE_RATE,
        data_keys.MALARIA.DISABILITY_WEIGHT,
    ]

    both_2019_and_neonatal_deleted = [
        data_keys.LRI.INCIDENCE_RATE,
        data_keys.LRI.DISABILITY_WEIGHT,
    ]

    no_age = [
        data_keys.POPULATION.CRUDE_BIRTH_RATE,
    ]

    if key in use_2019_data_keys:
        data = interface.get_measure(entity, key.measure, location, True)
        data = data.query("year_start == 2019")

    elif key in neonatal_deleted_keys:
        data = interface.get_measure(entity, key.measure, location)
        data.loc[data.reset_index()["age_start"] < metadata.NEONATAL_END_AGE, :] = 0

    elif key in both_2019_and_neonatal_deleted:
        data = interface.get_measure(entity, key.measure, location, True)
        data = data.query("year_start == 2019")
        data.loc[data.reset_index()["age_start"] < metadata.NEONATAL_END_AGE, :] = 0

    else:
        data = interface.get_measure(entity, key.measure, location)

    if key not in no_age:
        data = data.query("age_start < 5")
    # TODO: delete me
    # return data.droplevel("location")
    return data


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, "to_dict"):
        entity_metadata = entity_metadata.to_dict()
    if key == data_keys.WASTING.CATEGORIES:
        entity_metadata["cat2"] = "Wasting Between -3 SD and -2.5 SD (post-ensemble)"
        entity_metadata["cat2.5"] = "Wasting Between -2.5 SD and -2 SD (post-ensemble)"
    return entity_metadata


def load_categorical_paf(key: str, location: str) -> pd.DataFrame:
    try:
        risk = {
            data_keys.WASTING.PAF: data_keys.WASTING,
            data_keys.STUNTING.PAF: data_keys.STUNTING,
            data_keys.SAM_TREATMENT.PAF: data_keys.SAM_TREATMENT,
            data_keys.MAM_TREATMENT.PAF: data_keys.MAM_TREATMENT,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    distribution_type = get_data(risk.DISTRIBUTION, location)

    if distribution_type != "dichotomous" and "polytomous" not in distribution_type:
        raise NotImplementedError(
            f"Unrecognized distribution {distribution_type} for {risk.name}. Only dichotomous and "
            f"polytomous are recognized categorical distributions."
        )

    exp = get_data(risk.EXPOSURE, location)
    rr = get_data(risk.RELATIVE_RISK, location)

    # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
    sum_exp_x_rr = (
        (exp * rr)
        .groupby(list(set(rr.index.names) - {"parameter"}))
        .sum()
        .reset_index()
        .set_index(rr.index.names[:-1])
    )
    paf = (sum_exp_x_rr - 1) / sum_exp_x_rr

    if key == data_keys.SAM_TREATMENT.PAF or key == data_keys.MAM_TREATMENT.PAF:
        paf.loc[paf.query("age_start < .5").index] = 0

    return paf


def load_wasting_transition_rates(key: str, location: str) -> pd.DataFrame:
    """Read in wasting transition rates from flat file and expand to include all years."""
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    rates = pd.read_csv(paths.WASTING_TRANSITIONS_DATA_DIR / f"{location.lower()}.csv")
    rates = rates.rename({"parameter": "transition"}, axis=1)

    # explicitly add the youngest ages data with values of 0
    min_age = rates.reset_index()["age_start"].min()
    demography = demography.query("age_start < @min_age")
    youngest_ages_data = pd.DataFrame(
        0, columns=metadata.ARTIFACT_COLUMNS, index=demography.index
    )
    # add all transitions
    transitions = rates.reset_index()["transition"].unique()
    youngest_ages_data = expand_data(youngest_ages_data, "transition", transitions)

    youngest_ages_data = youngest_ages_data.drop(columns=["location"])
    rates["year_start"] = 2021
    rates["year_end"] = 2022
    rates = rates[youngest_ages_data.columns]
    rates = pd.concat([youngest_ages_data, rates])

    # update rate transitions into MAM to substates

    # update incidence transition names
    incidence_rates = rates.query("transition == 'inc_rate_mam'").copy()
    worse_mam_incidence_rates = incidence_rates.replace(
        {"transition": {"inc_rate_mam": "inc_rate_worse_mam"}}
    )
    rates = rates.replace({"transition": {"inc_rate_mam": "inc_rate_better_mam"}})
    rates = pd.concat([rates, worse_mam_incidence_rates])
    # update incidence transition values
    rates = rates.set_index(metadata.ARTIFACT_INDEX_COLUMNS + ["transition"]).sort_index()
    worse_mam_idx = rates.query("transition == 'inc_rate_worse_mam'").index
    better_mam_idx = rates.query("transition == 'inc_rate_better_mam'").index
    rates.loc[worse_mam_idx] = (
        rates.loc[worse_mam_idx] * data_values.WASTING.PROBABILITY_OF_CAT2
    )
    rates.loc[better_mam_idx] = rates.loc[better_mam_idx] * (
        1 - data_values.WASTING.PROBABILITY_OF_CAT2
    )

    # update remission transition names
    rates = rates.reset_index()
    remission_rates = rates.query("transition == 'ux_rem_rate_sam'").copy()
    worse_mam_remission_rates = remission_rates.replace(
        {"transition": {"ux_rem_rate_sam": "sam_to_worse_mam"}}
    )
    rates = rates.replace({"transition": {"ux_rem_rate_sam": "sam_to_better_mam"}})
    rates = pd.concat([rates, worse_mam_remission_rates])
    # update incidence transition values
    rates = rates.set_index(metadata.ARTIFACT_INDEX_COLUMNS + ["transition"]).sort_index()
    worse_mam_idx = rates.query("transition == 'sam_to_worse_mam'").index
    better_mam_idx = rates.query("transition == 'sam_to_better_mam'").index
    rates.loc[worse_mam_idx] = (
        rates.loc[worse_mam_idx] * data_values.WASTING.PROBABILITY_OF_CAT2
    )
    rates.loc[better_mam_idx] = rates.loc[better_mam_idx] * (
        1 - data_values.WASTING.PROBABILITY_OF_CAT2
    )

    return rates


def expand_data(data: pd.DataFrame, column_name: str, column_values: List) -> pd.DataFrame:
    """Equivalent to: For each column value, create a copy of data with a new column with this value. Concat these copies.
    Note: This transformation will reset the index of your data."""
    data = data.reset_index()
    if "index" in data.columns:
        data = data.drop("index", axis=1)
    new_values = pd.DataFrame({column_name: column_values}).set_index(
        pd.Index([1] * len(column_values))
    )
    data = data.set_index(pd.Index([1] * len(data))).join(new_values)
    return data


def load_wasting_birth_prevalence(key: str, location: str) -> pd.DataFrame:
    wasting_prevalence = (
        get_data(data_keys.WASTING.EXPOSURE, location)
        .query("age_end == 0.5")
        .droplevel(["age_start", "age_end"])
    )

    # read and process prevalence of low birth weight amongst infants who survive to 30 days
    lbwsg_exposure = get_data(data_keys.LBWSG.EXPOSURE, location)
    # use data from 1 to 5 month age group and sum all low birth weight category prevalences
    lbwsg_exposure = lbwsg_exposure.query(
        "parameter in @data_values.LBWSG.LOW_BIRTH_WEIGHT_CATEGORIES & age_start==0.01917808"
    )
    lbw_prevalence = lbwsg_exposure.groupby(metadata.ARTIFACT_INDEX_COLUMNS).sum()
    lbw_prevalence = lbw_prevalence.droplevel(
        ["age_start", "age_end", "year_start", "year_end"]
    )

    # calculate prevalences
    prev_cat1 = wasting_prevalence.query("parameter=='cat1'")
    prev_cat3 = wasting_prevalence.query("parameter=='cat3'")
    prev_cat4 = wasting_prevalence.query("parameter=='cat4'")
    # sum cat2 and cat2.5 for MAM
    prev_cat2 = wasting_prevalence.query("parameter=='cat2' or parameter=='cat2.5'")
    prev_cat2 = prev_cat2.groupby(["sex", "year_start", "year_end"]).sum()
    prev_cat2["parameter"] = "cat2"
    prev_cat2 = prev_cat2.set_index(["parameter"], append=True)

    # relative risk of LBW on wasting
    relative_risk_draws = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS, *data_values.LBWSG.RR_ON_WASTING
    )
    relative_risk = pd.DataFrame(
        [relative_risk_draws], columns=metadata.ARTIFACT_COLUMNS, index=lbw_prevalence.index
    )

    adequate_birth_weight_cat1_probability = prev_cat1 / (
        (relative_risk * lbw_prevalence) + (1 - lbw_prevalence)
    )
    adequate_birth_weight_cat2_probability = prev_cat2 / (
        (relative_risk * lbw_prevalence) + (1 - lbw_prevalence)
    )
    adequate_birth_weight_cat3_probability = prev_cat3 + (
        (
            prev_cat1.droplevel("parameter")
            + prev_cat2.droplevel("parameter")
            - adequate_birth_weight_cat1_probability.droplevel("parameter")
            - adequate_birth_weight_cat2_probability.droplevel("parameter")
        )
        * prev_cat3
        / (prev_cat3 + prev_cat4.droplevel("parameter"))
    )
    adequate_birth_weight_cat4_probability = prev_cat4 + (
        (
            prev_cat1.droplevel("parameter")
            + prev_cat2.droplevel("parameter")
            - adequate_birth_weight_cat1_probability.droplevel("parameter")
            - adequate_birth_weight_cat2_probability.droplevel("parameter")
        )
        * prev_cat4
        / (prev_cat3.droplevel("parameter") + prev_cat4)
    )

    low_birth_weight_cat1_probability = adequate_birth_weight_cat1_probability * relative_risk
    low_birth_weight_cat2_probability = adequate_birth_weight_cat2_probability * relative_risk
    low_birth_weight_cat3_probability = prev_cat3 + (
        (
            prev_cat1.droplevel("parameter")
            + prev_cat2.droplevel("parameter")
            - low_birth_weight_cat1_probability.droplevel("parameter")
            - low_birth_weight_cat2_probability.droplevel("parameter")
        )
        * prev_cat3
        / (prev_cat3 + prev_cat4.droplevel("parameter"))
    )
    low_birth_weight_cat4_probability = prev_cat4 + (
        (
            prev_cat1.droplevel("parameter")
            + prev_cat2.droplevel("parameter")
            - low_birth_weight_cat1_probability.droplevel("parameter")
            - low_birth_weight_cat2_probability.droplevel("parameter")
        )
        * prev_cat4
        / (prev_cat3.droplevel("parameter") + prev_cat4)
    )

    adequate_bw_prevalence = pd.concat(
        [
            adequate_birth_weight_cat1_probability,
            adequate_birth_weight_cat2_probability,
            adequate_birth_weight_cat3_probability,
            adequate_birth_weight_cat4_probability,
        ]
    )
    low_bw_prevalence = pd.concat(
        [
            low_birth_weight_cat1_probability,
            low_birth_weight_cat2_probability,
            low_birth_weight_cat3_probability,
            low_birth_weight_cat4_probability,
        ]
    )

    adequate_bw_prevalence["birth_weight_status"] = "adequate_birth_weight"
    low_bw_prevalence["birth_weight_status"] = "low_birth_weight"

    birth_prevalence = pd.concat([low_bw_prevalence, adequate_bw_prevalence])
    birth_prevalence = birth_prevalence.set_index(
        "birth_weight_status", append=True
    ).sort_index()

    # distribute probability of being initialized in MAM state
    # amongst worse MAM (cat2) and better MAM (cat2.5)
    cat2_rows = birth_prevalence.query("parameter=='cat2'").copy()
    # update cat2 rows
    birth_prevalence.loc[birth_prevalence.query("parameter=='cat2'").index] = (
        cat2_rows * data_values.WASTING.PROBABILITY_OF_CAT2
    )
    # create cat2.5 rows
    cat25_rows = cat2_rows * (1 - data_values.WASTING.PROBABILITY_OF_CAT2)
    cat25_rows = (
        cat25_rows.reset_index()
        .replace({"parameter": {"cat2": "cat2.5"}})
        .set_index(birth_prevalence.index.names)
    )

    birth_prevalence = pd.concat([birth_prevalence, cat25_rows]).sort_index()

    return birth_prevalence


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data)


def load_duration(key: str, location: str) -> pd.DataFrame:
    """Get duration by sampling 1000 draws from the provided distributions
    and convert from days to years. The duration will be the same for each
    demographic group."""
    try:
        distribution = {
            data_keys.DIARRHEA.DURATION: data_values.DIARRHEA_DURATION,
            data_keys.LRI.DURATION: data_values.LRI_DURATION,
            data_keys.MALARIA.DURATION: data_values.MALARIA_DURATION,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    duration_draws = (
        get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution)
        / metadata.YEAR_DURATION
    )

    duration = pd.DataFrame(
        [duration_draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index
    )

    if key == data_keys.LRI.DURATION:
        duration = duration.reset_index()
        duration["year_start"] = 2019
        duration["year_end"] = 2020
        duration = duration.set_index(
            ["location", "sex", "age_start", "age_end", "year_start", "year_end"]
        )
    # TODO: delete me
    # duration = duration.droplevel("location")

    return duration


def load_prevalence_from_incidence_and_duration(key: str, location: str) -> pd.DataFrame:
    try:
        cause = {
            data_keys.DIARRHEA.PREVALENCE: data_keys.DIARRHEA,
            data_keys.LRI.PREVALENCE: data_keys.LRI,
            data_keys.MALARIA.PREVALENCE: data_keys.MALARIA,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    incidence_rate = get_data(cause.INCIDENCE_RATE, location)
    duration = get_data(cause.DURATION, location)
    prevalence = incidence_rate * duration

    # get enn prevalence
    birth_prevalence = data_values.BIRTH_PREVALENCE_OF_ZERO
    enn_prevalence = prevalence.query("age_start == 0")
    enn_prevalence = (birth_prevalence + enn_prevalence) / 2
    all_other_prevalence = prevalence.query("age_start != 0.0")

    prevalence = pd.concat([enn_prevalence, all_other_prevalence]).sort_index()
    return prevalence


def load_neonatal_deleted_remission_from_duration(key: str, location: str) -> pd.DataFrame:
    """Calculate remission rate from duration and zero out neonatal age group data."""
    try:
        cause = {
            data_keys.DIARRHEA.REMISSION_RATE: data_keys.DIARRHEA,
            data_keys.LRI.REMISSION_RATE: data_keys.LRI,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")
    step_size = 4 / 365  # years
    duration = get_data(cause.DURATION, location)
    remission_rate = (-1 / step_size) * np.log(1 - step_size / duration)

    remission_rate.loc[
        remission_rate.index.get_level_values("age_start") < metadata.NEONATAL_END_AGE, :
    ] = 0
    return remission_rate


def load_neonatal_deleted_malaria_remission_from_duration(
    key: str, location: str
) -> pd.DataFrame:
    """Return 1 / duration with zero'd out neonatal age groups."""
    try:
        cause = {
            data_keys.MALARIA.REMISSION_RATE: data_keys.MALARIA,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    duration = get_data(cause.DURATION, location)
    data = 1 / duration
    data.loc[data.reset_index()["age_start"] < metadata.NEONATAL_END_AGE, :] = 0

    return data


def load_emr_from_csmr_and_prevalence(key: str, location: str) -> pd.DataFrame:
    try:
        cause = {
            data_keys.DIARRHEA.EMR: data_keys.DIARRHEA,
            data_keys.LRI.EMR: data_keys.LRI,
            data_keys.MALARIA.EMR: data_keys.MALARIA,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    csmr = get_data(cause.CSMR, location)
    prevalence = get_data(cause.PREVALENCE, location)
    data = (csmr / prevalence).fillna(0)
    data = data.replace([np.inf, -np.inf], 0)

    if key == data_keys.DIARRHEA.EMR:
        data.loc[data.index.get_level_values("age_start") < metadata.NEONATAL_END_AGE, :] = 0
    return data


def load_neonatal_deleted_csmr(key: str, location: str) -> pd.DataFrame:
    """Get GBD 2019 CSMR data with 2021 age groups and zero out neonatal age groups."""
    allowed_keys = [data_keys.DIARRHEA.CSMR, data_keys.LRI.CSMR, data_keys.MALARIA.CSMR]
    if key not in allowed_keys:
        raise ValueError(f"Unrecognized key {key}")

    data = load_standard_data(key, location)
    # data.loc[data.age_start < metadata.NEONATAL_END_AGE, :] = 0
    data.loc[data.reset_index()["age_start"] < metadata.NEONATAL_END_AGE, :] = 0
    return data


def load_post_neonatal_birth_prevalence(key: str, location: str) -> pd.DataFrame:
    """Return post neonatal data (1 month to 6 months) as birth prevalence."""
    try:
        cause = {
            data_keys.DIARRHEA.BIRTH_PREVALENCE: data_keys.DIARRHEA,
            data_keys.MALARIA.BIRTH_PREVALENCE: data_keys.MALARIA,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    prevalence = get_data(cause.PREVALENCE, location)
    is_post_neonatal = np.isclose(
        prevalence.reset_index()["age_start"], metadata.NEONATAL_END_AGE
    )
    post_neonatal_prevalence = prevalence[is_post_neonatal]
    data = post_neonatal_prevalence.droplevel(["age_start", "age_end"])

    return data


def load_underweight_exposure(key: str, location: str) -> pd.DataFrame:
    """Read in exposure distribution data (conditional on stunting
    and wasting) from file and transform. This data looks like standard
    categorical exposure distribution data but with stunting and wasting
    parameter values in the index."""
    location_id = utility_data.get_location_id(location)
    df = pd.read_csv(paths.UNDERWEIGHT_CONDITIONAL_DISTRIBUTIONS)
    df = df.query("location_id==@location_id").drop("location_id", axis=1)
    # add early neonatal data by copying late neonatal
    early_neonatal = df[df["age_group_name"] == "late_neonatal"].copy()
    early_neonatal["age_group_name"] = "early_neonatal"
    df = pd.concat([early_neonatal, df])

    # add age start and age end data instead of age group name
    age_bins = get_data(data_keys.POPULATION.AGE_BINS, location).reset_index()
    age_bins["age_group_name"] = age_bins["age_group_name"].str.lower().str.replace(" ", "_")
    age_start_map = dict(zip(age_bins["age_group_name"], age_bins["age_start"]))
    age_end_map = dict(zip(age_bins["age_group_name"], age_bins["age_end"]))
    df["age_start"] = df["age_group_name"].map(age_start_map)
    df["age_end"] = df["age_group_name"].map(age_end_map)
    df = df.drop("age_group_name", axis=1)

    df["year_start"] = 2021
    df["year_end"] = df["year_start"] + 1

    # define index
    df = df.rename({"underweight_parameter": "parameter"}, axis=1)
    df = df.set_index(
        metadata.ARTIFACT_INDEX_COLUMNS
        + ["stunting_parameter", "wasting_parameter", "parameter"]
    )

    # add wasting cat2.5 data by duplicating wasting cat2 data
    cat2_rows = df.query("wasting_parameter=='cat2'").copy()
    new_cat_rows = (
        cat2_rows.reset_index()
        .replace({"wasting_parameter": {"cat2": "cat2.5"}})
        .set_index(df.index.names)
    )
    df = pd.concat([df, new_cat_rows]).sort_index()
    index_names = df.index.names

    # create missing rows and fill with 0
    def cartesian_product(elements: Dict) -> pd.DataFrame:
        """Create DataFrame with cartesian product of dictionary values as index"""
        index = pd.MultiIndex.from_product(elements.values(), names=elements.keys())
        return pd.DataFrame(index=index).reset_index()

    age_bins = get_data(data_keys.POPULATION.AGE_BINS, location).reset_index()[
        ["age_start", "age_end"]
    ]
    index_elements = {
        "sex": ["Male", "Female"],
        "age_start": age_bins["age_start"],
        "year_start": list([2021]),
        "stunting_parameter": ["cat1", "cat2", "cat3", "cat4"],
        "wasting_parameter": ["cat1", "cat2", "cat2.5", "cat3", "cat4"],
        "parameter": ["cat1", "cat2", "cat3", "cat4"],
    }
    complete_index = cartesian_product(index_elements)
    complete_index = complete_index.merge(age_bins, on=["age_start"])
    complete_index["year_end"] = complete_index["year_start"] + 1
    df_index = df.reset_index()[
        metadata.ARTIFACT_INDEX_COLUMNS
        + ["stunting_parameter", "wasting_parameter", "parameter"]
    ]
    merge_df = complete_index.merge(df_index, how="left", indicator=True)
    empty_missing_rows = merge_df.loc[merge_df["_merge"] == "left_only"].set_index(
        index_names
    )
    missing_rows = pd.DataFrame(
        0.0, columns=metadata.ARTIFACT_COLUMNS, index=empty_missing_rows.index
    )
    df = pd.concat([df, missing_rows]).sort_index()
    return df


def load_gbd_2021_exposure(key: str, location: str) -> pd.DataFrame:
    # Get national location id to use national data probabilities
    location_id = utility_data.get_location_id(location)
    entity_key = EntityKey(key)
    entity = utilities.get_entity(entity_key)

    data = load_standard_data(key, location)

    # if entity_key == data_keys.STUNTING.EXPOSURE:
    #     # Remove neonatal exposure
    #     neonatal_age_ends = data.index.get_level_values("age_end").unique().sort_values()[:2]
    #     data.loc[data.index.get_level_values("age_end").isin(neonatal_age_ends)] = 0.0
    #     data.loc[
    #         data.index.get_level_values("age_end").isin(neonatal_age_ends)
    #         & (data.index.get_level_values("parameter") == data_keys.STUNTING.CAT4)
    #     ] = 1.0
    if entity_key == data_keys.WASTING.EXPOSURE:
        # format probabilities of entering worse MAM state
        probabilities = pd.read_csv(paths.PROBABILITIES_OF_WORSE_MAM_EXPOSURE)
        # add early neonatal rows by duplicating late neonatal data
        enn_rows = probabilities.query("age_group_id==3").copy()
        enn_rows["age_group_id"] = 2
        probabilities = pd.concat([probabilities, enn_rows])
        probabilities = probabilities.query("location_id==@location_id").drop(
            ["Unnamed: 0", "location_id"], axis=1
        )
        probabilities["sex"] = probabilities["sex"].str.capitalize()
        # get age start and end from age group ID
        age_bins = utilities.get_gbd_age_bins()
        age_bins = age_bins.drop("age_group_name", axis=1)
        probabilities = probabilities.merge(age_bins, on="age_group_id").drop(
            "age_group_id", axis=1
        )
        # add year data
        probabilities["year_start"] = 2021
        probabilities["year_end"] = probabilities["year_start"] + 1

        probabilities = (
            pd.pivot_table(
                probabilities,
                values="exposure",
                index=metadata.ARTIFACT_INDEX_COLUMNS,
                columns="draw",
            )
            .sort_index()
            .reset_index()
        )

        # distribute probability of entering MAM state amongst worse MAM (cat2) and better MAM (cat2.5)
        rows_to_keep = data.query("parameter != 'cat2'")
        cat2_rows = data.query("age_start.isin([0., 0.01917808, 0.07671233, 0.5, 1.,2.])")
        cat2_rows = cat2_rows.query("parameter=='cat2'").copy().sort_index().reset_index()
        assert probabilities[metadata.ARTIFACT_INDEX_COLUMNS].equals(
            cat2_rows[metadata.ARTIFACT_INDEX_COLUMNS]
        )

        new_cat2_rows = cat2_rows.copy()
        new_cat2_rows[metadata.ARTIFACT_COLUMNS] = (
            cat2_rows[metadata.ARTIFACT_COLUMNS] * probabilities[metadata.ARTIFACT_COLUMNS]
        )
        new_cat2_rows = new_cat2_rows.set_index(
            metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
        ).sort_index()

        cat25_rows = cat2_rows.copy()
        cat25_rows["parameter"] = "cat2.5"
        cat25_rows[metadata.ARTIFACT_COLUMNS] = cat2_rows[metadata.ARTIFACT_COLUMNS] * (
            1 - probabilities[metadata.ARTIFACT_COLUMNS]
        )
        cat25_rows = cat25_rows.set_index(
            metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"]
        ).sort_index()

        data = pd.concat([rows_to_keep, new_cat2_rows, cat25_rows]).sort_index()

    return data


def load_wasting_rr(key: str, location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = pd.read_csv(paths.WASTING_RELATIVE_RISKS)
    data = data.query("location_id==@location_id").drop(["Unnamed: 0", "location_id"], axis=1)
    data["sex"] = data["sex"].str.capitalize()

    # get age start and end from age group ID
    age_bins = utilities.get_gbd_age_bins()
    age_bins = age_bins.drop("age_group_name", axis=1)
    data = data.merge(age_bins, on="age_group_id").drop("age_group_id", axis=1)
    data["year_start"] = 2021
    data["year_end"] = data["year_start"] + 1

    data = pd.pivot_table(
        data,
        values="value",
        index=metadata.ARTIFACT_INDEX_COLUMNS
        + ["affected_entity", "affected_measure", "parameter"],
        columns="draw",
    )
    data = data[metadata.ARTIFACT_COLUMNS]

    inc = data.query('affected_measure == "incidence_rate"')
    csmr = data.query('affected_measure == "cause_specific_mortality_rate"')
    emr = csmr.droplevel("affected_measure") / inc.droplevel("affected_measure")
    emr["affected_measure"] = "excess_mortality_rate"
    emr = emr.set_index("affected_measure", append=True).reorder_levels(inc.index.names)

    data = pd.concat([inc, emr])

    # add neonatal data with relative risks of 1
    # use stunting to get neonatal data
    neonatal_data = get_data(data_keys.STUNTING.RELATIVE_RISK, location).query(
        "age_start < .05"
    )
    cat25_rows = neonatal_data.query("parameter=='cat2'").copy().reset_index("parameter")
    cat25_rows["parameter"] = "cat2.5"
    cat25_rows = cat25_rows.set_index("parameter", append=True)
    neonatal_data = pd.concat([neonatal_data, cat25_rows]).sort_index()
    data = pd.concat([data, neonatal_data]).sort_index()

    return data


def load_gbd_2021_rr(key: str, location: str) -> pd.DataFrame:
    entity_key = EntityKey(key)
    entity = utilities.get_entity(entity_key)

    raw_data = load_standard_data(key, location)

    inc = raw_data.query('affected_measure == "incidence_rate"')
    csmr = raw_data.query('affected_measure == "cause_specific_mortality_rate"')
    emr = csmr.droplevel("affected_measure") / inc.droplevel("affected_measure")
    emr["affected_measure"] = "excess_mortality_rate"
    emr = emr.set_index("affected_measure", append=True).reorder_levels(inc.index.names)

    data = pd.concat([inc, emr])

    if key == data_keys.STUNTING.RELATIVE_RISK:
        # Remove neonatal relative risks
        neonatal_age_ends = data.index.get_level_values("age_end").unique().sort_values()[:2]
        data.loc[data.index.get_level_values("age_end").isin(neonatal_age_ends)] = 1.0

    return data


def load_cgf_paf(key: str, location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = pd.read_csv(paths.CGF_PAFS).query("location_id==@location_id")

    # add age start and age end data instead of age group name
    age_bins = get_data(data_keys.POPULATION.AGE_BINS, location).reset_index()
    age_bins["age_group_name"] = age_bins["age_group_name"].str.lower().str.replace(" ", "_")
    age_start_map = dict(zip(age_bins["age_group_name"], age_bins["age_start"]))
    age_end_map = dict(zip(age_bins["age_group_name"], age_bins["age_end"]))
    data["age_start"] = data["age_group_name"].map(age_start_map)
    data["age_end"] = data["age_group_name"].map(age_end_map)
    data = data.drop(["age_group_name", "location_id"], axis=1)
    data["year_start"] = 2021
    data["year_end"] = data["year_start"] + 1

    # Capitalize Sex
    data["sex"] = data["sex"].str.capitalize()

    # define index
    data = data.set_index(
        metadata.ARTIFACT_INDEX_COLUMNS + ["affected_entity", "affected_measure"]
    )
    data = data[metadata.ARTIFACT_COLUMNS]
    return data.sort_index()


def load_pem_disability_weight(key: str, location: str) -> pd.DataFrame:
    try:
        pem_sequelae = {
            data_keys.MODERATE_PEM.DISABILITY_WEIGHT: [
                sequelae.moderate_wasting_with_edema,
                sequelae.moderate_wasting_without_edema,
            ],
            data_keys.SEVERE_PEM.DISABILITY_WEIGHT: [
                sequelae.severe_wasting_with_edema,
                sequelae.severe_wasting_without_edema,
            ],
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    prevalence_disability_weight = []
    state_prevalence = []
    for s in pem_sequelae:
        sequela_prevalence = interface.get_measure(s, "prevalence", location)
        sequela_disability_weight = interface.get_measure(s, "disability_weight", location)

        prevalence_disability_weight += [sequela_prevalence * sequela_disability_weight]
        state_prevalence += [sequela_prevalence]
    # TODO: delete me
    # disability_weight = (
    #     (sum(prevalence_disability_weight) / sum(state_prevalence))
    #     .fillna(0)
    #     .droplevel("location")
    # )
    disability_weight = (sum(prevalence_disability_weight) / sum(state_prevalence)).fillna(0)
    return disability_weight


def load_pem_emr(key: str, location: str) -> pd.DataFrame:
    emr = load_standard_data(data_keys.PEM.EMR, location)
    return emr


def load_pem_csmr(key: str, location: str) -> pd.DataFrame:
    csmr = load_standard_data(data_keys.PEM.CSMR, location)
    return csmr


def load_pem_restrictions(key: str, location: str) -> pd.DataFrame:
    metadata = load_metadata(data_keys.PEM.RESTRICTIONS, location)
    return metadata


#####################
# MAM/SAM Treatment #
#####################


# noinspection PyUnusedLocal
def load_wasting_treatment_distribution(key: str, location: str) -> str:
    if key in [data_keys.SAM_TREATMENT.DISTRIBUTION, data_keys.MAM_TREATMENT.DISTRIBUTION]:
        return data_values.WASTING.DISTRIBUTION
    else:
        raise ValueError(f"Unrecognized key {key}")


# noinspection PyUnusedLocal
def load_wasting_treatment_categories(key: str, location: str) -> str:
    if key in [data_keys.SAM_TREATMENT.CATEGORIES, data_keys.MAM_TREATMENT.CATEGORIES]:
        return data_values.WASTING.CATEGORIES
    else:
        raise ValueError(f"Unrecognized key {key}")


def load_wasting_treatment_exposure(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.SAM_TREATMENT.EXPOSURE:
        parameter = "c_sam"
    elif key == data_keys.MAM_TREATMENT.EXPOSURE:
        parameter = "c_mam"
    else:
        raise ValueError(f"Unrecognized key {key}")

    treatment_coverage = utilities.get_wasting_treatment_parameter_data(parameter, location)

    idx = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    cat3 = pd.DataFrame({f"draw_{i}": 0.0 for i in range(0, metadata.DRAW_COUNT)}, index=idx)
    cat2 = (
        pd.DataFrame({f"draw_{i}": 1.0 for i in range(0, metadata.DRAW_COUNT)}, index=idx)
        * treatment_coverage
    )
    cat1 = 1 - cat2

    cat1["parameter"] = "cat1"
    cat2["parameter"] = "cat2"
    cat3["parameter"] = "cat3"

    exposure = pd.concat([cat1, cat2, cat3]).set_index("parameter", append=True).sort_index()

    # infants under 6 months of age should not receive treatment
    under_6_months_unexposed_idx = exposure.query("age_start < .5 & parameter=='cat1'").index
    under_6_months_exposed_idx = exposure.query("age_start < .5 & parameter!='cat1'").index
    exposure.loc[under_6_months_unexposed_idx] = 1
    exposure.loc[under_6_months_exposed_idx] = 0
    # TODO: delete me
    # return exposure.droplevel("location")
    return exposure


def load_sam_treatment_rr(key: str, location: str) -> pd.DataFrame:
    # tmrel is defined as baseline treatment (cat_2)
    if key != data_keys.SAM_TREATMENT.RELATIVE_RISK:
        raise ValueError(f"Unrecognized key {key}")

    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location).reset_index()
    sam_tx_efficacy, sam_tx_efficacy_tmrel = utilities.get_treatment_efficacy(
        demography, data_keys.WASTING.CAT1, location
    )

    # rr_t1 = t1 / t1_tmrel
    #       = (sam_tx_efficacy / sam_tx_duration) / (sam_tx_efficacy_tmrel / sam_tx_duration)
    #       = sam_tx_efficacy / sam_tx_efficacy_tmrel
    rr_sam_treated_remission = sam_tx_efficacy / sam_tx_efficacy_tmrel
    rr_sam_treated_remission[
        "affected_entity"
    ] = "severe_acute_malnutrition_to_mild_child_wasting"

    # rr_r2 = r2 / r2_tmrel
    #       = (1 - sam_tx_efficacy) * (r2_ux) / (1 - sam_tx_efficacy_tmrel) * (r2_ux)
    #       = (1 - sam_tx_efficacy) / (1 - sam_tx_efficacy_tmrel)
    rr_sam_untreated_remission = (1 - sam_tx_efficacy) / (1 - sam_tx_efficacy_tmrel)

    better_mam_rows = rr_sam_untreated_remission.copy()
    worse_mam_rows = rr_sam_untreated_remission.copy()
    better_mam_rows[
        "affected_entity"
    ] = "severe_acute_malnutrition_to_better_moderate_acute_malnutrition"
    worse_mam_rows[
        "affected_entity"
    ] = "severe_acute_malnutrition_to_worse_moderate_acute_malnutrition"
    rr_sam_untreated_remission = pd.concat([better_mam_rows, worse_mam_rows])

    rr = pd.concat([rr_sam_treated_remission, rr_sam_untreated_remission])

    rr["affected_measure"] = "transition_rate"
    rr = rr.set_index(["affected_entity", "affected_measure"], append=True)
    rr.index = rr.index.reorder_levels(
        [col for col in rr.index.names if col != "parameter"] + ["parameter"]
    )

    # no effect for simulants younger than 6 months
    rr.loc[rr.query("age_start < .5").index] = 1
    # TODO: delete me
    # return rr.droplevel("location").sort_index()
    return rr.sort_index()


def load_mam_treatment_rr(key: str, location: str) -> pd.DataFrame:
    # tmrel is defined as baseline treatment (cat_2)
    if key != data_keys.MAM_TREATMENT.RELATIVE_RISK:
        raise ValueError(f"Unrecognized key {key}")

    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location).reset_index()
    mam_tx_efficacy, mam_tx_efficacy_tmrel = utilities.get_treatment_efficacy(
        demography, data_keys.WASTING.CAT2, location
    )
    index = mam_tx_efficacy.index

    mam_ux_duration = data_values.WASTING.MAM_UX_RECOVERY_TIME_OVER_6MO
    mam_tx_duration = pd.Series(index=index)
    mam_tx_duration[
        index.get_level_values("age_start") < 0.5
    ] = data_values.WASTING.MAM_TX_RECOVERY_TIME_UNDER_6MO
    mam_tx_duration[0.5 <= index.get_level_values("age_start")] = get_random_variable_draws(
        mam_tx_duration[0.5 <= index.get_level_values("age_start")].index,
        *data_values.WASTING.MAM_TX_RECOVERY_TIME_OVER_6MO,
    )
    mam_tx_duration = pd.DataFrame(
        {f"draw_{i}": 1 for i in range(0, metadata.DRAW_COUNT)}, index=index
    ).multiply(mam_tx_duration, axis="index")

    # rr_r3 = r3 / r3_tmrel
    #       = (mam_tx_efficacy / mam_tx_duration) + (1 - mam_tx_efficacy / mam_ux_duration)
    #           / (mam_tx_efficacy_tmrel / mam_tx_duration) + (1 - mam_tx_efficacy_tmrel / mam_ux_duration)
    #       = (mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration)
    #           / (mam_tx_efficacy_tmrel * mam_ux_duration + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration)
    rr = (mam_tx_efficacy * mam_ux_duration + (1 - mam_tx_efficacy) * mam_tx_duration) / (
        mam_tx_efficacy_tmrel * mam_ux_duration
        + (1 - mam_tx_efficacy_tmrel) * mam_tx_duration
    )

    better_mam_rows = rr.copy()
    worse_mam_rows = rr.copy()
    better_mam_rows[
        "affected_entity"
    ] = "better_moderate_acute_malnutrition_to_mild_child_wasting"
    worse_mam_rows[
        "affected_entity"
    ] = "worse_moderate_acute_malnutrition_to_mild_child_wasting"
    rr = pd.concat([better_mam_rows, worse_mam_rows])

    rr["affected_measure"] = "transition_rate"
    rr = rr.set_index(["affected_entity", "affected_measure"], append=True)
    rr.index = rr.index.reorder_levels(
        [col for col in rr.index.names if col != "parameter"] + ["parameter"]
    )

    # no effect for simulants younger than 6 months
    rr.loc[rr.query("age_start < .5").index] = 1
    # TODO: delete me
    # return rr.droplevel("location").sort_index()
    return rr.sort_index()


def load_lbwsg_exposure(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.EXPOSURE:
        raise ValueError(f"Unrecognized key {key}")

    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
    data = utilities.load_lbwsg_exposure(location)
    # This category was a mistake in GBD 2019, so drop.
    extra_residual_category = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
    data = data.loc[data["parameter"] != extra_residual_category]
    idx_cols = ["location_id", "age_group_id", "year_id", "sex_id", "parameter"]
    data = data.set_index(idx_cols)[vi_globals.DRAW_COLUMNS]

    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    data = data.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    # normalize so all categories sum to 1
    total_exposure = data.groupby(["location_id", "age_group_id", "sex_id"]).transform("sum")
    data = (data / total_exposure).reset_index()
    data = reshape_to_vivarium_format(data, location)
    return data


def load_lbwsg_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK:
        raise ValueError(f"Unrecognized key {key}")

    data = load_standard_data(key, location)
    data = data.query("year_start == 2021").droplevel(["affected_entity", "affected_measure"])
    data = data[~data.index.duplicated()]
    return data


def load_lbwsg_interpolated_rr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR:
        raise ValueError(f"Unrecognized key {key}")

    rr = get_data(data_keys.LBWSG.RELATIVE_RISK, location).reset_index()
    rr["parameter"] = pd.Categorical(
        rr["parameter"], [f"cat{i}" for i in range(metadata.DRAW_COUNT)]
    )
    rr = (
        rr.sort_values("parameter")
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"])
        .stack()
        .unstack("parameter")
        .apply(np.log)
    )

    # get category midpoints
    def get_category_midpoints(lbwsg_type: str) -> pd.Series:
        categories = get_data(f"risk_factor.{data_keys.LBWSG.name}.categories", location)
        return utilities.get_intervals_from_categories(lbwsg_type, categories).apply(
            lambda x: x.mid
        )

    gestational_age_midpoints = get_category_midpoints("short_gestation")
    birth_weight_midpoints = get_category_midpoints("low_birth_weight")

    # build grid of gestational age and birth weight
    def get_grid(midpoints: pd.Series, endpoints: Tuple[float, float]) -> np.array:
        grid = np.append(np.unique(midpoints), endpoints)
        grid.sort()
        return grid

    gestational_age_grid = get_grid(gestational_age_midpoints, (0.0, 42.0))
    birth_weight_grid = get_grid(birth_weight_midpoints, (0.0, 4500.0))

    def make_interpolator(log_rr_for_age_sex_draw: pd.Series) -> RectBivariateSpline:
        # Use scipy.interpolate.griddata to extrapolate to grid using nearest neighbor interpolation
        log_rr_grid_nearest = griddata(
            (gestational_age_midpoints, birth_weight_midpoints),
            log_rr_for_age_sex_draw,
            (gestational_age_grid[:, None], birth_weight_grid[None, :]),
            method="nearest",
            rescale=True,
        )
        # return a RectBivariateSpline object from the extrapolated values on grid
        return RectBivariateSpline(
            gestational_age_grid, birth_weight_grid, log_rr_grid_nearest, kx=1, ky=1
        )

    log_rr_interpolator = (
        rr.apply(make_interpolator, axis="columns")
        .apply(lambda x: pickle.dumps(x).hex())
        .unstack()
    )
    return log_rr_interpolator


def load_lbwsg_paf(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.PAF:
        raise ValueError(f"Unrecognized key {key}")

    location_mapper = {
        "Sub-Saharan Africa": "sub-saharan_africa",
        "South Asia": "south_asia",
        "LMICs": "lmics",
        "Ethiopia": "ethiopia",
        "Nigeria": "nigeria",
        "India": "india",
        "Pakistan": "pakistan",
    }

    output_dir = paths.TEMPORARY_PAF_DIR  # / location_mapper[location]

    def get_age_and_sex(measure_str):
        age = measure_str.split("AGE_GROUP_")[1].split("SEX")[0][:-1]
        sex = measure_str.split("AGE_GROUP_")[1].split("SEX")[1][1:]

        return age + "," + sex

    df = pd.read_hdf(output_dir / "output.hdf")  # this is 4096_simulants.hdf for example
    df = df[[col for col in df.columns if "MEASURE" in col]].T
    df.columns = [f"draw_{i}" for i in range(metadata.DRAW_COUNT)]
    df = df.reset_index()
    df["demographics"] = df["index"].apply(get_age_and_sex)
    df = df.drop("index", axis=1)
    df[["age", "sex"]] = df["demographics"].str.split(",", expand=True)
    df = df.drop("demographics", axis=1)

    age_start_dict = {"early_neonatal": 0.0, "late_neonatal": 0.01917808}
    age_end_dict = {"early_neonatal": 0.01917808, "late_neonatal": 0.07671233}
    df["age_start"] = df["age"].replace(age_start_dict)
    df["age_end"] = df["age"].replace(age_end_dict)
    df["year_start"] = 2021
    df["year_end"] = 2022
    df = df.drop("age", axis=1)

    new_row_1 = [0] * metadata.DRAW_COUNT + ["Female", 0.07671233, 1.0, 2021, 2022]
    new_row_2 = [0] * metadata.DRAW_COUNT + ["Male", 0.07671233, 1.0, 2021, 2022]
    new_row_3 = [0] * metadata.DRAW_COUNT + ["Female", 1.0, 5.0, 2021, 2022]
    new_row_4 = [0] * metadata.DRAW_COUNT + ["Male", 1.0, 5.0, 2021, 2022]

    df.loc[len(df)] = new_row_1
    df.loc[len(df)] = new_row_2
    df.loc[len(df)] = new_row_3
    df.loc[len(df)] = new_row_4

    df = df.set_index(["sex", "age_start", "age_end", "year_start", "year_end"]).sort_index()
    return df


def load_sids_csmr(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.AFFECTED_UNMODELED_CAUSES.SIDS_CSMR:
        key = EntityKey(key)
        entity: Cause = utilities.get_entity(key)

        # get around the validation rejecting yll only causes
        entity.restrictions.yll_only = False
        entity.restrictions.yld_age_group_id_start = metadata.AGE_GROUP.LATE_NEONATAL_ID
        entity.restrictions.yld_age_group_id_end = metadata.AGE_GROUP.LATE_NEONATAL_ID
        # TODO: delete me
        # data = interface.get_measure(entity, key.measure, location).droplevel("location")
        data = interface.get_measure(entity, key.measure, location)
        return data
    else:
        raise ValueError(f"Unrecognized key {key}")


def load_neonatal_lri_csmr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_LRI_CSMR:
        raise ValueError(f"Unrecognized key {key}")

    data = load_standard_data(data_keys.LRI.CSMR, location)
    data.loc[data.index.get_level_values("age_start") >= metadata.NEONATAL_END_AGE, :] = 0
    return data


def load_neonatal_diarrhea_csmr(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.AFFECTED_UNMODELED_CAUSES.NEONATAL_DIARRHEAL_DISEASES_CSMR:
        raise ValueError(f"Unrecognized key {key}")

    data = load_standard_data(data_keys.DIARRHEA.CSMR, location)
    data.loc[data.index.get_level_values("age_start") >= metadata.NEONATAL_END_AGE, :] = 0
    return data


def load_intervention_distribution(key: str, location: str) -> str:
    try:
        return {
            data_keys.IFA_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_CHARACTERISTICS.DISTRIBUTION,
            data_keys.MMN_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_CHARACTERISTICS.DISTRIBUTION,
            data_keys.BEP_SUPPLEMENTATION.DISTRIBUTION: data_values.MATERNAL_CHARACTERISTICS.DISTRIBUTION,
            data_keys.IV_IRON.DISTRIBUTION: data_values.MATERNAL_CHARACTERISTICS.DISTRIBUTION,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")


def load_intervention_categories(key: str, location: str) -> str:
    try:
        return {
            data_keys.IFA_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_CHARACTERISTICS.CATEGORIES,
            data_keys.MMN_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_CHARACTERISTICS.CATEGORIES,
            data_keys.BEP_SUPPLEMENTATION.CATEGORIES: data_values.MATERNAL_CHARACTERISTICS.CATEGORIES,
            data_keys.IV_IRON.CATEGORIES: data_values.MATERNAL_CHARACTERISTICS.CATEGORIES,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")


def load_dichotomous_treatment_exposure(key: str, location: str, **kwargs) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.IFA_SUPPLEMENTATION.EXPOSURE: load_baseline_ifa_supplementation_coverage(
                location
            ),
            data_keys.MMN_SUPPLEMENTATION.EXPOSURE: data_values.MATERNAL_CHARACTERISTICS.BASELINE_MMN_COVERAGE,
            data_keys.BEP_SUPPLEMENTATION.EXPOSURE: data_values.MATERNAL_CHARACTERISTICS.BASELINE_BEP_COVERAGE,
            data_keys.IV_IRON.EXPOSURE: data_values.MATERNAL_CHARACTERISTICS.BASELINE_IV_IRON_COVERAGE,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")
    return load_dichotomous_exposure(location, distribution_data, is_risk=False, **kwargs)


def load_ifa_excess_shift(key: str, location: str) -> pd.DataFrame:
    birth_weight_shift = load_treatment_excess_shift(key, location)
    gestational_age_shift = load_excess_gestational_age_shift(key, location)
    return pd.concat([birth_weight_shift, gestational_age_shift])


def load_treatment_excess_shift(key: str, location: str) -> pd.DataFrame:
    try:
        distribution_data = {
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: data_values.MATERNAL_CHARACTERISTICS.IFA_BIRTH_WEIGHT_SHIFT,
            data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT: data_values.MATERNAL_CHARACTERISTICS.MMN_BIRTH_WEIGHT_SHIFT,
            data_keys.IV_IRON.EXCESS_SHIFT: data_values.MATERNAL_CHARACTERISTICS.IV_IRON_BIRTH_WEIGHT_SHIFT,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")
    return load_dichotomous_excess_shift(location, distribution_data)


def load_bep_excess_shift(key: str, location: str) -> pd.DataFrame:
    undernourished_distribution = (
        data_values.MATERNAL_CHARACTERISTICS.BEP_UNDERNOURISHED_BIRTH_WEIGHT_SHIFT
    )
    adequately_nourished_distribution = (
        data_values.MATERNAL_CHARACTERISTICS.BEP_ADEQUATELY_NOURISHED_BIRTH_WEIGHT_SHIFT
    )

    undernourished_shift = load_dichotomous_excess_shift(
        location, undernourished_distribution
    )
    adequately_nourished_shift = load_dichotomous_excess_shift(
        location, adequately_nourished_distribution
    )

    cat1_shift = undernourished_shift.copy()
    cat2_shift = adequately_nourished_shift.copy()
    cat3_shift = undernourished_shift.copy()
    cat4_shift = adequately_nourished_shift.copy()

    cat1_shift["maternal_bmi_anemia_exposure"] = "cat1"
    cat2_shift["maternal_bmi_anemia_exposure"] = "cat2"
    cat3_shift["maternal_bmi_anemia_exposure"] = "cat3"
    cat4_shift["maternal_bmi_anemia_exposure"] = "cat4"

    shift = pd.concat([cat1_shift, cat2_shift, cat3_shift, cat4_shift])
    shift = shift.set_index("maternal_bmi_anemia_exposure", append=True)

    return shift.sort_index()


def load_dichotomous_exposure(
    location: str,
    distribution_data: Union[float, pd.DataFrame],
    is_risk: bool,
) -> pd.DataFrame:
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    if type(distribution_data) == float:
        base_exposure = pd.Series(distribution_data, index=index)
        exposed = pd.DataFrame(
            {f"draw_{i}": base_exposure for i in range(metadata.DRAW_COUNT)}
        )
    else:
        exposed = distribution_data

    unexposed = 1 - exposed
    exposed["parameter"] = "cat1" if is_risk else "cat2"
    unexposed["parameter"] = "cat2" if is_risk else "cat1"

    exposure = (
        pd.concat([exposed, unexposed]).set_index("parameter", append=True).sort_index()
    )
    # TODO: delete me
    # return exposure.droplevel("location")
    return exposure


def load_dichotomous_excess_shift(
    location: str,
    distribution_data: Tuple,
) -> pd.DataFrame:
    """Load excess birth weight exposure shifts using distribution data."""
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    shift = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution_data)
    excess_shift = reshape_shift_data(shift, index, data_keys.LBWSG.BIRTH_WEIGHT_EXPOSURE)
    # TODO: delete me
    # return excess_shift.droplevel("location")
    return excess_shift


def load_excess_gestational_age_shift(key: str, location: str) -> pd.DataFrame:
    """Load excess gestational age shift data from IFA and MMS from file.
    Returns the sum of the shift data in the directories defined in data_dirs."""
    try:
        data_dirs = {
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: [paths.IFA_GA_SHIFT_DATA_DIR],
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1: [
                paths.MMS_GA_SHIFT_1_DATA_DIR
            ],
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2: [
                paths.MMS_GA_SHIFT_1_DATA_DIR,
                paths.MMS_GA_SHIFT_2_DATA_DIR,
            ],
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    all_shift_data = [
        pd.read_csv(data_dir / f"{location.lower()}.csv") for data_dir in data_dirs
    ]
    shifts = [
        pd.Series(shift_data["value"].values, index=shift_data["draw"])
        for shift_data in all_shift_data
    ]
    if len(shifts) > 1:
        shifts[1] = shifts[1].loc[shifts[1].notnull()]
    summed_shifts = sum(shifts)  # only sum more than one Series for subpop 2

    excess_shift = reshape_shift_data(
        summed_shifts, index, data_keys.LBWSG.GESTATIONAL_AGE_EXPOSURE
    )
    # TODO: delete me
    # return excess_shift.droplevel("location")
    return excess_shift


def reshape_shift_data(
    shift: pd.Series, index: pd.Index, target: TargetString
) -> pd.DataFrame:
    """Read in draw-level shift values and return a DataFrame where the data are the shift values,
    and the index is the passed index appended with affected entity/measure and parameter data.
    """
    exposed = pd.DataFrame([shift], index=index)
    exposed["parameter"] = "cat2"
    unexposed = pd.DataFrame([pd.Series(0.0, index=metadata.ARTIFACT_COLUMNS)], index=index)
    unexposed["parameter"] = "cat1"

    excess_shift = pd.concat([exposed, unexposed])
    excess_shift["affected_entity"] = target.name
    excess_shift["affected_measure"] = target.measure

    excess_shift = excess_shift.set_index(
        ["affected_entity", "affected_measure", "parameter"], append=True
    ).sort_index()
    return excess_shift


def load_risk_specific_shift(key: str, location: str) -> pd.DataFrame:
    try:
        key_group: data_keys.__AdditiveRisk = {
            data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.IFA_SUPPLEMENTATION,
            data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.MMN_SUPPLEMENTATION,
            data_keys.BEP_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.BEP_SUPPLEMENTATION,
            data_keys.IV_IRON.RISK_SPECIFIC_SHIFT: data_keys.IV_IRON,
            data_keys.MATERNAL_BMI_ANEMIA.RISK_SPECIFIC_SHIFT: data_keys.MATERNAL_BMI_ANEMIA,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    # p_exposed * exposed_shift
    exposure = get_data(key_group.EXPOSURE, location)
    excess_shift = get_data(key_group.EXCESS_SHIFT, location)

    risk_specific_shift = (
        (exposure * excess_shift)
        .groupby(metadata.ARTIFACT_INDEX_COLUMNS + ["affected_entity", "affected_measure"])
        .sum()
    )
    return risk_specific_shift


def load_baseline_ifa_supplementation_coverage(location: str) -> pd.DataFrame:
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    location_id = utility_data.get_location_id(location)
    data = pd.read_csv(paths.BASELINE_IFA_COVERAGE_CSV).drop("Unnamed: 0", axis=1)
    data = (
        data.query("location_id==@location_id")
        .drop("location_id", axis=1)
        .reset_index(drop=True)
    )

    draw_values = pd.pivot_table(data, values="value", columns="draw")
    coverage = pd.DataFrame(
        np.repeat(draw_values.values, len(index), axis=0), columns=draw_values.columns
    )
    coverage.index = index

    return coverage


def load_maternal_bmi_anemia_distribution(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_BMI_ANEMIA.DISTRIBUTION:
        raise ValueError(f"Unrecognized key {key}")
    return "ordered_polytomous"


def load_maternal_bmi_anemia_categories(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_BMI_ANEMIA.CATEGORIES:
        raise ValueError(f"Unrecognized key {key}")
    return {
        "cat4": "Pre-pregnancy/first trimester BMI exposure >= 18.5 and Early pregnancy "
        "“untreated” hemoglobin exposure >= 10g/dL",
        "cat3": "Pre-pregnancy/first trimester BMI exposure < 18.5 and Early pregnancy "
        "“untreated” hemoglobin exposure >= 10g/dL",
        "cat2": "Pre-pregnancy/first trimester BMI exposure >= 18.5 and Early pregnancy "
        "“untreated” hemoglobin exposure < 10g/dL",
        "cat1": "Pre-pregnancy/first trimester BMI exposure < 18.5 and Early pregnancy "
        "“untreated” hemoglobin exposure < 10g/dL",
    }


def load_maternal_bmi_anemia_exposure(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_BMI_ANEMIA.EXPOSURE:
        raise ValueError(f"Unrecognized key {key}")

    location_id = utility_data.get_location_id(location)
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index

    def _read_hgb_data(filename: str) -> pd.Series:
        raw_data = pd.read_csv(paths.RAW_DATA_DIR / filename)
        data = (
            raw_data.loc[raw_data["location_id"] == location_id, ["draw", "value"]]
            .set_index("draw")
            .squeeze()
        )
        data.index.name = None
        return data

    p_low_hgb = _read_hgb_data("pregnant_proportion_with_hgb_below_100.csv")
    p_low_bmi_given_low_hgb = _read_hgb_data(
        "prevalence_of_low_bmi_given_hemoglobin_below_10_age_weighted.csv"
    )
    p_low_bmi_given_high_hgb = _read_hgb_data(
        "prevalence_of_low_bmi_given_hemoglobin_above_10_age_weighted.csv"
    )

    cat4_exposure = pd.DataFrame(
        [(1 - p_low_hgb) * (1 - p_low_bmi_given_high_hgb)], index=index
    )
    cat4_exposure["parameter"] = "cat4"

    cat3_exposure = pd.DataFrame([(1 - p_low_hgb) * p_low_bmi_given_high_hgb], index=index)
    cat3_exposure["parameter"] = "cat3"

    cat2_exposure = pd.DataFrame([p_low_hgb * (1 - p_low_bmi_given_low_hgb)], index=index)
    cat2_exposure["parameter"] = "cat2"

    cat1_exposure = pd.DataFrame([p_low_hgb * p_low_bmi_given_low_hgb], index=index)
    cat1_exposure["parameter"] = "cat1"

    exposure = pd.concat([cat4_exposure, cat3_exposure, cat2_exposure, cat1_exposure])

    exposure = exposure.set_index(["parameter"], append=True).sort_index()
    # TODO: delete me
    # return exposure.droplevel("location")
    return exposure


def load_maternal_bmi_anemia_excess_shift(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.MATERNAL_BMI_ANEMIA.EXCESS_SHIFT:
        raise ValueError(f"Unrecognized key {key}")

    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    cat3_draws = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS,
        *data_values.MATERNAL_CHARACTERISTICS.BMI_ANEMIA_CAT3_BIRTH_WEIGHT_SHIFT,
    )
    cat2_draws = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS,
        *data_values.MATERNAL_CHARACTERISTICS.BMI_ANEMIA_CAT2_BIRTH_WEIGHT_SHIFT,
    )
    cat1_draws = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS,
        *data_values.MATERNAL_CHARACTERISTICS.BMI_ANEMIA_CAT1_BIRTH_WEIGHT_SHIFT,
    )

    cat4_shift = pd.DataFrame(0.0, columns=metadata.ARTIFACT_COLUMNS, index=index)
    cat4_shift["parameter"] = "cat4"

    cat3_shift = pd.DataFrame([cat3_draws], index=index)
    cat3_shift["parameter"] = "cat3"

    cat2_shift = pd.DataFrame([cat2_draws], index=index)
    cat2_shift["parameter"] = "cat2"

    cat1_shift = pd.DataFrame([cat1_draws], index=index)
    cat1_shift["parameter"] = "cat1"

    excess_shift = pd.concat([cat4_shift, cat3_shift, cat2_shift, cat1_shift])
    excess_shift["affected_entity"] = data_keys.LBWSG.BIRTH_WEIGHT_EXPOSURE.name
    excess_shift["affected_measure"] = data_keys.LBWSG.BIRTH_WEIGHT_EXPOSURE.measure

    excess_shift = excess_shift.set_index(
        ["affected_entity", "affected_measure", "parameter"], append=True
    ).sort_index()
    # TODO: delete me
    # return excess_shift.droplevel("location")
    return excess_shift


def load_sqlns_risk_ratios(key: str, location: str) -> pd.DataFrame:
    """Load effects of SQ-LNS treatment on wasting incidence and stunting prevalence ratios."""
    if key != data_keys.SQLNS_TREATMENT.RISK_RATIOS:
        raise ValueError(f"Unrecognized key {key}")

    # generate draws using distribution parameters for each row
    risk_ratios = pd.read_csv(paths.SQLNS_RISK_RATIOS)
    distributions = get_lognorm_from_quantiles(
        risk_ratios["median"], risk_ratios["lower"], risk_ratios["upper"]
    )
    draws = get_random_variable_draws(
        metadata.ARTIFACT_COLUMNS, "sqlns_risk_ratios", distributions
    )

    # reshape
    index_cols = ["age_start", "age_end", "affected_outcome"]
    draw_columns = pd.DataFrame(draw for draw in draws).T
    draw_columns.columns = metadata.ARTIFACT_COLUMNS
    data = pd.concat([risk_ratios[index_cols], draw_columns], axis=1).set_index(index_cols)

    return data


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column="age", split_column_prefix="age")
    df = vi_utils.split_interval(df, interval_column="year", split_column_prefix="year")
    df = vi_utils.sort_hierarchical_data(df)
    # TODO: delete me
    # df.index = df.index.droplevel("location")
    return df


def fetch_subnational_ids(location: str) -> List[int]:
    location_id = utility_data.get_location_id(location)
    location_metadata = gbd.get_location_path_to_global()
    subnational_location_metadata = location_metadata.loc[
        (location_metadata["path_to_top_parent"].apply(lambda x: str(location_id) in x))
        & (location_metadata["location_id"] != location_id)
    ]
    subnational_location_ids = subnational_location_metadata["location_id"].tolist()
    return subnational_location_ids


def get_national_location_id(location_id: int) -> int:
    location_metadata = gbd.get_location_metadata(location_id)
    path_to_parent = location_metadata.loc[location_metadata.location_id == location_id][
        "path_to_top_parent"
    ].to_list()
    national_location_id = int([loc_id.split(",")[3] for loc_id in path_to_parent][0])
    return national_location_id
