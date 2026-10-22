from typing import Tuple

import pandas as pd
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_gbd_access.gbd.measures import get_birth_exposure
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_nutrition_optimization_child.constants import data_keys, metadata
from vivarium_gates_nutrition_optimization_child.data import utilities

# Anemia estimates for the pregnant population exist only under release 33, not the
# GBD 2023 compare release the other pulls use.
ANEMIA_RELEASE_ID = 33
MODERATE_SEVERE_ANEMIA_REI_ID = 432
ALL_CAUSES_ID = 294
PREGNANT_POPULATION_GROUP_ID = 16
# 15-19 through 50-54 years; under 15 the REI 432 threshold is 110 g/L, not 100 g/L.
PREGNANCY_AGE_GROUP_IDS = list(range(8, 16))


# The release, year and ages are parameters so that changing one changes the cache key.
@vi_utils.cache
def get_pregnant_moderate_severe_anemia_prevalence(
    location: str,
    release_id: int = ANEMIA_RELEASE_ID,
    year_id: int = metadata.GBD_EXTRACT_YEAR,
    age_group_ids: Tuple[int, ...] = tuple(PREGNANCY_AGE_GROUP_IDS),
) -> pd.DataFrame:
    """Pull the prevalence of hemoglobin below 100 g/L among pregnant women by age.

    Parameters
    ----------
    location
        The location to pull.
    release_id
        The GBD release to pull from.
    year_id
        The year to pull.
    age_group_ids
        The age groups to pull.

    Returns
    -------
        One row per age group, with draws of the prevalence and the size of the pregnant
        population in a ``population`` column.
    """
    # Called directly because the vivarium_gbd_access wrapper drops the population column.
    from ihme_cc_get_estimates import get_machinery_estimates

    location_id = utility_data.resolve_location(location)
    data = get_machinery_estimates(
        entity="impairment-cause",
        entity_id=[(MODERATE_SEVERE_ANEMIA_REI_ID, ALL_CAUSES_ID)],
        release_id=release_id,
        estimates="draws",
        measure_id=vi_globals.MEASURES["Prevalence"],
        metric_id=vi_globals.METRICS["Rate"],
        location_id=location_id,
        sex_id=gbd_constants.SEX.FEMALE,
        population_group_id=PREGNANT_POPULATION_GROUP_ID,
        age_group_id=list(age_group_ids),
        year_id=year_id,
    )
    return data


@vi_utils.cache
def load_2021_lbwsg_birth_exposure(location: str) -> pd.DataFrame:
    """Pull the GBD 2021 LBWSG birth exposure draws for a location."""
    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
    location_id = utility_data.resolve_location(location)
    data = get_birth_exposure(
        entity.gbd_id,
        location_id,
        2022,
        "draws",
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
    )
    return data
