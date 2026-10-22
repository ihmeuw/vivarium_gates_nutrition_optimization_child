from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################

PROJECT_NAME = "vivarium_gates_nutrition_optimization_child"
CLUSTER_PROJECT = "proj_simscience"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

YEAR_DURATION: float = 365.25

LOCATIONS = [
    "Sub-Saharan Africa",
    "South Asia",
    "LMICs",
    "Ethiopia",
    "India",
    "Nigeria",
    "Pakistan",
]

# The LBWSG PAF workflow's phases, in the order they must run, and the
# environment each needs. The phases alternate between two environments whose
# dependencies conflict, so each runs as its own invocation of
# data/lbwsg_paf/code/run_paf_sim_step.py, driven by run_paf_sim.py. A new phase
# also needs a branch in run_paf_sim_step.py's _run_one_location.
PAF_PHASES: tuple[tuple[str, str], ...] = (
    ("initial-artifact", "artifact"),
    ("enn-paf", "simulation"),
    ("enn-artifact", "artifact"),
    ("lnn-paf", "simulation"),
    ("final-artifact", "artifact"),
)
PAF_PHASE_ENVIRONMENTS: dict[str, str] = dict(PAF_PHASES)
PAF_SIM_PEAK_MEMORY_GB = 3
PAF_SIM_MAX_RUNTIME = "01:00:00"

ARTIFACT_INDEX_COLUMNS = [
    "location",
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]
# Placeholder for index columns except location
DEMOGRAPHIC_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]
# Index columns + subnational
SUBNATIONAL_INDEX_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
    "subnational",
]

DRAW_COUNT = 250
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])
GBD_EXTRACT_YEAR = 2023


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()


class __AgeGroup(NamedTuple):
    BIRTH_ID = 164
    EARLY_NEONATAL_ID = 2
    LATE_NEONATAL_ID = 3
    POST_NEONATAL = 4
    YEARS_1_TO_4 = 5
    MONTHS_1_TO_5 = 388
    MONTHS_6_TO_11 = 389
    MONTHS_12_TO_23 = 238
    YEARS_2_TO_4 = 34

    NEONATAL_AGE_GROUP_IDS = [EARLY_NEONATAL_ID, LATE_NEONATAL_ID]

    ## Could make this through vivarium_inputs instead of defining it here
    GBD_2023 = {
        EARLY_NEONATAL_ID,
        LATE_NEONATAL_ID,
        MONTHS_1_TO_5,
        MONTHS_6_TO_11,
        MONTHS_12_TO_23,
        YEARS_2_TO_4,
    }


AGE_GROUP = __AgeGroup()

NEONATAL_END_AGE = 0.076712

SUBNATIONAL_LOCATION_DICT = {
    "Ethiopia": [
        "Addis Ababa",
        "Afar",
        "Amhara",
        "Benishangul-Gumuz",
        "Dire Dawa",
        "Gambella",
        "Harari",
        "Oromia",
        "Somali",
        "Southern Nations, Nationalities, and Peoples",
        "Tigray",
    ],
    "Nigeria": [
        "Abia",
        "Adamawa",
        "Akwa Ibom",
        "Anambra",
        "Bauchi",
        "Bayelsa",
        "Benue",
        "Borno",
        "Cross River",
        "Delta",
        "Ebonyi",
        "Edo",
        "Ekiti",
        "Enugu",
        "FCT (Abuja)",
        "Gombe",
        "Imo",
        "Jigawa",
        "Kaduna",
        "Kano",
        "Katsina",
        "Kebbi",
        "Kogi",
        "Kwara",
        "Lagos",
        "Nasarawa",
        "Niger",
        "Ogun",
        "Ondo",
        "Osun",
        "Oyo",
        "Plateau",
        "Rivers",
        "Sokoto",
        "Taraba",
        "Yobe",
        "Zamfara",
    ],
    "Pakistan": [
        "Azad Jammu & Kashmir",
        "Balochistan",
        "Gilgit-Baltistan",
        "Islamabad Capital Territory",
        "Khyber Pakhtunkhwa",
        "Punjab",
        "Sindh",
    ],
}
