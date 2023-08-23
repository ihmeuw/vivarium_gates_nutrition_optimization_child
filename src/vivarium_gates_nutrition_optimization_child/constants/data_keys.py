from enum import Enum
from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"
    CRUDE_BIRTH_RATE: str = "covariate.live_births_by_sex.estimate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


##########
# Causes #
##########


class __DiarrhealDiseases(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DURATION: TargetString = TargetString("cause.diarrheal_diseases.duration")
    PREVALENCE: TargetString = TargetString("cause.diarrheal_diseases.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.diarrheal_diseases.incidence_rate")
    REMISSION_RATE: TargetString = TargetString("cause.diarrheal_diseases.remission_rate")
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.diarrheal_diseases.disability_weight"
    )
    EMR: TargetString = TargetString("cause.diarrheal_diseases.excess_mortality_rate")
    CSMR: TargetString = TargetString(
        "cause.diarrheal_diseases.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString("cause.diarrheal_diseases.restrictions")
    BIRTH_PREVALENCE: TargetString = TargetString("cause.diarrheal_diseases.birth_prevalence")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "diarrheal_diseases"

    @property
    def log_name(self):
        return "diarrheal diseases"


DIARRHEA = __DiarrhealDiseases()


class __Measles(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.measles.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.measles.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.measles.disability_weight")
    EMR: TargetString = TargetString("cause.measles.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.measles.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.measles.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "measles"

    @property
    def log_name(self):
        return "measles"


MEASLES = __Measles()


class __LowerRespiratoryInfections(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DURATION: TargetString = TargetString("cause.lower_respiratory_infections.duration")
    PREVALENCE: TargetString = TargetString("cause.lower_respiratory_infections.prevalence")
    INCIDENCE_RATE: TargetString = TargetString(
        "cause.lower_respiratory_infections.incidence_rate"
    )
    REMISSION_RATE: TargetString = TargetString(
        "cause.lower_respiratory_infections.remission_rate"
    )
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.lower_respiratory_infections.disability_weight"
    )
    EMR: TargetString = TargetString(
        "cause.lower_respiratory_infections.excess_mortality_rate"
    )
    CSMR: TargetString = TargetString(
        "cause.lower_respiratory_infections.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString(
        "cause.lower_respiratory_infections.restrictions"
    )

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "lower_respiratory_infections"

    @property
    def log_name(self):
        return "lower respiratory infections"


LRI = __LowerRespiratoryInfections()


class __ProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    EMR: TargetString = TargetString(
        "cause.protein_energy_malnutrition.excess_mortality_rate"
    )
    CSMR: TargetString = TargetString(
        "cause.protein_energy_malnutrition.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString(
        "cause.protein_energy_malnutrition.restrictions"
    )


PEM = __ProteinEnergyMalnutrition()


class __ModerateProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.moderate_protein_energy_malnutrition.disability_weight"
    )
    EMR: TargetString = TargetString(
        "cause.moderate_protein_energy_malnutrition.excess_mortality_rate"
    )
    CSMR: TargetString = TargetString(
        "cause.moderate_protein_energy_malnutrition.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString(
        "cause.moderate_protein_energy_malnutrition.restrictions"
    )

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "moderate_protein_energy_malnutrition"

    @property
    def log_name(self):
        return "moderate protein energy malnutrition"


MODERATE_PEM = __ModerateProteinEnergyMalnutrition()


class __SevereProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.severe_protein_energy_malnutrition.disability_weight"
    )
    EMR: TargetString = TargetString(
        "cause.severe_protein_energy_malnutrition.excess_mortality_rate"
    )
    CSMR: TargetString = TargetString(
        "cause.severe_protein_energy_malnutrition.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString(
        "cause.severe_protein_energy_malnutrition.restrictions"
    )

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "severe_protein_energy_malnutrition"

    @property
    def log_name(self):
        return "severe protein energy malnutrition"


SEVERE_PEM = __SevereProteinEnergyMalnutrition()


################
# Risk Factors #
################


class __Wasting(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = "risk_factor.child_wasting.distribution"
    ALT_DISTRIBUTION: TargetString = "alternative_risk_factor.child_wasting.distribution"
    CATEGORIES: TargetString = "risk_factor.child_wasting.categories"
    EXPOSURE: TargetString = "risk_factor.child_wasting.exposure"
    RELATIVE_RISK: TargetString = "risk_factor.child_wasting.relative_risk"
    PAF: TargetString = "risk_factor.child_wasting.population_attributable_fraction"

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT4 = "cat4"
    CAT3 = "cat3"
    CAT2 = "cat2"
    CAT1 = "cat1"

    @property
    def name(self):
        return "child_wasting"

    @property
    def log_name(self):
        return "child wasting"


WASTING = __Wasting()


class __Stunting(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = "risk_factor.child_stunting.distribution"
    ALT_DISTRIBUTION: TargetString = "alternative_risk_factor.child_stunting.distribution"
    CATEGORIES: TargetString = "risk_factor.child_stunting.categories"
    EXPOSURE: TargetString = "risk_factor.child_stunting.exposure"
    RELATIVE_RISK: TargetString = "risk_factor.child_stunting.relative_risk"
    PAF: TargetString = "risk_factor.child_stunting.population_attributable_fraction"

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT4 = "cat4"
    CAT3 = "cat3"
    CAT2 = "cat2"
    CAT1 = "cat1"

    @property
    def name(self):
        return "child_stunting"

    @property
    def log_name(self):
        return "child stunting"


STUNTING = __Stunting()


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = "risk_factor.low_birth_weight_and_short_gestation.exposure"
    DISTRIBUTION: TargetString = (
        "risk_factor.low_birth_weight_and_short_gestation.distribution"
    )
    CATEGORIES: TargetString = "risk_factor.low_birth_weight_and_short_gestation.categories"
    RELATIVE_RISK: TargetString = (
        "risk_factor.low_birth_weight_and_short_gestation.relative_risk"
    )
    RELATIVE_RISK_INTERPOLATOR: TargetString = (
        "risk_factor.low_birth_weight_and_short_gestation.relative_risk_interpolator"
    )

    PAF: TargetString = (
        "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction"
    )

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    BIRTH_WEIGHT_EXPOSURE = TargetString("risk_factor.birth_weight.birth_exposure")
    GESTATIONAL_AGE_EXPOSURE = TargetString("risk_factor.gestational_age.birth_exposure")

    @property
    def name(self):
        return "low_birth_weight_and_short_gestation"

    @property
    def log_name(self):
        return "low birth weight and short gestation"


LBWSG = __LowBirthWeightShortGestation()


class __AffectedUnmodeledCauses(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    URI_CSMR: TargetString = TargetString(
        "cause.upper_respiratory_infections.cause_specific_mortality_rate"
    )
    OTITIS_MEDIA_CSMR: TargetString = TargetString(
        "cause.otitis_media.cause_specific_mortality_rate"
    )
    MENINGITIS_CSMR: TargetString = TargetString(
        "cause.meningitis.cause_specific_mortality_rate"
    )
    ENCEPHALITIS_CSMR: TargetString = TargetString(
        "cause.encephalitis.cause_specific_mortality_rate"
    )
    NEONATAL_PRETERM_BIRTH_CSMR: TargetString = TargetString(
        "cause.neonatal_preterm_birth.cause_specific_mortality_rate"
    )
    NEONATAL_ENCEPHALOPATHY_CSMR: TargetString = TargetString(
        "cause.neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.cause_specific_mortality_rate"
    )
    NEONATAL_SEPSIS_CSMR: TargetString = TargetString(
        "cause.neonatal_sepsis_and_other_neonatal_infections.cause_specific_mortality_rate"
    )
    NEONATAL_JAUNDICE_CSMR: TargetString = TargetString(
        "cause.hemolytic_disease_and_other_neonatal_jaundice.cause_specific_mortality_rate"
    )
    OTHER_NEONATAL_DISORDERS_CSMR: TargetString = TargetString(
        "cause.other_neonatal_disorders.cause_specific_mortality_rate"
    )
    SIDS_CSMR: TargetString = TargetString(
        "cause.sudden_infant_death_syndrome.cause_specific_mortality_rate"
    )
    NEONATAL_LRI_CSMR: TargetString = TargetString(
        "cause.neonatal_lower_respiratory_infections.cause_specific_mortality_rate"
    )
    NEONATAL_DIARRHEAL_DISEASES_CSMR: TargetString = TargetString(
        "cause.neonatal_diarrheal_diseases.cause_specific_mortality_rate"
    )

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "affected_unmodeled_causes"

    @property
    def log_name(self):
        return "affected unmodeled causes"


AFFECTED_UNMODELED_CAUSES = __AffectedUnmodeledCauses()


class CGFCategories(Enum):
    UNEXPOSED = "unexposed"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class __AdditiveRisk(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString
    DISTRIBUTION: TargetString
    CATEGORIES: TargetString
    # analogous to excess mortality rate
    EXCESS_SHIFT: TargetString
    # analogous to cause specific mortality rate
    RISK_SPECIFIC_SHIFT: TargetString

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT1 = "cat1"
    CAT2 = "cat2"

    @property
    def name(self):
        return self.EXPOSURE.name

    @property
    def log_name(self):
        return self.name.replace("_", " ")


def _get_additive_risk_keys(treatment_type: str) -> __AdditiveRisk:
    return __AdditiveRisk(
        EXPOSURE=TargetString(f"risk_factor.{treatment_type}.exposure"),
        DISTRIBUTION=TargetString(f"risk_factor.{treatment_type}.distribution"),
        CATEGORIES=TargetString(f"risk_factor.{treatment_type}.categories"),
        EXCESS_SHIFT=TargetString(f"risk_factor.{treatment_type}.excess_shift"),
        RISK_SPECIFIC_SHIFT=TargetString(f"risk_factor.{treatment_type}.risk_specific_shift"),
    )


IFA_SUPPLEMENTATION = _get_additive_risk_keys("iron_folic_acid_supplementation")
BEP_SUPPLEMENTATION = _get_additive_risk_keys("balanced_energy_protein_supplementation")
IV_IRON = _get_additive_risk_keys("iv_iron")
MATERNAL_BMI_ANEMIA = _get_additive_risk_keys("maternal_bmi_anemia")


class __MMN_Supplementation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.exposure")
    DISTRIBUTION: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.distribution")
    CATEGORIES: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.categories")
    EXCESS_SHIFT: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.excess_shift")
    EXCESS_GA_SHIFT_SUBPOP_1: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.excess_gestational_age_shift_subpop_1")
    EXCESS_GA_SHIFT_SUBPOP_2: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.excess_gestational_age_shift_subpop_2")
    RISK_SPECIFIC_SHIFT: TargetString = TargetString("risk_factor.multiple_micronutrient_supplementation.risk_specific_shift")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT1 = "cat1"
    CAT2 = "cat2"

    @property
    def name(self):
        return self.EXPOSURE.name

    @property
    def log_name(self):
        return self.name.replace("_", " ")

MMN_SUPPLEMENTATION = __MMN_Supplementation()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    DIARRHEA,
    MEASLES,
    LRI,
    STUNTING,
    WASTING,
    MODERATE_PEM,
    SEVERE_PEM,
    LBWSG,
    AFFECTED_UNMODELED_CAUSES,
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    BEP_SUPPLEMENTATION,
    IV_IRON,
    MATERNAL_BMI_ANEMIA,
]
