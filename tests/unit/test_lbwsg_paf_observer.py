"""Guard the exposure weights ``LBWSGPAFObserver`` averages its relative risks over.

Ported from vivarium_gates_lsff_2026's tests. The weights are the whole
calculation, and three things about them are easy to get wrong without raising:

- the distribution has to be birth prevalence, not the neonatal exposure key,
  which GBD has already depleted by mortality;
- it has to be the prevalence of the stratum's own sex, not of both pooled;
- late neonatal weights have to be birth prevalence depleted by the early
  neonatal survival this simulation produced.
"""

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.components.lbwsg import (
    LBWSGPAFObserver,
    calculate_mortality_weights,
)
from vivarium_gates_nutrition_optimization_child.constants import data_keys

TARGET = "cause.affected_unmodeled.cause_specific_mortality_rate"
RELATIVE_RISKS = {"cat1": 3.0, "cat2": 1.0}
BIRTH_PREVALENCE = {
    ("Female", "cat1"): 0.2,
    ("Female", "cat2"): 0.8,
    ("Male", "cat1"): 0.5,
    ("Male", "cat2"): 0.5,
}
AGE_BIN_EDGES = [0.0, 0.01917808, 0.07671233, 1.0]


class FakePopulationView:
    """The slice of ``PopulationView`` the observer uses."""

    def __init__(self, population: pd.DataFrame):
        self.population = population

    def get(self, index, attributes, **kwargs):
        subset = self.population.loc[index]
        if isinstance(attributes, str):
            return subset[attributes]
        return subset[list(attributes)]


class IndexIgnoringView(FakePopulationView):
    """A view that returns the whole population whatever index it is given."""

    def get(self, index, attributes, **kwargs):
        return super().get(self.population.index, attributes, **kwargs)


class FakeRiskEffect:
    relative_risk_name = "lbwsg_relative_risk"


def make_population(dead: Optional[Dict[Tuple[str, str], int]] = None) -> pd.DataFrame:
    """Four early neonatal simulants per (sex, category), with ``dead`` of them not alive."""
    dead = dead or {}
    rows = []
    for sex in ("Female", "Male"):
        for category in RELATIVE_RISKS:
            for i in range(4):
                rows.append(
                    {
                        "sex": sex,
                        "age": 0.001,
                        "lbwsg_category": category,
                        "is_alive": i >= dead.get((sex, category), 0),
                        FakeRiskEffect.relative_risk_name: RELATIVE_RISKS[category],
                    }
                )
    return pd.DataFrame(rows)


def make_birth_exposure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sex": sex, "parameter": category, "value": value}
            for (sex, category), value in BIRTH_PREVALENCE.items()
        ]
    )


def make_observer(population: pd.DataFrame) -> LBWSGPAFObserver:
    observer = LBWSGPAFObserver(TARGET)
    observer._population_view = FakePopulationView(population)
    observer.birth_exposure = make_birth_exposure()
    observer.risk_effect = FakeRiskEffect()
    observer.pop_size = len(population)
    observer.age_bin_edges = AGE_BIN_EDGES
    return observer


def paf_from(weights: Dict[str, float]) -> float:
    mean_rr = np.average(
        [RELATIVE_RISKS[category] for category in weights],
        weights=list(weights.values()),
    )
    return (mean_rr - 1) / mean_rr


def test_mortality_weights_are_the_living_fraction_of_the_birth_cohort():
    population = make_population(dead={("Female", "cat1"): 2, ("Male", "cat2"): 1})
    observer = make_observer(population)

    weights = calculate_mortality_weights(observer, "Female")

    assert list(weights["sex"].unique()) == ["Female"]
    proportion_alive = weights.set_index("lbwsg_category")["proportion_alive"]
    assert proportion_alive["cat1"] == 0.5
    assert proportion_alive["cat2"] == 1.0


def test_paf_uses_the_stratum_sex_birth_prevalence():
    population = make_population()
    observer = make_observer(population)

    pafs = {
        sex: observer.calculate_paf(population[population["sex"] == sex])
        for sex in ("Female", "Male")
    }

    assert pafs["Female"] == pytest.approx(paf_from({"cat1": 0.2, "cat2": 0.8}))
    assert pafs["Male"] == pytest.approx(paf_from({"cat1": 0.5, "cat2": 0.5}))
    pooled = paf_from({"cat1": 0.7, "cat2": 1.3})
    assert pooled != pytest.approx(pafs["Female"])
    assert pooled != pytest.approx(pafs["Male"])


def test_paf_weights_are_birth_prevalence_depleted_by_simulated_mortality():
    population = make_population(dead={("Female", "cat1"): 2})
    observer = make_observer(population)
    living_females = population[(population["sex"] == "Female") & population["is_alive"]]

    paf = observer.calculate_paf(living_females)

    assert paf == pytest.approx(paf_from({"cat1": 0.2 * 0.5, "cat2": 0.8}))


def test_paf_refuses_to_pool_sexes():
    population = make_population()
    observer = make_observer(population)

    with pytest.raises(ValueError, match=r"\['Female', 'Male'\]"):
        observer.calculate_paf(population)


def test_paf_of_an_empty_stratum_is_zero():
    population = make_population()
    observer = make_observer(population)

    assert observer.calculate_paf(population.iloc[:0]) == 0.0


def test_paf_reads_the_stratum_it_is_handed():
    """Sex, category and relative risk come from the stratum, not an index lookup."""
    population = make_population()
    observer = make_observer(population)
    observer._population_view = IndexIgnoringView(population)

    paf = observer.calculate_paf(population[population["sex"] == "Female"])

    assert paf == pytest.approx(paf_from({"cat1": 0.2, "cat2": 0.8}))


def test_results_updater_keeps_each_stratum_first_non_zero_value():
    """Repeat early neonatal observations are ignored; late neonatal is recorded later."""
    observer = make_observer(make_population())
    start = _results({"early_neonatal": 0.0, "late_neonatal": 0.0})
    first_step = _results({"early_neonatal": 0.3, "late_neonatal": 0.0})
    second_step = _results({"early_neonatal": 0.25, "late_neonatal": 0.0})
    third_step = _results({"early_neonatal": 0.0, "late_neonatal": 0.2})

    results = start
    for step in (first_step, second_step, third_step):
        results = observer.results_updater(results, step)

    assert results["value"].to_dict() == {
        ("early_neonatal", "Female"): 0.3,
        ("late_neonatal", "Female"): 0.2,
    }


def test_results_updater_aligns_on_the_stratum_index():
    observer = make_observer(make_population())
    start = _results({"early_neonatal": 0.0, "late_neonatal": 0.0})
    reordered = _results({"late_neonatal": 0.2, "early_neonatal": 0.3})

    results = observer.results_updater(start, reordered)

    assert results.index.equals(start.index)
    assert results["value"].to_dict() == {
        ("early_neonatal", "Female"): 0.3,
        ("late_neonatal", "Female"): 0.2,
    }


def test_observation_refuses_a_cohort_spanning_two_age_groups():
    population = make_population()
    observer = make_observer(population)
    observer.on_time_step_prepare(SimpleNamespace(index=population.index))

    population.loc[0, "age"] = 0.02
    with pytest.raises(ValueError, match="more than one age group"):
        observer.on_time_step_prepare(SimpleNamespace(index=population.index))


def test_dead_simulants_do_not_count_toward_the_age_group_check():
    population = make_population(dead={("Female", "cat1"): 1})
    population.loc[0, "age"] = 0.02
    observer = make_observer(population)

    observer.on_time_step_prepare(SimpleNamespace(index=population.index))


def test_setup_refuses_birth_exposure_that_misses_a_category():
    observer = LBWSGPAFObserver(TARGET)
    birth_exposure = make_birth_exposure()
    data = {
        data_keys.LBWSG.BIRTH_EXPOSURE: birth_exposure[birth_exposure["parameter"] == "cat1"],
        data_keys.LBWSG.CATEGORIES: {"cat1": "first", "cat2": "second"},
    }
    builder = SimpleNamespace(data=SimpleNamespace(load=lambda key: data[key]))

    with pytest.raises(ValueError, match=r"only in categories: \['cat2'\]"):
        observer.setup(builder)


def _results(values: Dict[str, float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(age_group, "Female") for age_group in values], names=["age_group", "sex"]
    )
    return pd.DataFrame({"value": list(values.values())}, index=index)
