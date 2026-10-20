from typing import Callable

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vivarium_inputs", exc_type=ImportError)

from vivarium_gates_nutrition_optimization_child.constants import (
    data_keys,
    metadata,
    paths,
)
from vivarium_gates_nutrition_optimization_child.data import extra_gbd, loader

LOCATION = "Ethiopia"
ETHIOPIA = 179
GBD_DRAW_COUNT = 100
AGE_BINS = [
    (0.0, 0.01917808),
    (0.01917808, 0.07671233),
    (0.07671233, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
]
PREGNANT_POPULATION = [40.0, 300.0, 250.0, 180.0, 120.0, 60.0, 20.0, 5.0]


def _anemia_prevalence() -> pd.DataFrame:
    """Return a pull of pregnant anemia prevalence in the shape GBD returns it.

    Rows are shuffled and draw columns are in lexical order, so the loader must key both
    by label.
    """
    ages = extra_gbd.PREGNANCY_AGE_GROUP_IDS
    draws = {
        name: [
            0.1 + 0.02 * a + 0.001 * int(name.removeprefix("draw_")) for a in range(len(ages))
        ]
        for name in sorted(f"draw_{d}" for d in range(GBD_DRAW_COUNT))
    }
    data = pd.DataFrame(
        {
            "population_group_id": extra_gbd.PREGNANT_POPULATION_GROUP_ID,
            "location_id": ETHIOPIA,
            "measure_id": 5,
            "year_id": metadata.GBD_EXTRACT_YEAR,
            "metric_id": 3,
            "cause_id": extra_gbd.ALL_CAUSES_ID,
            "rei_id": extra_gbd.MODERATE_SEVERE_ANEMIA_REI_ID,
            "sex_id": 2,
            "age_group_id": ages,
            "population": PREGNANT_POPULATION,
            **draws,
            "gbd_process_version_id": 20405,
        }
    )
    return data.sample(frac=1, random_state=0).reset_index(drop=True)


def _expected_p_low_hgb() -> np.ndarray:
    """Return the population-weighted prevalence per artifact draw."""
    data = _anemia_prevalence()
    weights = data["population"] / data["population"].sum()
    gbd_draws = np.array([(data[f"draw_{d}"] * weights).sum() for d in range(GBD_DRAW_COUNT)])
    return gbd_draws[np.arange(metadata.DRAW_COUNT) % GBD_DRAW_COUNT]


def _companion_draws(filename: str) -> np.ndarray:
    """Return Ethiopia's artifact draws from a packaged low-BMI-given-hemoglobin CSV."""
    data = pd.read_csv(paths.RAW_DATA_DIR / filename)
    values = data[data["location_id"] == ETHIOPIA].set_index("draw")["value"]
    return values[metadata.ARTIFACT_COLUMNS].to_numpy()


@pytest.fixture
def install_pull(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.DataFrame], None]:
    """Patch the demography key and location lookup; return a setter for the GBD pull."""
    demography = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                (LOCATION, sex, age_start, age_end, 2023, 2024)
                for sex in ["Female", "Male"]
                for age_start, age_end in AGE_BINS
            ],
            names=["location", "sex", "age_start", "age_end", "year_start", "year_end"],
        )
    )

    def _get_data(lookup_key: str, location: str, *args, **kwargs) -> pd.DataFrame:
        assert lookup_key == data_keys.POPULATION.DEMOGRAPHY
        return demography

    monkeypatch.setattr(loader, "get_data", _get_data)
    monkeypatch.setattr(
        loader.utility_data, "resolve_location", {LOCATION: ETHIOPIA}.__getitem__
    )

    def _install(data: pd.DataFrame) -> None:
        monkeypatch.setattr(
            loader.extra_gbd,
            "get_pregnant_moderate_severe_anemia_prevalence",
            lambda location: data,
        )

    return _install


def _load_exposure() -> pd.DataFrame:
    return loader.load_maternal_bmi_anemia_exposure(
        data_keys.MATERNAL_BMI_ANEMIA.EXPOSURE, LOCATION
    )


def test_exposure_takes_low_hemoglobin_from_the_weighted_gbd_pull(
    install_pull: Callable[[pd.DataFrame], None],
) -> None:
    install_pull(_anemia_prevalence())

    exposure = _load_exposure()

    assert exposure.shape == (4 * 2 * len(AGE_BINS), metadata.DRAW_COUNT)
    assert list(exposure.columns) == list(metadata.ARTIFACT_COLUMNS)
    assert exposure.notna().all().all()
    by_category = exposure.groupby("parameter")
    np.testing.assert_array_equal(by_category.nunique().to_numpy(), 1)
    category_values = by_category.mean()
    np.testing.assert_allclose(category_values.sum().to_numpy(), 1.0)
    low_hgb = category_values.loc[["cat1", "cat2"]].sum()
    np.testing.assert_allclose(low_hgb.to_numpy(), _expected_p_low_hgb())
    np.testing.assert_allclose(
        (category_values.loc["cat1"] / low_hgb).to_numpy(),
        _companion_draws("prevalence_of_low_bmi_given_hemoglobin_below_10_age_weighted.csv"),
    )
    np.testing.assert_allclose(
        (category_values.loc["cat3"] / (1 - low_hgb)).to_numpy(),
        _companion_draws("prevalence_of_low_bmi_given_hemoglobin_above_10_age_weighted.csv"),
    )


def _drop_age_group(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(index=0)


def _duplicate_age_group(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([data, data.iloc[[0]]], ignore_index=True)


def _add_age_group(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([data, data.iloc[[0]].assign(age_group_id=7)], ignore_index=True)


@pytest.mark.parametrize("defect", [_drop_age_group, _duplicate_age_group, _add_age_group])
def test_exposure_rejects_a_pull_without_one_row_per_age_group(
    install_pull: Callable[[pd.DataFrame], None],
    defect: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    install_pull(defect(_anemia_prevalence()))

    with pytest.raises(ValueError, match="one row per age group"):
        _load_exposure()


def _blank_draw(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[0, "draw_3"] = np.nan
    return data


def _blank_population(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[0, "population"] = np.nan
    return data


def _zero_population(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[0, "population"] = 0.0
    return data


def _drop_draws(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=data.filter(like="draw_").columns)


@pytest.mark.parametrize(
    "defect", [_blank_draw, _blank_population, _zero_population, _drop_draws]
)
def test_exposure_rejects_a_pull_with_unusable_values(
    install_pull: Callable[[pd.DataFrame], None],
    defect: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    install_pull(defect(_anemia_prevalence()))

    with pytest.raises(ValueError, match="no draws, a missing value"):
        _load_exposure()
