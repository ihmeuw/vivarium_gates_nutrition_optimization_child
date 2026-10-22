"""Check how LBWSG PAF simulation results become the artifact's LBWSG PAF key."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.constants import metadata, paths
from vivarium_gates_nutrition_optimization_child.data import lbwsg_paf_results
from vivarium_gates_nutrition_optimization_child.utilities import sanitize_location

EARLY = lbwsg_paf_results.EARLY_NEONATAL
LATE = lbwsg_paf_results.LATE_NEONATAL


def make_results(draws: Optional[List[int]] = None, seed: int = 4) -> pd.DataFrame:
    """Results in a shuffled draw order, with a value that encodes its own draw."""
    draws = list(range(metadata.DRAW_COUNT)) if draws is None else draws
    order = np.random.default_rng(seed).permutation(draws)
    rows = [
        {
            "age_group": age_group,
            "sex": sex,
            "input_draw": draw,
            "random_seed": 0,
            "value": base + draw / 10_000 + (0.001 if sex == "Male" else 0.0),
        }
        for draw in order
        for sex in lbwsg_paf_results.SEXES
        for age_group, base in [(EARLY, 0.8), (LATE, 0.7)]
    ]
    results = pd.DataFrame(rows)
    results["age_group"] = pd.Categorical(results["age_group"], [EARLY, LATE])
    return results


def test_draw_k_holds_input_draw_k_whatever_the_row_order():
    paf = lbwsg_paf_results.format_lbwsg_paf(make_results())

    early_female = paf.xs(("Female", 0.0), level=["sex", "age_start"]).iloc[0]
    expected = 0.8 + np.arange(metadata.DRAW_COUNT) / 10_000
    np.testing.assert_array_equal(early_female.to_numpy(), expected)
    assert list(paf.columns) == list(metadata.ARTIFACT_COLUMNS)


def test_formatted_paf_has_the_artifact_shape():
    paf = lbwsg_paf_results.format_lbwsg_paf(make_results())

    assert paf.shape == (8, metadata.DRAW_COUNT)
    assert paf.index.names == ["sex", "age_start", "age_end", "year_start", "year_end"]
    ages = sorted(
        set(
            zip(
                paf.index.get_level_values("age_start"), paf.index.get_level_values("age_end")
            )
        )
    )
    assert ages == [
        (0.0, 0.01917808),
        (0.01917808, 0.07671233),
        (0.07671233, 1.0),
        (1.0, 5.0),
    ]
    assert set(paf.index.get_level_values("year_start")) == {metadata.GBD_EXTRACT_YEAR}
    assert set(paf.index.get_level_values("year_end")) == {metadata.GBD_EXTRACT_YEAR + 1}
    post_neonatal = paf[paf.index.get_level_values("age_start") >= 0.07671233]
    assert (post_neonatal == 0).all().all()


def test_early_neonatal_pass_may_leave_late_neonatal_at_zero():
    results = make_results()
    results.loc[results["age_group"] == LATE, "value"] = 0.0

    with pytest.raises(ValueError, match="outside"):
        lbwsg_paf_results.format_lbwsg_paf(results)
    paf = lbwsg_paf_results.format_lbwsg_paf(results, require_late_neonatal=False)
    late = paf[paf.index.get_level_values("age_start") == 0.01917808]
    assert (late == 0).all().all()


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda df: df.iloc[1:], "missing a value"),
        (lambda df: df[df["input_draw"] != 7], "missing draws"),
        (lambda df: df[df["sex"] != "Male"], "Missing strata"),
        (lambda df: pd.concat([df, df.iloc[:1]]), "more than one value"),
        (lambda df: df.assign(random_seed=df.index % 2), "more than one random_seed"),
        (lambda df: df.assign(value=df["value"] + 0.5), "outside"),
        (lambda df: df.drop(columns="input_draw"), "missing columns"),
    ],
)
def test_incomplete_or_invalid_results_are_refused(corrupt, message):
    with pytest.raises(ValueError, match=message):
        lbwsg_paf_results.format_lbwsg_paf(corrupt(make_results()))


def test_extra_draws_are_refused():
    results = make_results(draws=list(range(metadata.DRAW_COUNT + 1)))

    with pytest.raises(ValueError, match="unexpected draws"):
        lbwsg_paf_results.format_lbwsg_paf(results)


def test_non_zero_post_neonatal_results_are_refused():
    results = pd.concat(
        [
            make_results(),
            pd.DataFrame(
                [{"age_group": "1_to_5_months", "sex": "Male", "input_draw": 0, "value": 0.1}]
            ),
        ]
    )

    with pytest.raises(ValueError, match="outside the neonatal"):
        lbwsg_paf_results.format_lbwsg_paf(results)


@pytest.mark.parametrize(
    "location, directory",
    [
        ("Ethiopia", "ethiopia"),
        ("Sub-Saharan Africa", "sub-saharan_africa"),
        ("Kenya", "kenya"),
    ],
)
def test_output_path_uses_the_sanitized_location(location, directory):
    path = lbwsg_paf_results.get_output_path(location)

    assert path.parent == paths.LBWSG_PAF_OUTPUT_DIR / directory
    assert path.name == f"{lbwsg_paf_results.LBWSG_PAF_MEASURE}.parquet"


def test_early_neonatal_check_passes_on_identical_values_in_any_order():
    results = make_results()

    lbwsg_paf_results.check_early_neonatal_unchanged(results, make_results(seed=9))


def test_early_neonatal_check_refuses_any_difference():
    results = make_results()
    changed = results.assign(value=results["value"] + 1e-15)

    with pytest.raises(ValueError, match="different early neonatal PAFs"):
        lbwsg_paf_results.check_early_neonatal_unchanged(results, changed)


@pytest.mark.parametrize(
    "output_file",
    sorted(
        paths.LBWSG_PAF_OUTPUT_DIR.glob(f"*/{lbwsg_paf_results.LBWSG_PAF_MEASURE}.parquet")
    ),
    ids=lambda path: path.parent.name,
)
def test_committed_outputs_are_complete(output_file: Path):
    results = pd.read_parquet(output_file)

    paf = lbwsg_paf_results.format_lbwsg_paf(results)

    assert output_file.parent.name in {sanitize_location(loc) for loc in metadata.LOCATIONS}
    by_age = paf.groupby(level=["sex", "age_start"]).mean().mean(axis=1)
    for sex in lbwsg_paf_results.SEXES:
        assert by_age[(sex, 0.01917808)] < by_age[(sex, 0.0)]
