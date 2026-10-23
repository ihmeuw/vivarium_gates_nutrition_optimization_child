import itertools
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from vivarium_gates_nutrition_optimization_child.constants import metadata, paths
from vivarium_gates_nutrition_optimization_child.data.lbwsg_paf_results import (
    LBWSG_PAF_MEASURE,
    collect_paf_results,
    format_lbwsg_paf,
    get_output_path,
)

STRATA = list(itertools.product(["early_neonatal", "late_neonatal"], ["Female", "Male"]))
ENN_START, LNN_START, LNN_END = 0.0, 0.01917808, 0.07671233


def _value(age_group: str, sex: str, draw: int) -> float:
    """Return a PAF in (0, 1) that encodes its stratum and input draw."""
    return 0.5 + 0.1 * STRATA.index((age_group, sex)) + 0.0001 * draw


def _results(draws: Iterable[int] = range(metadata.DRAW_COUNT)) -> pd.DataFrame:
    """Return long-format PAF simulation results for the given input draws."""
    return pd.DataFrame(
        [
            {
                "age_group": age_group,
                "sex": sex,
                "input_draw": draw,
                "random_seed": 0,
                "value": _value(age_group, sex, draw),
            }
            for age_group, sex in STRATA
            for draw in draws
        ]
    )


def test_format_puts_input_draw_k_in_column_draw_k() -> None:
    results = _results().sample(frac=1, random_state=0)

    paf = format_lbwsg_paf(results)

    for draw in [0, 17, 249]:
        assert paf.loc[("Male", LNN_START, LNN_END, 2023, 2024), f"draw_{draw}"] == (
            _value("late_neonatal", "Male", draw)
        )
        assert paf.loc[("Female", ENN_START, LNN_START, 2023, 2024), f"draw_{draw}"] == (
            _value("early_neonatal", "Female", draw)
        )


def test_format_matches_the_artifact_layout() -> None:
    paf = format_lbwsg_paf(_results())

    assert list(paf.columns) == list(metadata.ARTIFACT_COLUMNS)
    assert list(paf.index.names) == ["sex", "age_start", "age_end", "year_start", "year_end"]
    assert sorted(
        set(
            zip(
                paf.index.get_level_values("age_start"), paf.index.get_level_values("age_end")
            )
        )
    ) == [
        (ENN_START, LNN_START),
        (LNN_START, LNN_END),
        (LNN_END, 1.0),
        (1.0, 5.0),
    ]
    assert set(paf.index.get_level_values("year_start")) == {2023}
    post_neonatal = paf.loc[paf.index.get_level_values("age_start") >= LNN_END]
    assert len(post_neonatal) == 4
    assert (post_neonatal == 0.0).all().all()


def test_format_drops_zero_results_outside_the_neonatal_age_groups() -> None:
    results = _results()
    post_neonatal = results.assign(age_group="1-5_months", value=0.0)

    paf = format_lbwsg_paf(pd.concat([results, post_neonatal]))

    pd.testing.assert_frame_equal(paf, format_lbwsg_paf(results))


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda r: r.loc[r["input_draw"] != 7], "missing draws"),
        (lambda r: pd.concat([r, _results([250])]), "unexpected draws"),
        (
            lambda r: r.loc[~((r["sex"] == "Male") & (r["age_group"] == "late_neonatal"))],
            "Missing strata",
        ),
        (lambda r: pd.concat([r, r.iloc[:1]]), "more than one value"),
        (
            lambda r: pd.concat([r, r.iloc[:1].assign(random_seed=1, input_draw=0)]),
            "more than one random seed",
        ),
        (
            lambda r: r.assign(value=r["value"].where(r["input_draw"] != 3, 1.0)),
            r"outside \(0, 1\)",
        ),
        (
            lambda r: pd.concat([r, r.iloc[:1].assign(age_group="1-5_months")]),
            "non-zero outside",
        ),
        (lambda r: r.drop(columns="input_draw"), "missing columns"),
    ],
)
def test_format_refuses_incomplete_or_inconsistent_results(corrupt, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        format_lbwsg_paf(corrupt(_results()))


def test_collect_reads_every_shard_in_stratum_and_draw_order(tmp_path: Path) -> None:
    results = _results()
    measure_dir = tmp_path / LBWSG_PAF_MEASURE
    measure_dir.mkdir()
    shuffled = results.sample(frac=1, random_state=1).astype(
        {"age_group": "category", "sex": "category"}
    )
    shuffled.iloc[:500].to_parquet(measure_dir / "b.parquet")
    shuffled.iloc[500:].to_parquet(measure_dir / "a.parquet")

    collected = collect_paf_results(tmp_path)

    expected = results.sort_values(["age_group", "sex", "input_draw"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(collected, expected)


def test_collect_raises_without_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=LBWSG_PAF_MEASURE):
        collect_paf_results(tmp_path)


def test_output_path_uses_the_sanitized_location() -> None:
    assert get_output_path("Ethiopia") == paths.LBWSG_PAF_OUTPUT_DIR / "ethiopia.parquet"


@pytest.mark.parametrize(
    "output_file", sorted(paths.LBWSG_PAF_OUTPUT_DIR.glob("*.parquet")), ids=lambda p: p.stem
)
def test_tracked_outputs_format_as_the_artifact_key(output_file: Path) -> None:
    paf = format_lbwsg_paf(pd.read_parquet(output_file))

    assert not paf.isna().any().any()
