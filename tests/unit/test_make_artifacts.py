"""Check the national subset make_artifacts builds for the LBWSG PAF simulation."""

import sys
import types
from typing import List, Tuple

import pytest

pytest.importorskip("vivarium.cluster_tools", exc_type=ImportError)

from click.testing import CliRunner

import vivarium_gates_nutrition_optimization_child.data as data
from vivarium_gates_nutrition_optimization_child.constants import data_keys
from vivarium_gates_nutrition_optimization_child.tools import cli, make_artifacts


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, bool, bool]]:
    """Replace the artifact builder with one that records what it would write."""
    calls: List[Tuple[str, bool, bool]] = []
    builder = types.ModuleType("vivarium_gates_nutrition_optimization_child.data.builder")
    builder.open_artifact = lambda path, location: None
    builder.load_and_write_data = lambda artifact, key, location, replace, subnationals: (
        calls.append((key, replace, subnationals))
    )
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    monkeypatch.setattr(data, "builder", builder, raising=False)
    return calls


def test_paf_subset_skips_the_paf_unless_it_is_replaced(written):
    make_artifacts.build_single_location_artifact(
        "x.hdf", "Ethiopia", (), False, for_lbwsg_pafs=True
    )

    expected = [
        key
        for group in data_keys.LBWSG_PAF_KEY_GROUPS
        for key in group
        if key != data_keys.LBWSG.PAF
    ]
    assert [key for key, _, _ in written] == expected

    written.clear()
    make_artifacts.build_single_location_artifact(
        "x.hdf", "Ethiopia", (data_keys.LBWSG.PAF,), False, for_lbwsg_pafs=True
    )
    assert (data_keys.LBWSG.PAF, True, False) in written


def test_full_build_includes_the_birth_exposure(written):
    make_artifacts.build_single_location_artifact("x.hdf", "Ethiopia", (), True)

    keys = [key for key, _, _ in written]
    assert data_keys.LBWSG.BIRTH_EXPOSURE in keys
    assert data_keys.LBWSG.PAF in keys


def test_cli_builds_the_paf_subset_nationally(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(cli, "build_artifacts", lambda *args: calls.append(args))

    result = CliRunner().invoke(
        cli.make_artifacts, ["-l", "Ethiopia", "--for-lbwsg-pafs", "-o", "out"]
    )

    assert result.exit_code == 0, result.output
    location, _, _, _, _, fetch_subnationals, for_lbwsg_pafs = calls[0]
    assert (location, fetch_subnationals, for_lbwsg_pafs) == ("Ethiopia", False, True)


def test_cli_refuses_the_paf_subset_for_all_locations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "build_artifacts", lambda *args: pytest.fail("built"))

    result = CliRunner().invoke(cli.make_artifacts, ["-l", "all", "--for-lbwsg-pafs"])

    assert result.exit_code != 0
    assert "--for-lbwsg-pafs builds one location" in result.output
