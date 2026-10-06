from typing import Callable, Dict, List

import pytest

pytest.importorskip("vivarium_inputs", exc_type=ImportError)

from vivarium_gates_nutrition_optimization_child.data import loader

ETHIOPIA = 179
ETHIOPIA_PATH = [1, 166, 174, ETHIOPIA]
SUBNATIONALS = {44853: "Addis Ababa", 44854: "Afar", 44855: "Amhara"}


@pytest.fixture
def resolve_location(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Patch utility_data.resolve_location to map "Ethiopia" to its id and record calls."""
    calls: List[str] = []

    def _resolve(location_name: str) -> int:
        calls.append(location_name)
        return {"Ethiopia": ETHIOPIA}[location_name]

    monkeypatch.setattr(loader.utility_data, "resolve_location", _resolve)
    return calls


@pytest.fixture
def location_id_parents(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    """Patch utility_data.get_location_id_parents with Ethiopia's subnational paths."""
    calls: List[int] = []
    parents: Dict[int, List[int]] = {
        loc_id: ETHIOPIA_PATH + [loc_id] for loc_id in SUBNATIONALS
    }
    parents[ETHIOPIA] = ETHIOPIA_PATH

    def _parents(location_id: int) -> Dict[int, List[int]]:
        calls.append(location_id)
        return {location_id: parents[location_id]}

    monkeypatch.setattr(loader.utility_data, "get_location_id_parents", _parents)
    return calls


@pytest.fixture
def most_detailed_locations(monkeypatch: pytest.MonkeyPatch) -> Callable[[set], List[int]]:
    """Patch gbd.get_most_detailed_locations to return a chosen set and record calls."""
    calls: List[int] = []

    def _install(result: set) -> List[int]:
        def _most_detailed(location_id: int) -> set:
            calls.append(location_id)
            return set(result)

        monkeypatch.setattr(loader.gbd, "get_most_detailed_locations", _most_detailed)
        return calls

    return _install


def test_get_national_location_id_from_name(
    resolve_location: List[str], location_id_parents: List[int]
) -> None:
    assert loader.get_national_location_id("Ethiopia") == ETHIOPIA
    assert resolve_location == ["Ethiopia"]
    assert location_id_parents == []


def test_get_national_location_id_from_subnational_list(
    resolve_location: List[str], location_id_parents: List[int]
) -> None:
    subnational_ids = sorted(SUBNATIONALS)
    assert loader.get_national_location_id(subnational_ids) == ETHIOPIA
    assert location_id_parents == [subnational_ids[0]]
    assert resolve_location == []


def test_get_national_location_id_from_int(
    resolve_location: List[str], location_id_parents: List[int]
) -> None:
    assert loader.get_national_location_id(44854) == ETHIOPIA
    assert location_id_parents == [44854]
    assert resolve_location == []


def test_get_national_location_id_of_national_id_is_itself(
    location_id_parents: List[int],
) -> None:
    assert loader.get_national_location_id(ETHIOPIA) == ETHIOPIA


def test_fetch_subnational_ids_returns_sorted_ints_without_parent(
    resolve_location: List[str], most_detailed_locations: Callable[[set], List[int]]
) -> None:
    calls = most_detailed_locations({44855, ETHIOPIA, 44853, 44854})

    result = loader.fetch_subnational_ids("Ethiopia")

    assert result == [44853, 44854, 44855]
    assert all(type(loc_id) is int for loc_id in result)
    assert resolve_location == ["Ethiopia"]
    assert calls == [ETHIOPIA]


def test_fetch_subnational_ids_of_most_detailed_location_is_empty(
    resolve_location: List[str], most_detailed_locations: Callable[[set], List[int]]
) -> None:
    most_detailed_locations({ETHIOPIA})
    assert loader.fetch_subnational_ids("Ethiopia") == []
