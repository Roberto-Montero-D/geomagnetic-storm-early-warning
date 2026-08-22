from src.feature_screening.manifests import (
    PHASE3_EXPERIMENT_ORDER,
    PHASE3_EXTRATREES_PARAMS,
    PHASE3_FEATURE_SETS,
)
from src.features.integrated import PRIMARY_FEATURE_COLUMNS
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


def test_phase3_experiment_order_is_frozen():
    assert PHASE3_EXPERIMENT_ORDER == ("A", "B", "C", "D", "E")


def test_a_is_exactly_raw_family():
    assert PHASE3_FEATURE_SETS["A"] == tuple(PRIMARY_RAW_FEATURE_COLUMNS)


def test_sets_are_strictly_cumulative():
    previous = ()
    for name in PHASE3_EXPERIMENT_ORDER:
        current = PHASE3_FEATURE_SETS[name]
        if previous:
            assert current[:len(previous)] == previous
            assert len(current) > len(previous)
        previous = current


def test_e_equals_complete_primary_feature_manifest():
    assert PHASE3_FEATURE_SETS["E"] == tuple(PRIMARY_FEATURE_COLUMNS)
    assert len(PHASE3_FEATURE_SETS["E"]) == 93


def test_no_feature_set_contains_duplicates():
    for columns in PHASE3_FEATURE_SETS.values():
        assert len(columns) == len(set(columns))


def test_phase3_extratrees_configuration_reuses_frozen_b3():
    assert PHASE3_EXTRATREES_PARAMS == {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": None,
        "random_state": 42,
    }
