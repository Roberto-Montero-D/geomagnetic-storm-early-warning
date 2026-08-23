from src.feature_screening.freeze import (
    PHASE3_SELECTED_EXPERIMENT,
    PHASE3_SELECTED_FEATURES,
    PHASE3_SELECTION_STATUS,
)
from src.feature_screening.manifests import PHASE3_FEATURE_SETS


def test_phase3_selected_experiment_is_frozen_to_a():
    assert PHASE3_SELECTED_EXPERIMENT == "A"


def test_phase3_selected_features_exactly_match_manifest_a():
    assert PHASE3_SELECTED_FEATURES == tuple(PHASE3_FEATURE_SETS["A"])


def test_phase3_selected_feature_contract_has_ten_unique_features():
    assert len(PHASE3_SELECTED_FEATURES) == 10
    assert len(set(PHASE3_SELECTED_FEATURES)) == 10


def test_phase3_selection_status_is_frozen():
    assert PHASE3_SELECTION_STATUS == "frozen"
