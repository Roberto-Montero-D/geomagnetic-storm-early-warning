from src.model_selection.contract import (
    PHASE5_ADVANCE_PER_FAMILY, PHASE5_CONFIGURATIONS, PHASE5_CONFIG_IDS,
    PHASE5_CONFIRMATION_FOLDS, PHASE5_FEATURES, PHASE5_IMBALANCE_EXPERIMENT,
    PHASE5_MAX_FAR_PER_DAY, PHASE5_RANDOM_STATE, PHASE5_SCREENING_FOLD,
    PHASE5_STACKING_STATUS, validate_phase5_contract,
)
from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES


def test_phase5_contract_validates():
    validate_phase5_contract()


def test_exactly_27_unique_frozen_configurations():
    assert len(PHASE5_CONFIGURATIONS)==27
    assert len(set(PHASE5_CONFIG_IDS))==27


def test_exactly_nine_configurations_per_family():
    counts={}
    for cfg in PHASE5_CONFIGURATIONS:
        counts[cfg.family]=counts.get(cfg.family,0)+1
    assert counts=={"extratrees":9,"lightgbm":9,"xgboost":9}


def test_phase5_inherits_phase3_feature_freeze():
    assert PHASE5_FEATURES==tuple(PHASE3_SELECTED_FEATURES)


def test_phase5_inherits_phase4_none_strategy():
    assert PHASE5_IMBALANCE_EXPERIMENT=="none"


def test_phase5_temporal_contract_is_development_only():
    assert PHASE5_SCREENING_FOLD=="screening"
    assert PHASE5_CONFIRMATION_FOLDS==("walk_forward_1","walk_forward_2")


def test_phase5_operational_constraint_remains_frozen():
    assert PHASE5_MAX_FAR_PER_DAY==0.2


def test_phase5_random_seed_is_frozen():
    assert PHASE5_RANDOM_STATE==42
    assert all(
        cfg.as_dict()["random_state"]==42
        for cfg in PHASE5_CONFIGURATIONS
    )


def test_one_candidate_advances_per_family():
    assert PHASE5_ADVANCE_PER_FAMILY==1


def test_stacking_is_not_pre_authorized():
    assert PHASE5_STACKING_STATUS=="gated_not_authorized"
