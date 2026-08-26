import pandas as pd

from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    assign_temporal_periods,
)
from src.evaluation.phase6_freeze import (
    PHASE6_FROZEN_DECISION,
    PHASE6_SELECTED_THRESHOLD,
)
from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES
from src.final_test.contract import (
    PHASE8_FROZEN_CONTRACT,
    PHASE8_FINAL_TEST_END_EXCLUSIVE,
    PHASE8_FINAL_TEST_START,
    PHASE8_FEATURES,
    PHASE8_MODEL_CONFIG_ID,
    PHASE8_OPERATIONAL_THRESHOLD,
    validate_phase8_contract,
)
from src.imbalance.freeze import PHASE4_FROZEN_DECISION


def test_phase6_machine_readable_handoff_is_frozen():
    assert PHASE6_FROZEN_DECISION.config_id == "lightgbm_lr0.1_leaves127"
    assert PHASE6_SELECTED_THRESHOLD == 0.10
    assert PHASE6_FROZEN_DECISION.threshold == 0.10
    assert PHASE6_FROZEN_DECISION.oof_rows == 25_873
    assert PHASE6_FROZEN_DECISION.n_detected_events == 21
    assert PHASE6_FROZEN_DECISION.n_events == 31
    assert PHASE6_FROZEN_DECISION.protected_final_test_scored is False


def test_phase8_contract_matches_phase3_feature_freeze():
    assert PHASE8_FEATURES == tuple(PHASE3_SELECTED_FEATURES)
    assert len(PHASE8_FEATURES) == 10


def test_phase8_contract_matches_phase4_imbalance_freeze():
    assert PHASE8_FROZEN_CONTRACT.imbalance_experiment == "none"
    assert (
        PHASE8_FROZEN_CONTRACT.imbalance_experiment
        == PHASE4_FROZEN_DECISION.experiment
    )
    assert PHASE8_FROZEN_CONTRACT.use_resampling is False
    assert PHASE8_FROZEN_CONTRACT.class_weight is None


def test_phase8_contract_matches_phase5_phase6_model_and_threshold():
    assert PHASE8_MODEL_CONFIG_ID == PHASE6_FROZEN_DECISION.config_id
    assert PHASE8_MODEL_CONFIG_ID == "lightgbm_lr0.1_leaves127"
    assert PHASE8_OPERATIONAL_THRESHOLD == PHASE6_FROZEN_DECISION.threshold
    assert PHASE8_OPERATIONAL_THRESHOLD == 0.10


def test_phase8_primary_truth_and_alert_contract_are_unchanged():
    contract = PHASE8_FROZEN_CONTRACT

    assert contract.experiment_id == "t5_h6"
    assert contract.storm_threshold == 5.0
    assert contract.horizon_hours == 6
    assert contract.event_termination_hours == 6
    assert contract.alert_cooldown_hours == 3
    assert contract.max_far_per_day == 0.2


def test_phase8_final_test_bounds_match_phase1_split_contract():
    times = pd.DatetimeIndex(
        [
            PHASE8_FINAL_TEST_START,
            PHASE8_FINAL_TEST_END_EXCLUSIVE - pd.Timedelta(hours=1),
        ]
    )

    splits = assign_temporal_periods(times)

    assert splits["period"].eq(PERIOD_FINAL_TEST).all()
    assert splits["is_final_test"].all()


def test_phase8_contract_forbids_post_test_retuning():
    assert PHASE8_FROZEN_CONTRACT.single_use is True
    assert PHASE8_FROZEN_CONTRACT.results_may_trigger_retuning is False


def test_phase8_contract_self_validation_passes():
    validate_phase8_contract()
