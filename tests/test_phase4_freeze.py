from src.imbalance.freeze import (
    PHASE4_CLASS_WEIGHT,
    PHASE4_FROZEN_DECISION,
    PHASE4_SCREENING_ADVANCERS,
    PHASE4_SELECTED_EXPERIMENT,
    PHASE4_USE_RESAMPLING,
)


def test_phase4_selected_experiment_is_frozen_to_none():
    assert PHASE4_SELECTED_EXPERIMENT == "none"
    assert PHASE4_FROZEN_DECISION.experiment == "none"


def test_phase4_freeze_disables_all_imbalance_intervention():
    assert PHASE4_USE_RESAMPLING is False
    assert PHASE4_CLASS_WEIGHT is None
    assert PHASE4_FROZEN_DECISION.use_resampling is False
    assert PHASE4_FROZEN_DECISION.class_weight is None


def test_phase4_freeze_preserves_screening_decision_trail():
    assert PHASE4_SCREENING_ADVANCERS == (
        "undersample_10_to_1",
        "none",
        "class_weight_1",
    )
    assert (
        PHASE4_FROZEN_DECISION.screening_advancers
        == PHASE4_SCREENING_ADVANCERS
    )


def test_phase4_frozen_decision_is_immutable():
    try:
        PHASE4_FROZEN_DECISION.experiment = "undersample_10_to_1"
    except Exception:
        pass
    else:
        raise AssertionError("Frozen Phase 4 decision must be immutable.")

    assert PHASE4_FROZEN_DECISION.experiment == "none"
