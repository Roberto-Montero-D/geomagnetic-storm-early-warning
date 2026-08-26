import pandas as pd
import pytest

import src.phase7.thresholds as phase7_thresholds
from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    get_phase7_experiment,
)
from src.phase7.oof import Phase7OOFPredictions


def _oof(experiment_id: str) -> Phase7OOFPredictions:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="h",
        name="timestamp",
    )

    table = pd.DataFrame(
        {
            "probability": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 0, 1, 1],
            "storm_id": [pd.NA, pd.NA, 1, 1],
            "fold": ["walk_forward_1"] * 4,
        },
        index=index,
    )

    return Phase7OOFPredictions(
        experiment_id=experiment_id,
        config_id="lightgbm_lr0.1_leaves127",
        table=table,
    )


def test_every_registered_experiment_can_be_resolved():
    for experiment in PHASE7_EXPERIMENTS:
        assert (
            phase7_thresholds._resolve_experiment(experiment)
            == experiment
        )


def test_rejects_modified_experiment_object():
    frozen = get_phase7_experiment("t5_h6")

    modified = type(frozen)(
        experiment_id=frozen.experiment_id,
        threshold=frozen.threshold,
        horizon_hours=12,
        is_primary_control=frozen.is_primary_control,
    )

    with pytest.raises(ValueError, match="frozen registry"):
        phase7_thresholds._resolve_experiment(modified)


def test_rejects_oof_from_different_experiment():
    with pytest.raises(ValueError, match="does not match"):
        phase7_thresholds._validate_oof_experiment(
            _oof("t5_h3"),
            get_phase7_experiment("t5_h6"),
        )


def test_passes_experiment_horizon_to_phase6_optimizer(
    monkeypatch,
):
    captured = {}

    def fake_events(kp_intervals, experiment):
        captured["event_experiment"] = experiment.experiment_id
        return pd.DataFrame(
            {
                "event_id": [1],
                "start_time": [pd.Timestamp("2020-01-01")],
                "end_time": [pd.Timestamp("2020-01-01 03:00")],
            }
        )

    class FakeResult:
        selected_threshold = 0.42
        global_threshold_table = pd.DataFrame()
        fold_threshold_table = pd.DataFrame()
        fold_selected_thresholds = {}
        stability_thresholds = ()

    def fake_optimize(oof, events, **kwargs):
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        phase7_thresholds,
        "build_phase7_events",
        fake_events,
    )
    monkeypatch.setattr(
        phase7_thresholds,
        "optimize_phase6_threshold",
        fake_optimize,
    )

    result = phase7_thresholds.optimize_phase7_threshold(
        _oof("t5_h24"),
        pd.DataFrame(),
        "t5_h24",
    )

    assert result.experiment_id == "t5_h24"
    assert result.selected_threshold == 0.42
    assert captured["event_experiment"] == "t5_h24"
    assert captured["horizon_hours"] == 24
    assert captured["cooldown_hours"] == 3
    assert captured["max_far_per_day"] == 0.2


def test_threshold_recalibration_uses_experiment_specific_events(
    monkeypatch,
):
    captured = []

    def fake_events(kp_intervals, experiment):
        captured.append(
            (
                experiment.experiment_id,
                experiment.threshold,
            )
        )
        return pd.DataFrame(
            {
                "event_id": [1],
                "start_time": [pd.Timestamp("2020-01-01")],
                "end_time": [pd.Timestamp("2020-01-01 03:00")],
            }
        )

    class FakeResult:
        selected_threshold = 0.1
        global_threshold_table = pd.DataFrame()
        fold_threshold_table = pd.DataFrame()
        fold_selected_thresholds = {}
        stability_thresholds = ()

    monkeypatch.setattr(
        phase7_thresholds,
        "build_phase7_events",
        fake_events,
    )
    monkeypatch.setattr(
        phase7_thresholds,
        "optimize_phase6_threshold",
        lambda *args, **kwargs: FakeResult(),
    )

    phase7_thresholds.optimize_phase7_threshold(
        _oof("t6_h6"),
        pd.DataFrame(),
        "t6_h6",
    )

    assert captured == [("t6_h6", 6.0)]