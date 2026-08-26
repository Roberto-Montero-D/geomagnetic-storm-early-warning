import copy

import numpy as np
import pytest

from src.phase7.analysis import (
    PHASE7_ANALYSIS_COLUMNS,
    build_experiment_summary,
    build_fold_diagnostics,
    build_horizon_comparison,
    build_severity_comparison,
    validate_phase7_analysis,
)
from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    PHASE7_EXPERIMENT_IDS,
    PHASE7_PRIMARY_CONTROL_ID,
)


def _summary(
    experiment_id,
    *,
    selected_threshold=0.10,
    n_events=10,
    n_detected_events=8,
    n_alert_episodes=220,
    false_alarm_episodes=200,
    valid_exposure_hours=24000,
    fold_1=0.08,
    fold_2=0.12,
    stability=(0.10, 0.11),
):
    spec = next(
        experiment
        for experiment in PHASE7_EXPERIMENTS
        if experiment.experiment_id == experiment_id
    )
    return {
        "experiment_id": experiment_id,
        "storm_threshold": spec.threshold,
        "horizon_hours": spec.horizon_hours,
        "selected_config_id": "lightgbm_lr0.1_leaves127",
        "selected_threshold": selected_threshold,
        "event_recall": n_detected_events / n_events,
        "n_events": n_events,
        "n_detected_events": n_detected_events,
        "n_alert_episodes": n_alert_episodes,
        "false_alarm_episodes": false_alarm_episodes,
        "valid_exposure_hours": valid_exposure_hours,
        "far_per_day": false_alarm_episodes / (valid_exposure_hours / 24.0),
        "fold_selected_thresholds": {
            "walk_forward_1": fold_1,
            "walk_forward_2": fold_2,
        },
        "stability_thresholds": list(stability),
        "protected_final_test_scored": False,
    }


def _all_summaries():
    return [_summary(experiment_id) for experiment_id in PHASE7_EXPERIMENT_IDS]


def test_build_summary_contains_all_six_frozen_experiments():
    table = build_experiment_summary(_all_summaries())
    assert tuple(table.columns) == PHASE7_ANALYSIS_COLUMNS
    assert table["experiment_id"].tolist() == list(PHASE7_EXPERIMENT_IDS)
    assert len(table) == 6
    assert table.loc[
        table["is_primary_control"], "experiment_id"
    ].tolist() == [PHASE7_PRIMARY_CONTROL_ID]


def test_horizon_comparison_is_t5_only_and_in_horizon_order():
    table = build_experiment_summary(_all_summaries())
    horizon = build_horizon_comparison(table)
    assert horizon["experiment_id"].tolist() == [
        "t5_h3",
        "t5_h6",
        "t5_h12",
        "t5_h24",
    ]
    assert horizon["storm_threshold"].eq(5.0).all()
    assert horizon["horizon_hours"].tolist() == [3, 6, 12, 24]


def test_severity_comparison_is_h6_only_and_in_threshold_order():
    table = build_experiment_summary(_all_summaries())
    severity = build_severity_comparison(table)
    assert severity["experiment_id"].tolist() == [
        "t5_h6",
        "t6_h6",
        "t7_h6",
    ]
    assert severity["horizon_hours"].eq(6).all()
    assert severity["storm_threshold"].tolist() == [5.0, 6.0, 7.0]


def test_rejects_recall_that_is_not_global_event_ratio():
    summaries = _all_summaries()
    summaries[0]["event_recall"] = 0.99
    with pytest.raises(ValueError, match="event recall does not equal"):
        build_experiment_summary(summaries)


def test_rejects_inconsistent_far_per_day():
    summaries = _all_summaries()
    summaries[0]["far_per_day"] = 0.01
    with pytest.raises(ValueError, match="FAR/day does not equal"):
        build_experiment_summary(summaries)


def test_rejects_far_above_frozen_limit():
    summaries = _all_summaries()
    summary = summaries[0]
    summary["false_alarm_episodes"] = 210
    summary["valid_exposure_hours"] = 24000
    summary["far_per_day"] = 210 / 1000.0
    with pytest.raises(ValueError, match="violates FAR/day limit"):
        build_experiment_summary(summaries)


def test_rejects_final_test_scoring():
    summaries = _all_summaries()
    summaries[0]["protected_final_test_scored"] = True
    with pytest.raises(ValueError, match="Final Test"):
        build_experiment_summary(summaries)


def test_rejects_registry_identity_drift():
    summaries = _all_summaries()
    summaries[0]["horizon_hours"] = 99
    with pytest.raises(ValueError, match="horizon differs from frozen registry"):
        build_experiment_summary(summaries)


def test_rejects_missing_or_duplicate_experiments():
    summaries = _all_summaries()
    with pytest.raises(ValueError, match="exactly the six frozen experiments"):
        build_experiment_summary(summaries[:-1])

    duplicated = summaries + [copy.deepcopy(summaries[-1])]
    with pytest.raises(ValueError, match="Duplicate Phase 7 summary"):
        build_experiment_summary(duplicated)


def test_fold_diagnostics_remain_separate_and_are_not_averaged():
    summaries = _all_summaries()
    summaries[0]["fold_selected_thresholds"] = {
        "walk_forward_1": 0.04,
        "walk_forward_2": 0.08,
    }
    table = build_experiment_summary(summaries)
    diagnostics = build_fold_diagnostics(table)
    rows = diagnostics.loc[diagnostics["experiment_id"].eq("t5_h3")]
    assert rows["fold"].tolist() == ["walk_forward_1", "walk_forward_2"]
    assert rows["selected_threshold"].tolist() == [0.04, 0.08]
    assert len(diagnostics) == 12


def test_target_prevalence_is_optional_but_validated_when_provided():
    prevalence = {
        experiment_id: 0.01 + index * 0.001
        for index, experiment_id in enumerate(PHASE7_EXPERIMENT_IDS)
    }
    table = build_experiment_summary(
        _all_summaries(), target_prevalence=prevalence
    )
    assert np.allclose(
        table["target_prevalence"].to_numpy(dtype=float),
        np.array([prevalence[e] for e in PHASE7_EXPERIMENT_IDS]),
    )

    bad = dict(prevalence)
    bad["t5_h3"] = 1.5
    with pytest.raises(ValueError, match="target prevalence must lie"):
        build_experiment_summary(_all_summaries(), target_prevalence=bad)


def test_validate_phase7_analysis_rejects_column_drift():
    table = build_experiment_summary(_all_summaries())
    bad = table.rename(columns={"event_recall": "recall"})
    with pytest.raises(ValueError, match="columns differ"):
        validate_phase7_analysis(bad)
