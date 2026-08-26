"""Tests for the Phase 6 executable artifact layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from scripts.run_phase6_threshold_selection import (
    write_phase6_artifacts,
)


def _oof_result():
    table = pd.DataFrame(
        {
            "probability": [0.2, 0.8],
            "target": [0, 1],
            "storm_id": pd.array(
                [pd.NA, 7],
                dtype="Int64",
            ),
            "fold": [
                "walk_forward_1",
                "walk_forward_2",
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2019-01-01 00:00",
                "2021-01-01 00:00",
            ],
            name="timestamp",
        ),
    )

    return SimpleNamespace(
        config_id=(
            "lightgbm_lr0.1_leaves127"
        ),
        table=table,
    )


def _threshold_result():
    global_table = pd.DataFrame(
        {
            "threshold": [0.50, 0.51],
            "event_recall": [0.8, 0.7],
            "false_alarm_rate_per_day": [
                0.20,
                0.18,
            ],
            "n_events": [10, 10],
            "n_detected_events": [8, 7],
            "n_alert_episodes": [12, 10],
            "n_false_alarm_episodes": [2, 1],
            "valid_exposure_hours": [
                240,
                240,
            ],
            "far_feasible": [True, True],
            "in_stability_region": [
                True,
                True,
            ],
        }
    )

    fold_table = pd.DataFrame(
        {
            "fold": [
                "walk_forward_1",
                "walk_forward_2",
            ],
            "threshold": [0.50, 0.50],
            "event_recall": [0.8, 0.8],
            "false_alarm_rate_per_day": [
                0.2,
                0.2,
            ],
            "n_events": [5, 5],
            "n_detected_events": [4, 4],
            "n_alert_episodes": [6, 6],
            "n_false_alarm_episodes": [1, 1],
            "valid_exposure_hours": [
                120,
                120,
            ],
            "far_feasible": [True, True],
            "in_stability_region": [
                True,
                True,
            ],
        }
    )

    return SimpleNamespace(
        selected_threshold=0.50,
        global_threshold_table=global_table,
        fold_threshold_table=fold_table,
        fold_selected_thresholds={
            "walk_forward_1": 0.48,
            "walk_forward_2": None,
        },
        stability_thresholds=(
            0.50,
            0.51,
        ),
    )


def test_writer_emits_complete_phase6_artifact_set(
    tmp_path,
):
    write_phase6_artifacts(
        _oof_result(),
        _threshold_result(),
        tmp_path,
    )

    expected = {
        "oof_predictions.csv",
        "global_threshold_curve.csv",
        "fold_threshold_curves.csv",
        "fold_selected_thresholds.csv",
        "stability_thresholds.csv",
        "phase6_selection_summary.json",
    }

    assert {
        path.name
        for path in tmp_path.iterdir()
    } == expected


def test_written_oof_preserves_timestamp_and_contract(
    tmp_path,
):
    write_phase6_artifacts(
        _oof_result(),
        _threshold_result(),
        tmp_path,
    )

    frame = pd.read_csv(
        tmp_path
        / "oof_predictions.csv"
    )

    assert list(frame.columns) == [
        "timestamp",
        "probability",
        "target",
        "storm_id",
        "fold",
    ]

    assert frame["fold"].tolist() == [
        "walk_forward_1",
        "walk_forward_2",
    ]


def test_summary_records_frozen_selection_and_test_isolation(
    tmp_path,
):
    write_phase6_artifacts(
        _oof_result(),
        _threshold_result(),
        tmp_path,
    )

    with (
        tmp_path
        / "phase6_selection_summary.json"
    ).open(
        encoding="utf-8",
    ) as handle:
        summary = json.load(
            handle
        )

    assert summary[
        "selected_config_id"
    ] == "lightgbm_lr0.1_leaves127"

    assert summary[
        "selected_threshold"
    ] == 0.50

    assert summary[
        "max_far_per_day"
    ] == 0.2

    assert summary[
        "stability_min_far_per_day"
    ] == 0.15

    assert summary[
        "protected_final_test_scored"
    ] is False

    assert summary[
        "oof_folds"
    ] == [
        "walk_forward_1",
        "walk_forward_2",
    ]


def test_summary_serializes_missing_fold_threshold_as_null(
    tmp_path,
):
    write_phase6_artifacts(
        _oof_result(),
        _threshold_result(),
        tmp_path,
    )

    with (
        tmp_path
        / "phase6_selection_summary.json"
    ).open(
        encoding="utf-8",
    ) as handle:
        summary = json.load(
            handle
        )

    assert (
        summary[
            "fold_selected_thresholds"
        ][
            "walk_forward_2"
        ]
        is None
    )
