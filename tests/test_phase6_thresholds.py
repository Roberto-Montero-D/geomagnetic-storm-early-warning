"""Tests for fold-aware Phase 6 threshold optimization."""

from __future__ import annotations

import pandas as pd
import pytest

import src.evaluation.phase6_thresholds as phase6


def _oof(
    fold_a_probability,
    fold_b_probability,
):
    a_index = pd.date_range(
        "2019-01-01 00:00",
        periods=len(fold_a_probability),
        freq="h",
    )
    b_index = pd.date_range(
        "2021-01-01 00:00",
        periods=len(fold_b_probability),
        freq="h",
    )

    index = a_index.append(b_index).rename("timestamp")

    return pd.DataFrame(
        {
            "probability": (
                list(fold_a_probability)
                + list(fold_b_probability)
            ),
            "target": [0] * len(index),
            "storm_id": pd.array(
                [pd.NA] * len(index),
                dtype="Int64",
            ),
            "fold": (
                ["walk_forward_1"] * len(a_index)
                + ["walk_forward_2"] * len(b_index)
            ),
        },
        index=index,
    )


def _events(rows=None):
    if rows is None:
        rows = []

    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "start_time",
            "end_time",
            "boundary_status",
        ],
    )


def test_alert_episodes_cannot_bridge_fold_boundaries():
    oof = _oof(
        [0.9],
        [0.9],
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        _events(),
        thresholds=[0.5],
        max_far_per_day=100.0,
        stability_min_far_per_day=0.0,
        cooldown_hours=3,
    )

    row = result.global_threshold_table.iloc[0]

    assert row["n_alert_episodes"] == 2
    assert row["n_false_alarm_episodes"] == 2
    assert row["valid_exposure_hours"] == 2
    assert row["false_alarm_rate_per_day"] == pytest.approx(24.0)


def test_global_far_uses_total_false_alarms_over_total_exposure():
    oof = _oof(
        [0.9] + [0.0] * 23,
        [0.0] * 24,
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        _events(),
        thresholds=[0.5],
        max_far_per_day=1.0,
        stability_min_far_per_day=0.0,
    )

    row = result.global_threshold_table.iloc[0]

    assert row["n_false_alarm_episodes"] == 1
    assert row["valid_exposure_hours"] == 48
    assert row["false_alarm_rate_per_day"] == pytest.approx(0.5)


def test_global_event_recall_is_event_weighted_not_fold_mean():
    oof = _oof(
        [0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0],
    )

    events = _events(
        [
            (
                1,
                pd.Timestamp("2019-01-01 00:00"),
                pd.Timestamp("2019-01-01 00:00"),
                "complete",
            ),
            (
                2,
                pd.Timestamp("2021-01-01 00:00"),
                pd.Timestamp("2021-01-01 00:00"),
                "complete",
            ),
            (
                3,
                pd.Timestamp("2021-01-01 02:00"),
                pd.Timestamp("2021-01-01 02:00"),
                "complete",
            ),
        ]
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        events,
        thresholds=[0.5],
        max_far_per_day=100.0,
        stability_min_far_per_day=0.0,
    )

    row = result.global_threshold_table.iloc[0]

    assert row["n_events"] == 3
    assert row["n_detected_events"] == 2
    assert row["event_recall"] == pytest.approx(2.0 / 3.0)


def test_selects_lowest_global_far_feasible_threshold():
    # The two above-threshold predictions in WF1 are deliberately separated
    # by more than the 1-hour cooldown. At tau=0.3 they therefore form two
    # false-alarm episodes; at tau>=0.5 only the 0.9 prediction remains.
    oof = _oof(
        [0.9, 0.0, 0.4] + [0.0] * 21,
        [0.0] * 24,
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        _events(),
        thresholds=[0.3, 0.5, 0.8],
        max_far_per_day=0.5,
        stability_min_far_per_day=0.0,
        cooldown_hours=1,
    )

    assert result.selected_threshold == 0.5

    table = (
        result.global_threshold_table
        .set_index("threshold")
    )

    assert (
        table.loc[
            0.3,
            "false_alarm_rate_per_day",
        ]
        == pytest.approx(1.0)
    )

    assert (
        table.loc[
            0.5,
            "false_alarm_rate_per_day",
        ]
        == pytest.approx(0.5)
    )

    assert not bool(
        table.loc[
            0.3,
            "far_feasible",
        ]
    )

    assert bool(
        table.loc[
            0.5,
            "far_feasible",
        ]
    )


def test_reports_fold_specific_thresholds_as_diagnostics():
    oof = _oof(
        [0.9] + [0.0] * 23,
        [0.6] + [0.0] * 23,
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        _events(),
        thresholds=[0.5, 0.7],
        max_far_per_day=0.0,
        stability_min_far_per_day=0.0,
    )

    assert result.fold_selected_thresholds == {
        "walk_forward_1": None,
        "walk_forward_2": 0.7,
    }


def test_stability_region_is_inclusive():
    oof = _oof(
        [0.9] + [0.0] * 119,
        [0.0] * 120,
    )

    result = phase6.optimize_phase6_threshold(
        oof,
        _events(),
        thresholds=[0.5, 0.95],
        max_far_per_day=0.2,
        stability_min_far_per_day=0.1,
    )

    table = result.global_threshold_table.set_index("threshold")

    assert table.loc[0.5, "false_alarm_rate_per_day"] == pytest.approx(0.1)
    assert bool(table.loc[0.5, "in_stability_region"])
    assert not bool(table.loc[0.95, "in_stability_region"])
    assert result.stability_thresholds == (0.5,)


def test_rejects_non_contract_oof_columns():
    oof = _oof([0.1], [0.2]).drop(columns=["storm_id"])

    with pytest.raises(
        ValueError,
        match="frozen Phase 6 OOF contract",
    ):
        phase6.optimize_phase6_threshold(
            oof,
            _events(),
            thresholds=[0.5],
        )
