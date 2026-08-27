"""Tests for Phase 9.4 pre-onset physical-state diagnostics."""
import numpy as np
import pandas as pd

from src.final_test.diagnostics_physical_state import (
    PHYSICAL_FEATURES,
    conditional_2025_feature_shift,
    conditional_recall_summary,
    pre_onset_physical_state_table,
)


def _dataset():
    idx = pd.date_range(
        "2024-12-31 12:00",
        periods=48,
        freq="h",
        name="prediction_time",
    )
    data = {}
    for i, feature in enumerate(PHYSICAL_FEATURES):
        data[feature] = np.arange(len(idx), dtype=float) + i
    return pd.DataFrame(data, index=idx)


def _context():
    return pd.DataFrame(
        {
            "event_id": [1, 2],
            "start_time": [
                pd.Timestamp("2024-12-31 23:00"),
                pd.Timestamp("2025-01-01 16:00"),
            ],
            "year": [2024, 2025],
            "detected": [True, False],
            "prior_events_72h": [0, 1],
            "background_kp_active_hours_12h": [0, 2],
        }
    )


def test_pre_onset_window_excludes_event_onset():
    dataset = _dataset()
    out = pre_onset_physical_state_table(_context(), dataset)

    event = out.loc[out["event_id"] == 2].iloc[0]
    start = pd.Timestamp("2025-01-01 16:00")

    assert event["pre_onset_prediction_rows"] == 6
    assert event["speed__last"] == dataset.at[
        start - pd.Timedelta(hours=1), "speed"
    ]
    assert event["speed__last"] != dataset.at[start, "speed"]


def test_context_strata_and_comparison_period_are_deterministic():
    out = pre_onset_physical_state_table(_context(), _dataset())

    first = out.loc[out["event_id"] == 1].iloc[0]
    second = out.loc[out["event_id"] == 2].iloc[0]

    assert first["context_stratum"] == "isolated_quiet"
    assert first["comparison_period"] == "2022_2024"
    assert second["context_stratum"] == "recurrent_active"
    assert second["comparison_period"] == "2025"


def test_conditional_recall_uses_event_counts_only():
    state = pre_onset_physical_state_table(_context(), _dataset())
    summary = conditional_recall_summary(state)

    prior = summary.loc[
        (summary["context_stratum"] == "all_events")
        & (summary["comparison_period"] == "2022_2024")
    ].iloc[0]
    current = summary.loc[
        (summary["context_stratum"] == "all_events")
        & (summary["comparison_period"] == "2025")
    ].iloc[0]

    assert prior["n_events"] == 1
    assert prior["n_detected"] == 1
    assert prior["event_recall"] == 1.0
    assert current["n_events"] == 1
    assert current["n_detected"] == 0
    assert current["event_recall"] == 0.0


def test_feature_shift_is_descriptive_and_period_scoped():
    state = pre_onset_physical_state_table(_context(), _dataset())
    shift = conditional_2025_feature_shift(state)

    row = shift.loc[
        (shift["context_stratum"] == "all_events")
        & (shift["metric"] == "speed__last")
    ].iloc[0]

    assert row["n_2022_2024"] == 1
    assert row["n_2025"] == 1
    assert row["median_2025"] > row["median_2022_2024"]
