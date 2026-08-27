"""Tests for Phase 9.5 signal-timing diagnostics."""
import numpy as np
import pandas as pd

from src.final_test.diagnostics_signal_timing import (
    onset_aligned_event_trajectory,
    timing_window_contrast,
)


def _dataset():
    idx = pd.date_range(
        "2025-01-01 00:00",
        periods=24,
        freq="h",
        name="prediction_time",
    )
    return pd.DataFrame(
        {
            "bz_gsm": np.arange(24, dtype=float),
            "bt": np.arange(24, dtype=float) + 10,
            "speed": np.arange(24, dtype=float) + 400,
            "flow_pressure": np.arange(24, dtype=float) + 2,
            "kp_lag_1h": np.arange(24, dtype=float) / 10,
            "bz_neg_x_speed": np.arange(24, dtype=float) * 2,
        },
        index=idx,
    )


def _context():
    return pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [
                pd.Timestamp("2025-01-01 16:00")
            ],
            "year": [2025],
            "detected": [True],
            "prior_events_72h": [0],
            "background_kp_active_hours_12h": [0],
        }
    )


def _predictions():
    idx = pd.date_range(
        "2025-01-01 00:00",
        periods=24,
        freq="h",
        name="prediction_time",
    )
    return pd.DataFrame(
        {
            "probability": np.linspace(
                0.0, 1.0, len(idx)
            )
        },
        index=idx,
    )


def test_trajectory_is_strictly_pre_onset():
    out = onset_aligned_event_trajectory(
        _context(),
        _dataset(),
        _predictions(),
    )

    assert out["relative_hour"].min() == -12
    assert out["relative_hour"].max() == -1
    assert len(out) == 12
    assert (
        out["prediction_time"] < out["storm_start"]
    ).all()


def test_trajectory_uses_exact_prediction_timestamp():
    out = onset_aligned_event_trajectory(
        _context(),
        _dataset(),
        _predictions(),
    )

    row = out.loc[out["relative_hour"] == -1].iloc[0]
    expected_time = pd.Timestamp("2025-01-01 15:00")

    assert row["prediction_time"] == expected_time
    assert row["speed"] == _dataset().at[
        expected_time, "speed"
    ]
    assert row["probability"] == _predictions().at[
        expected_time, "probability"
    ]


def test_window_contrast_uses_predeclared_halves():
    trajectory = onset_aligned_event_trajectory(
        _context(),
        _dataset(),
        _predictions(),
    )
    contrast = timing_window_contrast(trajectory)

    speed = contrast.loc[
        contrast["metric"] == "speed"
    ].iloc[0]

    assert speed["n_early"] == 6
    assert speed["n_warning"] == 6
    assert speed["median_warning"] > speed["median_early"]
