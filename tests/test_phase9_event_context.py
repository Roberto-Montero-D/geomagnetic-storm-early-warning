import pandas as pd
from src.final_test.diagnostics_recurrence import (
    event_context_table,
    clustered_vs_isolated_recall,
)


def _events():
    return pd.DataFrame({
        "event_id": [1, 2, 3],
        "start_time": pd.to_datetime([
            "2022-01-01 12:00",
            "2022-01-02 00:00",
            "2022-01-06 00:00",
        ]),
        "end_time": pd.to_datetime([
            "2022-01-01 15:00",
            "2022-01-02 03:00",
            "2022-01-06 03:00",
        ]),
    })


def _dataset():
    idx = pd.date_range("2021-12-31", "2022-01-06", freq="h")
    return pd.DataFrame({"kp_lag_1h": 2.0}, index=idx)


def _outcomes():
    return pd.DataFrame({
        "event_id": [1, 2, 3],
        "detected": [True, False, True],
        "year": [2022, 2022, 2022],
    })


def test_prior_event_counts_are_strictly_before_start():
    out = event_context_table(_events(), _outcomes(), _dataset())
    second = out.loc[out["event_id"] == 2].iloc[0]
    assert second["prior_events_24h"] == 1


def test_isolated_event_has_no_prior_event_72h():
    out = event_context_table(_events(), _outcomes(), _dataset())
    third = out.loc[out["event_id"] == 3].iloc[0]
    assert third["prior_events_72h"] == 0


def test_background_window_excludes_event_start():
    data = _dataset()
    data.loc[pd.Timestamp("2022-01-02 00:00"), "kp_lag_1h"] = 9.0
    out = event_context_table(_events(), _outcomes(), data)
    second = out.loc[out["event_id"] == 2].iloc[0]
    assert second["background_kp_max_12h"] == 2.0


def test_recurrence_strata_preserve_total_partition():
    out = event_context_table(_events(), _outcomes(), _dataset())
    strata = clustered_vs_isolated_recall(out).set_index("stratum")
    assert (
        strata.loc["isolated_72h", "n_events"]
        + strata.loc["recurrent_72h", "n_events"]
        == 3
    )
