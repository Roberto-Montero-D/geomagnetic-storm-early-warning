import pandas as pd

from src.final_test.diagnostics_operational import (
    event_outcomes,
    yearly_operational_decomposition,
)


def _events():
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "start_time": pd.to_datetime(
                [
                    "2022-01-02 00:00",
                    "2022-06-01 00:00",
                    "2024-01-02 00:00",
                ]
            ),
            "end_time": pd.to_datetime(
                [
                    "2022-01-02 03:00",
                    "2022-06-01 03:00",
                    "2024-01-02 03:00",
                ]
            ),
            "boundary_status": ["complete"] * 3,
        }
    )


def _episodes():
    return pd.DataFrame(
        {
            "first_alert_time": pd.to_datetime(
                [
                    "2022-01-01 22:00",
                    "2022-03-01 00:00",
                    "2024-01-01 22:00",
                    "2024-02-01 00:00",
                ]
            ),
            "classification": [
                "early_detection",
                "false_alarm",
                "early_detection",
                "false_alarm",
            ],
            "associated_event_id": [1.0, pd.NA, 3.0, pd.NA],
            "lead_time": [
                pd.Timedelta(hours=2),
                pd.NaT,
                pd.Timedelta(hours=2),
                pd.NaT,
            ],
        }
    )


def test_yearly_event_recall_uses_unique_events():
    result = yearly_operational_decomposition(_events(), _episodes())
    row = result.loc[result["year"] == 2022].iloc[0]

    assert row["n_events"] == 2
    assert row["n_detected_events"] == 1
    assert row["n_missed_events"] == 1
    assert row["event_recall"] == 0.5


def test_far_per_day_uses_exact_calendar_exposure():
    result = yearly_operational_decomposition(_events(), _episodes())

    y2022 = result.loc[result["year"] == 2022].iloc[0]
    y2024 = result.loc[result["year"] == 2024].iloc[0]

    assert y2022["exposure_days"] == 365.0
    assert y2022["far_per_day"] == 1 / 365
    assert y2024["exposure_days"] == 366.0
    assert y2024["far_per_day"] == 1 / 366


def test_event_outcomes_has_one_row_per_event():
    result = event_outcomes(_events(), _episodes())

    assert len(result) == 3
    assert result["event_id"].is_unique
    assert int(result["detected"].sum()) == 2


def test_missed_event_has_no_detection_time():
    result = event_outcomes(_events(), _episodes())
    missed = result.loc[result["event_id"] == 2].iloc[0]

    assert not missed["detected"]
    assert pd.isna(missed["first_detection_time"])


def test_early_event_records_lead_time():
    result = event_outcomes(_events(), _episodes())
    detected = result.loc[result["event_id"] == 1].iloc[0]

    assert detected["early_detected"]
    assert detected["max_early_lead_hours"] == 2.0
