import inspect

import numpy as np
import pandas as pd

import src.final_test.diagnostics as diagnostics


def _predictions():
    index = pd.DatetimeIndex(
        [
            "2022-01-01 00:00",
            "2022-01-01 01:00",
            "2023-01-01 00:00",
            "2023-01-01 01:00",
        ],
        name="prediction_time",
    )
    return pd.DataFrame(
        {
            "probability": [0.1, 0.8, 0.2, 0.7],
            "target": [0.0, 1.0, 0.0, 1.0],
        },
        index=index,
    )


def _episodes():
    return pd.DataFrame(
        {
            "first_alert_time": pd.to_datetime(
                [
                    "2022-01-01 00:00",
                    "2022-02-01 00:00",
                    "2023-01-01 00:00",
                ]
            ),
            "classification": [
                "false_alarm",
                "early_detection",
                "late_detection",
            ],
            "lead_time": [
                pd.NaT,
                pd.Timedelta(hours=3),
                pd.NaT,
            ],
        }
    )


def test_yearly_probability_diagnostics_are_year_scoped():
    result = diagnostics.yearly_probability_diagnostics(
        _predictions()
    )

    assert result["year"].tolist() == [2022, 2023]
    assert result["known_target_rows"].tolist() == [2, 2]


def test_calibration_table_uses_fixed_ten_bins():
    result = diagnostics.calibration_table(
        _predictions()
    )

    assert len(result) == 10
    assert result["bin_id"].tolist() == list(range(10))


def test_episode_diagnostics_preserve_classifications():
    result = diagnostics.yearly_episode_diagnostics(
        _episodes()
    )

    row_2022 = result.loc[
        result["year"] == 2022
    ].iloc[0]

    assert row_2022["alert_episodes"] == 2
    assert row_2022["false_alarm_episodes"] == 1
    assert row_2022["early_detection_episodes"] == 1


def test_lead_summary_uses_early_detections_only():
    result = diagnostics.lead_time_summary(
        _episodes()
    )

    assert result.iloc[0]["n"] == 1
    assert result.iloc[0]["median_hours"] == 3.0


def test_diagnostics_module_has_no_threshold_search_symbols():
    forbidden = {
        "optimize_phase6_threshold",
        "DEFAULT_THRESHOLD_GRID",
        "threshold_selection",
        "make_phase5_model_by_id",
        "fit",
        "predict_proba",
    }

    for name in forbidden:
        assert not hasattr(diagnostics, name)


def test_public_diagnostics_accept_no_threshold_argument():
    for function in (
        diagnostics.yearly_probability_diagnostics,
        diagnostics.calibration_table,
        diagnostics.yearly_episode_diagnostics,
        diagnostics.lead_time_summary,
    ):
        assert "threshold" not in inspect.signature(
            function
        ).parameters
