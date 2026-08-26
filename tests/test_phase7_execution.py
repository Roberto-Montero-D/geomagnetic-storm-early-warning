from scripts.run_phase7_experiments import (
    PHASE7_EXECUTION_EXPERIMENTS,
)
from src.phase7.contract import (
    PHASE7_PRIMARY_CONTROL_ID,
)
import pandas as pd
from scripts.run_phase7_experiments import _selected_row
def test_phase7_execution_set_is_frozen_and_excludes_control():
    ids = tuple(
        experiment.experiment_id
        for experiment in PHASE7_EXECUTION_EXPERIMENTS
    )

    assert ids == (
        "t5_h3",
        "t5_h12",
        "t5_h24",
        "t6_h6",
        "t7_h6",
    )

    assert PHASE7_PRIMARY_CONTROL_ID not in ids

class _ThresholdResult:
    selected_threshold = 0.10
    global_threshold_table = pd.DataFrame(
        {
            "threshold": [0.10],
            "event_recall": [0.75],
            "false_alarm_rate_per_day": [0.19],
            "n_events": [20],
            "n_detected_events": [15],
            "n_alert_episodes": [30],
            "n_false_alarm_episodes": [10],
            "valid_exposure_hours": [1263],
            "far_feasible": [True],
            "in_stability_region": [True],
        }
    )


def test_selected_row_uses_canonical_phase6_schema():
    row = _selected_row(_ThresholdResult())

    assert row["event_recall"] == 0.75
    assert row["n_false_alarm_episodes"] == 10
    assert row["valid_exposure_hours"] == 1263
    assert row["false_alarm_rate_per_day"] == 0.19