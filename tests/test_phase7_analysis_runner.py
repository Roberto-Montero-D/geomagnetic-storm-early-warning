import json

import pandas as pd
import pytest

from scripts.run_phase7_analysis import (
    NON_CONTROL_IDS,
    build_phase6_control_summary,
    load_phase7_non_control_summaries,
    load_target_prevalence,
    run_phase7_analysis,
)
from src.phase7.contract import (
    PHASE7_EXPERIMENT_IDS,
    PHASE7_MODEL_CONFIG_ID,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _phase6_artifacts(root):
    root.mkdir(parents=True, exist_ok=True)

    _write_json(
        root / "phase6_selection_summary.json",
        {
            "selected_config_id": PHASE7_MODEL_CONFIG_ID,
            "selected_threshold": 0.10,
            "fold_selected_thresholds": {
                "walk_forward_1": 0.07,
                "walk_forward_2": 0.16,
            },
            "stability_thresholds": [0.10, 0.11, 0.12, 0.13],
            "protected_final_test_scored": False,
        },
    )

    pd.DataFrame(
        {
            "threshold": [0.09, 0.10],
            "event_recall": [20 / 31, 21 / 31],
            "n_events": [31, 31],
            "n_detected_events": [20, 21],
            "n_alert_episodes": [220, 222],
            "n_false_alarm_episodes": [201, 200],
            "valid_exposure_hours": [25873, 25873],
            "false_alarm_rate_per_day": [
                201 / (25873 / 24.0),
                200 / (25873 / 24.0),
            ],
        }
    ).to_csv(
        root / "global_threshold_curve.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01",
                periods=4,
                freq="h",
            ),
            "probability": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 0, 1, 1],
            "storm_id": [None, None, 1, 1],
            "fold": ["walk_forward_1"] * 4,
        }
    ).to_csv(
        root / "oof_predictions.csv",
        index=False,
    )


def _non_control_summary(experiment_id, index):
    spec = {
        "t5_h3": (5.0, 3),
        "t5_h12": (5.0, 12),
        "t5_h24": (5.0, 24),
        "t6_h6": (6.0, 6),
        "t7_h6": (7.0, 6),
    }[experiment_id]

    n_events = 10 + index
    n_detected = 8 + index
    false_alarms = 190 + index
    exposure = 25873

    return {
        "experiment_id": experiment_id,
        "storm_threshold": spec[0],
        "horizon_hours": spec[1],
        "selected_config_id": PHASE7_MODEL_CONFIG_ID,
        "selected_threshold": 0.05 + index * 0.01,
        "event_recall": n_detected / n_events,
        "n_events": n_events,
        "n_detected_events": n_detected,
        "n_alert_episodes": 210 + index,
        "false_alarm_episodes": false_alarms,
        "valid_exposure_hours": exposure,
        "far_per_day": false_alarms / (exposure / 24.0),
        "fold_selected_thresholds": {
            "walk_forward_1": 0.04 + index * 0.01,
            "walk_forward_2": 0.08 + index * 0.01,
        },
        "stability_thresholds": [
            0.05 + index * 0.01,
        ],
        "protected_final_test_scored": False,
    }


def _phase7_artifacts(root):
    summaries = [
        _non_control_summary(experiment_id, index)
        for index, experiment_id in enumerate(NON_CONTROL_IDS)
    ]

    _write_json(
        root / "phase7_execution_summary.json",
        {
            "selected_config_id": PHASE7_MODEL_CONFIG_ID,
            "executed_experiments": list(NON_CONTROL_IDS),
            "primary_control_executed": False,
            "protected_final_test_scored": False,
            "experiments": summaries,
        },
    )

    for index, summary in enumerate(summaries):
        experiment_id = summary["experiment_id"]
        experiment_dir = root / experiment_id

        _write_json(
            experiment_dir / "selection_summary.json",
            summary,
        )

        target = [0, 1, 0, 1]
        if index % 2:
            target = [0, 0, 0, 1]

        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01",
                    periods=4,
                    freq="h",
                ),
                "probability": [0.1, 0.2, 0.3, 0.4],
                "target": target,
                "storm_id": [None, 1, None, 2],
                "fold": ["walk_forward_1"] * 4,
            }
        ).to_csv(
            experiment_dir / "oof_predictions.csv",
            index=False,
        )

    return summaries


def test_phase6_control_is_converted_to_phase7_schema(tmp_path):
    phase6 = tmp_path / "phase6"
    _phase6_artifacts(phase6)

    result = build_phase6_control_summary(phase6)

    assert result["experiment_id"] == "t5_h6"
    assert result["storm_threshold"] == 5.0
    assert result["horizon_hours"] == 6
    assert result["selected_threshold"] == 0.10
    assert result["n_detected_events"] == 21
    assert result["n_events"] == 31
    assert result["false_alarm_episodes"] == 200
    assert result["protected_final_test_scored"] is False


def test_non_control_loader_requires_individual_and_aggregate_agreement(
    tmp_path,
):
    phase7 = tmp_path / "phase7"
    _phase7_artifacts(phase7)

    loaded = load_phase7_non_control_summaries(phase7)

    assert [item["experiment_id"] for item in loaded] == list(
        NON_CONTROL_IDS
    )

    path = (
        phase7
        / NON_CONTROL_IDS[0]
        / "selection_summary.json"
    )
    individual = json.loads(path.read_text(encoding="utf-8"))
    individual["selected_threshold"] = 0.99
    _write_json(path, individual)

    with pytest.raises(
        ValueError,
        match="aggregate and individual summaries differ",
    ):
        load_phase7_non_control_summaries(phase7)


def test_target_prevalence_reads_all_six_oof_artifacts(tmp_path):
    phase6 = tmp_path / "phase6"
    phase7 = tmp_path / "phase7"
    _phase6_artifacts(phase6)
    _phase7_artifacts(phase7)

    prevalence = load_target_prevalence(
        phase6,
        phase7,
    )

    assert set(prevalence) == set(PHASE7_EXPERIMENT_IDS)
    assert prevalence["t5_h6"] == 0.5
    assert prevalence["t5_h3"] == 0.5
    assert prevalence["t5_h12"] == 0.25


def test_analysis_runner_writes_controlled_comparison_artifacts(
    tmp_path,
):
    phase6 = tmp_path / "phase6"
    phase7 = tmp_path / "phase7"
    output = tmp_path / "analysis"

    _phase6_artifacts(phase6)
    _phase7_artifacts(phase7)

    summary = run_phase7_analysis(
        phase6,
        phase7,
        output,
    )

    assert summary["experiment_id"].tolist() == list(
        PHASE7_EXPERIMENT_IDS
    )

    assert (
        output / "experiment_summary.csv"
    ).is_file()
    assert (
        output / "horizon_comparison.csv"
    ).is_file()
    assert (
        output / "severity_comparison.csv"
    ).is_file()
    assert (
        output / "fold_threshold_diagnostics.csv"
    ).is_file()
    assert (
        output / "phase7_analysis_summary.json"
    ).is_file()

    horizon = pd.read_csv(
        output / "horizon_comparison.csv"
    )
    severity = pd.read_csv(
        output / "severity_comparison.csv"
    )

    assert horizon["experiment_id"].tolist() == [
        "t5_h3",
        "t5_h6",
        "t5_h12",
        "t5_h24",
    ]
    assert severity["experiment_id"].tolist() == [
        "t5_h6",
        "t6_h6",
        "t7_h6",
    ]

    payload = json.loads(
        (
            output / "phase7_analysis_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["protected_final_test_scored"] is False
    assert (
        payload["comparison_policy"]["cross_task_ranking_authorized"]
        is False
    )


def test_rejects_any_final_test_flag_in_source_artifacts(tmp_path):
    phase6 = tmp_path / "phase6"
    phase7 = tmp_path / "phase7"

    _phase6_artifacts(phase6)
    _phase7_artifacts(phase7)

    path = phase7 / "phase7_execution_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protected_final_test_scored"] = True
    _write_json(path, payload)

    with pytest.raises(
        ValueError,
        match="Final Test isolation",
    ):
        load_phase7_non_control_summaries(phase7)
