from pathlib import Path
import pandas as pd

from scripts.run_phase2_baselines import write_phase2_results
from src.evaluation.cross_fold import CrossFoldEvaluation


def _result():
    metrics=pd.DataFrame({
        "fold":["screening","walk_forward_1"],
        "baseline":["B0_persistence","B2_logistic"],
        "threshold":[0.5,0.7],
        "event_recall":[0.4,0.5],
        "false_alarm_rate_per_day":[0.1,0.2],
        "median_lead_time":[pd.Timedelta(hours=2),pd.Timedelta(hours=3)],
        "n_alert_episodes":[2,3],
        "n_false_alarm_episodes":[1,1],
        "n_early_detections":[1,2],
        "n_late_detections":[0,0],
        "valid_exposure_hours":[240,240],
    })
    return CrossFoldEvaluation(
        selected_thresholds={
            "B0_persistence":0.5,
            "B1_physical":0.5,
            "B2_logistic":0.7,
            "B3_extratrees":0.8,
        },
        fold_metrics=metrics,
        threshold_tables={
            "B2_logistic":pd.DataFrame({
                "threshold":[0.6,0.7],
                "false_alarm_rate_per_day":[0.3,0.2],
                "far_feasible":[False,True],
            }),
            "B3_extratrees":pd.DataFrame({
                "threshold":[0.7,0.8],
                "false_alarm_rate_per_day":[0.4,0.1],
                "far_feasible":[False,True],
            }),
        },
    )


def test_writer_creates_expected_development_artifacts(tmp_path):
    write_phase2_results(_result(),tmp_path)
    assert (tmp_path/"baseline_fold_metrics.csv").exists()
    assert (tmp_path/"baseline_selected_thresholds.csv").exists()
    assert (tmp_path/"b2_logistic_threshold_curve.csv").exists()
    assert (tmp_path/"b3_extratrees_threshold_curve.csv").exists()


def test_threshold_artifact_contains_all_four_baselines(tmp_path):
    write_phase2_results(_result(),tmp_path)
    table=pd.read_csv(tmp_path/"baseline_selected_thresholds.csv")
    assert table["baseline"].tolist()==[
        "B0_persistence","B1_physical","B2_logistic","B3_extratrees"
    ]


def test_writer_does_not_export_prediction_timestamps_or_targets(tmp_path):
    write_phase2_results(_result(),tmp_path)
    for path in tmp_path.glob("*.csv"):
        table=pd.read_csv(path)
        forbidden={"prediction_time","target","y_true","final_test"}
        assert forbidden.isdisjoint(table.columns)
