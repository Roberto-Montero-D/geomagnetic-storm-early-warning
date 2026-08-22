import pandas as pd

from scripts.run_phase3_screening import write_phase3_screening_results
from src.feature_screening.screening import (
    Phase3ScreeningResult,
    ScreeningExperimentResult,
)


def _experiment(name, n_features, threshold):
    return ScreeningExperimentResult(
        experiment=name,
        n_features=n_features,
        threshold=threshold,
        event_recall=0.5,
        false_alarm_rate_per_day=0.15,
        pr_auc=0.2,
        operationally_feasible=True,
        validation_probability=pd.Series(dtype=float),
        threshold_table=pd.DataFrame(
            {
                "threshold": [0.01, threshold],
                "event_recall": [0.9, 0.5],
                "false_alarm_rate_per_day": [0.4, 0.15],
                "far_feasible": [False, True],
            }
        ),
    )


def _result():
    experiments = {
        "A": _experiment("A", 10, 0.07),
        "B": _experiment("B", 70, 0.08),
        "C": _experiment("C", 75, 0.08),
        "D": _experiment("D", 90, 0.09),
        "E": _experiment("E", 93, 0.09),
    }
    ranking = pd.DataFrame(
        {
            "experiment": ["C", "B", "D", "E", "A"],
            "n_features": [75, 70, 90, 93, 10],
            "threshold": [0.08, 0.08, 0.09, 0.09, 0.07],
            "event_recall": [0.7, 0.65, 0.6, 0.55, 0.5],
            "false_alarm_rate_per_day": [0.18, 0.17, 0.16, 0.15, 0.14],
            "pr_auc": [0.3, 0.29, 0.28, 0.27, 0.26],
            "operationally_feasible": [True] * 5,
        }
    )
    return Phase3ScreeningResult(
        experiments=experiments,
        ranking=ranking,
        advancing_experiments=("C", "B", "D"),
    )


def test_writer_creates_expected_screening_artifacts(tmp_path):
    write_phase3_screening_results(_result(), tmp_path)

    expected = {
        "screening_ranking.csv",
        "screening_advancing_experiments.csv",
        "screening_metrics.csv",
        "screening_a_threshold_curve.csv",
        "screening_b_threshold_curve.csv",
        "screening_c_threshold_curve.csv",
        "screening_d_threshold_curve.csv",
        "screening_e_threshold_curve.csv",
    }
    assert {path.name for path in tmp_path.glob("*.csv")} == expected


def test_advancing_artifact_preserves_frozen_ranking_order(tmp_path):
    write_phase3_screening_results(_result(), tmp_path)
    table = pd.read_csv(tmp_path / "screening_advancing_experiments.csv")
    assert table["rank"].tolist() == [1, 2, 3]
    assert table["experiment"].tolist() == ["C", "B", "D"]


def test_screening_artifacts_do_not_export_target_or_prediction_timeline(tmp_path):
    write_phase3_screening_results(_result(), tmp_path)

    forbidden = {
        "prediction_time",
        "target",
        "y_true",
        "final_test",
        "validation_probability",
    }
    for path in tmp_path.glob("*.csv"):
        table = pd.read_csv(path)
        assert forbidden.isdisjoint(table.columns)
