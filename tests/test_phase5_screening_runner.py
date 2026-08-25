from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.run_phase5_screening import (
    write_phase5_screening_results,
)
from src.model_selection.contract import (
    PHASE5_CONFIGURATIONS,
)


def _result():
    configurations = {}

    for i, config in enumerate(
        PHASE5_CONFIGURATIONS,
        start=1,
    ):
        configurations[config.config_id] = SimpleNamespace(
            threshold=0.10 + i * 0.001,
            event_recall=0.5,
            false_alarm_rate_per_day=0.19,
            pr_auc=0.2,
            operationally_feasible=True,
            threshold_table=pd.DataFrame(
                {
                    "threshold": [0.10, 0.20],
                    "event_recall": [0.7, 0.5],
                    "false_alarm_rate_per_day": [0.25, 0.19],
                    "far_feasible": [False, True],
                }
            ),
        )

    family_rankings = {}
    advancing = []

    for family in (
        "extratrees",
        "lightgbm",
        "xgboost",
    ):
        configs = [
            config
            for config in PHASE5_CONFIGURATIONS
            if config.family == family
        ]

        ranking = pd.DataFrame(
            {
                "config_id": [
                    config.config_id
                    for config in configs
                ],
                "family": [family] * len(configs),
                "threshold": [0.2] * len(configs),
                "event_recall": [0.5] * len(configs),
                "false_alarm_rate_per_day":
                    [0.19] * len(configs),
                "pr_auc": [0.2] * len(configs),
                "operationally_feasible":
                    [True] * len(configs),
            }
        )

        family_rankings[family] = ranking
        advancing.append(configs[0].config_id)

    return SimpleNamespace(
        configurations=configurations,
        family_rankings=family_rankings,
        advancing_configurations=tuple(advancing),
    )


def test_writer_emits_complete_phase5_artifact_set(
    tmp_path: Path,
):
    result = _result()
    write_phase5_screening_results(
        result,
        tmp_path,
    )

    assert (
        tmp_path / "dependency_versions.csv"
    ).exists()
    assert (
        tmp_path / "screening_metrics.csv"
    ).exists()
    assert (
        tmp_path
        / "screening_advancing_configurations.csv"
    ).exists()

    for family in (
        "extratrees",
        "lightgbm",
        "xgboost",
    ):
        assert (
            tmp_path
            / f"screening_ranking_{family}.csv"
        ).exists()

    for config in PHASE5_CONFIGURATIONS:
        assert (
            tmp_path
            / (
                "screening_"
                f"{config.config_id}"
                "_threshold_curve.csv"
            )
        ).exists()

    # 27 threshold curves
    # + 3 family rankings
    # + metrics
    # + advancing
    # + dependency versions
    assert len(list(tmp_path.glob("*.csv"))) == 33


def test_metrics_contains_every_config_once(
    tmp_path: Path,
):
    result = _result()
    write_phase5_screening_results(
        result,
        tmp_path,
    )

    metrics = pd.read_csv(
        tmp_path / "screening_metrics.csv"
    )

    assert len(metrics) == 27
    assert metrics["config_id"].is_unique
    assert tuple(metrics["config_id"]) == tuple(
        config.config_id
        for config in PHASE5_CONFIGURATIONS
    )


def test_advancement_has_one_winner_per_family(
    tmp_path: Path,
):
    result = _result()
    write_phase5_screening_results(
        result,
        tmp_path,
    )

    table = pd.read_csv(
        tmp_path
        / "screening_advancing_configurations.csv"
    )

    assert len(table) == 3
    assert tuple(table["family"]) == (
        "extratrees",
        "lightgbm",
        "xgboost",
    )
    assert tuple(table["rank"]) == (1, 2, 3)


def test_dependency_versions_are_recorded(
    tmp_path: Path,
):
    result = _result()
    write_phase5_screening_results(
        result,
        tmp_path,
    )

    table = pd.read_csv(
        tmp_path / "dependency_versions.csv"
    )

    assert set(table["package"]) == {
        "scikit-learn",
        "lightgbm",
        "xgboost",
    }
    assert table["version"].notna().all()


def test_writer_does_not_export_raw_predictions(
    tmp_path: Path,
):
    result = _result()
    write_phase5_screening_results(
        result,
        tmp_path,
    )

    names = {
        path.name.lower()
        for path in tmp_path.iterdir()
    }

    assert not any(
        "probability" in name
        for name in names
    )
    assert not any(
        "prediction" in name
        for name in names
    )
