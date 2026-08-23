from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from scripts.run_phase4_screening import write_phase4_screening_results
from src.imbalance.contract import PHASE4_EXPERIMENT_NAMES


def _result():
    experiments={}
    for i,name in enumerate(PHASE4_EXPERIMENT_NAMES):
        experiments[name]=SimpleNamespace(
            threshold=0.10+i*0.01,
            event_recall=0.5,
            false_alarm_rate_per_day=0.19,
            pr_auc=0.2,
            operationally_feasible=True,
            threshold_table=pd.DataFrame({
                "threshold":[0.1,0.2],
                "event_recall":[0.8,0.5],
                "false_alarm_rate_per_day":[0.3,0.19],
                "far_feasible":[False,True],
            }),
        )
    ranking=pd.DataFrame({
        "experiment":list(PHASE4_EXPERIMENT_NAMES),
        "threshold":[0.2]*len(PHASE4_EXPERIMENT_NAMES),
        "event_recall":[0.5]*len(PHASE4_EXPERIMENT_NAMES),
        "false_alarm_rate_per_day":[0.19]*len(PHASE4_EXPERIMENT_NAMES),
        "pr_auc":[0.2]*len(PHASE4_EXPERIMENT_NAMES),
        "operationally_feasible":[True]*len(PHASE4_EXPERIMENT_NAMES),
    })
    return SimpleNamespace(
        experiments=experiments,
        ranking=ranking,
        advancing_experiments=PHASE4_EXPERIMENT_NAMES[:3],
    )


def test_writer_emits_complete_aggregate_artifact_set(tmp_path: Path):
    result=_result()
    write_phase4_screening_results(result,tmp_path)

    assert (tmp_path/"screening_ranking.csv").exists()
    assert (tmp_path/"screening_metrics.csv").exists()
    assert (tmp_path/"screening_advancing_experiments.csv").exists()

    for name in PHASE4_EXPERIMENT_NAMES:
        assert (tmp_path/f"screening_{name}_threshold_curve.csv").exists()

    # 17 threshold curves + 3 aggregate tables.
    assert len(list(tmp_path.glob("*.csv")))==20


def test_advancing_artifact_preserves_frozen_rank_order(tmp_path: Path):
    result=_result()
    write_phase4_screening_results(result,tmp_path)
    table=pd.read_csv(tmp_path/"screening_advancing_experiments.csv")
    assert tuple(table["experiment"])==PHASE4_EXPERIMENT_NAMES[:3]
    assert tuple(table["rank"])==(1,2,3)


def test_metrics_contains_every_configuration_once(tmp_path: Path):
    result=_result()
    write_phase4_screening_results(result,tmp_path)
    metrics=pd.read_csv(tmp_path/"screening_metrics.csv")
    assert tuple(metrics["experiment"])==PHASE4_EXPERIMENT_NAMES
    assert len(metrics)==17
    assert metrics["experiment"].is_unique


def test_writer_does_not_export_raw_validation_probabilities(tmp_path: Path):
    result=_result()
    write_phase4_screening_results(result,tmp_path)
    names={p.name for p in tmp_path.iterdir()}
    assert not any("probab" in name.lower() for name in names)
    assert not any("prediction" in name.lower() for name in names)
