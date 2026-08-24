from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from scripts.run_phase4_confirmation import write_phase4_confirmation_results
from src.imbalance.confirmation import PHASE4_ADVANCING_EXPERIMENTS
from src.imbalance.contract import PHASE4_CONFIRMATION_FOLDS


def _result():
    folds={}
    for fold_name in PHASE4_CONFIRMATION_FOLDS:
        for experiment in PHASE4_ADVANCING_EXPERIMENTS:
            folds[(fold_name,experiment)]=SimpleNamespace(
                threshold=0.2,
                event_recall=0.5,
                false_alarm_rate_per_day=0.18,
                pr_auc=0.3,
                operationally_feasible=True,
                threshold_table=pd.DataFrame({
                    "threshold":[0.1,0.2],
                    "event_recall":[0.7,0.5],
                    "false_alarm_rate_per_day":[0.25,0.18],
                    "far_feasible":[False,True],
                }),
            )
    ranking=pd.DataFrame({
        "experiment":list(PHASE4_ADVANCING_EXPERIMENTS),
        "confirmation_feasible":[True]*3,
        "minimum_event_recall":[0.5]*3,
        "mean_event_recall":[0.5]*3,
        "mean_pr_auc":[0.3]*3,
        "mean_false_alarm_rate_per_day":[0.18]*3,
    })
    return SimpleNamespace(
        folds=folds,
        ranking=ranking,
        selected_experiment=PHASE4_ADVANCING_EXPERIMENTS[0],
    )


def test_writer_emits_exact_confirmation_artifact_set(tmp_path: Path):
    result=_result()
    write_phase4_confirmation_results(result,tmp_path)

    assert (tmp_path/"confirmation_fold_metrics.csv").exists()
    assert (tmp_path/"confirmation_ranking.csv").exists()
    assert (tmp_path/"confirmation_selected_experiment.csv").exists()

    for fold_name in PHASE4_CONFIRMATION_FOLDS:
        for experiment in PHASE4_ADVANCING_EXPERIMENTS:
            assert (
                tmp_path/f"{fold_name}_{experiment}_threshold_curve.csv"
            ).exists()

    # 6 curves + 3 aggregate files.
    assert len(list(tmp_path.glob("*.csv")))==9


def test_fold_metrics_contains_exactly_six_confirmations(tmp_path: Path):
    result=_result()
    write_phase4_confirmation_results(result,tmp_path)
    table=pd.read_csv(tmp_path/"confirmation_fold_metrics.csv")
    assert len(table)==6
    assert set(table["fold"])==set(PHASE4_CONFIRMATION_FOLDS)
    assert set(table["experiment"])==set(PHASE4_ADVANCING_EXPERIMENTS)
    assert not table.duplicated(["fold","experiment"]).any()


def test_selected_strategy_artifact_is_single_frozen_candidate(tmp_path: Path):
    result=_result()
    write_phase4_confirmation_results(result,tmp_path)
    table=pd.read_csv(tmp_path/"confirmation_selected_experiment.csv")
    assert len(table)==1
    assert table.loc[0,"selected_experiment"] in PHASE4_ADVANCING_EXPERIMENTS


def test_runner_does_not_export_raw_probabilities(tmp_path: Path):
    result=_result()
    write_phase4_confirmation_results(result,tmp_path)
    names={p.name.lower() for p in tmp_path.iterdir()}
    assert not any("probab" in name for name in names)
    assert not any("prediction" in name for name in names)
