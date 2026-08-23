import pandas as pd
from scripts.run_phase3_confirmation import write_phase3_confirmation_results
from src.feature_screening.confirmation import (
 PHASE3_ADVANCING_EXPERIMENTS,PHASE3_CONFIRMATION_FOLDS,
 ConfirmationFoldResult,Phase3ConfirmationResult)

def _result():
    d={}
    for f in PHASE3_CONFIRMATION_FOLDS:
      for e in PHASE3_ADVANCING_EXPERIMENTS:
        d[(e,f)]=ConfirmationFoldResult(e,f,0.1,0.5,0.15,0.25,True,
          pd.DataFrame({"threshold":[.01,.1],"event_recall":[.7,.5],
          "false_alarm_rate_per_day":[.3,.15],"far_feasible":[False,True]}))
    ranking=pd.DataFrame({"experiment":["A","E","C"],
      "confirmation_feasible":[True]*3,"minimum_event_recall":[.5,.4,.3],
      "mean_event_recall":[.6,.5,.4],"mean_pr_auc":[.3,.29,.28],
      "mean_false_alarm_rate_per_day":[.15,.14,.13]})
    return Phase3ConfirmationResult(d,ranking,"A")

def test_writer_creates_expected_confirmation_artifacts(tmp_path):
    write_phase3_confirmation_results(_result(),tmp_path)
    expected={"confirmation_ranking.csv","confirmation_fold_metrics.csv",
      "confirmation_selected_experiment.csv"}
    expected|={f"{f}_{e.lower()}_threshold_curve.csv"
      for f in PHASE3_CONFIRMATION_FOLDS for e in PHASE3_ADVANCING_EXPERIMENTS}
    assert {p.name for p in tmp_path.glob("*.csv")}==expected

def test_writer_does_not_export_prediction_or_target_timeline(tmp_path):
    write_phase3_confirmation_results(_result(),tmp_path)
    forbidden={"prediction_time","target","y_true","validation_probability","final_test"}
    for p in tmp_path.glob("*.csv"):
        assert forbidden.isdisjoint(pd.read_csv(p).columns)
