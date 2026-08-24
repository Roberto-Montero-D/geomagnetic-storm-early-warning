from pathlib import Path
import pandas as pd

from src.imbalance.freeze import (
    PHASE4_CLASS_WEIGHT,
    PHASE4_FROZEN_DECISION,
    PHASE4_SCREENING_ADVANCERS,
    PHASE4_SELECTED_EXPERIMENT,
    PHASE4_USE_RESAMPLING,
)


ROOT=Path(__file__).resolve().parents[1]
SCREENING=ROOT/"results"/"phase4"/"screening"
CONFIRMATION=ROOT/"results"/"phase4"/"confirmation"


def test_frozen_selection_matches_committed_confirmation_artifact():
    table=pd.read_csv(CONFIRMATION/"confirmation_selected_experiment.csv")
    assert len(table)==1
    assert table.loc[0,"selected_experiment"]==PHASE4_SELECTED_EXPERIMENT
    assert PHASE4_SELECTED_EXPERIMENT=="none"


def test_frozen_advancers_match_committed_screening_artifact():
    table=pd.read_csv(SCREENING/"screening_advancing_experiments.csv")
    observed=tuple(
        table.sort_values("rank")["experiment"].astype(str)
    )
    assert observed==PHASE4_SCREENING_ADVANCERS


def test_selected_strategy_is_first_committed_confirmation_rank():
    table=pd.read_csv(CONFIRMATION/"confirmation_ranking.csv")
    assert table.iloc[0]["experiment"]==PHASE4_SELECTED_EXPERIMENT


def test_selected_strategy_was_feasible_in_both_confirmation_folds():
    metrics=pd.read_csv(CONFIRMATION/"confirmation_fold_metrics.csv")
    selected=metrics.loc[
        metrics["experiment"]==PHASE4_SELECTED_EXPERIMENT
    ]
    assert len(selected)==2
    assert selected["operationally_feasible"].astype(bool).all()


def test_every_selected_fold_respects_frozen_far_constraint():
    metrics=pd.read_csv(CONFIRMATION/"confirmation_fold_metrics.csv")
    selected=metrics.loc[
        metrics["experiment"]==PHASE4_SELECTED_EXPERIMENT
    ]
    assert (selected["false_alarm_rate_per_day"] <= 0.2 + 1e-12).all()


def test_downstream_imbalance_contract_is_none():
    assert PHASE4_FROZEN_DECISION.experiment=="none"
    assert PHASE4_USE_RESAMPLING is False
    assert PHASE4_CLASS_WEIGHT is None
