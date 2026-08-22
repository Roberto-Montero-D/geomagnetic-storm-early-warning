"""Phase 3 initial A-E feature screening evaluator.

This module is development-only and is intended for the frozen screening fold:
1996-2016 -> 2017-2018.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.extratrees import make_extratrees_model
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    PERIOD_INITIAL_TRAIN,
    PERIOD_VALIDATION_1,
)
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
    _validate_threshold_grid,
)
from src.feature_screening.manifests import (
    PHASE3_EXPERIMENT_ORDER,
    PHASE3_EXTRATREES_PARAMS,
    PHASE3_FEATURE_SETS,
)


@dataclass(frozen=True)
class ScreeningExperimentResult:
    experiment: str
    n_features: int
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    validation_probability: pd.Series
    threshold_table: pd.DataFrame


@dataclass(frozen=True)
class Phase3ScreeningResult:
    experiments: dict[str, ScreeningExperimentResult]
    ranking: pd.DataFrame
    advancing_experiments: tuple[str, ...]

def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)

def _events_for_validation(
    events: pd.DataFrame,
    validation_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(validation_index) == 0:
        return events.iloc[0:0].copy()
    starts = pd.to_datetime(events["start_time"], errors="raise")
    return events.loc[
        (starts >= validation_index.min())
        & (starts <= validation_index.max())
    ].copy()


def _validate_screening_fold(
    fold: DevelopmentFold,
    splits: pd.DataFrame,
) -> None:
    for role, index in (
        ("train", fold.train_index),
        ("validation", fold.validation_index),
    ):
        if not index.isin(splits.index).all():
            raise ValueError(
                f"screening {role} contains timestamps absent from splits."
            )

        periods = splits.loc[index, "period"]
        if (periods == PERIOD_FINAL_TEST).any():
            raise ValueError(
                f"screening {role} touches the protected Final Test."
            )

    train_periods = splits.loc[fold.train_index, "period"]
    validation_periods = splits.loc[fold.validation_index, "period"]

    if not train_periods.eq(PERIOD_INITIAL_TRAIN).all():
        raise ValueError(
            "Phase 3 screening train must contain only initial_train rows."
        )

    if not validation_periods.eq(PERIOD_VALIDATION_1).all():
        raise ValueError(
            "Phase 3 screening validation must contain only validation_1 rows."
        )

    train_years=set(fold.train_index.year)
    validation_years=set(fold.validation_index.year)
    if train_years and (min(train_years) < 1996 or max(train_years) > 2016):
        raise ValueError("Phase 3 screening train must be confined to 1996-2016.")
    if validation_years and (
        min(validation_years) < 2017 or max(validation_years) > 2018
    ):
        raise ValueError(
            "Phase 3 screening validation must be confined to 2017-2018."
        )


def _fit_experiment(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    experiment: str,
) -> tuple[pd.Series, pd.Series]:
    features=PHASE3_FEATURE_SETS[experiment]
    x_train,y_train,x_validation,y_validation=get_development_xy(
        dataset,fold,features
    )
    if x_train.isna().any().any():
        raise AssertionError(
            f"Phase 3 {experiment} training predictors contain missing values."
        )
    if x_validation.isna().any().any():
        raise AssertionError(
            f"Phase 3 {experiment} validation predictors contain missing values."
        )
    if len(np.unique(y_train.to_numpy())) != 2:
        raise ValueError(
            f"Phase 3 {experiment} training target must contain both classes."
        )

    model=make_extratrees_model(
        n_estimators=PHASE3_EXTRATREES_PARAMS["n_estimators"],
        max_depth=PHASE3_EXTRATREES_PARAMS["max_depth"],
        random_state=PHASE3_EXTRATREES_PARAMS["random_state"],
    )
    model.fit(x_train,y_train.astype(int))
    probability=pd.Series(
        model.predict_proba(x_validation)[:,1],
        index=x_validation.index,
        name="probability",
        dtype=float,
    )
    return probability,y_validation.astype(int)


def _select_single_fold_threshold(
    probability: pd.Series,
    events: pd.DataFrame,
    *,
    thresholds,
    max_far_per_day: float,
    cooldown_hours: int,
    horizon_hours: int,
    progress: bool = False,
    experiment: str | None = None,
) -> tuple[float | None,pd.DataFrame]:
    thresholds=_validate_threshold_grid(thresholds)
    total_thresholds = len(thresholds)
    try:
        max_far = float(max_far_per_day)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_far_per_day must be numeric.") from exc

    if not np.isfinite(max_far) or max_far < 0:
        raise ValueError(
            "max_far_per_day must be finite and non-negative."
        )
    rows=[]
    for number, threshold in enumerate(thresholds, start=1):
        if progress:
            label = experiment or "experiment"
            _progress(
                True,
                f"      {label}: threshold "
                f"{number:02d}/{total_thresholds} (tau={threshold:.2f})",
            )
        _,metrics=evaluate_operational_series(
            probability,
            events,
            threshold=threshold,
            cooldown_hours=cooldown_hours,
            horizon_hours=horizon_hours,
        )
        far=metrics.false_alarm_rate_per_day
        rows.append({
            "threshold":threshold,
            "event_recall":metrics.event_recall,
            "false_alarm_rate_per_day":far,
            "far_feasible":bool(
                not pd.isna(far) and far <= max_far
            ),
        })
    table=pd.DataFrame(rows)
    feasible=table.loc[table["far_feasible"],"threshold"]
    selected=None if feasible.empty else float(feasible.iloc[0])
    return selected,table


def evaluate_screening_experiment(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    experiment: str | None = None,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float=DEFAULT_MAX_FAR_PER_DAY,
    cooldown_hours: int=3,
    horizon_hours: int=6,
    progress: bool = False,
) -> ScreeningExperimentResult:
    if experiment not in PHASE3_FEATURE_SETS:
        raise ValueError(f"unknown Phase 3 experiment: {experiment}")

    _validate_screening_fold(fold,splits)
    probability,y_validation=_fit_experiment(dataset,fold,experiment)
    scoped_events=_events_for_validation(events,probability.index)

    pr_auc=float(
        average_precision_score(
            y_validation.to_numpy(),
            probability.to_numpy(),
        )
    )

    selected,table=_select_single_fold_threshold(
        probability,
        scoped_events,
        thresholds=thresholds,
        max_far_per_day=max_far_per_day,
        cooldown_hours=cooldown_hours,
        horizon_hours=horizon_hours,
        progress=progress,
        experiment=experiment,
    )

    if selected is None:
        return ScreeningExperimentResult(
            experiment=experiment,
            n_features=len(PHASE3_FEATURE_SETS[experiment]),
            threshold=None,
            event_recall=np.nan,
            false_alarm_rate_per_day=np.nan,
            pr_auc=pr_auc,
            operationally_feasible=False,
            validation_probability=probability,
            threshold_table=table,
        )
    
    _progress(
        progress,
        f"    Fitting experiment {experiment} "
        f"({len(PHASE3_FEATURE_SETS[experiment])} features)...",
    )

    _,metrics=evaluate_operational_series(
        probability,
        scoped_events,
        threshold=selected,
        cooldown_hours=cooldown_hours,
        horizon_hours=horizon_hours,
    )
    return ScreeningExperimentResult(
        experiment=experiment,
        n_features=len(PHASE3_FEATURE_SETS[experiment]),
        threshold=selected,
        event_recall=metrics.event_recall,
        false_alarm_rate_per_day=metrics.false_alarm_rate_per_day,
        pr_auc=pr_auc,
        operationally_feasible=True,
        validation_probability=probability,
        threshold_table=table,
    )


def rank_screening_experiments(
    experiments: dict[str,ScreeningExperimentResult],
) -> tuple[pd.DataFrame,tuple[str,...]]:
    rows=[]
    order_index={name:i for i,name in enumerate(PHASE3_EXPERIMENT_ORDER)}
    for name in PHASE3_EXPERIMENT_ORDER:
        result=experiments[name]
        rows.append({
            "experiment":name,
            "n_features":result.n_features,
            "threshold":result.threshold,
            "event_recall":result.event_recall,
            "false_alarm_rate_per_day":result.false_alarm_rate_per_day,
            "pr_auc":result.pr_auc,
            "operationally_feasible":result.operationally_feasible,
            "_order":order_index[name],
        })
    ranking=pd.DataFrame(rows)
    ranking=ranking.sort_values(
        by=[
            "operationally_feasible",
            "event_recall",
            "pr_auc",
            "false_alarm_rate_per_day",
            "_order",
        ],
        ascending=[False,False,False,True,True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    feasible=ranking.loc[ranking["operationally_feasible"],"experiment"]
    n_advance=min(3,len(feasible))
    advancing=tuple(feasible.iloc[:n_advance])
    ranking=ranking.drop(columns="_order")
    return ranking,advancing


def evaluate_phase3_screening(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    progress: bool = False,
    max_far_per_day: float=DEFAULT_MAX_FAR_PER_DAY,
) -> Phase3ScreeningResult:
    results={}
    for experiment in PHASE3_EXPERIMENT_ORDER:
        results[experiment]=evaluate_screening_experiment(
            dataset,
            fold,
            events,
            splits,
            experiment,
            thresholds=thresholds,
            max_far_per_day=max_far_per_day,
            progress=progress,
        )
    _progress(progress, "    Ranking A-E screening experiments...")
    ranking,advancing=rank_screening_experiments(results)
    return Phase3ScreeningResult(results,ranking,advancing)
