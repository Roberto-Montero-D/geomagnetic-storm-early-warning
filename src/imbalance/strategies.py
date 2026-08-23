"""Training-only implementations of the frozen Phase 4 imbalance strategies."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .contract import ImbalanceExperiment, PHASE4_RANDOM_STATE


@dataclass(frozen=True)
class PreparedTrainingData:
    x: pd.DataFrame
    y: pd.Series
    class_weight: dict[int, float] | None


def _validate_training_xy(x: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("Phase 4 training inputs must be a DataFrame and Series.")
    if len(x) != len(y):
        raise ValueError("Training predictors and target must have equal length.")
    if not x.index.equals(y.index):
        raise ValueError("Training predictor and target indices must match exactly.")
    if x.isna().any().any() or y.isna().any():
        raise ValueError("Phase 4 imbalance handling does not accept missing training values.")
    classes=set(pd.unique(y))
    if classes != {0,1}:
        raise ValueError("Phase 4 training target must contain exactly binary classes {0, 1}.")


def _as_pandas(x_res, y_res, columns):
    x_out=pd.DataFrame(x_res,columns=columns)
    y_out=pd.Series(np.asarray(y_res,dtype=int),name="target")
    return x_out,y_out


def _undersampling_strategy(ratio: str) -> float:
    negatives_per_positive=int(ratio.split(":")[0])
    return 1.0 / negatives_per_positive


def prepare_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    experiment: ImbalanceExperiment,
) -> PreparedTrainingData:
    """Apply one frozen imbalance strategy to training data only.

    The function has no validation-data argument by design. Resampled outputs use
    a fresh RangeIndex because synthetic/deleted samples no longer correspond
    one-to-one with prediction timestamps.
    """
    _validate_training_xy(x_train,y_train)

    strategy=experiment.strategy
    parameter=experiment.parameter

    if strategy == "none":
        return PreparedTrainingData(x_train.copy(),y_train.copy(),None)

    if strategy == "class_weighting":
        weight=float(parameter)
        return PreparedTrainingData(
            x_train.copy(),y_train.copy(),{0:1.0,1:weight}
        )

    if strategy == "random_undersampling":
        sampler=RandomUnderSampler(
            sampling_strategy=_undersampling_strategy(str(parameter)),
            random_state=PHASE4_RANDOM_STATE,
        )
    elif strategy == "smote":
        sampler=SMOTE(
            sampling_strategy="auto",
            k_neighbors=int(parameter),
            random_state=PHASE4_RANDOM_STATE,
        )
    elif strategy == "borderline_smote":
        sampler=BorderlineSMOTE(
            sampling_strategy="auto",
            k_neighbors=int(parameter),
            random_state=PHASE4_RANDOM_STATE,
        )
    elif strategy == "smote_enn":
        sampler=SMOTEENN(
            sampling_strategy="auto",
            random_state=PHASE4_RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown Phase 4 imbalance strategy: {strategy!r}")

    x_res,y_res=sampler.fit_resample(x_train,y_train)
    x_out,y_out=_as_pandas(x_res,y_res,x_train.columns)

    return PreparedTrainingData(x_out,y_out,None)
