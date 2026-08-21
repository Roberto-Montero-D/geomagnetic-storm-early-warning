"""B3 ExtraTrees baseline with predeclared configuration."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS

EXTRATREES_FEATURES=tuple(PRIMARY_RAW_FEATURE_COLUMNS)
DEFAULT_N_ESTIMATORS=100
DEFAULT_MAX_DEPTH=10
DEFAULT_RANDOM_STATE=42

@dataclass(frozen=True)
class ExtraTreesFoldResult:
    fold_name: str
    model: ExtraTreesClassifier
    validation_probability: pd.Series

def make_extratrees_model(*, n_estimators: int=DEFAULT_N_ESTIMATORS,
                          max_depth: int|None=DEFAULT_MAX_DEPTH,
                          random_state: int=DEFAULT_RANDOM_STATE):
    if not isinstance(n_estimators,int) or isinstance(n_estimators,bool):
        raise TypeError("n_estimators must be an integer.")
    if n_estimators <= 0: raise ValueError("n_estimators must be positive.")
    if max_depth is not None:
        if not isinstance(max_depth,int) or isinstance(max_depth,bool):
            raise TypeError("max_depth must be an integer or None.")
        if max_depth <= 0: raise ValueError("max_depth must be positive or None.")
    return ExtraTreesClassifier(n_estimators=n_estimators,max_depth=max_depth,
        class_weight=None,random_state=random_state,n_jobs=-1)

def fit_extratrees_fold(dataset: pd.DataFrame, fold: DevelopmentFold, *,
                        n_estimators: int=DEFAULT_N_ESTIMATORS,
                        max_depth: int|None=DEFAULT_MAX_DEPTH,
                        random_state: int=DEFAULT_RANDOM_STATE):
    x_train,y_train,x_validation,_=get_development_xy(dataset,fold,EXTRATREES_FEATURES)
    if x_train.isna().any().any(): raise AssertionError("B3 training data contains missing raw predictors.")
    if x_validation.isna().any().any(): raise AssertionError("B3 validation data contains missing raw predictors.")
    if len(np.unique(y_train.to_numpy())) != 2:
        raise ValueError("B3 training target must contain both classes.")
    model=make_extratrees_model(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state)
    model.fit(x_train,y_train.astype(int))
    probability=pd.Series(model.predict_proba(x_validation)[:,1],
                          index=x_validation.index,name="probability",dtype=float)
    if not probability.between(0,1).all():
        raise AssertionError("B3 validation probabilities must lie in [0, 1].")
    return ExtraTreesFoldResult(fold.name,model,probability)
