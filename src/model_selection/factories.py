"""Deterministic factories for the frozen Phase 5 model grid."""
from __future__ import annotations
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from sklearn.ensemble import ExtraTreesClassifier
from .contract import ModelConfiguration, PHASE5_CONFIGURATIONS, PHASE5_CONFIG_IDS, PHASE5_IMBALANCE_EXPERIMENT, PHASE5_RANDOM_STATE

SUPPORTED_FAMILIES=("extratrees","lightgbm","xgboost")
FAMILY_FIXED_PARAMS={
 "extratrees":{"class_weight":None,"n_jobs":-1},
 "lightgbm":{"objective":"binary","class_weight":None,"n_jobs":-1,"verbosity":-1},
 "xgboost":{"objective":"binary:logistic","n_jobs":-1,"verbosity":0},
}

def configuration_by_id(config_id):
    if config_id not in PHASE5_CONFIG_IDS:
        raise KeyError(f"Unknown frozen Phase 5 configuration: {config_id}")
    return next(c for c in PHASE5_CONFIGURATIONS if c.config_id==config_id)

def dependency_versions():
    out={}
    for p in ("scikit-learn","lightgbm","xgboost"):
        try: out[p]=version(p)
        except PackageNotFoundError: out[p]="NOT_INSTALLED"
    return out

def assert_phase5_dependencies():
    missing=[p for p,v in dependency_versions().items() if v=="NOT_INSTALLED"]
    if missing: raise RuntimeError("Missing Phase 5 dependencies: "+", ".join(missing))

def _validate_configuration(config):
    if config not in PHASE5_CONFIGURATIONS:
        raise ValueError("Phase 5 models may only be created from frozen configurations.")
    if config.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported Phase 5 model family: {config.family}")
    if config.as_dict().get("random_state") != PHASE5_RANDOM_STATE:
        raise ValueError("Phase 5 random_state drift detected.")
    if PHASE5_IMBALANCE_EXPERIMENT != "none":
        raise AssertionError("Phase 5 requires frozen Phase 4 `none` strategy.")

def make_phase5_model(config)->Any:
    _validate_configuration(config)
    tuned=config.as_dict(); fixed=dict(FAMILY_FIXED_PARAMS[config.family])
    overlap=set(tuned)&set(fixed)
    if overlap: raise AssertionError(f"Frozen tuned/fixed parameter overlap: {sorted(overlap)}")
    params={**tuned,**fixed}
    if config.family=="extratrees": return ExtraTreesClassifier(**params)
    if config.family=="lightgbm":
        try: cls=import_module("lightgbm").LGBMClassifier
        except ImportError as e: raise RuntimeError("LightGBM is required for Phase 5.") from e
        return cls(**params)
    if config.family=="xgboost":
        try: cls=import_module("xgboost").XGBClassifier
        except ImportError as e: raise RuntimeError("XGBoost is required for Phase 5.") from e
        return cls(**params)
    raise AssertionError("unreachable")

def make_phase5_model_by_id(config_id):
    return make_phase5_model(configuration_by_id(config_id))
