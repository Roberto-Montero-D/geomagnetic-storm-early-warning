"""Frozen Phase 5 model-selection contract.

This module resolves implementation details left open by MASTER_PROTOCOL_v1.3
before any Phase 5 model-selection results are inspected.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES
from src.imbalance.freeze import PHASE4_FROZEN_DECISION

PHASE5_RANDOM_STATE = 42
PHASE5_MAX_FAR_PER_DAY = 0.2
PHASE5_SCREENING_FOLD = "screening"
PHASE5_CONFIRMATION_FOLDS = ("walk_forward_1","walk_forward_2")
PHASE5_THRESHOLD_MIN = 0.01
PHASE5_THRESHOLD_MAX = 0.99
PHASE5_THRESHOLD_STEP = 0.01

# Pre-results advancement rule:
# advance the best configuration from each model family. This guarantees that
# temporal confirmation compares families rather than allowing one family to
# occupy every confirmation slot.
PHASE5_ADVANCE_PER_FAMILY = 1

# Screening and confirmation use the same operational hierarchy.
PHASE5_RANKING_RULE = (
    "operationally feasible first; highest Event Recall; highest PR-AUC; "
    "lowest FAR/day; frozen configuration order"
)

# Confirmation winner hierarchy across WF1/WF2.
PHASE5_CONFIRMATION_RANKING_RULE = (
    "feasible in both confirmation folds; highest minimum Event Recall; "
    "highest mean Event Recall; highest mean PR-AUC; lowest mean FAR/day; "
    "frozen candidate order"
)

# Stacking remains gated exactly as the protocol says: it is not automatically
# authorized. Error complementarity must first be demonstrated between the
# confirmed ExtraTrees and LightGBM candidates.
PHASE5_STACKING_STATUS = "gated_not_authorized"

EXTRATREES_GRID = {
    "n_estimators": (100,200,500),
    "max_depth": (10,20,None),
}
LIGHTGBM_GRID = {
    "learning_rate": (0.01,0.05,0.1),
    "num_leaves": (31,63,127),
}
XGBOOST_GRID = {
    "learning_rate": (0.01,0.05,0.1),
    "max_depth": (3,6,9),
}

@dataclass(frozen=True)
class ModelConfiguration:
    config_id: str
    family: str
    params: tuple[tuple[str,object], ...]

    def as_dict(self) -> dict[str,object]:
        return dict(self.params)


def _configs():
    out=[]
    for n,d in product(EXTRATREES_GRID["n_estimators"],EXTRATREES_GRID["max_depth"]):
        depth="none" if d is None else str(d)
        out.append(ModelConfiguration(
            f"extratrees_n{n}_d{depth}","extratrees",
            (("n_estimators",n),("max_depth",d),("random_state",PHASE5_RANDOM_STATE)),
        ))
    for lr,leaves in product(LIGHTGBM_GRID["learning_rate"],LIGHTGBM_GRID["num_leaves"]):
        out.append(ModelConfiguration(
            f"lightgbm_lr{lr:g}_leaves{leaves}","lightgbm",
            (("learning_rate",lr),("num_leaves",leaves),("random_state",PHASE5_RANDOM_STATE)),
        ))
    for lr,d in product(XGBOOST_GRID["learning_rate"],XGBOOST_GRID["max_depth"]):
        out.append(ModelConfiguration(
            f"xgboost_lr{lr:g}_d{d}","xgboost",
            (("learning_rate",lr),("max_depth",d),("random_state",PHASE5_RANDOM_STATE)),
        ))
    return tuple(out)

PHASE5_CONFIGURATIONS=_configs()
PHASE5_CONFIG_IDS=tuple(x.config_id for x in PHASE5_CONFIGURATIONS)
PHASE5_FEATURES=tuple(PHASE3_SELECTED_FEATURES)
PHASE5_IMBALANCE_EXPERIMENT=PHASE4_FROZEN_DECISION.experiment


def validate_phase5_contract() -> None:
    assert len(PHASE5_CONFIGURATIONS)==27
    assert len(set(PHASE5_CONFIG_IDS))==27
    counts={}
    for cfg in PHASE5_CONFIGURATIONS:
        counts[cfg.family]=counts.get(cfg.family,0)+1
        assert cfg.as_dict()["random_state"]==PHASE5_RANDOM_STATE
    assert counts=={"extratrees":9,"lightgbm":9,"xgboost":9}
    assert PHASE5_IMBALANCE_EXPERIMENT=="none"
