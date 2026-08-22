"""Frozen cumulative Phase 3 feature-set manifests."""
from __future__ import annotations

from src.features.dynamics import DYNAMIC_FEATURE_COLUMNS
from src.features.integrated import PRIMARY_FEATURE_COLUMNS
from src.features.interactions import INTERACTION_FEATURE_COLUMNS
from src.features.persistence import PERSISTENCE_FEATURE_COLUMNS
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS
from src.features.rolling import rolling_feature_names

PHASE3_EXPERIMENT_ORDER = ("A", "B", "C", "D", "E")

PHASE3_FEATURE_SETS = {
    "A": tuple(PRIMARY_RAW_FEATURE_COLUMNS),
    "B": (
        *PRIMARY_RAW_FEATURE_COLUMNS,
        *rolling_feature_names(),
    ),
    "C": (
        *PRIMARY_RAW_FEATURE_COLUMNS,
        *rolling_feature_names(),
        *PERSISTENCE_FEATURE_COLUMNS,
    ),
    "D": (
        *PRIMARY_RAW_FEATURE_COLUMNS,
        *rolling_feature_names(),
        *PERSISTENCE_FEATURE_COLUMNS,
        *DYNAMIC_FEATURE_COLUMNS,
    ),
    "E": (
        *PRIMARY_RAW_FEATURE_COLUMNS,
        *rolling_feature_names(),
        *PERSISTENCE_FEATURE_COLUMNS,
        *DYNAMIC_FEATURE_COLUMNS,
        *INTERACTION_FEATURE_COLUMNS,
    ),
}

PHASE3_EXTRATREES_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": None,
    "random_state": 42,
}


def validate_phase3_feature_sets() -> None:
    previous: tuple[str, ...] = ()
    for name in PHASE3_EXPERIMENT_ORDER:
        columns = PHASE3_FEATURE_SETS[name]
        if len(columns) != len(set(columns)):
            raise AssertionError(f"Phase 3 set {name} contains duplicate features.")
        if previous and columns[: len(previous)] != previous:
            raise AssertionError(
                f"Phase 3 set {name} is not a cumulative extension."
            )
        previous = columns

    if PHASE3_FEATURE_SETS["E"] != tuple(PRIMARY_FEATURE_COLUMNS):
        raise AssertionError(
            "Phase 3 set E must equal the complete frozen primary manifest."
        )


validate_phase3_feature_sets()
