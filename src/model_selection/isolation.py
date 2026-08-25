"""Temporal/isolation guards for Phase 5 screening."""
from __future__ import annotations
import pandas as pd

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    PERIOD_INITIAL_TRAIN,
    PERIOD_VALIDATION_1,
)
from .contract import PHASE5_CONFIGURATIONS, PHASE5_SCREENING_FOLD


def validate_phase5_screening_fold(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    splits: pd.DataFrame,
) -> None:
    """Require the exact frozen Initial Train -> Validation 1 contract."""
    if fold.name != PHASE5_SCREENING_FOLD:
        raise ValueError(
            "Phase 5 screening requires the frozen `screening` fold."
        )
    if not dataset.index.equals(splits.index):
        raise ValueError(
            "Phase 5 screening dataset and temporal splits must align exactly."
        )
    if dataset.index.has_duplicates:
        raise ValueError("Phase 5 screening dataset index must be unique.")

    missing_train=fold.train_index.difference(dataset.index)
    missing_val=fold.validation_index.difference(dataset.index)
    if len(missing_train) or len(missing_val):
        raise ValueError(
            "Phase 5 screening fold contains timestamps absent from dataset."
        )

    if fold.train_index.intersection(fold.validation_index).size:
        raise ValueError(
            "Phase 5 screening train and validation indices must not overlap."
        )

    train_periods=set(
        splits.loc[fold.train_index,"period"].astype(str).unique()
    )
    val_periods=set(
        splits.loc[fold.validation_index,"period"].astype(str).unique()
    )

    if PERIOD_FINAL_TEST in train_periods or PERIOD_FINAL_TEST in val_periods:
        raise ValueError(
            "Phase 5 screening must never access protected Final Test rows."
        )
    if train_periods != {PERIOD_INITIAL_TRAIN}:
        raise ValueError(
            "Phase 5 screening training rows must contain only Initial Train."
        )
    if val_periods != {PERIOD_VALIDATION_1}:
        raise ValueError(
            "Phase 5 screening validation rows must contain only Validation 1 "
            "(2017-2018)."
        )

    if len(fold.train_index) and len(fold.validation_index):
        if fold.train_index.max() >= fold.validation_index.min():
            raise ValueError(
                "Phase 5 screening must be strictly chronological."
            )


def assert_identical_phase5_screening_indices(
    observed: dict[str, tuple[pd.DatetimeIndex,pd.DatetimeIndex]],
    fold: DevelopmentFold,
) -> None:
    """Audit that all 27 configurations saw exactly the same rows."""
    expected_ids={c.config_id for c in PHASE5_CONFIGURATIONS}
    if set(observed) != expected_ids:
        missing=sorted(expected_ids-set(observed))
        extra=sorted(set(observed)-expected_ids)
        raise AssertionError(
            f"Phase 5 screening index audit mismatch; missing={missing}, "
            f"extra={extra}."
        )

    for config_id,(train_index,val_index) in observed.items():
        if not pd.DatetimeIndex(train_index).equals(fold.train_index):
            raise AssertionError(
                f"{config_id} did not use the canonical screening training rows."
            )
        if not pd.DatetimeIndex(val_index).equals(fold.validation_index):
            raise AssertionError(
                f"{config_id} did not use the canonical screening validation rows."
            )
