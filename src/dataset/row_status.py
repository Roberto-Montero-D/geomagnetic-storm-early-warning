"""Canonical row-status classification for Phase 1 datasets.

Phase 1.3 classifies rows without dropping or imputing them.

A row is supervised-eligible only when:
- its retrospective target is known; and
- all frozen primary predictor features are present.

Feature incompleteness is intentionally not labeled "warm-up" here. Missing
features may arise from insufficient history, genuine source gaps, or other
upstream missingness. Diagnosing those causes belongs to later audit/reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.integrated import PRIMARY_FEATURE_COLUMNS


ROW_STATUS_ELIGIBLE = "eligible"
ROW_STATUS_UNKNOWN_TARGET = "unknown_target"
ROW_STATUS_FEATURE_INCOMPLETE = "feature_incomplete"
ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET = (
    "feature_incomplete_unknown_target"
)

ROW_STATUS_VALUES = (
    ROW_STATUS_ELIGIBLE,
    ROW_STATUS_UNKNOWN_TARGET,
    ROW_STATUS_FEATURE_INCOMPLETE,
    ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET,
)

ROW_STATUS_COLUMNS = (
    "target_known",
    "features_complete",
    "n_missing_features",
    "supervised_eligible",
    "row_status",
)


def build_row_status(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic row-level supervised-eligibility metadata.

    The input must be the canonical assembled dataset containing all frozen
    primary feature columns and ``target``.

    No rows or values are modified. The returned status frame has exactly the
    same index as ``dataset``.
    """

    required = (*PRIMARY_FEATURE_COLUMNS, "target")
    missing_columns = [
        column
        for column in required
        if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "dataset is missing required canonical columns: "
            f"{missing_columns}"
        )

    if dataset.index.has_duplicates:
        raise ValueError(
            "dataset index must be unique."
        )

    feature_frame = dataset.loc[
        :,
        list(PRIMARY_FEATURE_COLUMNS),
    ]

    n_missing_features = feature_frame.isna().sum(
        axis=1
    ).astype("int64")
    features_complete = n_missing_features.eq(0)
    target_known = dataset["target"].notna()
    supervised_eligible = (
        features_complete & target_known
    )

    row_status = np.select(
        [
            supervised_eligible,
            (~features_complete) & target_known,
            features_complete & (~target_known),
            (~features_complete) & (~target_known),
        ],
        [
            ROW_STATUS_ELIGIBLE,
            ROW_STATUS_FEATURE_INCOMPLETE,
            ROW_STATUS_UNKNOWN_TARGET,
            ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET,
        ],
        default="__invalid__",
    )

    status = pd.DataFrame(
        {
            "target_known": target_known.astype(bool),
            "features_complete": features_complete.astype(bool),
            "n_missing_features": n_missing_features,
            "supervised_eligible": supervised_eligible.astype(bool),
            "row_status": pd.Series(
                row_status,
                index=dataset.index,
                dtype="string",
            ),
        },
        index=dataset.index,
    )

    if (status["row_status"] == "__invalid__").any():
        raise AssertionError(
            "Row-status classification was not exhaustive."
        )

    if tuple(status.columns) != ROW_STATUS_COLUMNS:
        raise AssertionError(
            "Row-status columns are not in deterministic order."
        )

    if not status.index.equals(dataset.index):
        raise AssertionError(
            "Row-status classification did not preserve the dataset index."
        )

    return status
