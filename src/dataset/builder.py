"""Canonical Phase 1 dataset assembly.

This module composes the frozen prediction grid, primary causal feature frame,
and retrospective target into one row-preserving dataset.

It deliberately does NOT:
- impute features;
- drop rows with missing features;
- drop unknown targets;
- assign supervised eligibility;
- assign temporal splits;
- fit or transform model-side preprocessing.

Those responsibilities belong to later Phase 1 checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from src.features.integrated import (
    PRIMARY_FEATURE_COLUMNS,
    build_primary_feature_frame,
)
from src.targets.event_window import (
    DEFAULT_TARGET_HORIZON_HOURS,
    DEFAULT_TARGET_THRESHOLD,
    build_event_window_target,
)
from time import perf_counter

TARGET_COLUMN = "target"

FEATURE_AUDIT_COLUMNS = (
    "raw_information_time",
    "rolling_information_time",
    "persistence_information_time",
    "dynamics_information_time",
    "interaction_information_time",
    "maximum_feature_information_time",
    "information_cutoff",
)

TARGET_AUDIT_COLUMNS = (
    "future_window_start",
    "future_window_end",
    "expected_future_hours",
    "observed_future_hours",
    "missing_future_hours",
    "positive_future_hours",
    "target_status",
)

DATASET_AUDIT_COLUMNS = (
    *FEATURE_AUDIT_COLUMNS,
    *TARGET_AUDIT_COLUMNS,
)


def _validate_prediction_times(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(
        prediction_times,
        name="prediction_time",
    )

    if index.hasnans:
        raise ValueError(
            "prediction_times must not contain NaT."
        )

    if index.has_duplicates:
        raise ValueError(
            "prediction_times must be unique."
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "prediction_times must be monotonically increasing."
        )

    if len(index) and (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError(
            "prediction_times must be aligned to whole hours."
        )

    if index.tz is not None:
        raise ValueError(
            "prediction_times must be timezone-naive."
        )

    return index


def build_canonical_dataset(
    omni: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    threshold: float = DEFAULT_TARGET_THRESHOLD,
    horizon_hours: int = DEFAULT_TARGET_HORIZON_HOURS,
    return_audit: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the canonical row-preserving Phase 1 dataset.

    Output dataset columns are exactly:

        93 frozen primary feature columns
        + target

    The prediction timestamp remains the DataFrame index named
    ``prediction_time``.

    When ``return_audit=True``, a second frame is returned with feature
    provenance and target-ground-truth audit metadata. Audit metadata is kept
    separate from the predictor matrix so it cannot accidentally become X.

    Every requested prediction timestamp must survive assembly exactly once,
    regardless of missing feature values or an unknown target.
    """

    prediction_index = _validate_prediction_times(
        prediction_times
    )
    dataset_start = perf_counter()

    feature_start = perf_counter()
    if progress is not None:
        progress("[dataset 1/3] Building primary causal feature frame...")

    features, feature_audit = build_primary_feature_frame(
        omni,
        kp_intervals,
        prediction_index,
        return_audit=True,
        progress=progress,
    )

    if progress is not None:
        progress(
            "[dataset 1/3] Primary causal feature frame complete "
            f"[{perf_counter() - feature_start:.1f} s]"
        )

    target_start = perf_counter()
    if progress is not None:
        progress("[dataset 2/3] Building future event-window target...")

    target, target_audit = build_event_window_target(
        kp_intervals,
        prediction_index,
        threshold=threshold,
        horizon_hours=horizon_hours,
        return_audit=True,
    )

    if progress is not None:
        progress(
            "[dataset 2/3] Future event-window target complete "
            f"[{perf_counter() - target_start:.1f} s]"
        )

    if progress is not None:
        progress("[dataset 3/3] Validating and assembling canonical dataset...")

    components = {
        "features": features.index,
        "feature_audit": feature_audit.index,
        "target": target.index,
        "target_audit": target_audit.index,
    }
    for name, index in components.items():
        if not index.equals(prediction_index):
            raise AssertionError(
                f"{name} index does not match the requested "
                "prediction-time universe."
            )

    if tuple(features.columns) != tuple(
        PRIMARY_FEATURE_COLUMNS
    ):
        raise AssertionError(
            "Integrated feature columns do not match the frozen "
            "primary feature manifest."
        )

    dataset = features.copy()
    dataset[TARGET_COLUMN] = target

    expected_dataset_columns = (
        *PRIMARY_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
    if tuple(dataset.columns) != expected_dataset_columns:
        raise AssertionError(
            "Canonical dataset columns are not in the frozen order."
        )

    if not dataset.index.equals(prediction_index):
        raise AssertionError(
            "Canonical dataset did not preserve all requested rows."
        )

    dataset.index.name = "prediction_time"

    if not return_audit:
        if progress is not None:
            progress(
                "[dataset 3/3] Canonical dataset complete: "
                f"{len(dataset):,} rows x {len(dataset.columns)} columns "
                f"[{perf_counter() - dataset_start:.1f} s total]"
            )
        return dataset

    audit = pd.concat(
        [feature_audit, target_audit],
        axis=1,
    )

    if audit.columns.has_duplicates:
        duplicates = audit.columns[
            audit.columns.duplicated()
        ].tolist()
        raise AssertionError(
            f"Duplicate canonical audit columns: {duplicates}"
        )

    if tuple(audit.columns) != DATASET_AUDIT_COLUMNS:
        raise AssertionError(
            "Canonical dataset audit columns are not in the "
            "expected deterministic order."
        )

    if not audit.index.equals(prediction_index):
        raise AssertionError(
            "Canonical audit did not preserve all requested rows."
        )

    audit.index.name = "prediction_time"
    if progress is not None:
        progress(
            "[dataset 3/3] Canonical dataset complete: "
            f"{len(dataset):,} rows x {len(dataset.columns)} columns "
            f"[{perf_counter() - dataset_start:.1f} s total]"
        )

    return dataset, audit
