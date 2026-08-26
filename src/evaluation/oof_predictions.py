"""Phase 6 out-of-fold predictions for the frozen Phase 5 winner.

Phase 6 does not perform model selection. It refits exactly one frozen
Phase 5 configuration on each frozen confirmation training window and emits
validation-only predictions for operational threshold optimization.

The protected Final Test is never exposed through this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.dataset.temporal_splits import PERIOD_FINAL_TEST
from src.model_selection.confirmation import (
    PHASE5_CONFIRMATION_FOLDS,
    _validate_confirmation_fold,
)
from src.model_selection.contract import PHASE5_FEATURES
from src.model_selection.factories import make_phase5_model_by_id


PHASE6_SELECTED_CONFIG_ID = "lightgbm_lr0.1_leaves127"

PHASE6_OOF_COLUMNS = (
    "probability",
    "target",
    "storm_id",
    "fold",
)


@dataclass(frozen=True)
class Phase6OOFPredictions:
    """Validated Phase 6 validation-only prediction artifact."""

    config_id: str
    table: pd.DataFrame


def _storm_id_for_predictions(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
) -> pd.Series:
    """Map prediction timestamps to canonical active storm IDs.

    ``storm_id`` is metadata only. A prediction row receives an event ID when
    its timestamp lies inside that event's canonical active interval
    [start_time, end_time]. Rows outside active storms remain missing.

    Operational association and lead-time logic continue to use the canonical
    event table directly; this column never replaces that logic.
    """

    result = pd.Series(
        pd.array(
            [pd.NA] * len(index),
            dtype="Int64",
        ),
        index=index,
        name="storm_id",
    )

    if len(index) == 0 or events.empty:
        return result

    required = {
        "event_id",
        "start_time",
        "end_time",
    }
    missing = required.difference(
        events.columns
    )
    if missing:
        raise ValueError(
            "events is missing required columns: "
            f"{sorted(missing)}"
        )

    event_ids = pd.to_numeric(
        events["event_id"],
        errors="raise",
    )
    starts = pd.to_datetime(
        events["start_time"],
        errors="raise",
    )
    ends = pd.to_datetime(
        events["end_time"],
        errors="raise",
    )

    if event_ids.isna().any():
        raise ValueError(
            "event_id must not contain missing values."
        )
    if starts.isna().any() or ends.isna().any():
        raise ValueError(
            "event start/end times must not be missing."
        )
    if (ends < starts).any():
        raise ValueError(
            "event end_time cannot precede start_time."
        )

    previous_end = None

    for event_id, start, end in zip(
        event_ids.astype(int),
        starts,
        ends,
    ):
        if previous_end is not None and start <= previous_end:
            raise ValueError(
                "canonical events must be strictly non-overlapping."
            )

        mask = (
            (index >= start)
            & (index <= end)
        )

        if mask.any():
            result.loc[mask] = int(
                event_id
            )

        previous_end = end

    return result


def _validate_oof_table(
    table: pd.DataFrame,
    folds: dict[str, DevelopmentFold],
    splits: pd.DataFrame,
) -> None:
    """Validate the complete Phase 6 OOF contract."""

    if tuple(table.columns) != PHASE6_OOF_COLUMNS:
        raise AssertionError(
            "Phase 6 OOF columns drifted from the frozen contract."
        )

    if not isinstance(
        table.index,
        pd.DatetimeIndex,
    ):
        raise AssertionError(
            "Phase 6 OOF index must be a DatetimeIndex."
        )

    if table.index.name != "timestamp":
        raise AssertionError(
            "Phase 6 OOF index must be named timestamp."
        )

    if table.index.has_duplicates:
        raise AssertionError(
            "Phase 6 OOF timestamps must be unique."
        )

    if not table.index.is_monotonic_increasing:
        raise AssertionError(
            "Phase 6 OOF timestamps must be chronologically ordered."
        )

    if not table.index.isin(
        splits.index
    ).all():
        raise AssertionError(
            "Phase 6 OOF contains timestamps absent from temporal splits."
        )

    periods = splits.loc[
        table.index,
        "period",
    ]

    if periods.eq(
        PERIOD_FINAL_TEST
    ).any():
        raise AssertionError(
            "Protected Final Test entered Phase 6 OOF predictions."
        )

    expected_parts = []

    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise AssertionError(
                "Missing frozen Phase 6 source fold: "
                f"{fold_name}"
            )

        fold = folds[
            fold_name
        ]

        expected_parts.append(
            pd.DataFrame(
                {
                    "fold": fold_name,
                },
                index=fold.validation_index,
            )
        )

    expected = pd.concat(
        expected_parts,
        axis=0,
    ).sort_index(
        kind="mergesort"
    )

    expected.index = pd.DatetimeIndex(
        expected.index,
        name="timestamp",
    )

    if not table.index.equals(
        expected.index
    ):
        raise AssertionError(
            "Phase 6 OOF timestamps are not exactly the frozen "
            "confirmation validation timestamps."
        )

    actual_fold = table[
        "fold"
    ].astype(str)

    if not actual_fold.equals(
        expected["fold"].astype(str)
    ):
        raise AssertionError(
            "Phase 6 OOF fold labels do not match the frozen "
            "confirmation validation windows."
        )

    probability = pd.to_numeric(
        table["probability"],
        errors="raise",
    )

    if probability.isna().any():
        raise AssertionError(
            "Phase 6 OOF probabilities must not be missing."
        )

    if not np.isfinite(
        probability.to_numpy(
            dtype=float
        )
    ).all():
        raise AssertionError(
            "Phase 6 OOF probabilities must be finite."
        )

    if (
        (probability < 0)
        | (probability > 1)
    ).any():
        raise AssertionError(
            "Phase 6 OOF probabilities must lie in [0, 1]."
        )

    target = pd.to_numeric(
        table["target"],
        errors="raise",
    )

    if target.isna().any():
        raise AssertionError(
            "Phase 6 OOF target must not be missing."
        )

    if not target.isin(
        [0, 1]
    ).all():
        raise AssertionError(
            "Phase 6 OOF target must be binary."
        )


def generate_phase6_oof_predictions(
    dataset: pd.DataFrame,
    folds: dict[str, DevelopmentFold],
    events: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    config_id: str = PHASE6_SELECTED_CONFIG_ID,
    progress: bool = False,
) -> Phase6OOFPredictions:
    """Generate the frozen Phase 6 OOF prediction table.

    Exactly two refits are performed:
    - walk_forward_1: train through Validation 1, predict Validation 2;
    - walk_forward_2: train through Validation 2, predict Validation 3.

    The selected configuration is frozen by the committed Phase 5 decision.
    """

    if config_id != PHASE6_SELECTED_CONFIG_ID:
        raise ValueError(
            "Phase 6 may only use the frozen Phase 5 winner: "
            f"{PHASE6_SELECTED_CONFIG_ID}."
        )

    parts = []

    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise AssertionError(
                "Missing required Phase 6 fold: "
                f"{fold_name}"
            )

        fold = folds[
            fold_name
        ]

        _validate_confirmation_fold(
            fold,
            splits,
            fold_name,
        )

        (
            x_train,
            y_train,
            x_validation,
            y_validation,
        ) = get_development_xy(
            dataset,
            fold,
            PHASE5_FEATURES,
        )

        if (
            x_train.isna().any().any()
            or x_validation.isna().any().any()
        ):
            raise AssertionError(
                "Phase 6 predictors contain missing values."
            )

        if progress:
            print(
                "    "
                f"{fold_name} / {config_id}: "
                f"fit {len(x_train):,} rows -> "
                f"predict {len(x_validation):,} rows",
                flush=True,
            )

        model = make_phase5_model_by_id(
            config_id
        )

        model.fit(
            x_train,
            y_train.astype(int),
        )

        probability = pd.Series(
            model.predict_proba(
                x_validation
            )[:, 1],
            index=x_validation.index,
            name="probability",
            dtype=float,
        )

        part = pd.DataFrame(
            {
                "probability": probability,
                "target": (
                    y_validation
                    .astype(int)
                ),
                "storm_id": (
                    _storm_id_for_predictions(
                        x_validation.index,
                        events,
                    )
                ),
                "fold": fold_name,
            },
            index=x_validation.index,
        )

        part.index = pd.DatetimeIndex(
            part.index,
            name="timestamp",
        )

        parts.append(
            part
        )

    table = pd.concat(
        parts,
        axis=0,
    ).sort_index(
        kind="mergesort"
    )

    table = table.loc[
        :,
        list(
            PHASE6_OOF_COLUMNS
        ),
    ]

    _validate_oof_table(
        table,
        folds,
        splits,
    )

    return Phase6OOFPredictions(
        config_id=config_id,
        table=table,
    )
