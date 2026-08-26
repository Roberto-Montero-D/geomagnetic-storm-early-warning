"""Phase 7 positive-control equivalence against frozen Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.oof_predictions import (
    PHASE6_SELECTED_CONFIG_ID,
    generate_phase6_oof_predictions,
)
from src.phase7.contract import (
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
    build_phase7_events,
)
from src.phase7.oof import (
    assert_phase7_primary_control_dataset,
    generate_phase7_oof_predictions,
)


@dataclass(frozen=True)
class Phase7PositiveControlResult:
    """Audit result for the Phase 6 -> Phase 7 t5_h6 control."""

    experiment_id: str
    config_id: str
    oof_rows: int
    max_probability_abs_diff: float


def assert_phase7_primary_control_oof(
    base_dataset: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    folds,
    splits: pd.DataFrame,
    *,
    progress: bool = False,
    probability_atol: float = 1e-15,
) -> Phase7PositiveControlResult:
    """Require Phase 7 t5_h6 to reproduce frozen Phase 6 OOF output."""

    if PHASE7_MODEL_CONFIG_ID != PHASE6_SELECTED_CONFIG_ID:
        raise AssertionError(
            "Phase 7 model ID does not match the frozen Phase 6 model."
        )

    # First prove the experiment-specific t5_h6 target is exactly the
    # canonical target already used by Phase 6.
    assert_phase7_primary_control_dataset(
        base_dataset,
        kp_intervals,
    )

    phase6_events = build_phase7_events(
        kp_intervals,
        PHASE7_PRIMARY_CONTROL_ID,
    )

    if progress:
        print(
            "    generating frozen Phase 6 control OOF...",
            flush=True,
        )

    phase6 = generate_phase6_oof_predictions(
        base_dataset,
        folds,
        phase6_events,
        splits,
        config_id=PHASE6_SELECTED_CONFIG_ID,
        progress=progress,
    )

    if progress:
        print(
            "    generating Phase 7 t5_h6 OOF...",
            flush=True,
        )

    phase7 = generate_phase7_oof_predictions(
        base_dataset,
        kp_intervals,
        folds,
        splits,
        PHASE7_PRIMARY_CONTROL_ID,
        progress=progress,
    )

    if phase7.experiment_id != PHASE7_PRIMARY_CONTROL_ID:
        raise AssertionError(
            "Phase 7 positive control returned the wrong experiment ID."
        )

    if phase6.config_id != phase7.config_id:
        raise AssertionError(
            "Phase 6 and Phase 7 model configuration IDs differ."
        )

    left = phase6.table
    right = phase7.table

    if len(left) != len(right):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF row counts differ."
        )

    if not left.index.equals(right.index):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF timestamps differ."
        )

    if not left["fold"].astype(str).equals(
        right["fold"].astype(str)
    ):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF fold labels differ."
        )

    if not left["target"].equals(right["target"]):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF targets differ."
        )

    if not left["storm_id"].equals(right["storm_id"]):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF storm IDs differ."
        )

    p6 = pd.to_numeric(
        left["probability"],
        errors="raise",
    ).to_numpy(dtype=float)

    p7 = pd.to_numeric(
        right["probability"],
        errors="raise",
    ).to_numpy(dtype=float)

    abs_diff = np.abs(p6 - p7)

    max_abs_diff = (
        float(abs_diff.max())
        if len(abs_diff)
        else 0.0
    )

    if not np.allclose(
        p6,
        p7,
        rtol=0.0,
        atol=probability_atol,
    ):
        raise AssertionError(
            "Phase 6 and Phase 7 OOF probabilities differ: "
            f"max absolute difference={max_abs_diff:.17g}, "
            f"allowed={probability_atol:.17g}."
        )

    return Phase7PositiveControlResult(
        experiment_id=PHASE7_PRIMARY_CONTROL_ID,
        config_id=phase7.config_id,
        oof_rows=len(right),
        max_probability_abs_diff=max_abs_diff,
    )