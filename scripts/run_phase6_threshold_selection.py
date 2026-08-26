"""Run frozen Phase 6 OOF prediction and operational threshold selection.

This runner reuses the canonical preprocessing/development-fold builder from
Phase 5, refits only the frozen Phase 5 winner on WF1/WF2, and writes
development-only Phase 6 artifacts. The protected Final Test is never scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from scripts.run_phase5_screening import (
    _progress,
    build_phase5_screening_inputs,
)
from src.evaluation.oof_predictions import (
    PHASE6_SELECTED_CONFIG_ID,
    generate_phase6_oof_predictions,
)
from src.evaluation.phase6_thresholds import (
    DEFAULT_STABILITY_MIN_FAR_PER_DAY,
    optimize_phase6_threshold,
)
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
)


def _json_number(value):
    """Return a JSON-safe scalar while preserving missing diagnostics."""

    if value is None or pd.isna(value):
        return None
    return float(value)


def write_phase6_artifacts(
    oof_result,
    threshold_result,
    output_dir: Path,
) -> None:
    """Write the complete development-only Phase 6 audit artifact set."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_result.table.to_csv(
        output_dir / "oof_predictions.csv",
        index=True,
    )

    threshold_result.global_threshold_table.to_csv(
        output_dir / "global_threshold_curve.csv",
        index=False,
    )

    threshold_result.fold_threshold_table.to_csv(
        output_dir / "fold_threshold_curves.csv",
        index=False,
    )

    fold_rows = [
        {
            "fold": fold,
            "selected_threshold": threshold,
        }
        for fold, threshold in (
            threshold_result
            .fold_selected_thresholds
            .items()
        )
    ]

    pd.DataFrame(
        fold_rows,
        columns=[
            "fold",
            "selected_threshold",
        ],
    ).to_csv(
        output_dir
        / "fold_selected_thresholds.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "threshold": list(
                threshold_result
                .stability_thresholds
            )
        }
    ).to_csv(
        output_dir
        / "stability_thresholds.csv",
        index=False,
    )

    summary = {
        "selected_config_id": (
            oof_result.config_id
        ),
        "selected_threshold": _json_number(
            threshold_result
            .selected_threshold
        ),
        "max_far_per_day": float(
            DEFAULT_MAX_FAR_PER_DAY
        ),
        "stability_min_far_per_day": float(
            DEFAULT_STABILITY_MIN_FAR_PER_DAY
        ),
        "threshold_grid_min": float(
            DEFAULT_THRESHOLD_GRID[0]
        ),
        "threshold_grid_max": float(
            DEFAULT_THRESHOLD_GRID[-1]
        ),
        "threshold_grid_size": int(
            len(DEFAULT_THRESHOLD_GRID)
        ),
        "oof_rows": int(
            len(oof_result.table)
        ),
        "oof_folds": [
            str(value)
            for value in (
                oof_result.table["fold"]
                .drop_duplicates()
                .tolist()
            )
        ],
        "fold_selected_thresholds": {
            str(fold): _json_number(
                threshold
            )
            for fold, threshold in (
                threshold_result
                .fold_selected_thresholds
                .items()
            )
        },
        "stability_thresholds": [
            float(value)
            for value in (
                threshold_result
                .stability_thresholds
            )
        ],
        "protected_final_test_scored": False,
    }

    with (
        output_dir
        / "phase6_selection_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def run_phase6_threshold_selection(
    fmt: Path,
    lst: Path,
    output_dir: Path,
):
    """Execute the frozen Phase 6 development-only selection workflow."""

    (
        dataset,
        splits,
        folds,
        events,
    ) = build_phase5_screening_inputs(
        fmt,
        lst,
    )

    _progress(
        "[Phase 6 1/3] Generating frozen-model "
        "OOF predictions..."
    )
    oof_start = perf_counter()

    oof_result = (
        generate_phase6_oof_predictions(
            dataset,
            folds,
            events,
            splits,
            config_id=(
                PHASE6_SELECTED_CONFIG_ID
            ),
            progress=True,
        )
    )

    _progress(
        "      OOF prediction generation complete: "
        f"{len(oof_result.table):,} rows "
        f"[{perf_counter() - oof_start:.1f} s]"
    )

    _progress(
        "[Phase 6 2/3] Sweeping frozen "
        "operational threshold grid..."
    )
    threshold_start = perf_counter()

    threshold_result = (
        optimize_phase6_threshold(
            oof_result.table,
            events,
            thresholds=(
                DEFAULT_THRESHOLD_GRID
            ),
            max_far_per_day=(
                DEFAULT_MAX_FAR_PER_DAY
            ),
            stability_min_far_per_day=(
                DEFAULT_STABILITY_MIN_FAR_PER_DAY
            ),
            cooldown_hours=3,
            horizon_hours=6,
            progress=True,
        )
    )

    _progress(
        "      threshold sweep complete "
        f"[{perf_counter() - threshold_start:.1f} s]"
    )

    _progress(
        "[Phase 6 3/3] Writing "
        "development-only artifacts..."
    )
    write_start = perf_counter()

    write_phase6_artifacts(
        oof_result,
        threshold_result,
        output_dir,
    )

    _progress(
        "      artifact writing complete "
        f"[{perf_counter() - write_start:.1f} s]"
    )

    return (
        oof_result,
        threshold_result,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Phase 6 OOF threshold selection "
            "without scoring the protected Final Test."
        )
    )

    parser.add_argument(
        "--omni-fmt",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--omni-lst",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/phase6/threshold_selection"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    (
        oof_result,
        threshold_result,
    ) = run_phase6_threshold_selection(
        args.omni_fmt,
        args.omni_lst,
        args.output_dir,
    )

    print()
    print(
        "Phase 6 operational threshold "
        "selection complete."
    )
    print(
        f"Results written to: {args.output_dir}"
    )
    print(
        "Frozen configuration: "
        f"{oof_result.config_id}"
    )
    print(
        "Selected global threshold: "
        f"{threshold_result.selected_threshold}"
    )
    print(
        "Fold diagnostic thresholds: "
        f"{threshold_result.fold_selected_thresholds}"
    )
    print(
        "Stability thresholds: "
        f"{threshold_result.stability_thresholds}"
    )


if __name__ == "__main__":
    main()
