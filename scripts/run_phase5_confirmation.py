"""Run the frozen Phase 5 walk-forward confirmation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from scripts.run_phase5_screening import (
    _progress,
    build_phase5_screening_inputs,
)
from src.model_selection.confirmation import (
    PHASE5_CONFIRMATION_CANDIDATES,
    PHASE5_CONFIRMATION_FOLDS,
    Phase5ConfirmationResult,
    evaluate_confirmation_fold,
    rank_confirmation_candidates,
)


def _format_index_range(index) -> str:
    """Return a compact timestamp range for one materialized fold index."""
    if len(index) == 0:
        return "empty"

    return (
        f"{pd.Timestamp(index[0])} -> "
        f"{pd.Timestamp(index[-1])}"
    )


def evaluate_confirmation_with_progress(
    dataset,
    folds,
    events,
    splits,
):
    """Evaluate the six frozen confirmation fits with visible context."""

    results = {}

    total = (
        len(PHASE5_CONFIRMATION_CANDIDATES)
        * len(PHASE5_CONFIRMATION_FOLDS)
    )
    fit_number = 0

    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise AssertionError(
                f"Missing required Phase 5 confirmation fold: {fold_name}"
            )

        fold = folds[fold_name]

        _progress(
            f"    {fold_name}: "
            f"{len(fold.train_index):,} train rows, "
            f"{len(fold.validation_index):,} validation rows"
        )
        _progress(
            "             train range: "
            f"{_format_index_range(fold.train_index)}"
        )
        _progress(
            "             validation range: "
            f"{_format_index_range(fold.validation_index)}"
        )

        for config_id in PHASE5_CONFIRMATION_CANDIDATES:
            fit_number += 1
            fit_start = perf_counter()

            _progress(
                f"    [{fit_number:02d}/{total:02d}] "
                f"{fold_name} / {config_id}"
            )
            _progress(
                "             fitting and evaluating "
                f"{len(fold.train_index):,} -> "
                f"{len(fold.validation_index):,} rows..."
            )

            result = evaluate_confirmation_fold(
                dataset,
                fold,
                events,
                splits,
                fold_name,
                config_id,
            )

            results[
                (fold_name, config_id)
            ] = result

            _progress(
                "             completed in "
                f"{perf_counter() - fit_start:.1f} s"
            )
            _progress(
                "             result: "
                f"threshold={result.threshold:.4f}, "
                f"event_recall={result.event_recall:.4f}, "
                "FAR/day="
                f"{result.false_alarm_rate_per_day:.4f}, "
                f"PR-AUC={result.pr_auc:.4f}, "
                "feasible="
                f"{result.operationally_feasible}"
            )

    ranking = rank_confirmation_candidates(
        results
    )

    feasible = ranking[
        ranking.feasible_both_folds
    ]

    selected = (
        None
        if feasible.empty
        else str(
            feasible.iloc[0].config_id
        )
    )

    return Phase5ConfirmationResult(
        results,
        ranking,
        selected,
    )


def write_confirmation_results(
    result,
    output_dir,
):
    """Write development-only Phase 5 confirmation artifacts."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        for config_id in PHASE5_CONFIRMATION_CANDIDATES:
            item = result.fold_results[
                (fold_name, config_id)
            ]

            rows.append(
                {
                    "fold_name": fold_name,
                    "config_id": config_id,
                    "threshold": item.threshold,
                    "event_recall": item.event_recall,
                    "false_alarm_rate_per_day": (
                        item.false_alarm_rate_per_day
                    ),
                    "pr_auc": item.pr_auc,
                    "operationally_feasible": (
                        item.operationally_feasible
                    ),
                }
            )

            item.threshold_table.to_csv(
                output_dir
                / (
                    f"confirmation_{fold_name}_"
                    f"{config_id}_threshold_curve.csv"
                ),
                index=False,
            )

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        / "confirmation_fold_metrics.csv",
        index=False,
    )

    result.ranking.to_csv(
        output_dir
        / "confirmation_ranking.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "selected_config_id": (
                    result.selected_config_id
                )
            }
        ]
    ).to_csv(
        output_dir
        / "confirmation_selected_model.csv",
        index=False,
    )


def run_phase5_confirmation(
    fmt,
    lst,
    output_dir,
):
    """Run the frozen Phase 5 WF1/WF2 confirmation."""

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
        "[7/7] Evaluating 6 frozen Phase 5 "
        "walk-forward confirmation fits..."
    )

    confirmation_start = perf_counter()

    result = (
        evaluate_confirmation_with_progress(
            dataset,
            folds,
            events,
            splits,
        )
    )

    _progress(
        "      confirmation evaluation complete "
        f"[{perf_counter() - confirmation_start:.1f} s]"
    )

    _progress(
        "Writing development-only Phase 5 "
        "confirmation artifacts..."
    )

    write_confirmation_results(
        result,
        output_dir,
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Phase 5 WF1/WF2 confirmation."
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
            "results/phase5/confirmation"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_phase5_confirmation(
        args.omni_fmt,
        args.omni_lst,
        args.output_dir,
    )

    print()
    print(
        "Phase 5 walk-forward confirmation complete."
    )
    print(
        f"Results written to: {args.output_dir}"
    )
    print(
        "Selected configuration: "
        f"{result.selected_config_id}"
    )


if __name__ == "__main__":
    main()
