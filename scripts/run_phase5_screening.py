"""Run the official Phase 5 initial model-selection screening.

This runner evaluates only the frozen screening fold:
    Initial Train -> Validation 1

It never evaluates Validation 2, Validation 3, or the protected Final Test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.baselines.framework import build_development_folds
from src.data.kp import build_kp_intervals
from src.data.omni import load_omni
from src.dataset.builder import build_canonical_dataset
from src.dataset.prediction_grid import build_prediction_grid
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    assign_temporal_periods,
)
from src.definitions.events import identify_events
from src.model_selection.contract import (
    PHASE5_CONFIGURATIONS,
    PHASE5_SCREENING_FOLD,
)
from src.model_selection.factories import (
    assert_phase5_dependencies,
    dependency_versions,
)
from src.model_selection.isolation import (
    assert_identical_phase5_screening_indices,
    validate_phase5_screening_fold,
)
from src.model_selection.screening import (
    Phase5ScreeningResult,
    advance_family_winners,
    evaluate_model_configuration,
)


def _progress(message: str) -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


def _timed_stage(
    label: str,
    function,
    *,
    completion_message=None,
):
    """Run one preprocessing stage with elapsed-time logging."""
    _progress(label)
    start = perf_counter()

    result = function()

    elapsed = perf_counter() - start

    if completion_message is None:
        suffix = ""
    else:
        suffix = f": {completion_message(result)}"

    _progress(
        f"      complete{suffix} [{elapsed:.1f} s]"
    )

    return result


def build_phase5_screening_inputs(
    fmt_path: Path,
    lst_path: Path,
):
    """Build the canonical development-only Phase 5 inputs with timings."""

    stage_start = perf_counter()
    _progress("[1/7] Checking Phase 5 dependencies...")
    assert_phase5_dependencies()
    _progress(
        f"      complete [{perf_counter() - stage_start:.1f} s]"
    )

    omni = _timed_stage(
        "[2/7] Loading OMNI source data...",
        lambda: load_omni(
            fmt_path,
            lst_path,
        ),
        completion_message=lambda frame: (
            f"{len(frame):,} hourly rows"
        ),
    )

    kp_intervals = _timed_stage(
        "[3/7] Building canonical Kp intervals...",
        lambda: build_kp_intervals(
            omni
        ),
        completion_message=lambda frame: (
            f"{len(frame):,} canonical intervals"
        ),
    )

    _progress(
        "[4/7] Building canonical prediction grid and dataset..."
    )
    stage_start = perf_counter()

    grid = build_prediction_grid()

    _progress(
        f"      prediction grid: {len(grid):,} timestamps"
    )

    dataset = build_canonical_dataset(
        omni,
        kp_intervals,
        grid,
        progress=lambda message: _progress(
            f"      {message}"
        ),
    )

    _progress(
        "      [stage 4] canonical prediction grid and dataset "
        f"complete [{perf_counter() - stage_start:.1f} s]"
    )

    _progress(
        "[5/7] Building row status, temporal splits, and development folds..."
    )
    stage_start = perf_counter()

    status = build_row_status(
        dataset
    )

    splits = assign_temporal_periods(
        dataset.index
    )

    folds = build_development_folds(
        dataset,
        status,
        splits,
    )

    _progress(
        "      complete: "
        f"{len(folds)} development folds "
        f"[{perf_counter() - stage_start:.1f} s]"
    )

    _progress(
        "[6/7] Identifying canonical storm events..."
    )
    stage_start = perf_counter()

    events = identify_events(
        kp_intervals
    )

    _progress(
        "      complete: "
        f"{len(events):,} canonical events "
        f"[{perf_counter() - stage_start:.1f} s]"
    )

    return (
        dataset,
        splits,
        folds,
        events,
    )


def evaluate_phase5_screening_with_progress(
    dataset: pd.DataFrame,
    fold,
    events: pd.DataFrame,
    splits: pd.DataFrame,
) -> Phase5ScreeningResult:
    """Evaluate all 27 frozen configurations with visible progress."""

    validate_phase5_screening_fold(
        dataset,
        fold,
        splits,
    )

    results = {}
    observed_indices = {}

    total = len(
        PHASE5_CONFIGURATIONS
    )

    for i, config in enumerate(
        PHASE5_CONFIGURATIONS,
        start=1,
    ):
        model_start = perf_counter()

        _progress(
            f"    [{i:02d}/{total:02d}] "
            f"{config.family} / "
            f"{config.config_id}"
        )

        results[
            config.config_id
        ] = evaluate_model_configuration(
            dataset,
            fold,
            events,
            config.config_id,
        )

        _progress(
            "             completed in "
            f"{perf_counter() - model_start:.1f} s"
        )

        # The evaluator reads the canonical materialized fold directly.
        # Record the exact supplied indices for the post-run equality audit.
        observed_indices[
            config.config_id
        ] = (
            fold.train_index.copy(),
            fold.validation_index.copy(),
        )

    assert_identical_phase5_screening_indices(
        observed_indices,
        fold,
    )

    family_rankings, advancing = (
        advance_family_winners(
            results
        )
    )

    return Phase5ScreeningResult(
        configurations=results,
        family_rankings=family_rankings,
        advancing_configurations=advancing,
    )


def write_phase5_screening_results(
    result: Phase5ScreeningResult,
    output_dir: Path,
) -> None:
    """Write development-only Phase 5 screening artifacts."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Dependency versions make the model implementation auditable.
    versions = dependency_versions()

    pd.DataFrame(
        {
            "package": list(
                versions.keys()
            ),
            "version": list(
                versions.values()
            ),
        }
    ).to_csv(
        output_dir
        / "dependency_versions.csv",
        index=False,
    )

    # One aggregate row per frozen configuration.
    metrics_rows = []

    for config in PHASE5_CONFIGURATIONS:
        item = result.configurations[
            config.config_id
        ]

        metrics_rows.append(
            {
                "config_id": (
                    config.config_id
                ),
                "family": config.family,
                "threshold": (
                    item.threshold
                ),
                "event_recall": (
                    item.event_recall
                ),
                "false_alarm_rate_per_day": (
                    item.false_alarm_rate_per_day
                ),
                "pr_auc": (
                    item.pr_auc
                ),
                "operationally_feasible": (
                    item.operationally_feasible
                ),
            }
        )

        item.threshold_table.to_csv(
            output_dir
            / (
                "screening_"
                f"{config.config_id}"
                "_threshold_curve.csv"
            ),
            index=False,
        )

    pd.DataFrame(
        metrics_rows
    ).to_csv(
        output_dir
        / "screening_metrics.csv",
        index=False,
    )

    # Family-local rankings preserve the precommitted selection mechanism.
    for (
        family,
        ranking,
    ) in result.family_rankings.items():
        ranking.to_csv(
            output_dir
            / (
                "screening_ranking_"
                f"{family}.csv"
            ),
            index=False,
        )

    advancing_rows = []

    for (
        rank,
        config_id,
    ) in enumerate(
        result.advancing_configurations,
        start=1,
    ):
        family = next(
            config.family
            for config
            in PHASE5_CONFIGURATIONS
            if (
                config.config_id
                == config_id
            )
        )

        advancing_rows.append(
            {
                "rank": rank,
                "family": family,
                "config_id": config_id,
            }
        )

    pd.DataFrame(
        advancing_rows
    ).to_csv(
        output_dir
        / (
            "screening_"
            "advancing_configurations.csv"
        ),
        index=False,
    )


def run_phase5_screening(
    fmt_path: Path,
    lst_path: Path,
    output_dir: Path,
) -> Phase5ScreeningResult:
    """Run the frozen Phase 5 initial model-screening workflow."""

    (
        dataset,
        splits,
        folds,
        events,
    ) = build_phase5_screening_inputs(
        fmt_path,
        lst_path,
    )

    if (
        PHASE5_SCREENING_FOLD
        not in folds
    ):
        raise AssertionError(
            "Canonical development folds lack "
            "the Phase 5 screening fold."
        )

    if (
        splits[
            "period"
        ].eq(
            PERIOD_FINAL_TEST
        ).sum()
        == 0
    ):
        raise AssertionError(
            "Canonical split table unexpectedly "
            "lacks protected Final Test rows."
        )

    _progress(
        "[7/7] Evaluating 27 frozen Phase 5 "
        "model configurations..."
    )

    result = (
        evaluate_phase5_screening_with_progress(
            dataset,
            folds[
                PHASE5_SCREENING_FOLD
            ],
            events,
            splits,
        )
    )

    _progress(
        "Writing development-only Phase 5 "
        "screening artifacts..."
    )

    write_phase5_screening_results(
        result,
        output_dir,
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Phase 5 model "
            "screening on Initial Train -> "
            "Validation 1 only."
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
            "results/phase5/screening"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_phase5_screening(
        args.omni_fmt,
        args.omni_lst,
        args.output_dir,
    )

    print()
    print(
        "Phase 5 initial model "
        "screening run complete."
    )
    print(
        f"Results written to: "
        f"{args.output_dir}"
    )
    print(
        "Advancing family winners:"
    )

    if (
        result.advancing_configurations
    ):
        for (
            rank,
            config_id,
        ) in enumerate(
            result.advancing_configurations,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{config_id}"
            )
    else:
        print(
            "  None — no model family "
            "produced an operationally "
            "feasible configuration."
        )


if __name__ == "__main__":
    main()
