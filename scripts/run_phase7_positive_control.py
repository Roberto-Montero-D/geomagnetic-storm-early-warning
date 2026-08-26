"""Run the Phase 7 t5_h6 positive-control equivalence gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from scripts.run_phase5_screening import (
    _progress,
    build_phase5_screening_inputs,
)
from src.data.kp import build_kp_intervals
from src.data.omni import load_omni
from src.phase7.positive_control import (
    assert_phase7_primary_control_oof,
)


def run_phase7_positive_control(
    fmt_path: Path,
    lst_path: Path,
):
    """Execute the Phase 6 <-> Phase 7 t5_h6 equivalence gate."""

    _progress(
        "[Phase 7 control 1/3] Building canonical development inputs..."
    )
    start = perf_counter()

    (
        dataset,
        splits,
        folds,
        _,
    ) = build_phase5_screening_inputs(
        fmt_path,
        lst_path,
    )

    _progress(
        "      canonical development inputs complete "
        f"[{perf_counter() - start:.1f} s]"
    )

    _progress(
        "[Phase 7 control 2/3] Rebuilding canonical Kp intervals..."
    )
    start = perf_counter()

    omni = load_omni(
        fmt_path,
        lst_path,
    )
    kp_intervals = build_kp_intervals(
        omni
    )

    _progress(
        f"      complete: {len(kp_intervals):,} intervals "
        f"[{perf_counter() - start:.1f} s]"
    )

    _progress(
        "[Phase 7 control 3/3] Comparing Phase 6 and Phase 7 t5_h6 OOF..."
    )
    start = perf_counter()

    result = assert_phase7_primary_control_oof(
        dataset,
        kp_intervals,
        folds,
        splits,
        progress=True,
    )

    _progress(
        "      positive-control equivalence PASSED "
        f"[{perf_counter() - start:.1f} s]"
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify that Phase 7 t5_h6 exactly reproduces "
            "the frozen Phase 6 development-only OOF result."
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

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_phase7_positive_control(
        args.omni_fmt,
        args.omni_lst,
    )

    print()
    print("Phase 7 positive-control gate PASSED.")
    print(f"Experiment: {result.experiment_id}")
    print(f"Frozen configuration: {result.config_id}")
    print(f"OOF rows: {result.oof_rows:,}")
    print(
        "Maximum probability absolute difference: "
        f"{result.max_probability_abs_diff:.17g}"
    )
    print("Protected Final Test scored: False")


if __name__ == "__main__":
    main()