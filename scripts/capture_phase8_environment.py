"""Capture the frozen Phase 8 software/repository provenance.

This runner does not load OMNI/Kp data and cannot score the Final Test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.final_test.provenance import (
    capture_phase8_environment,
    write_phase8_environment_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture Phase 8 environment provenance before protected "
            "Final Test execution."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase8/environment"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()

    record = capture_phase8_environment(repo_root)
    write_phase8_environment_artifacts(
        record,
        args.output_dir,
        repo_root=repo_root,
    )

    print("Phase 8 environment/provenance capture PASSED.")
    print(f"Git commit: {record.git_commit}")
    print(f"Git branch: {record.git_branch}")
    print(f"Clean worktree: {record.git_worktree_clean}")
    print(
        "Direct dependency versions match freeze: "
        f"{record.direct_package_versions_match}"
    )
    print(
        "Python major/minor matches freeze: "
        f"{record.python_major_minor_match}"
    )
    print("Protected Final Test scored: False")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
