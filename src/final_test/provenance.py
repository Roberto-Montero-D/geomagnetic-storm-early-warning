"""Phase 8 environment/provenance capture.

This module records software and repository provenance only. It does not import
or access the canonical dataset, OMNI/Kp data, model predictions, or Final Test
outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys


PHASE8_DIRECT_PACKAGE_VERSIONS = {
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "scipy": "1.18.0",
    "PyYAML": "6.0.3",
    "requests": "2.34.2",
    "beautifulsoup4": "4.15.0",
    "scikit-learn": "1.9.0",
    "imbalanced-learn": "0.14.2",
    "lightgbm": "4.7.0",
    "xgboost": "3.4.1",
    "matplotlib": "3.11.1",
    "seaborn": "0.13.2",
    "shap": "0.52.0",
    "pytest": "9.1.1",
    "jupyter": "1.1.1",
}

PHASE8_PYTHON_MAJOR_MINOR = (3, 14)


@dataclass(frozen=True)
class Phase8EnvironmentRecord:
    python_version: str
    python_executable: str
    platform: str
    implementation: str
    git_commit: str
    git_branch: str
    git_worktree_clean: bool
    direct_package_versions: dict[str, str]
    direct_package_versions_match: bool
    python_major_minor_match: bool
    protected_final_test_scored: bool


def _run_text(
    command: list[str],
    *,
    cwd: Path,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def git_provenance(repo_root: Path) -> tuple[str, str, bool]:
    """Return HEAD SHA, branch, and clean-worktree flag."""

    commit = _run_text(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
    )
    branch = _run_text(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
    )
    status = _run_text(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
    )

    if not commit:
        raise RuntimeError("Could not resolve Git commit.")
    if not branch:
        raise RuntimeError("Could not resolve Git branch.")

    return commit, branch, status == ""


def installed_direct_versions() -> dict[str, str]:
    """Return installed versions for every frozen direct dependency."""

    result: dict[str, str] = {}

    for package in PHASE8_DIRECT_PACKAGE_VERSIONS:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "NOT_INSTALLED"

    return result


def validate_direct_versions(
    installed: dict[str, str],
) -> None:
    """Require exact equality with the pre-Final-Test dependency freeze."""

    if set(installed) != set(PHASE8_DIRECT_PACKAGE_VERSIONS):
        raise ValueError(
            "Installed direct-package set differs from Phase 8 freeze."
        )

    mismatches = {
        package: {
            "expected": expected,
            "installed": installed[package],
        }
        for package, expected in PHASE8_DIRECT_PACKAGE_VERSIONS.items()
        if installed[package] != expected
    }

    if mismatches:
        detail = "; ".join(
            f"{package}: expected {values['expected']}, "
            f"installed {values['installed']}"
            for package, values in sorted(mismatches.items())
        )
        raise RuntimeError(
            "Phase 8 direct dependency mismatch: " + detail
        )


def capture_phase8_environment(
    repo_root: Path,
) -> Phase8EnvironmentRecord:
    """Capture and validate the environment without touching project data."""

    repo_root = Path(repo_root).resolve()

    installed = installed_direct_versions()
    validate_direct_versions(installed)

    python_match = (
        sys.version_info.major,
        sys.version_info.minor,
    ) == PHASE8_PYTHON_MAJOR_MINOR

    if not python_match:
        raise RuntimeError(
            "Phase 8 Python runtime mismatch: "
            f"expected {PHASE8_PYTHON_MAJOR_MINOR[0]}."
            f"{PHASE8_PYTHON_MAJOR_MINOR[1]}, "
            f"running {sys.version_info.major}.{sys.version_info.minor}."
        )

    commit, branch, clean = git_provenance(repo_root)

    if branch != "main":
        raise RuntimeError(
            f"Phase 8 must run from main; current branch is {branch!r}."
        )

    if not clean:
        raise RuntimeError(
            "Phase 8 environment capture requires a clean Git worktree."
        )

    return Phase8EnvironmentRecord(
        python_version=sys.version,
        python_executable=sys.executable,
        platform=platform.platform(),
        implementation=platform.python_implementation(),
        git_commit=commit,
        git_branch=branch,
        git_worktree_clean=clean,
        direct_package_versions=installed,
        direct_package_versions_match=True,
        python_major_minor_match=True,
        protected_final_test_scored=False,
    )


def write_phase8_environment_artifacts(
    record: Phase8EnvironmentRecord,
    output_dir: Path,
    *,
    repo_root: Path,
) -> None:
    """Write provenance artifacts without accessing scientific data."""

    if not isinstance(record, Phase8EnvironmentRecord):
        raise TypeError("record must be Phase8EnvironmentRecord.")

    if record.protected_final_test_scored:
        raise ValueError(
            "Environment provenance cannot report Final Test scoring."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        output_dir / "environment_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(record),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    pip_freeze = _run_text(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=repo_root,
    )
    (output_dir / "pip-freeze.txt").write_text(
        pip_freeze + "\n",
        encoding="utf-8",
    )

    pip_list = _run_text(
        [sys.executable, "-m", "pip", "list"],
        cwd=repo_root,
    )
    (output_dir / "pip-list.txt").write_text(
        pip_list + "\n",
        encoding="utf-8",
    )

    (output_dir / "python-platform.txt").write_text(
        "\n".join(
            [
                record.python_version,
                record.platform,
                record.python_executable,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
