from pathlib import Path

import pytest

import src.final_test.provenance as provenance


def _matching_versions():
    return dict(
        provenance.PHASE8_DIRECT_PACKAGE_VERSIONS
    )


def test_frozen_direct_versions_are_exact():
    installed = _matching_versions()

    provenance.validate_direct_versions(installed)


def test_dependency_version_mismatch_is_rejected():
    installed = _matching_versions()
    installed["lightgbm"] = "999.0"

    with pytest.raises(
        RuntimeError,
        match="lightgbm",
    ):
        provenance.validate_direct_versions(installed)


def test_missing_direct_package_is_rejected():
    installed = _matching_versions()
    installed["numpy"] = "NOT_INSTALLED"

    with pytest.raises(
        RuntimeError,
        match="numpy",
    ):
        provenance.validate_direct_versions(installed)


def test_git_provenance_clean_main(monkeypatch, tmp_path):
    outputs = iter(
        [
            "abcdef123456",
            "main",
            "",
        ]
    )

    monkeypatch.setattr(
        provenance,
        "_run_text",
        lambda command, cwd: next(outputs),
    )

    commit, branch, clean = provenance.git_provenance(
        tmp_path
    )

    assert commit == "abcdef123456"
    assert branch == "main"
    assert clean is True


def test_capture_rejects_dirty_worktree(monkeypatch, tmp_path):
    monkeypatch.setattr(
        provenance,
        "installed_direct_versions",
        _matching_versions,
    )
    monkeypatch.setattr(
        provenance,
        "git_provenance",
        lambda repo_root: (
            "abcdef",
            "main",
            False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="clean Git worktree",
    ):
        provenance.capture_phase8_environment(tmp_path)


def test_capture_rejects_non_main_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        provenance,
        "installed_direct_versions",
        _matching_versions,
    )
    monkeypatch.setattr(
        provenance,
        "git_provenance",
        lambda repo_root: (
            "abcdef",
            "experiment",
            True,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="must run from main",
    ):
        provenance.capture_phase8_environment(tmp_path)


def test_capture_record_explicitly_says_final_test_unscored(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        provenance,
        "installed_direct_versions",
        _matching_versions,
    )
    monkeypatch.setattr(
        provenance,
        "git_provenance",
        lambda repo_root: (
            "abcdef",
            "main",
            True,
        ),
    )

    record = provenance.capture_phase8_environment(tmp_path)

    assert record.protected_final_test_scored is False
    assert record.git_worktree_clean is True
    assert record.git_branch == "main"
    assert record.direct_package_versions_match is True


def test_artifact_writer_creates_provenance_only(
    monkeypatch,
    tmp_path,
):
    record = provenance.Phase8EnvironmentRecord(
        python_version="3.14.7",
        python_executable="python",
        platform="Windows-test",
        implementation="CPython",
        git_commit="abcdef",
        git_branch="main",
        git_worktree_clean=True,
        direct_package_versions=_matching_versions(),
        direct_package_versions_match=True,
        python_major_minor_match=True,
        protected_final_test_scored=False,
    )

    monkeypatch.setattr(
        provenance,
        "_run_text",
        lambda command, cwd: (
            "fake-package==1.0"
            if "freeze" in command
            else "Package Version\nfake-package 1.0"
        ),
    )

    output = tmp_path / "environment"
    provenance.write_phase8_environment_artifacts(
        record,
        output,
        repo_root=tmp_path,
    )

    assert (
        output / "environment_manifest.json"
    ).is_file()
    assert (output / "pip-freeze.txt").is_file()
    assert (output / "pip-list.txt").is_file()
    assert (
        output / "python-platform.txt"
    ).is_file()

    names = {
        path.name
        for path in output.iterdir()
    }
    assert names == {
        "environment_manifest.json",
        "pip-freeze.txt",
        "pip-list.txt",
        "python-platform.txt",
    }
