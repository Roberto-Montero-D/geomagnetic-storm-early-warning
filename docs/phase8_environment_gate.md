# Phase 8.3 — Environment and Provenance Gate

**Status:** PRE-FINAL-TEST PROVENANCE GATE  
**Protected Final Test scored:** NO

Phase 8.3 captures the exact software and repository state immediately before
the single-use protected Final Test execution.

The capture path does not import or access the canonical scientific dataset,
OMNI/Kp source files, protected predictions, or protected outcomes.

## Required Environment

The direct dependency versions must exactly match the Phase 8 freeze in
`requirements-lock-phase8.txt`.

The runtime must use Python 3.14.x and the repository must be:

```text
branch = main
working tree = clean
```

The exact commit SHA is recorded in the environment manifest.

## Generated Provenance

The capture runner writes:

```text
results/phase8/environment/
    environment_manifest.json
    pip-freeze.txt
    pip-list.txt
    python-platform.txt
```

These files are audit provenance only. They contain no scientific outcomes.

## Development Positive-Control Recheck

After the environment capture succeeds, rerun the already-committed Phase 7
`t5_h6` positive-control gate using the real OMNI source.

That recheck is development-only. It must again report:

```text
Maximum probability absolute difference = 0
Protected Final Test scored = False
```

If the positive control does not reproduce exactly, stop. Do not execute the
protected Final Test.

## Authorization Rule

Phase 8.4 becomes eligible only when all of the following are true:

```text
Phase 8.1 focused tests pass
Phase 8.2 focused tests pass
Phase 8.3 focused tests pass
full repository test suite passes
Git branch is main
Git worktree is clean
direct dependencies match the frozen versions
environment provenance has been captured
Phase 7 positive control reproduces exactly
```

Phase 8.3 itself does not authorize any methodological change.
