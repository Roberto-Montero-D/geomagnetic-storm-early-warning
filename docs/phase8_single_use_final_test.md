# Phase 8.4 — Single-Use Protected Final Test

**Status before execution:** IMPLEMENTATION ONLY — DO NOT RUN REAL DATA YET.

This step adds the only authorized protected scoring path. It uses the frozen
Phase 8 contract: 10 frozen features, no resampling, LightGBM
`lightgbm_lr0.1_leaves127`, and operational threshold `tau=0.10`.

The runner refuses execution unless the explicit
`--execute-protected-final-test` flag is supplied and requires `main` with a
clean worktree.

## Pre-execution rule

First run focused tests and the full suite. Then commit and push this Phase 8.4
implementation. Only from that clean immutable commit may the real protected
2022–2025 command be executed once.

## Outputs

The one-time execution writes probabilities/targets, associated alert episodes,
and the frozen primary/secondary metrics. After results are visible, no model,
feature, imbalance, threshold, horizon, or event-definition changes are
permitted in response to performance.

Any genuine software defect discovered during execution must be documented
before any rerun is considered.
