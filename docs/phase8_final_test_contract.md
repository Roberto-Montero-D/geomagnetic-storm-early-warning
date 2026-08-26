# Phase 8 — Protected Final Test Contract

**Status:** FROZEN BEFORE FINAL TEST OUTCOME ACCESS  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Protected interval:** 2022-01-01 through 2025-12-31  
**Final Test scored:** NO

Phase 8 is a single-use evaluation of the unchanged primary system selected in
Phases 0–7. This contract is frozen before any Phase 8 metric-producing runner
is implemented or executed.

## Frozen Primary Handoff

```text
experiment             = t5_h6
T                      = 5
H                      = 6 h
Z                      = 6 h
C                      = 3 h
maximum FAR/day        = 0.2

features               = Phase 3 Experiment A
selected feature count = 10
imbalance              = none
model                  = lightgbm_lr0.1_leaves127
operational tau        = 0.10

training window        = [1996-01-01, 2022-01-01)
Final Test window      = [2022-01-01, 2026-01-01)
```

The Phase 8 model must be fit once on all supervised-eligible pre-2022 rows
using the frozen primary feature tuple and frozen model specification, then
evaluated once on the protected 2022–2025 rows.

## Forbidden Actions

After this contract is frozen, Phase 8 must not:

- perform feature selection;
- change the 10 selected predictors;
- introduce AE, Dst, or CME predictors;
- change imbalance treatment;
- compare model families or hyperparameters;
- recalibrate `tau`;
- substitute a Phase 7 alternative H/T experiment;
- inspect Final Test outcomes before the dedicated scoring step;
- use Final Test results to modify any methodological choice.

## Machine-Readable Sources

Phase 8 must import:

```text
src.feature_screening.freeze.PHASE3_SELECTED_FEATURES
src.imbalance.freeze.PHASE4_FROZEN_DECISION
src.evaluation.phase6_freeze.PHASE6_FROZEN_DECISION
src.final_test.contract.PHASE8_FROZEN_CONTRACT
```

The Phase 8 contract deliberately contains no fitting, prediction, scoring, or
metric code.

## Single-Use Rule

The protected Final Test is used once. After results are produced, the project
moves to Phase 9 interpretation/scientific audit. Final Test findings may be
explained and audited, but may not trigger retraining, reselection, threshold
changes, or a second "improved" protected evaluation.
