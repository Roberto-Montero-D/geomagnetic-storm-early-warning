# Project Status Ledger

**Canonical implementation status:** through Phase 6  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Protected Final Test:** LOCKED / UNSCORED

This is the current implementation-status ledger. Historical protocol, contract, decision, and closure documents should not be rewritten merely because later phases complete.

| Phase | Scope | Status | Frozen handoff |
|---|---|---|---|
| 0 | Causality and temporal semantics | COMPLETE | 93-feature causal universe |
| 1 | Dataset, row status, temporal splits | COMPLETE | development folds + protected Final Test |
| 2 | Baselines/evaluation | COMPLETE | canonical operational evaluator |
| 3 | Feature screening | COMPLETE / FROZEN | Experiment A — 10 raw features |
| 4 | Imbalance experiments | COMPLETE / FROZEN | `none` |
| 5 | Model selection | COMPLETE / FROZEN | `lightgbm_lr0.1_leaves127` |
| 6 | OOF threshold selection | COMPLETE / FROZEN | global `tau=0.10` |
| 7 | Horizon/severity experiments | NEXT | not yet executed |
| 8 | Protected Final Test | LOCKED | no outcome access authorized |
| 9 | Interpretation/scientific audit | PENDING | — |

## Current Primary Configuration

```text
T = 5
H = 6 h
Z = 6 h
C = 3 h
maximum FAR/day = 0.2
information cutoff = t - 1h

causal universe = 93 features
selected inputs = Phase 3 Experiment A = 10 raw features
imbalance       = none
model           = lightgbm_lr0.1_leaves127
threshold       = 0.10
```

## Phase 5 Freeze

```text
ExtraTrees winner = extratrees_n100_dnone
LightGBM winner   = lightgbm_lr0.1_leaves127
XGBoost winner    = xgboost_lr0.1_d9

selected model = lightgbm_lr0.1_leaves127

worst-fold Event Recall = 0.7333
mean Event Recall       = 0.7417
mean PR-AUC             = 0.2792
mean FAR/day            = 0.1917
```

## Phase 6 Freeze

```text
OOF rows = 25,873
tau      = 0.10

Event Recall = 21 / 31 = 0.6774
FAR/day      = 0.1855

tau=0.09 FAR/day = 0.20036 -> infeasible
tau=0.10 FAR/day = 0.18552 -> feasible

stability thresholds = 0.10, 0.11, 0.12, 0.13
WF1 diagnostic tau   = 0.07
WF2 diagnostic tau   = 0.16
```

## Protected Final Test

```text
interval = 2022–2025
protected_final_test_scored = false
```

No Phase 0–6 selection decision uses protected Final Test outcomes.

## Next Authorized Work

Phase 7 horizon/severity experiments are next. The primary Phase 3–6 decisions remain frozen. Phase 8 Final Test evaluation remains locked.
