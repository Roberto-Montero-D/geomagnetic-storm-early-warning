# Project Status Ledger

**Canonical implementation status:** through Phase 7  
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
| 7 | Horizon/severity experiments | COMPLETE / FROZEN | primary `t5_h6` unchanged |
| 8 | Protected Final Test | LOCKED | no outcome access authorized yet |
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

## Phase 7 Freeze

Phase 7 executed the six pre-authorized truth configurations without changing the frozen predictor/model stack:

```text
t5_h3
t5_h6   <- primary control
t5_h12
t5_h24
t6_h6
t7_h6
```

The `t5_h6` positive control reproduced Phase 6 OOF probabilities exactly:

```text
OOF rows = 25,873
maximum probability absolute difference = 0
```

Controlled horizon comparison (`T=5`):

```text
t5_h3  -> 21/31 = 0.6774, FAR/day 0.1939
t5_h6  -> 21/31 = 0.6774, FAR/day 0.1855
t5_h12 -> 26/31 = 0.8387, FAR/day 0.1929
t5_h24 -> 28/31 = 0.9032, FAR/day 0.1939
```

Controlled severity comparison (`H=6 h`):

```text
t5_h6 -> 21/31 = 0.6774
t6_h6 ->  4/4  = 1.0000
t7_h6 ->  1/2  = 0.5000
```

Severity results are sample-size limited and are not used to replace the primary task.

Phase 7 does not reopen the Phase 3 feature set, Phase 4 imbalance decision, Phase 5 model, Phase 6 threshold rule, or the primary `T=5, H=6 h` configuration.

## Protected Final Test

```text
interval = 2022–2025
protected_final_test_scored = false
```

No Phase 0–7 selection decision uses protected Final Test outcomes.

## Next Authorized Work

Phase 8 protected Final Test evaluation is next.

The Final Test remains locked until the dedicated Phase 8 execution path is implemented and audited. The primary configuration entering Phase 8 is unchanged:

```text
feature set           = Phase 3 Experiment A
selected inputs       = 10 raw features
imbalance strategy    = none
model                 = lightgbm_lr0.1_leaves127
operational threshold = 0.10
T                     = 5
H                     = 6 h
Z                     = 6 h
C                     = 3 h
maximum FAR/day       = 0.2
```
