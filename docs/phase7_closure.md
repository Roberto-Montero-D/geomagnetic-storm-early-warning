# Phase 7 — Closure

**Status:** COMPLETE AND FROZEN  
**Frozen primary configuration:** `t5_h6`  
**Frozen model:** `lightgbm_lr0.1_leaves127`  
**Frozen primary operational threshold:** `tau = 0.10`  
**Protected Final Test:** UNTOUCHED

Phase 7 is formally closed.

Phase 7 executed only the six pre-authorized horizon/severity truth configurations:

```text
t5_h3
t5_h6
t5_h12
t5_h24
t6_h6
t7_h6
```

`t5_h6` served as the positive control and reproduced the frozen Phase 6 OOF prediction path exactly:

```text
OOF rows = 25,873
maximum probability absolute difference = 0
```

The controlled development comparisons were:

```text
horizon: T=5, vary H
severity: H=6 h, vary T
```

Cross-task ranking was not authorized.

## Frozen Phase 7 Findings

Horizon comparison:

```text
t5_h3  -> Event Recall 21/31 = 0.6774, FAR/day 0.1939
t5_h6  -> Event Recall 21/31 = 0.6774, FAR/day 0.1855
t5_h12 -> Event Recall 26/31 = 0.8387, FAR/day 0.1929
t5_h24 -> Event Recall 28/31 = 0.9032, FAR/day 0.1939
```

Severity comparison:

```text
t5_h6 -> 21/31 = 0.6774
t6_h6 ->  4/4  = 1.0000
t7_h6 ->  1/2  = 0.5000
```

The severity results are explicitly sample-size limited and do not support strong comparative claims.

Longer T=5 horizons increase both Event Recall and target prevalence. They define different operational warning questions and do not replace the frozen primary `H=6 h` configuration.

## Primary Configuration Remains Unchanged

The definitive primary configuration entering the protected Final Test remains:

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
information cutoff    = t - 1h
```

No Phase 7 result authorizes re-optimization of features, imbalance handling, model, hyperparameters, primary threshold, T, or H.

## Final Test Firewall

All Phase 7 generation, threshold recalibration, and analysis were development-only.

```text
protected_final_test_scored = false
```

The protected interval remains:

```text
2022-01-01 through 2025-12-31
```

Phase 7 closure does not itself score or inspect Final Test outcomes.

## Authorized Handoff

Phase 7 hands the unchanged primary configuration to the protected Final Test stage.

Any Final Test execution must:

1. use the frozen primary `t5_h6` task;
2. use the frozen 10-feature Phase 3 input set;
3. use `none` imbalance treatment;
4. use `lightgbm_lr0.1_leaves127`;
5. use global `tau=0.10`;
6. make no further development-result-driven selections;
7. score the protected Final Test only through the protocol-authorized Phase 8 path.

Phase 7 is closed.
