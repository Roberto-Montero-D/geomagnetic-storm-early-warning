# Phase 6 — Closure

**Status:** COMPLETE AND FROZEN  
**Frozen model:** `lightgbm_lr0.1_leaves127`  
**Frozen operational threshold:** `tau = 0.10`  
**Protected Final Test:** UNTOUCHED

Phase 6 is formally closed.

The definitive primary configuration entering later phases is:

```text
feature set           = Phase 3 Experiment A
imbalance strategy    = none
model                 = lightgbm_lr0.1_leaves127
operational threshold = 0.10
T                     = 5
H                     = 6 h
Z                     = 6 h
C                     = 3 h
maximum FAR/day       = 0.2
```

Selection boundary:

```text
tau = 0.09 -> FAR/day = 0.20036 -> infeasible
tau = 0.10 -> FAR/day = 0.18552 -> feasible
Event Recall at 0.10 = 21 / 31 = 0.6774
```

Frozen diagnostics:

```text
stability thresholds = 0.10, 0.11, 0.12, 0.13
WF1 diagnostic tau   = 0.07
WF2 diagnostic tau   = 0.16
```

Later primary analyses must not use later results to change the Phase 3 feature set, Phase 4 imbalance strategy, Phase 5 model, Phase 6 grid/rule, or global `tau=0.10`.

Any protocol-defined Phase 7 alternative horizon/severity experiment must be labeled as such and must not silently replace the primary configuration.

The official summary records `protected_final_test_scored = false`. Phase 6 closure does not itself authorize Final Test evaluation.
