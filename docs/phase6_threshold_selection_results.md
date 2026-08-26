# Phase 6 — OOF Operational Threshold Selection Results

**Status:** OFFICIAL DEVELOPMENT-ONLY RESULTS  
**Frozen model:** `lightgbm_lr0.1_leaves127`  
**Selected threshold:** `tau = 0.10`  
**Protected Final Test:** NOT SCORED

## OOF Run

```text
WF1 rows   = 17,442
WF2 rows   =  8,431
total OOF  = 25,873
```

The frozen grid contains 99 thresholds from `0.01` through `0.99`.

## Selection Boundary

| Threshold | Event Recall | Detected / Events | FAR/day | Feasible |
|---:|---:|---:|---:|---|
| 0.09 | 0.6452 | 20 / 31 | 0.20036 | no |
| **0.10** | **0.6774** | **21 / 31** | **0.18552** | **yes** |

The frozen minimum-feasible rule therefore selects:

```text
tau = 0.10
```

At selection:

```text
Event Recall         = 0.677419
detected events      = 21 / 31
FAR/day              = 0.185522
alert episodes       = 222
false-alarm episodes = 200
OOF exposure         = 25,873 hours
```

## Stability Diagnostic

For predefined `0.15 <= FAR/day <= 0.20`:

| Threshold | Event Recall | Detected / Events | FAR/day |
|---:|---:|---:|---:|
| 0.10 | 0.6774 | 21 / 31 | 0.1855 |
| 0.11 | 0.6774 | 21 / 31 | 0.1725 |
| 0.12 | 0.6774 | 21 / 31 | 0.1688 |
| 0.13 | 0.7097 | 22 / 31 | 0.1568 |

The higher observed recall at `0.13` is a post-selection diagnostic and does not authorize changing the frozen rule or threshold.

Fold-specific diagnostic thresholds:

```text
WF1 = 0.07
WF2 = 0.16
```

Their difference indicates temporal variation in the probability-to-FAR relationship and should be reported later, not tuned away.

## Audit Trail

Official artifacts are under:

```text
results/phase6/threshold_selection/
```

The official summary records:

```text
protected_final_test_scored = false
```

These are development OOF results, not Final Test performance.
