# Phase 7 — Horizon/Severity Development Results

**Status:** OFFICIAL DEVELOPMENT-ONLY RESULTS  
**Frozen model:** `lightgbm_lr0.1_leaves127`  
**Primary control:** `t5_h6`  
**Protected Final Test:** NOT SCORED

Phase 7 evaluates pre-authorized changes to retrospective truth while preserving the frozen Phase 3–6 predictor/model framework.

The six registered experiments are:

```text
t5_h3
t5_h6   <- frozen primary control
t5_h12
t5_h24
t6_h6
t7_h6
```

Comparisons are controlled:

- horizon comparison: hold `T=5` fixed and vary `H`;
- severity comparison: hold `H=6 h` fixed and vary `T`;
- cross-task ranking is not authorized.

Changing `T` or `H` changes the prediction problem. Results must therefore not be interpreted as six interchangeable model candidates.

## Primary Control

`t5_h6` is the frozen Phase 6 primary configuration.

The Phase 7 positive-control gate reproduced the Phase 6 OOF output exactly:

```text
experiment        = t5_h6
model             = lightgbm_lr0.1_leaves127
OOF rows          = 25,873
max probability absolute difference = 0
protected Final Test scored          = false
```

The frozen Phase 6 operating point is:

```text
tau               = 0.10
Event Recall      = 21 / 31 = 0.677419
FAR/day           = 0.185522
alert episodes    = 222
false alarms      = 200
OOF exposure      = 25,873 h
WF1 diagnostic tau = 0.07
WF2 diagnostic tau = 0.16
stability thresholds = 0.10, 0.11, 0.12, 0.13
```

## Horizon Comparison — T = 5

| Experiment | H | Selected tau | Event Recall | Detected / Events | FAR/day | Target prevalence |
|---|---:|---:|---:|---:|---:|---:|
| `t5_h3` | 3 h | 0.05 | 0.677419 | 21 / 31 | 0.193870 | 0.008194 |
| `t5_h6` | 6 h | 0.10 | 0.677419 | 21 / 31 | 0.185522 | 0.011827 |
| `t5_h12` | 12 h | 0.13 | 0.838710 | 26 / 31 | 0.192942 | 0.018939 |
| `t5_h24` | 24 h | 0.20 | 0.903226 | 28 / 31 | 0.193870 | 0.032466 |

At approximately the same frozen operational FAR/day budget:

```text
H = 3 h  -> 21 / 31 detected
H = 6 h  -> 21 / 31 detected
H = 12 h -> 26 / 31 detected
H = 24 h -> 28 / 31 detected
```

The observed development recall therefore increases at longer horizons, especially from 6 h to 12 h and 24 h.

This is not evidence that the same prediction task becomes intrinsically easier because the model changed. The model did not change. The target prevalence rises materially with `H`:

```text
H = 3 h  -> 0.8194%
H = 6 h  -> 1.1827%
H = 12 h -> 1.8939%
H = 24 h -> 3.2466%
```

A longer event-in-window target gives more prediction timestamps positive truth and asks a different operational question.

The primary `H=6 h` configuration therefore remains frozen. Phase 7 does not authorize replacing it with `H=12 h` or `H=24 h`.

## Severity Comparison — H = 6 h

| Experiment | T | Selected tau | Event Recall | Detected / Events | FAR/day | Target prevalence |
|---|---:|---:|---:|---:|---:|---:|
| `t5_h6` | 5 | 0.10 | 0.677419 | 21 / 31 | 0.185522 | 0.011827 |
| `t6_h6` | 6 | 0.02 | 1.000000 | 4 / 4 | 0.190160 | 0.002048 |
| `t7_h6` | 7 | 0.01 | 0.500000 | 1 / 2 | 0.055656 | 0.000734 |

The severity experiment is strongly sample-size limited.

```text
T = 5 -> 31 eligible development events
T = 6 ->  4 eligible development events
T = 7 ->  2 eligible development events
```

The `t6_h6` value `4/4 = 1.0` must not be interpreted as evidence that the system is generally perfect for Kp >= 6 storms. Likewise, the `t7_h6` value `1/2 = 0.5` is based on only two eligible development events.

These variants are exploratory generalization diagnostics, not replacements for the primary `T=5, H=6 h` task.

## Threshold Diagnostics

Selected global thresholds:

```text
t5_h3  = 0.05
t5_h6  = 0.10
t5_h12 = 0.13
t5_h24 = 0.20
t6_h6  = 0.02
t7_h6  = 0.01
```

Fold-specific diagnostic thresholds show temporal variation:

```text
             WF1    WF2
t5_h3       0.04   0.08
t5_h6       0.07   0.16
t5_h12      0.11   0.19
t5_h24      0.18   0.23
t6_h6       0.02   0.04
t7_h6       0.01   0.01
```

The predefined `0.15 <= FAR/day <= 0.20` stability regions were:

```text
t5_h3  = 0.05, 0.06, 0.07
t5_h6  = 0.10, 0.11, 0.12, 0.13
t5_h12 = 0.13, 0.14, 0.15, 0.16
t5_h24 = 0.20, 0.21, 0.22
t6_h6  = 0.02
t7_h6  = none
```

`t7_h6` has no threshold in the predefined stability region. This reinforces the limited diagnostic value of the T=7 development experiment.

## Scientific Interpretation

Phase 7 supports three conclusions.

First, the frozen model retains useful signal when the event-in-window horizon changes. Development Event Recall rises substantially for the 12 h and 24 h T=5 tasks while respecting the same FAR/day constraint.

Second, longer horizons materially increase target prevalence. The horizon result must therefore be described as performance on different operational warning windows, not as a direct model-quality ranking.

Third, the T=6 and T=7 development event populations are too small for strong severity-dependent performance claims. Their results should remain descriptive and explicitly denominator-qualified.

No Phase 7 result reopens:

```text
Phase 3 feature set        = Experiment A / 10 raw features
Phase 4 imbalance strategy = none
Phase 5 model              = lightgbm_lr0.1_leaves127
Phase 6 primary threshold  = 0.10
primary T                  = 5
primary H                  = 6 h
```

## Audit Trail

Development-only artifacts are under:

```text
results/phase7/experiments/
results/phase7/analysis/
```

The analysis summary records:

```text
primary_control_id = t5_h6
protected_final_test_scored = false
cross_task_ranking_authorized = false
```

The protected 2022–2025 Final Test remains unopened.

## Artifact Retention Note

The repository intentionally retains the compact frozen cross-experiment
summaries rather than every per-experiment OOF prediction file and 99-point
threshold curve. Those larger files are reproducible intermediates generated
by the committed Phase 7 runners; they are not required to reconstruct the
frozen headline comparison.