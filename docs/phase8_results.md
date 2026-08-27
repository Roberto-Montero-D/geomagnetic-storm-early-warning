# Phase 8 — Protected Final Test Results

**Status:** OFFICIAL / FINAL / OUT-OF-SAMPLE  
**Execution commit:** `8c773b1804feabb5cbc1c8dcc08c5340fb20c236`  
**Protected interval:** 2022-01-01 through 2025-12-31  
**Primary experiment:** `t5_h6`  
**Final Test consumed:** YES  
**Retuning from these results:** FORBIDDEN

Phase 8 executed the frozen primary system exactly once on the protected
2022–2025 Final Test.

The frozen configuration was:

```text
T                      = 5
H                      = 6 h
Z                      = 6 h
C                      = 3 h
maximum FAR/day        = 0.2

selected features      = Phase 3 Experiment A
selected feature count = 10
imbalance strategy     = none
model                  = lightgbm_lr0.1_leaves127
operational threshold  = 0.10

training window        = [1996-01-01, 2022-01-01)
Final Test window      = [2022-01-01, 2026-01-01)
```

Before execution, the environment/provenance gate passed on commit
`8c773b1804feabb5cbc1c8dcc08c5340fb20c236`, and the Phase 7 `t5_h6`
development positive control again reproduced the frozen Phase 6 OOF
probabilities exactly:

```text
OOF rows = 25,873
maximum probability absolute difference = 0
protected Final Test scored = false
```

## Official Final Test Metrics

| Metric | Final Test result |
|---|---:|
| Event Recall | **0.5430463576** |
| Detected events | **82 / 151** |
| FAR/day | **0.3680874062** |
| Median early lead time | **3.0 h** |
| Alert episodes | 611 |
| False-alarm episodes | 525 |
| Early-detection episodes | 41 |
| Late-detection episodes | 45 |
| PR-AUC | **0.4959636205** |
| ROC-AUC | **0.8825873610** |
| Brier score | **0.0396522111** |

The primary operational statement is:

> The frozen system detected 82 of 151 protected geomagnetic-storm events
> (Event Recall = 54.3%) with a median early-warning lead time of 3 hours,
> while producing 0.368 false-alarm episodes per day.

## Primary Operational Constraint

The frozen protocol required:

```text
FAR/day <= 0.20
```

The protected Final Test produced:

```text
FAR/day = 0.3680874062
```

Therefore the pre-specified operational false-alarm constraint **did not
generalize to the protected Final Test**.

This result is not corrected post hoc. The threshold remains the frozen
development-selected value:

```text
tau = 0.10
```

No alternative threshold is substituted into the confirmatory result.

## Interpretation

The result separates two aspects of performance.

### Event-level operational performance

```text
Event Recall = 0.5430
FAR/day      = 0.3681
```

The frozen operating point missed 69 of 151 protected storm events and exceeded
the allowed false-alarm burden.

### Probability discrimination

```text
PR-AUC  = 0.4960
ROC-AUC = 0.8826
Brier   = 0.03965
```

The probability model retained substantial ranking/discrimination information
out of sample even though the development-calibrated operational threshold did
not maintain the target FAR/day.

The protected result therefore does not support the claim that the primary
system satisfies the frozen operational constraint in 2022–2025. It does
support the narrower conclusion that the selected predictor/model retains
meaningful out-of-sample probabilistic discrimination.

## Episode and Event Accounting

The Final Test produced:

```text
611 total alert episodes
525 false-alarm episodes
 41 early-detection episodes
 45 late-detection episodes
```

The number of detection episodes may exceed the number of uniquely detected
events because multiple alert episodes may associate with the same canonical
storm event. Event Recall counts each event at most once.

## Immutable Result Artifacts

The one-time runner wrote:

```text
results/phase8/final_test/final_test_predictions.csv
results/phase8/final_test/final_test_alert_episodes.csv
results/phase8/final_test/final_test_metrics.json
```

The environment/provenance gate wrote:

```text
results/phase8/environment/environment_manifest.json
results/phase8/environment/pip-freeze.txt
results/phase8/environment/pip-list.txt
results/phase8/environment/python-platform.txt
```

These artifacts are the authoritative Phase 8 execution record.

## Scientific Status

Phase 8 is confirmatory and closed.

Any analysis performed after this point is post-hoc diagnostic interpretation.
Such analyses may explain error modes, temporal regime changes, calibration
behavior, and operating-point instability, but they may not redefine the
official Phase 8 result.
