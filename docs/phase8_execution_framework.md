# Phase 8.1 — Outcome-Blind Execution Framework

**Status:** IMPLEMENTATION GATE — NO FINAL TEST METRICS  
**Protected interval:** 2022–2025  
**Frozen experiment:** `t5_h6`

Phase 8.1 implements only the structural fit/predict path required before the
single-use protected evaluation.

It is deliberately outcome-blind on the Final Test side.

## Training Materialization

Training rows must satisfy:

```text
1996-01-01 <= prediction_time < 2022-01-01
supervised_eligible = true
```

The training matrix contains exactly the 10 frozen Phase 3 Experiment A
features. Training targets are allowed because they are pre-2022 development
truth.

## Final Test Prediction Materialization

Prediction rows must satisfy:

```text
period = final_test
2022-01-01 <= prediction_time < 2026-01-01
features_complete = true
```

Importantly, Final Test `target_known` and `supervised_eligible` are not used to
decide which rows receive a probability. This prevents outcome availability
from affecting the prediction set before the scoring layer is authorized.

The materialization API does not return Final Test targets.

## Frozen Model Path

Exactly one model may be created:

```text
lightgbm_lr0.1_leaves127
```

using the already-validated Phase 5 model factory.

The generated artifact contains:

```text
timestamp index
probability
```

and nothing else.

It does not contain:

```text
target
storm_id
alert
classification
metric
```

The frozen operational threshold `tau=0.10` is carried as immutable metadata
only. Phase 8.1 does not apply it to construct alerts and does not perform any
threshold search.

## Gate

Phase 8.1 is accepted only after its focused tests and the complete repository
test suite pass.

Passing Phase 8.1 does not authorize protected scoring. The next step is the
Phase 8.2 isolation/dry-run gate.
