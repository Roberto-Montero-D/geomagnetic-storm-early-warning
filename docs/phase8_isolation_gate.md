# Phase 8.2 — Isolation and Synthetic Dry-Run Gate

**Status:** PRE-SCORING SAFETY GATE  
**Real protected Final Test scored:** NO

Phase 8.2 validates that the Phase 8.1 execution framework is structurally
isolated from protected outcomes before any real 2022–2025 metric-producing
code exists.

## Required Invariants

### 1. Protected outcome isolation

Changing Final Test target values must not change:

```text
pre-2022 training predictors
pre-2022 training targets
Final Test prediction timestamps
Final Test predictor matrix
Final Test probabilities
```

The prediction set is allowed to depend on predictor availability. It is not
allowed to depend on Final Test target availability or supervised eligibility.

### 2. Training isolation

Changing post-2022 predictors must not change the pre-2022 training sample.

The fit window remains:

```text
[1996-01-01, 2022-01-01)
```

### 3. Frozen decision path

The public Phase 8.1 APIs accept no model ID or threshold candidate arguments.

The only authorized model ID remains:

```text
lightgbm_lr0.1_leaves127
```

and the only authorized operational threshold remains:

```text
tau = 0.10
```

No threshold grid, threshold optimizer, event evaluator, or Final Test scoring
function is imported or exposed by the materialization/prediction modules.

### 4. Synthetic dry run

A synthetic dataset exercises:

```text
materialization
    -> frozen model factory
    -> model fit
    -> predict_proba
    -> probability-only artifact
```

without producing:

```text
target
storm_id
alert episodes
event classifications
Event Recall
FAR/day
lead time
```

## Meaning of a Pass

Passing Phase 8.2 establishes that the execution machinery is outcome-blind
before scoring.

It does not authorize the real Final Test yet.

The next gate is Phase 8.3 environment/provenance capture and a final
pre-execution audit.
