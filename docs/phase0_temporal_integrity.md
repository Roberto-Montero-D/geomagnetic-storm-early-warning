# Phase 0 Temporal Integrity and Leakage Audit

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Passed and frozen

## 1. Purpose

Phase 0.9 verifies that the causal contracts established in Phases 0.1–0.8
remain valid when the canonical feature and target pipelines are composed.

It introduces no new predictor, target, event, alert, split, or model rule.

Canonical integration test:

```text
tests/test_phase0_temporal_integrity.py
```

## 2. Global Predictor Invariant

For every prediction time `t`:

```text
maximum_feature_information_time <= t - 1h
```

The integrated 93-feature frame exposes provenance and is tested against this
constraint over multiple prediction times.

## 3. Predictor / Target Separation

The project intentionally separates:

```text
X = causally available predictor information
y = retrospective future ground truth
```

The integration suite verifies both directions.

### Future OMNI

Changing OMNI measurements that are unavailable at prediction time cannot
change the historical feature vector.

### Future Kp

Changing retrospective Kp inside the target window `(t, t+H]` may change the
target because future Kp defines ground truth.

The same future Kp mutation must not change predictor features.

### Historical predictor-eligible Kp

Changing a completed Kp interval that is eligible for a causal Kp lag may
change the relevant predictor feature.

That historical mutation must not change the future target.

### Unfinished Kp intervals

A Kp interval whose `interval_end` is later than the predictor query time must
not enter a causal Kp lag.

These tests establish the required separation:

```text
future truth may define y
future truth may not define X

eligible historical information may define X
historical information outside the target window may not redefine y
```

## 4. Target Integrity

The composed pipeline re-verifies the frozen target window:

```text
(t, t+H]
```

Therefore:

```text
t is excluded
t+H is included
```

Incomplete future ground truth is not silently converted to a negative.

```text
known future positive                         -> 1
complete known all-negative horizon           -> 0
otherwise                                     -> NaN
```

A known positive remains positive even when other future states are missing.

## 5. Split-Boundary Integrity

Temporal semantics are independent of dataset partition labels.

The integration suite explicitly constructs predictions around:

```text
2021-12-31 23:00
2022-01-01 00:00
2022-01-01 01:00
```

crossing into the protected Final Test calendar period.

The same causal cutoff and target rules continue to apply on both sides of the
calendar boundary.

This test validates temporal semantics only. It does **not** constitute use or
evaluation of the real protected 2022–2025 Final Test dataset.

## 6. Prediction-Grid Invariance

Feature and target definitions depend on physical timestamps, not on the
density of requested prediction rows.

The suite verifies that a timestamp produces the same `X` and `y` when built:

```text
alone
```

or as part of a sparse prediction grid.

This protects against accidental row-count-based feature semantics.

## 7. Frozen Manifest Recheck

The integration suite confirms that the canonical primary feature frame
remains:

```text
93 features
```

with the frozen deterministic manifest and no duplicate feature names.

## 8. Test Evidence

Before Phase 0 closure:

```text
python -m pytest tests/test_phase0_temporal_integrity.py -v
python -m pytest -v
```

both passed locally.

The pushed integration test was then audited against the canonical
implementation.

The repository therefore records:

```text
leakage_tests_passing: true
```

This flag means the frozen Phase 0 temporal/leakage test suite passed.

It does not mean that later training, resampling, cross-validation, threshold
selection, or Final Test code is automatically leakage-free. Those later
pipelines must continue to obey the frozen Phase 0 contracts and require their
own implementation tests.

## 9. What Phase 0.9 Does Not Prove

Phase 0.9 does not claim that:

```text
model training has occurred
feature screening has occurred
imbalance experiments have occurred
OOF threshold selection has occurred
the real Final Test has been inspected
the real Final Test has been evaluated
final scientific performance is known
```

Those belong to later protocol phases.

## 10. Closure Rule

Phase 0 temporal infrastructure is considered ready for Phase 1 because the
canonical source semantics, feature construction, target construction, event
and alert definitions, and cross-pipeline temporal invariants are implemented,
documented, and tested.

Any later code that bypasses these canonical components must independently
demonstrate equivalent temporal semantics before it can be used in the frozen
experiment.
