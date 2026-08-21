# Phase 0 Completion Checklist

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Phase:** 0 — Causality and temporal infrastructure  
**Status:** Complete

## Completion Summary

Phase 0 was completed before model training or performance-driven feature
selection.

The final primary experimental source universe is:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

Primary exclusions frozen during Phase 0:

```text
AE
Dst
CDAW/LASCO CME-derived predictors
```

The canonical primary feature manifest contains:

```text
93 features
```

The canonical primary target is:

```text
y_event(t) = max(Kp[t+1:t+H]) >= T

T = 5
H = 6h
window = (t, t+H]
```

## Phase 0 Checklist

| Checkpoint | Result | Frozen outcome |
|---|---|---|
| 0.1 OMNI timestamp semantics | PASS | Raw timestamp is hourly period start; eligibility uses `period_end <= t-1h` |
| 0.2 Kp temporal semantics | PASS | Canonical 3h intervals; predictor mapping uses completed intervals only |
| 0.3 Source availability audit | PASS | AE/Dst/CDAW CME-derived predictors excluded from primary causal universe |
| 0.4 Temporal cutoff infrastructure | PASS | Canonical `t-1h` predictor cutoff implemented and tested |
| 0.5 Event definition | PASS | `T=5`, `Z=6h`, censoring and missing-ground-truth semantics frozen |
| 0.6 Alert definition | PASS | `C=3h`, episode association and operational classifications frozen |
| 0.7 Causal feature infrastructure | PASS | 93-feature raw/rolling/persistence/dynamics/interactions manifest frozen |
| 0.8 Target construction | PASS | Future storm-condition target `(t,t+6h]`; incomplete unknown truth -> `NaN` |
| 0.9 Leakage / temporal integrity | PASS | Composed X/y invariants and split-boundary semantics tested |

## Phase 0 Frozen Parameters

```text
T = 5
H = 6h
Z = 6h
C = 3h
max FAR/day = 0.2
```

Primary development coverage:

```text
1996–2021
```

Protected Final Test:

```text
2022–2025
```

The Final Test remains:

```text
protected = true
single_use = true
```

## Causal Feature Contract

For prediction time `t`:

```text
maximum_feature_information_time <= t - 1h
```

No feature-layer:

```text
forward fill
backward fill
nearest-row substitution
implicit interpolation
fallback to an older raw timestamp
```

is permitted unless a future protocol amendment explicitly changes the rule.

The feature family order is:

```text
raw
rolling
persistence
dynamics
interactions
```

with counts:

```text
10 + 60 + 5 + 15 + 3 = 93
```

## Target Contract

The historical name `y_event` refers to the frozen mathematical target:

```text
future storm-condition presence
```

not storm-onset-only prediction.

Target truth is retrospective and is kept separate from predictor-side
availability.

Missing future truth follows:

```text
any known positive                    -> 1
complete all-negative future horizon  -> 0
otherwise                             -> NaN
```

## Protected-Test Status at Phase 0 Closure

Phase 0 did not evaluate model performance on the protected 2022–2025 Final
Test.

A synthetic/calendar-boundary temporal-integrity test crosses the date
2022-01-01 only to verify that timestamp semantics do not change at a split
boundary.

This is not inspection of the real Final Test observations or labels.

## Ready for the Next Protocol Phase

Phase 0 closure means the temporal/data-contract foundation is ready.

It does **not** mean the project has completed:

```text
baseline evaluation
feature screening
walk-forward model validation
imbalance handling
model selection
operational threshold optimization
horizon/severity experiments
protected Final Test evaluation
scientific interpretation
```

Those remain governed by the frozen master protocol.

## Canonical Phase 0 Documentation

```text
MASTER_PROTOCOL_v1.3.md
docs/temporal_contract.md
docs/event_definition.md
docs/alert_definition.md
docs/cme_availability.md
docs/feature_definition.md
docs/target_definition.md
docs/phase0_temporal_integrity.md
docs/phase0_completion_checklist.md
```

Phase 0 decisions must not be modified later in response to model performance.
Only verified implementation bugs or source-semantic corrections may be
handled under the protocol freeze rule.
