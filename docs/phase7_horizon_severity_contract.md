# Phase 7 Horizon and Severity Experiment Contract

**Status:** Frozen before Phase 7 results are inspected  
**Protocol basis:** `MASTER_PROTOCOL_v1.3.md`  
**Scope:** Phase 7.0-7.3 contract, registry, truth construction, and invariance checks

## 1. Purpose

Phase 7 measures how the already-selected system behaves when the forecast
horizon or storm-severity threshold changes. It is not a new feature-screening,
imbalance-screening, or model-selection phase.

## 2. Authorized experiments

Exactly these six configurations are authorized:

| Experiment | Kp threshold T | Horizon H |
| --- | ---: | ---: |
| `t5_h3` | 5 | 3 h |
| `t5_h6` | 5 | 6 h |
| `t5_h12` | 5 | 12 h |
| `t5_h24` | 5 | 24 h |
| `t6_h6` | 6 | 6 h |
| `t7_h6` | 7 | 6 h |

No additional H/T configuration may be added after Phase 7 results are seen.

## 3. Frozen components

The following remain fixed across all Phase 7 experiments:

- Phase 3 selected experiment: `A`.
- Predictor set: the 10 primary raw features.
- Phase 4 imbalance decision: `none`.
- Phase 5 model configuration: `lightgbm_lr0.1_leaves127`.
- Model random state and all model hyperparameters.
- Development temporal folds.
- Event termination quiet period: `Z = 6 h`.
- Alert cooldown: `C = 3 h`.
- Maximum operational FAR: `0.2 false alarm episodes/day`.
- Threshold candidate grid: `0.01, 0.02, ..., 0.99`.
- Predictor-source universe and all causal availability rules.

AE, Dst, CME variables, new engineered features, new balancing strategies,
new model families, and new hyperparameter searches are not authorized.

## 4. What may change

Only experiment truth and its operational calibration may change:

1. `T`, according to the registry.
2. `H`, according to the registry.
3. The target labels implied by `(T, H)`.
4. The canonical event universe implied by `T`.
5. The fitted parameters learned by refitting the frozen model specification
   on the experiment-specific labels.
6. The OOF-selected operational probability threshold for that experiment.

Changing H or T must never change predictor values, predictor provenance,
feature membership, model specification, imbalance strategy, or split
boundaries.

## 5. Target semantics

For prediction time `t`:

`y(t; T,H) = 1` iff at least one known retrospective Kp state in `(t, t+H]`
satisfies `Kp >= T`.

The required hourly states are exactly:

`t+1h, t+2h, ..., t+H`.

A known positive is sufficient even if another required future state is
missing. A negative is valid only when all H required future states are known
and below T. Otherwise the target is unknown.

Phase 7 reuses `src.targets.event_window.build_event_window_target`; it does
not introduce a second target implementation.

## 6. Event semantics

For each severity threshold T, event truth is rebuilt with the existing
canonical `identify_events` implementation:

- event onset: first canonical Kp state satisfying `Kp >= T`;
- termination: only after 6 consecutive valid hourly states below T;
- missing Kp does not count as quiet time;
- boundary censoring semantics remain unchanged.

H does not affect event segmentation.

## 7. Predictor invariance

Phase 7 must build or reuse one causal feature matrix. H and T are truth-side
parameters only.

For identical prediction timestamps, all six experiments must therefore use
identical predictor columns and identical predictor values.

The selected model input is exactly the frozen Phase 3 Experiment A feature
tuple.

## 8. Development-only rule

Phase 7 model fitting, OOF prediction generation, threshold calibration, and
all comparisons are development-only. The protected Final Test remains
unavailable for Phase 7 decisions.

## 9. Later Phase 7 fitting interpretation

A changed target defines a changed supervised learning problem. Therefore each
authorized `(T,H)` experiment refits the exact frozen Phase 5 model
specification on that experiment's labels.

This is not model re-selection or hyperparameter optimization.

## 10. Positive control

`t5_h6` is the Phase 6 primary configuration. When the later Phase 7 OOF and
threshold machinery is implemented, it must reproduce the frozen Phase 6
control before the other five experiments are accepted.

## 11. Phase 7.0-7.3 acceptance criteria

Before any Phase 7 model is fitted:

- the registry contains exactly the six authorized experiments;
- experiment IDs and `(T,H)` pairs are unique;
- `t5_h6` is marked as the primary control;
- all experiments use the same frozen feature tuple;
- target construction is parameterized by T and H;
- event construction is parameterized by T and keeps Z=6 h;
- H changes target truth but not event truth;
- T can change both target truth and event truth;
- H/T cannot alter predictor values;
- the existing full test suite remains green.

No Phase 7 performance result is authorized until these conditions pass.
