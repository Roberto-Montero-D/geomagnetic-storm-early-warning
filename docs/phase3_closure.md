# Phase 3 Closure Record

**Phase:** Feature Screening  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** COMPLETE — Phase 3 formally closed after frozen initial screening, walk-forward confirmation, final feature-set freeze, and passing repository tests.

## 1. Scope of Phase 3

Phase 3 compared the five frozen cumulative feature experiments:

```text
A = Raw
B = Raw + Rolling
C = Raw + Rolling + Persistence
D = Raw + Rolling + Persistence + Dynamics
E = Raw + Rolling + Persistence + Dynamics + Interactions
```

All experiments used the frozen unbalanced ExtraTrees configuration:

```text
n_estimators = 100
max_depth = 10
class_weight = None
random_state = 42
```

No model-family search, imbalance treatment, Final Test access, or post-result feature editing occurred in Phase 3.

## 2. Initial Screening

Initial screening used only:

```text
Train:      1996–2016
Validation: 2017–2018
```

The official screening advanced:

```text
1. A
2. E
3. C
```

The permanent screening record is:

```text
docs/phase3_initial_screening_results.md
```

The authoritative empirical artifacts are:

```text
results/phase3/screening/screening_ranking.csv
results/phase3/screening/screening_advancing_experiments.csv
results/phase3/screening/screening_metrics.csv
results/phase3/screening/screening_a_threshold_curve.csv
results/phase3/screening/screening_b_threshold_curve.csv
results/phase3/screening/screening_c_threshold_curve.csv
results/phase3/screening/screening_d_threshold_curve.csv
results/phase3/screening/screening_e_threshold_curve.csv
```

These CSVs are authoritative for exact empirical values.

## 3. Walk-Forward Confirmation

Only the frozen advancing candidates A/E/C were confirmed on:

```text
WF1: 1996–2018 -> 2019–2020
WF2: 1996–2020 -> 2021
```

The confirmation evaluator enforced exact atomic-period membership and explicit Final Test exclusion.

The official confirmation selected:

```text
Experiment A
```

The authoritative empirical artifacts are:

```text
results/phase3/confirmation/confirmation_ranking.csv
results/phase3/confirmation/confirmation_fold_metrics.csv
results/phase3/confirmation/confirmation_selected_experiment.csv
results/phase3/confirmation/walk_forward_1_a_threshold_curve.csv
results/phase3/confirmation/walk_forward_1_e_threshold_curve.csv
results/phase3/confirmation/walk_forward_1_c_threshold_curve.csv
results/phase3/confirmation/walk_forward_2_a_threshold_curve.csv
results/phase3/confirmation/walk_forward_2_e_threshold_curve.csv
results/phase3/confirmation/walk_forward_2_c_threshold_curve.csv
```

These CSVs remain authoritative for exact threshold, Event Recall, FAR/day, and PR-AUC values.

## 4. Frozen Phase 3 Output

The immutable Phase 3 software contract is:

```text
src/feature_screening/freeze.py
```

with:

```text
PHASE3_SELECTED_EXPERIMENT = "A"
PHASE3_SELECTED_FEATURES = PHASE3_FEATURE_SETS["A"]
PHASE3_SELECTION_STATUS = "frozen"
```

`PHASE3_SELECTED_FEATURES` is the only Phase 3 feature-set input allowed to enter later phases.

Later phases must import the frozen contract rather than rebuilding or re-ranking Phase 3 feature candidates.

## 5. Phase 4 Input Boundary

Phase 4 may change only the class-imbalance treatment defined by the frozen protocol.

Phase 4 must keep fixed:

- target definition;
- horizon and event definition;
- temporal splits;
- protected Final Test isolation;
- Phase 3 selected feature set;
- causal feature computation;
- operational alert/event evaluation semantics;
- FAR/day constraint;
- Phase 3 selection outcome.

Phase 4 must not:

- re-run A–E feature selection;
- add or remove Phase 3 features based on Phase 4 performance;
- use Validation 2/3 to revise the Phase 3 feature freeze;
- use any 2022–2025 Final Test outcome;
- alter the Phase 3 confirmation ranking rule retrospectively.

## 6. Protected Final Test

The 2022–2025 Final Test remains locked.

Phase 3 used development periods only. No Phase 3 model fitting, threshold selection, feature ranking, advancement decision, confirmation decision, or exported result artifact depended on protected Final Test outcomes.

## 7. Closure Checklist

- [x] Phase 3 contract frozen before empirical A–E screening.
- [x] Exact A–E cumulative manifests implemented and tested.
- [x] Initial screening evaluator implemented.
- [x] Initial screening split isolation hardened.
- [x] Reproducible A–E runner implemented and tested.
- [x] Official A–E screening run completed.
- [x] Threshold boundaries audited.
- [x] Advancing candidates A/E/C frozen.
- [x] Permanent screening results record committed.
- [x] Walk-forward confirmation evaluator implemented.
- [x] Confirmation temporal isolation hardened.
- [x] Reproducible confirmation runner implemented and tested.
- [x] Official WF1/WF2 confirmation run completed.
- [x] Confirmation threshold boundaries and ranking audited.
- [x] Experiment A selected under the precommitted ranking rule.
- [x] Final Phase 3 feature contract frozen in code.
- [x] Protected Final Test remained locked.
- [x] Phase 3 outputs are ready to hand off to Phase 4.

## 8. Closure Statement

Phase 3 is formally complete.

The selected feature set is **Experiment A (10 raw causal predictors)**.

Any future change to that feature set would constitute a new protocol version or a separate post-hoc experiment and must not be presented as part of the frozen Phase 3 workflow.
