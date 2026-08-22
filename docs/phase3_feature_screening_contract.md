# Phase 3 Feature Screening Contract Freeze

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Frozen before Phase 3 empirical feature-set performance inspection.

## 1. Purpose

Phase 3 determines whether the frozen causal feature families add operational
value beyond the raw predictor family. It does not introduce new sources,
change the target, change temporal splits, tune model hyperparameters, or use
the protected 2022–2025 Final Test.

## 2. Cumulative Screening Experiments

The five experiments are cumulative and use the canonical Phase 0/1 feature
manifests.

| Experiment | Included families |
|---|---|
| A | raw |
| B | raw + rolling |
| C | raw + rolling + persistence |
| D | raw + rolling + persistence + dynamics |
| E | raw + rolling + persistence + dynamics + interactions |

The canonical family order is:

```text
raw -> rolling -> persistence -> dynamics -> interactions
```

No individual feature is selected or removed inside a family during Phase 3.

## 3. Screening Split

The initial screening comparison uses only:

```text
Train:      1996–2016
Validation: 2017–2018
```

Validation 2 (2019–2020), Validation 3 (2021), and Final Test (2022–2025)
must not influence which feature sets advance from initial screening.

## 4. Screening Model Freeze

To isolate the effect of feature families, every A–E experiment uses the same
unbalanced ExtraTrees configuration already frozen for B3:

```text
n_estimators = 100
max_depth = 10
class_weight = None
random_state = 42
```

No Phase 5 hyperparameter search is performed in Phase 3.

## 5. Screening Metrics

Each A–E experiment records:

```text
Event Recall
FAR/day
PR-AUC
```

PR-AUC is computed from the complete Validation 1 target/probability series
using average precision (`sklearn.metrics.average_precision_score`), after
restricting evaluation to the explicit supervised validation rows.

Operational Event Recall and FAR/day use the canonical Phase 0 event/alert
machinery.

## 6. Phase 3 Screening Threshold

For each A–E probability series independently:

```text
tau in {0.01, 0.02, ..., 0.99}
```

Select the minimum threshold satisfying:

```text
FAR/day <= 0.2
```

using Validation 1 only.

This is a Phase 3 development screening threshold. It is not the definitive
Phase 6 global OOF operational threshold.

If no threshold satisfies the constraint, the experiment is operationally
infeasible and cannot advance.

## 7. Initial Screening Advancement Rule

Exactly **three** feature sets advance from A–E initial screening when at least
three are operationally feasible.

Ranking among feasible sets is frozen as:

1. higher Event Recall;
2. higher PR-AUC;
3. lower FAR/day;
4. smaller feature set (earlier experiment A < B < C < D < E).

If fewer than three feature sets satisfy `FAR/day <= 0.2`, all feasible sets
advance. If none are feasible, Phase 3 stops and the operational failure is
reported rather than relaxing the FAR constraint.

This resolves the protocol's previously underspecified “2–3 best feature
sets” language before Phase 3 performance is inspected.

## 8. Walk-Forward Confirmation

The advancing sets are confirmed on the two later development folds:

```text
1996–2018 -> 2019–2020
1996–2020 -> 2021
```

Each confirmation fold trains a fresh model using only that fold's training
rows. Each fold receives its own development-only threshold selected from that
fold's validation probabilities under `FAR/day <= 0.2`.

The final Phase 3 feature-set choice is based on consistency across the two
confirmation folds plus the original screening result.

## 9. Final Phase 3 Selection Rule

A candidate is confirmation-feasible only if it satisfies `FAR/day <= 0.2` in
**both** confirmation folds.

Among confirmation-feasible candidates, rank by:

1. highest minimum Event Recall across the two confirmation folds;
2. highest mean Event Recall across the two confirmation folds;
3. highest mean PR-AUC across the two confirmation folds;
4. lowest mean FAR/day across the two confirmation folds;
5. smaller feature set.

The top-ranked candidate becomes the frozen Phase 3 feature set.

If no candidate satisfies the FAR constraint in both confirmation folds, Phase
3 records operational instability and stops for protocol review rather than
silently changing the constraint or selection rule.

The initial 2017–2018 screening metrics are retained for transparency but are
not included numerically in the final confirmation ranking; they were already
used to select the candidates entering confirmation.

## 10. Leakage Protection

Phase 3 must preserve all existing protections:

- no target column in predictors;
- no validation rows used for model fitting;
- no later validation period used during initial A–E screening;
- no Final Test row used in fitting, threshold selection, ranking, or metrics;
- no AE, Dst, or CME-derived predictor;
- no post-result modification of A–E manifests, model configuration, ranking
  rules, or FAR constraint.

## 11. Freeze Boundary

This document is frozen before the first official A–E empirical screening run.
Implementation bugs may be corrected, but methodological rules above must not
be changed because of Phase 3 performance.
