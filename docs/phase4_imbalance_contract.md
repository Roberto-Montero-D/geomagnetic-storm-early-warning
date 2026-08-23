# Phase 4 Imbalance-Handling Contract

**Status:** Frozen before empirical Phase 4 evaluation.

Phase 4 asks which pre-specified class-imbalance treatment maximizes robust operational Event Recall while satisfying `FAR/day <= 0.2`.

## Fixed inputs
Phase 4 uses only `PHASE3_SELECTED_FEATURES` (Experiment A, 10 predictors). Target, causal cutoff, event/alert definitions, temporal splits, threshold grid, FAR constraint, and protected Final Test isolation remain unchanged. ExtraTrees remains fixed at 100 trees, max depth 10, random state 42; model/hyperparameter selection belongs to Phase 5.

## Frozen grid
The protocol expands to 17 configurations: none; class positive weights 1/3/5/10/20/50; random undersampling negative:positive 10:1/5:1/2:1; SMOTE k=3/5/7; Borderline-SMOTE k=3/5/7; and SMOTE-ENN. Resampling is training-only; validation data are never resampled or used to fit a sampler.

## Screening
Use Initial Train -> Validation 1 only. For each configuration choose the lowest threshold satisfying `FAR/day <= 0.2`; otherwise mark infeasible. Rank by feasibility, Event Recall, PR-AUC, lower FAR/day, then frozen experiment order. Exactly the top 3 configurations advance.

## Confirmation
Refit the three advancing configurations independently on WF1 and WF2. Each fold uses its own minimum feasible development threshold. Final ranking requires feasibility in both folds, then highest worst-fold Event Recall, mean Event Recall, mean PR-AUC, lower mean FAR/day, and frozen experiment order.

Calibration and threshold stability are diagnostic only and cannot override selection. No Phase 4 result may change the feature set, grid, ranking, advancement count, FAR constraint, model structure, or protected Final Test policy.
