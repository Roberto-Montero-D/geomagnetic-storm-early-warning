# Phase 5 — Model Selection Contract Freeze

**Status:** FROZEN BEFORE PHASE 5 PERFORMANCE INSPECTION

Phase 5 compares the model families and hyperparameter grids already specified
by `MASTER_PROTOCOL_v1.3.md`.

## Inherited decisions

Phase 5 inherits, without reopening:

- the Phase 3 selected feature set;
- the Phase 4 imbalance strategy `none`;
- the canonical target/event/alert definitions;
- `FAR/day <= 0.2`;
- the existing temporal development folds;
- complete protection of the 2022–2025 Final Test.

## Frozen model grid

### ExtraTrees
- `n_estimators`: 100, 200, 500
- `max_depth`: 10, 20, None

### LightGBM
- `learning_rate`: 0.01, 0.05, 0.1
- `num_leaves`: 31, 63, 127

### XGBoost
- `learning_rate`: 0.01, 0.05, 0.1
- `max_depth`: 3, 6, 9

There are exactly 27 configurations: nine per family.

## Reproducibility defaults

`random_state = 42` is frozen for all model families.

Other library parameters not named by the Master Protocol will remain at the
library default unless a compatibility or deterministic-execution parameter is
required. Such parameters must not be tuned from Phase 5 performance.

No class weighting or resampling is permitted.

## Screening

All 27 configurations are trained on the canonical screening training rows and
evaluated on Validation 1 only.

For each configuration, threshold selection uses the existing canonical
development operational evaluator and the frozen threshold grid 0.01–0.99.

A configuration is operationally feasible only if a threshold satisfies:

`FAR/day <= 0.2`

Within each model family, screening ranking is:

1. operationally feasible first;
2. highest Event Recall;
3. highest PR-AUC;
4. lowest FAR/day;
5. frozen configuration order.

Exactly the highest-ranked configuration from each family advances. This rule
is frozen pre-results so confirmation compares ExtraTrees, LightGBM, and
XGBoost rather than allowing one family to occupy every confirmation position.

## Walk-forward confirmation

The three advancing family winners are evaluated independently on WF1 and WF2.

The confirmation ranking is:

1. feasible in both folds;
2. highest minimum (worst-fold) Event Recall;
3. highest mean Event Recall;
4. highest mean PR-AUC;
5. lowest mean FAR/day;
6. frozen candidate order.

The winner becomes the frozen Phase 5 model/configuration for Phase 6.

## Stacking gate

The Master Protocol permits stacking only if ExtraTrees and LightGBM demonstrate
complementary errors.

Therefore stacking is **not authorized by default** in Phase 5.

After individual-model confirmation, a separate pre-results diagnostic contract
must define an error-complementarity criterion before any stacked model is fit.
If that gate was not frozen before inspecting the relevant error diagnostic,
stacking is skipped.

No stacking result may be used to retroactively change the individual-model
screening or confirmation rules.

## Final Test

The protected 2022–2025 Final Test must not be used for:

- fitting;
- hyperparameter selection;
- threshold selection;
- candidate advancement;
- stacking authorization;
- model selection.

It remains locked until the protocol-defined Final Test phase.
