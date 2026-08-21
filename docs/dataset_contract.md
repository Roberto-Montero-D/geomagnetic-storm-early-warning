# Phase 1 Dataset Contract

## Scope

Phase 1 converts the frozen Phase 0 causal components into a deterministic, row-preserving supervised-learning dataset infrastructure. It adds no model fitting and no performance-driven scientific decisions.

## Canonical prediction universe

The canonical primary grid is hourly and half-open:

```text
[1996-01-01 00:00, 2026-01-01 00:00)
```

`prediction_time` is unique, monotonically increasing, timezone-naive, and aligned to whole hours. The grid exists independently of source-data availability; missing OMNI/Kp data must not silently remove prediction timestamps.

## Canonical assembled dataset

For each requested prediction timestamp, `build_canonical_dataset()` composes the already-frozen feature and target builders. The dataset contains exactly:

```text
93 PRIMARY_FEATURE_COLUMNS
+ target
```

The timestamp remains the index named `prediction_time`.

The builder does not independently redefine features or targets. It does not impute values, drop rows, assign eligibility, assign temporal splits, fit preprocessing, or fit models.

## Audit metadata isolation

With audit output enabled, feature provenance and target-ground-truth metadata are returned in a second frame. They are deliberately separated from the predictor/target frame so fields such as `information_cutoff`, `maximum_feature_information_time`, `future_window_end`, or `target_status` cannot accidentally become predictors.

## Missingness and unknown truth

All requested timestamps survive assembly. Missing feature values and unknown targets remain explicit.

Phase 1.3 classifies row state using:

```text
target_known
features_complete
n_missing_features
supervised_eligible
row_status
```

Frozen row-status values are:

```text
eligible
unknown_target
feature_incomplete
feature_incomplete_unknown_target
```

A row is `supervised_eligible` iff its target is known and all 93 frozen primary features are non-missing.

Feature incompleteness is not automatically called "warm-up" because missingness may arise from insufficient history, genuine source gaps, or other upstream availability effects.

## Causality and leakage invariants

All feature construction remains subject to the Phase 0 cutoff contract:

```text
maximum_feature_information_time <= prediction_time - 1h
```

Future retrospective Kp may change the target but must not change the 93 predictors. Phase 1 integration tests explicitly verify this invariant across the protected 2022 boundary.

## Temporal membership independence

Calendar split assignment depends only on `prediction_time`. Eligibility, feature missingness, target availability, and source availability cannot change period membership.

## Protected Final Test auditing

The Final Test (2022–2025) may be audited structurally before Phase 8. Permitted structural summaries include row count, feature completeness, and per-feature missingness.

The Phase 1 period-audit API redacts outcome-derived Final Test fields. It must not expose target prevalence, positive/negative counts, target-known counts, or supervised-eligibility summaries for Final Test rows.

## Non-claims

Phase 1 completion means the infrastructure is implemented and tested. It does not by itself claim empirical full-history row counts, class prevalence, missingness rates, or model performance. Such empirical results must be produced only by the appropriate later workflow and under the Final Test protection rules.
