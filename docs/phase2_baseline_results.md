# Phase 2 Baseline Results Record

**Phase:** Baselines  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Run type:** Development-only empirical baseline evaluation  
**Protected Final Test:** 2022–2025 not used for development train/validation or exported outcome artifacts  
**Status:** Empirical run complete and audited; final full-suite closure check pending.

## 1. Frozen Baselines Evaluated

- **B0 Persistence:** `Kp(t-1h) >= 5`
- **B1 Physical:** `Bz < -5 nT AND V > 500 km/s`
- **B2 Logistic Regression:** 10 raw primary predictors, train-only scaling, no class weighting
- **B3 ExtraTrees:** 10 raw primary predictors, `n_estimators=100`, `max_depth=10`, `class_weight=None`, `random_state=42`

No baseline configuration was changed after observing these results.

## 2. Phase 2 Baseline-Evaluation Thresholds

B0/B1 use the fixed `0.5` adapter threshold because their outputs are deterministic `0/1`.

B2/B3 use the frozen Phase 2 development-only threshold rule:

```text
tau in {0.01, 0.02, ..., 0.99}
select minimum tau such that aggregate FAR/day <= 0.2
```

Selected thresholds:

| Baseline | Threshold |
|---|---:|
| B0 Persistence | 0.50 |
| B1 Physical | 0.50 |
| B2 Logistic Regression | **0.07** |
| B3 ExtraTrees | **0.07** |

The probabilistic threshold boundary audit is:

| Baseline | FAR/day at 0.06 | FAR/day at 0.07 | First feasible |
|---|---:|---:|---:|
| B2 Logistic | 0.217703 | **0.188898** | **0.07** |
| B3 ExtraTrees | 0.216595 | **0.181696** | **0.07** |

Thus `0.07` is the minimum feasible grid threshold for both B2 and B3 under the aggregate Phase 2 FAR/day constraint.

These are **Phase 2 baseline-evaluation thresholds only**. They do not replace the definitive global OOF threshold procedure reserved for Phase 6.

## 3. Fold-Level Operational Results

| Fold | Baseline | Event Recall | FAR/day | Median Early Lead |
|---|---|---:|---:|---:|
| screening | B0 Persistence | 0.568182 | 0.027504 | 5 h |
| screening | B1 Physical | 0.500000 | 0.019253 | 3.5 h |
| screening | B2 Logistic | 0.500000 | 0.250287 | 3 h |
| screening | B3 ExtraTrees | 0.454545 | 0.243410 | 3 h |
| walk_forward_1 | B0 Persistence | 0.400000 | 0.012384 | — |
| walk_forward_1 | B1 Physical | 0.400000 | 0.009632 | 4 h |
| walk_forward_1 | B2 Logistic | 0.533333 | 0.112831 | 3 h |
| walk_forward_1 | B3 ExtraTrees | 0.666667 | 0.108703 | 3.5 h |
| walk_forward_2 | B0 Persistence | 0.312500 | 0.031313 | — |
| walk_forward_2 | B1 Physical | 0.062500 | 0.011387 | — |
| walk_forward_2 | B2 Logistic | 0.812500 | 0.219191 | 3.5 h |
| walk_forward_2 | B3 ExtraTrees | 0.812500 | 0.204958 | 4 h |

A missing median early lead means that baseline produced no Early Detection episodes in that fold; Late Detections may still contribute to Event Recall.

## 4. Aggregate FAR Audit for Selected Probabilistic Thresholds

Total valid validation exposure:

```text
17,452 + 17,442 + 8,431 = 43,325 hours
```

B2 false-alarm episodes at `tau=0.07`:

```text
182 + 82 + 77 = 341
FAR/day = 341 / (43,325 / 24) = 0.188898
```

B3 false-alarm episodes at `tau=0.07`:

```text
177 + 79 + 72 = 328
FAR/day = 328 / (43,325 / 24) = 0.181696
```

Both satisfy the frozen aggregate `FAR/day <= 0.2` constraint.

Individual folds may exceed 0.2 FAR/day. This does not violate the Phase 2 implemented rule, which selects one development baseline threshold using aggregate valid exposure while preserving fold boundaries for alert episode construction. The fold variation is retained as an observed temporal-stability result rather than altered post hoc.

## 5. Reproducibility Artifacts

The official runner generates:

```text
results/phase2/baseline_fold_metrics.csv
results/phase2/baseline_selected_thresholds.csv
results/phase2/b2_logistic_threshold_curve.csv
results/phase2/b3_extratrees_threshold_curve.csv
```

Expected shapes from the audited run:

- `baseline_fold_metrics.csv`: 12 rows = 3 folds × 4 baselines
- `baseline_selected_thresholds.csv`: 4 rows
- `b2_logistic_threshold_curve.csv`: 99 rows
- `b3_extratrees_threshold_curve.csv`: 99 rows

The result artifacts do not export `prediction_time`, `target`, `y_true`, or Final Test outcome columns.

## 6. Scientific Interpretation Boundary

These baseline results establish reference operating behavior only. They do not justify changing frozen baseline configurations, selecting Phase 3 features, choosing Phase 4 imbalance strategies, or replacing the Phase 6 global OOF threshold protocol.

Notable temporal differences—such as lower deterministic-baseline recall in later folds and higher B2/B3 recall in 2021—are recorded as empirical observations. Their physical/statistical cause is not inferred from Phase 2 alone.

## 7. Closure State

Completed:

- empirical development run;
- threshold feasibility audit;
- metric arithmetic audit;
- artifact structure audit;
- Final Test isolation audit.

Remaining before formal Phase 2 closure:

- final complete test-suite run after this results record is committed;
- set final closure flags only after that suite passes.
