# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Phases 0–4 Complete; Phase 5 In Progress
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp >= 5

## 1. Project Overview

This repository implements a scientifically controlled early-warning system for geomagnetic storms.

The operational question is:

> Given only information that would have been available at prediction time `t`, can the system issue a reliable warning that geomagnetic storm conditions will occur within the next `H` hours?

Phase 0 froze the causal predictor/target infrastructure. Phase 1 froze the canonical prediction grid, row-preserving dataset assembly, supervised-row status, temporal development folds, Final Test isolation, and protected descriptive auditing. Phase 2 now implements the frozen baseline family and its development-only operational evaluation infrastructure.

The primary predictor universe remains:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

AE, Dst, and CDAW/LASCO CME-derived predictors remain excluded from the primary causal feature matrix.

## 2. Frozen Primary Configuration

| Parameter | Primary value |
|---|---:|
| Storm threshold `T` | 5 |
| Forecast horizon `H` | 6 h |
| Event separation / termination `Z` | 6 h |
| Alert episode gap `C` | 3 h |
| Maximum FAR/day | 0.2 |
| Information cutoff | `t - 1h` |
| Canonical prediction grid | `[1996-01-01 00:00, 2026-01-01 00:00)` hourly |
| Protected Final Test | 2022–2025 |
| Primary causal features | 93 |

The protected Final Test remains single-use and must not influence development decisions.

## 3. Completed Infrastructure

### Phase 0 — causality and temporal infrastructure

Completed Phase 0 infrastructure includes verified OMNI timestamp semantics; canonical Kp intervals and conservative predictor-side `kp_asof`; AE/Dst and CME availability audits; temporal cutoff infrastructure; storm-event and alert-episode construction; the frozen 93-feature causal pipeline; canonical target construction; and leakage/temporal-integrity tests.

### Phase 1 — canonical dataset and temporal splits

Completed Phase 1 infrastructure includes:

- continuous canonical hourly prediction grid for 1996–2025;
- row-preserving assembly of 93 predictors plus `target`;
- separate feature/target audit metadata;
- explicit row status and supervised eligibility without implicit dropping or imputation;
- deterministic atomic calendar periods and expanding development folds;
- adversarial integration/isolation tests around the protected 2022 boundary;
- descriptive development-period auditing;
- structural-only Final Test auditing with outcome-derived fields redacted.

Phase 1 completion refers to tested infrastructure. It does **not** claim that empirical full-dataset counts, prevalence, or model results have already been inspected.

### Phase 2 — frozen baselines and development evaluation

Implemented Phase 2 infrastructure includes:

- protected development-fold baseline framework;
- B0 persistence baseline;
- B1 physical baseline with frozen `Bz < -5 nT AND V > 500 km/s`;
- B2 unbalanced Logistic Regression on raw primary predictors;
- B3 unbalanced ExtraTrees on raw primary predictors with frozen `n_estimators=100`, `max_depth=10`;
- canonical alert/event operational evaluation reuse;
- development-only baseline threshold handling for probabilistic baselines;
- fold-preserving cross-fold FAR evaluation;
- explicit rejection of protected Final Test timestamps in Phase 2 train and validation inputs.

The official development-only Phase 2 baseline tables were generated and audited, and the complete repository test suite passed afterward. Phase 2 is formally complete.

### Phase 3 — feature screening

Phase 3 is formally complete. Experiment A (10 raw causal predictors) is frozen as the selected feature set for subsequent phases.

## 4. Canonical Target

```text
y_event(t) = max(Kp[t+1:t+H]) >= T

T = 5
H = 6h
window = (t, t+H]
```

Despite the historical name `y_event`, this is a future storm-condition target, not an event-onset-only target.

## 5. Temporal Information Rule

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

Predictor-side Kp uses the most recent canonical interval with `interval_end <= q`. Future retrospective Kp may define target/event truth but may never enter the predictor matrix.

## 6. Canonical Dataset Contract

The Phase 1 dataset is indexed by `prediction_time` and contains exactly:

```text
93 frozen primary predictor columns
+
target
```

Audit/provenance metadata are returned separately and are not predictor columns.

Dataset assembly preserves every requested prediction timestamp exactly once. It does not impute, drop incomplete rows, assign eligibility, assign splits, fit preprocessing, or fit models.

Row status is classified separately using:

```text
target_known
features_complete
n_missing_features
supervised_eligible
row_status
```

A row is supervised-eligible only when the target is known and all 93 frozen predictors are present.

## 7. Temporal Validation

| Atomic period | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Validation 2 | 2019–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

Development folds are expanding-window:

```text
screening       1996–2016 -> 2017–2018
walk_forward_1  1996–2018 -> 2019–2020
walk_forward_2  1996–2020 -> 2021
```

Final Test rows are excluded from every development train and validation mask, including after intersection with supervised eligibility.

## 8. Final Test Audit Protection

Before Phase 8, Final Test auditing is structural only. Allowed diagnostics include calendar row count, feature completeness, and per-feature missingness.

The Phase 1 audit API redacts Final Test outcome-derived fields such as target-known counts, supervised eligibility, positive/negative counts, and target prevalence. Development periods may expose those statistics.

Phase 2 adds downstream defense in depth: the cross-fold baseline evaluator independently rejects any training or validation index touching the protected Final Test.

## 9. Phase 2 Baseline Evaluation Thresholds

B0 and B1 are deterministic binary baselines and use a fixed adapter threshold of `0.5` only to pass their `0/1` outputs through the common alert interface.

B2 and B3 produce probabilities. During Phase 2 baseline comparison, development-only baseline-evaluation thresholds may be selected under the frozen operational constraint:

```text
FAR/day <= 0.2
```

These Phase 2 thresholds exist only to make probabilistic baselines operationally comparable to B0/B1. They are **not** the final production/global OOF threshold.

The definitive global OOF operational threshold procedure remains reserved for **Phase 6**, exactly as specified in `MASTER_PROTOCOL_v1.3.md`.

## 10. Current Project Status

```text
Phase 0 — causality and temporal infrastructure     COMPLETE
Phase 1 — dataset construction and temporal splits COMPLETE
Phase 2 — baselines and development evaluation     COMPLETE
Phase 3 — feature screening                        COMPLETE
Phase 4 — imbalance experiments                    COMPLETE
Phase 5 — model selection                          IN PROGRESS
Phase 6 — OOF operational threshold selection      PENDING
Phase 7 — horizon/severity experiments             PENDING
Phase 8 — protected Final Test                      LOCKED
Phase 9 — interpretation/scientific audit           PENDING
```

## 11. Canonical Documentation

Phase 0 and Phase 1 documents remain canonical. Phase 2 adds:

```text
docs/phase2_baseline_configuration_freeze.md
docs/phase2_b1_protocol_gap.md
docs/phase2_b3_protocol_gap.md
docs/phase2_completion_checklist.md
docs/phase2_baseline_results.md
docs/phase3_feature_screening_contract.md
docs/phase3_initial_screening_results.md
docs/phase3_feature_selection.md
docs/phase3_closure.md
```

The two Phase 2 protocol-gap documents are retained as historical decision records and are marked resolved by the baseline configuration freeze.

Historical protocol versions remain unchanged. `MASTER_PROTOCOL_v1.3.md` is not rewritten merely to record implementation completion.

## 12. Next Step

Begin Phase 5 model selection using only the Phase 3 selected feature set and
the frozen Phase 4 imbalance treatment (`none`), without accessing protected
2022–2025 Final Test outcomes.
