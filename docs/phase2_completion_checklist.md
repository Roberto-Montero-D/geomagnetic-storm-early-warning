# Phase 2 Completion Checklist

**Phase:** Baselines  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implementation complete; empirical development run and final closure pending.

## 1. Protected Development Framework

- [x] Frozen Phase 1 development folds reused.
- [x] Only supervised-eligible rows enter baseline train/validation materialization.
- [x] Predictor manifests are explicit.
- [x] `target` cannot enter predictor matrices.
- [x] Protected 2022–2025 Final Test is excluded from development folds.
- [x] Cross-fold evaluation independently rejects Final Test timestamps in both train and validation.

## 2. Baselines

### B0 — Persistence

- [x] Uses canonical causal `kp_lag_1h`.
- [x] Frozen rule is `Kp(t-1h) >= 5`.
- [x] No fitting or tuning.
- [x] Missing predictor state is not silently converted to a negative.

### B1 — Physical

- [x] Uses only canonical causal `bz_gsm` and `speed`.
- [x] Frozen rule is `Bz < -5 nT AND V > 500 km/s`.
- [x] Threshold configuration was frozen before official empirical baseline evaluation.
- [x] Original protocol gap is retained as a resolved decision record.

### B2 — Logistic Regression

- [x] Uses the 10 raw primary predictors only.
- [x] Uses train-only `StandardScaler`.
- [x] Uses unbalanced Logistic Regression (`class_weight=None`).
- [x] Returns validation probabilities.
- [x] Validation targets/features cannot alter fitted preprocessing/model state.

### B3 — ExtraTrees

- [x] Uses the 10 raw primary predictors only.
- [x] Uses `n_estimators=100`.
- [x] Uses `max_depth=10`.
- [x] Uses `class_weight=None`.
- [x] Uses `random_state=42`.
- [x] Configuration was frozen before official empirical baseline evaluation.
- [x] Original protocol gap is retained as a resolved decision record.

## 3. Operational Evaluation

- [x] Canonical Phase 0 alert construction is reused.
- [x] Canonical event association is reused.
- [x] Event Recall is available.
- [x] FAR/day uses valid exposure.
- [x] Lead Time uses Early Detections.
- [x] Late detections are distinguished from early detections.
- [x] B0/B1 deterministic outputs are normalized through the common probability interface.
- [x] Validation folds remain separate operational timelines.
- [x] Alert episodes cannot bridge multi-year validation gaps.
- [x] Cross-fold FAR is aggregated as total false-alarm episodes / total valid exposure days.
- [x] Events beginning after a validation period ends are not counted as evaluable in that fold.

## 4. Phase 2 Baseline Threshold Semantics

- [x] B0/B1 use fixed adapter threshold `0.5` only for common alert-interface compatibility.
- [x] B2/B3 may use development-only baseline-evaluation thresholds.
- [x] Threshold grid is ordered and unique.
- [x] Primary candidate grid remains `0.01 ... 0.99`.
- [x] Operational constraint remains `FAR/day <= 0.2`.
- [x] Minimum feasible threshold rule is enforced.
- [x] Phase 2 baseline-evaluation thresholds are explicitly distinguished from the definitive Phase 6 global OOF operational threshold.

## 5. Leakage and Final Test Protection

- [x] Development prediction generation ignores rows outside explicit folds.
- [x] Validation targets cannot alter B0–B3 predictions.
- [x] Final Test-like appended rows cannot alter development predictions.
- [x] Cross-fold evaluator requires the canonical split table.
- [x] Final Test in training is rejected.
- [x] Final Test in validation is rejected.
- [x] Phase 2 threshold selection does not use protected Final Test outcomes.

## 6. Documentation and Configuration

- [x] B1 protocol gap marked resolved.
- [x] B3 protocol gap marked resolved.
- [x] Baseline configuration freeze document retained.
- [x] README synchronized with Phase 2 implementation state.
- [x] `config/config.yaml` records frozen B1/B3 baseline configurations.
- [x] `config/config.yaml` distinguishes Phase 2 baseline thresholds from Phase 6 global OOF threshold selection.
- [x] Phase 2 completion checklist added.
- [x] Historical `MASTER_PROTOCOL_v1.3.md` remains unchanged.

## 7. Remaining Closure Items

- [ ] Add reproducible real-data Phase 2 baseline runner.
- [ ] Run B0–B3 on the real protected development folds.
- [ ] Save development-only result artifacts.
- [ ] Audit result artifacts for Final Test leakage and metric consistency.
- [ ] Confirm selected B2/B3 baseline thresholds satisfy the frozen FAR/day rule.
- [ ] Confirm no protected 2022–2025 outcome-derived information appears in Phase 2 artifacts.
- [ ] Run the complete test suite after the empirical runner is added.
- [ ] Set `full_suite_passing_at_closure: true`.
- [ ] Set `empirical_development_run_complete: true`.
- [ ] Set `empirical_results_audited: true`.
- [ ] Set `phase_2_complete: true`.
- [ ] Update README status from implementation complete to Phase 2 complete.
- [ ] Begin Phase 3 only after all items above are complete.

## Closure Rule

Phase 2 is formally complete only when the baseline implementation,
development-only empirical run, result audit, documentation/configuration, and
full test suite are all complete without accessing protected Final Test
outcomes.
