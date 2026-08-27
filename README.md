# Geomagnetic Storm Early Warning System

**Status:** Protocol v1.3 complete — Phases 0–9 closed  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Protected Final Test:** consumed once (2022–2025)  
**Frozen Model:** `lightgbm_lr0.1_leaves127`  
**Frozen Operational Threshold:** `tau = 0.10`

## Project Overview

This repository implements a scientifically controlled early-warning experiment for geomagnetic storms:

> Given only information available at prediction time `t`, can the system warn that geomagnetic storm conditions will occur within the next `H` hours?

The project emphasizes causal timestamp semantics, temporal leakage prevention, walk-forward development, pre-specified operational evaluation, a single protected Final Test, and explicitly post-hoc scientific diagnostics.

The v1.3 experiment is closed. Future model improvements informed by the protected-test outcome require a new protocol/version and a new unseen confirmatory evaluation set.

## Frozen Primary Configuration

| Parameter | Frozen value |
|---|---:|
| Storm threshold `T` | 5 |
| Forecast horizon `H` | 6 h |
| Event separation / termination `Z` | 6 h |
| Alert episode gap `C` | 3 h |
| Maximum FAR/day | 0.2 |
| Information cutoff | `t - 1h` |
| Canonical prediction grid | `[1996-01-01, 2026-01-01)` hourly |
| Protected Final Test | 2022–2025, consumed once |
| Causal feature universe | 93 |
| Selected model-input set | Phase 3 Experiment A — 10 raw features |
| Imbalance strategy | `none` |
| Model | `lightgbm_lr0.1_leaves127` |
| Operational threshold | `0.10` |

The canonical dataset contains 93 causally eligible predictors. Phase 3 selected the 10-feature raw family for the frozen primary model; the remaining causal features stayed available for audit infrastructure and later descriptive diagnostics but were not silently reintroduced into model selection.

## Target and Temporal Semantics

```text
y_event(t) = max(Kp[t+1:t+H]) >= T
T = 5
H = 6 h
window = (t, t+H]
information_cutoff = t - 1 h
```

Future retrospective Kp defines target/event truth but never enters the predictor matrix at prediction time.

Development used temporal screening/walk-forward folds ending in 2021. The 2022–2025 interval was isolated from all feature, imbalance, model, hyperparameter, threshold, horizon, severity, and candidate-selection decisions before its one-time Phase 8 evaluation.

## Phase Summary

| Phase | Scope | Status |
|---|---|---|
| 0 | Causality and temporal semantics | COMPLETE |
| 1 | Dataset, row status, temporal splits | COMPLETE |
| 2 | Baselines/evaluation | COMPLETE |
| 3 | Feature screening | COMPLETE / FROZEN |
| 4 | Imbalance experiments | COMPLETE / FROZEN |
| 5 | Model selection | COMPLETE / FROZEN |
| 6 | OOF threshold selection | COMPLETE / FROZEN |
| 7 | Horizon/severity experiments | COMPLETE / FROZEN |
| 8 | Protected Final Test | COMPLETE / CONSUMED ONCE |
| 9 | Post-hoc interpretation/scientific audit | COMPLETE / CLOSED |

## Development Freeze

Phase 5 selected:

```text
model = lightgbm_lr0.1_leaves127
```

Phase 6 selected the lowest OOF threshold satisfying the development constraint `FAR/day <= 0.2`:

```text
tau = 0.09 -> FAR/day = 0.20036 -> infeasible
tau = 0.10 -> FAR/day = 0.18552 -> feasible
```

Therefore `tau = 0.10` was frozen before protected-test outcome access.

Phase 7 executed only the pre-authorized horizon/severity truth variants and did not reopen the primary configuration. The primary task remained `T=5`, `H=6 h`.

## Official Protected Final Test Result

Phase 8 executed the frozen primary system exactly once on 2022–2025. The execution commit was:

```text
8c773b1804feabb5cbc1c8dcc08c5340fb20c236
```

| Metric | Final Test |
|---|---:|
| Event Recall | **0.5430** |
| Detected events | **82 / 151** |
| FAR/day | **0.3681** |
| Median early lead time | **3.0 h** |
| PR-AUC | **0.4960** |
| ROC-AUC | **0.8826** |
| Brier score | **0.03965** |

The pre-specified operational constraint `FAR/day <= 0.20` **did not generalize** to the protected Final Test. The threshold was not changed after observing this result.

The confirmatory conclusion is therefore deliberately limited: the selected predictor/model retained meaningful out-of-sample probabilistic discrimination, but the frozen operating point did not satisfy the required false-alarm burden and detected 54.3% of canonical storm events.

See `docs/phase8_results.md` and `docs/phase8_closure.md` for the authoritative interpretation.

## Phase 9 Scientific Audit

Phase 9 is post-hoc and explanatory only. It examined operational/temporal behavior, physical error regimes, event context/recurrence, pre-onset physical state, and signal timing. These diagnostics do not redefine the Phase 8 result and are not permitted to retune v1.3.

SHAP/model-internal attribution was not reconstructed by refitting the protected-test estimator; the limitation is documented rather than violating the no-refit/no-retuning boundary.

See `docs/phase9_closure.md` for the complete diagnostic synthesis.

## Reproducibility and Tests

Run the test suite with:

```bash
python -m pytest -q
```

The repository contains dedicated tests for causal cutoff semantics, future-mutation leakage, event/alert definitions, temporal splits, development/final-test isolation, model/threshold freeze logic, Phase 8 scoring/provenance, and Phase 9 diagnostics.

The Phase 8 environment was frozen separately in `requirements-lock-phase8.txt`. The repository also records the one-time execution commit and official metrics. A compact archival summary is retained under `artifacts/phase8_final/`.

Raw data are not committed. Historical machine-readable development outputs under `results/` are retained where already tracked; new generated result directories remain ignored by default.

## Scientific Status

Protocol v1.3 is closed. Its official result must not be improved retrospectively by changing features, imbalance handling, model, threshold, task horizon/severity, or event/alert semantics.

Any successor system may use the Phase 8/9 findings as hypothesis-generating evidence, but it must be labeled as a new protocol/version and evaluated on new unseen confirmatory data.

The repository-level closure record is `docs/project_v1.3_closure.md`.