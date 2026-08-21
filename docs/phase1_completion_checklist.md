# Phase 1 Completion Checklist

**Phase:** 1 — Dataset Construction and Temporal Splits  
**Status:** COMPLETE after repository tests pass and this closure batch is merged.

## 1. Canonical prediction grid

- [x] Hourly prediction universe is independent of source availability.
- [x] Primary coverage is `[1996-01-01 00:00, 2026-01-01 00:00)`.
- [x] Timestamps are unique, monotonic, timezone-naive, and whole-hour aligned.
- [x] Half-open boundaries and leap-day preservation are tested.

## 2. Canonical dataset assembly

- [x] Dataset reuses the frozen primary feature builder.
- [x] Dataset reuses the frozen event-window target builder.
- [x] Dataset exposes exactly 93 predictors plus `target`.
- [x] Every requested prediction timestamp is preserved exactly once.
- [x] Missing features are preserved rather than imputed or dropped.
- [x] Unknown targets are preserved rather than dropped.
- [x] Feature/target audit metadata are isolated from X/y.
- [x] Future Kp mutation can change y without changing X.

## 3. Row eligibility/status

- [x] `target_known` is explicit.
- [x] `features_complete` is explicit.
- [x] `n_missing_features` is explicit and restricted to the 93 frozen predictors.
- [x] `supervised_eligible` requires known target plus complete predictors.
- [x] Four deterministic row-status classes are implemented.
- [x] Status classification does not mutate or drop dataset rows.
- [x] Feature incompleteness is not automatically mislabeled as warm-up.

## 4. Temporal splits

- [x] Atomic periods are deterministic and non-overlapping.
- [x] Screening fold is 1996–2016 -> 2017–2018.
- [x] Walk-forward fold 1 is 1996–2018 -> 2019–2020.
- [x] Walk-forward fold 2 is 1996–2020 -> 2021.
- [x] Final Test is 2022–2025.
- [x] Final Test is excluded from all development train/validation masks.

## 5. Integration and isolation

- [x] Grid, dataset, audit, row-status, split, and fold outputs remain index-aligned.
- [x] Eligibility cannot change calendar membership.
- [x] Missing/unknown rows survive the protected boundary.
- [x] Eligibility filtering cannot pull Final Test into development masks.
- [x] Future truth mutation cannot alter X, row status, or split assignment when target-known state is unchanged.
- [x] Final Test membership depends only on prediction time.

## 6. Dataset auditing

- [x] Development periods can report descriptive target/eligibility summaries.
- [x] Feature missingness is auditable by period.
- [x] Final Test structural feature completeness remains auditable.
- [x] Final Test outcome-derived fields are redacted by the audit API.
- [x] Changing Final Test targets cannot change the structural Final Test audit output.

## 7. Documentation closure

- [x] Dataset contract documented.
- [x] Temporal split contract documented.
- [x] README synchronized to Phase 1 complete / Phase 2 next.
- [x] Historical frozen master protocol remains unchanged because its Phase 1 scientific definitions already match the implementation.
- [x] No empirical Final Test outcome statistics are introduced by Phase 1 documentation.

## 8. Phase gate

Phase 2 may begin only after:

```text
python -m pytest -v
```

passes on the closure commit and a final repository consistency audit confirms the documentation matches the implemented Phase 1 contracts.

Phase 2 begins with the pre-specified baselines. The protected Final Test remains locked.
