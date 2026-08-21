# CME Availability and Causality Audit

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Phase:** 0.3  
**Status:** Complete — CME excluded from the primary causal feature set  
**Decision timing:** Before model training, feature screening, threshold optimization, or final-test inspection

---

## 1. Purpose

Phase 0.3 asked whether historical CME information could be reconstructed such that every CME-derived predictor at prediction time `t` was based only on information demonstrably available by the project cutoff:

```text
information_cutoff = t - 1h
```

The audit separated two causal requirements:

1. **Measurement causality:** given a CME candidate, can its physical properties be reconstructed using only observations available by the historical cutoff?
2. **Candidate-event causality:** can it be demonstrated that the CME candidate itself was known or deterministically identifiable at that historical time?

This distinction is central to the final decision.

## 2. Original Planned Role of CME

Earlier protocol versions allowed CME-derived features only when historical availability could be demonstrated.

Candidate features included recent-CME counts, time since CME activity, kinematic quantities, and related temporal descriptors.

Phase 0.3 was required to operationalize that condition before any CME feature could enter the primary model.

## 3. Source Audited

The primary source investigated was the **SOHO/LASCO CME Catalog maintained by the CDAW Data Center**.

The catalog provides event-level quantities such as:

```text
Date
Time
Central PA
Width
Linear Speed
2nd-order speeds
Acceleration
Mass
Kinetic Energy
MPA
Remarks
```

These fields must not be assumed to have been available at the event observation time. Several are retrospective products derived from multiple observations.

The project experiment remains bounded to 1996–2025.

## 4. Retrospective Catalog Problem

The physical observation time of a CME is not equivalent to its historical catalog-availability time.

The complete Phase 0.3 audit identified:

```text
explicit retrospective insertions = 3,410
```

Therefore:

```text
event_time <= t
```

does not imply:

```text
catalog_candidate_known_at_t == True
```

This is a candidate-identity availability problem, not merely a missing-data problem.

## 5. CDAW Field Causality Assessment

| CDAW quantity | Primary causal predictor? | Reason |
|---|---:|---|
| First C2 appearance | No, not by itself | Observation time does not prove historical candidate identification |
| Central PA | No | Historical availability of the final catalog geometry is not established |
| Width | No | Final catalog value may depend on later CME evolution |
| Halo classification | No | Uniform historical availability is not established |
| Linear speed | No, final catalog value | Multi-point retrospective fit |
| 2nd-order speeds | No | Retrospective multi-point fits |
| Acceleration | No | Retrospective multi-point fit |
| Mass | No | Derived product; historical availability not established |
| Kinetic energy | No | Depends on derived mass and speed |
| MPA | No | Retrospective catalog geometry |
| Timestamped height-time points | Potentially causal measurements | Individual measurement timestamps can be filtered by cutoff |

The audit therefore distinguishes usable raw measurement timestamps from retrospective catalog products.

## 6. Historical Operational / Alert Source Investigation

Historical operational products, including LASCO Halo CME Mail, were investigated because issuance timestamps provide stronger evidence of information availability than retrospective catalog event times.

The investigation showed that first observation and operational issuance are distinct times, reporting delays can vary, messages may contain contemporaneous estimates, and resends or delayed reports can occur.

However, the archive represents a selected halo/notable-CME population rather than a complete homogeneous CME census. It was therefore rejected as a dependency of the primary feature pipeline.

## 7. CDAW Height-Time `.yht` Measurements

Individual CDAW CME pages expose height-time (`.yht`) records containing timestamped measurements such as:

```text
HEIGHT
DATE
TIME
ANGLE
TEL
FC
COL
ROW
```

The files can also contain retrospective metadata such as fitted speed, acceleration, width, onset estimates, and quality information.

The Phase 0.3 rule is:

```text
timestamped measurement rows
    -> eligible for causal reconstruction after applying the cutoff

final retrospective metadata
    -> audit/validation only
    -> not primary predictors
```

## 8. Acquisition and Parser Audit

The Phase 0.3 implementation developed and tested CDAW acquisition and parsing logic under:

```text
src/data/cme_cdaw.py
scripts/audit_cdaw_yht.py
scripts/retry_cdaw_failures.py
tests/test_cme_cdaw.py
```

The implementation was designed to:

- discover exact `.yht` links from CDAW monthly catalog pages;
- preserve raw source ordering;
- parse CDAW height-time measurement rows;
- separate measurement data from retrospective metadata;
- cache source files for reproducibility and to avoid unnecessary downloads;
- identify explicitly retrospective catalog insertions;
- detect duplicate measurement timestamps;
- detect non-monotonic measurement ordering;
- reject invalid heights;
- fail loudly when a measurement header is found but valid measurements cannot be parsed.

These files are retained as audit/research infrastructure even though CME predictors are excluded from the primary model.

## 9. Cross-Era Validation

Before the full archive audit, the parser and acquisition logic were exercised across multiple LASCO eras, including representative months from 1996, 2000, 2005, 2010, 2015, 2020, and 2025.

This validation established that the `.yht` representation was sufficiently stable to justify the full 1996–2025 audit.

## 10. Full 1996–2025 Quantitative Audit

| Metric | Result |
|---|---:|
| Successful `.yht` records | 42,422 |
| Remaining failed records | 4 |
| Explicit retrospective insertions | 3,410 |
| Provisional non-inserted candidates | 39,012 |
| Provisional candidate fraction | 91.9617% |
| ≥2 measurements | 42,346 |
| ≥3 measurements | 41,705 |
| ≥4 measurements | 39,900 |
| ≥5 measurements | 37,394 |
| Fraction with ≥3 | 98.3098% |
| C2 only | 14,838 |
| C3 only | 545 |
| C2 + C3 | 27,039 |
| Duplicate-timestamp trajectories | 21 |
| Non-monotonic trajectories | 17 |
| Invalid heights | 0 |
| Median first→third point | 0.408056 h |
| P75 first→third point | 0.796667 h |
| P90 first→third point | 1.020833 h |
| P95 first→third point | 1.400000 h |
| Third point within 1 h | 88.0470% |
| Third point within 3 h | 99.3070% |
| Third point within 6 h | 99.8945% |
| Third point within 12 h | 99.9952% |

**Conclusion:** measurement reconstruction passed the audit. The height-time trajectories are sufficiently complete and timely to support causal kinematic reconstruction once a candidate CME is defined.

## 11. Known LASCO / Catalog Coverage Gaps

The audit encountered four unavailable monthly catalog pages:

```text
1998-07
1998-08
1998-09
1999-01
```

The critical interpretation is:

```text
missing LASCO/catalog coverage != zero CME activity
```

Any operational CME-count feature would therefore require explicit source-coverage state rather than interpreting absence as zero activity.

Because CME is excluded from the primary feature universe, no CME imputation or coverage-state policy is required for the primary experiment.

## 12. Causal Kinematic Reconstruction

For prediction time `t`:

```text
information_cutoff = t - 1h
```

For candidate CME `j`, define the causally eligible measurement set:

```text
D_j(t) = {(t_i, h_i) : t_i <= information_cutoff}
```

The audit supports a provisional minimum of three valid measurements for causal linear-speed reconstruction:

```text
N_min = 3
```

subject to valid heights and acceptable timestamp integrity.

Phase 0.3 therefore demonstrated that **measurement causality is technically solvable**.

This result is conditional on the CME candidate already being defined.

## 13. Candidate-Event Causality

The unresolved problem is candidate identity.

```text
timestamped LASCO measurements
        solve
measurement availability

but

current CDAW event membership
        contains
retrospective catalog knowledge
```

### 13.1 Current CDAW Version 2

Direct historical use was rejected because the event universe includes substantial retrospective additions and revisions.

### 13.2 Retained CDAW Version 1

Version 1 is a useful earlier catalog state, but it is still a manually curated retrospective catalog rather than a per-event operational issuance stream. It therefore does not prove that every candidate was historically known at its first measurement time.

### 13.3 Version 1 / Version 2 Splice

A catalog-version splice was rejected for the primary experiment because it would introduce a catalog-regime transition and would affect the protected evaluation era.

### 13.4 Reconstructing Candidates From Imagery

A fixed automated detector applied causally to historical LASCO imagery could in principle create a defensible candidate universe. That would constitute a substantial separate image-level CME-detection project and is outside the scope of the present protocol.

## 14. Candidate-Sensitive Features

Features especially sensitive to retrospective candidate completeness include:

```text
cme_count_24h
cme_count_48h
cme_count_72h
hours_since_last_cme
```

Their historical values can change when a CME is retrospectively added to the catalog even though the physical observations available at the historical prediction time have not changed.

Reconstructed kinematic features also remain candidate-dependent and are therefore excluded from the primary system despite the successful measurement-level audit.

## 15. Final Decision

CDAW/LASCO CME-derived predictors are **excluded from the primary causal feature set**.

This decision is not based on CME data quality or predictive performance.

The full Phase 0.3 audit demonstrated that timestamped LASCO height-time measurements are sufficiently complete and timely to support causal kinematic reconstruction after a CME candidate is defined.

The exclusion is instead caused by the absence of a uniform, historically defensible candidate-event availability rule across 1996–2025. The CDAW event universe is manually curated and retrospectively revised, and neither the investigated Version 1 nor Version 2 candidate universe provides per-event historical availability semantics sufficient for the primary causal standard of this project.

The primary operational feature universe is therefore restricted to:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

CDAW/LASCO acquisition, parsing, tests, and audit results are retained as reproducible methodological evidence and may support a future separately specified CME research extension.

## 16. Non-Performance-Driven Decision

No model was trained with CME predictors before this exclusion decision.

No CME feature was retained or removed based on:

- validation performance;
- Event Recall;
- FAR/day;
- PR-AUC;
- feature importance;
- SHAP;
- final-test performance.

The decision was based exclusively on Phase 0 source semantics, historical availability, reproducibility, and leakage control.

## 17. Implications for the Primary Protocol

```text
PRIMARY FEATURE SOURCES
    OMNI solar-wind measurements
    causal Kp history

INGESTED / AUDIT-ONLY
    AE
    Dst

RESEARCH / AUDIT-ONLY
    CDAW/LASCO CME

NOT PRIMARY FEATURES
    AE
    Dst
    CME
```

CME is removed from primary feature screening, primary model inputs, threshold optimization inputs, and the protected final-test feature universe.

## 18. Future Work

CME remote-sensing information may be revisited only under a new protocol/version if a homogeneous timestamped operational CME event stream is established or a frozen automated CME detector is applied causally to historical timestamped imagery.

Such an extension may compare:

```text
OMNI + causal Kp
vs.
OMNI + causal Kp + causally defined remote-sensing CME information
```

without altering the present experiment.
