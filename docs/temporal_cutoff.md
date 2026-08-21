# Temporal Cutoff and Predictor Availability

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implemented, tested, and frozen

## 1. Purpose

This document defines the project's generic predictor-side temporal cutoff for interval-based observations.

## 2. Canonical Information Cutoff

For prediction time `t`:

```text
information_cutoff(t) = t - 1h
```

## 3. Interval-Based Eligibility

For an observation whose source timestamp `s` is the start of an interval of duration `d`:

```text
period_start = s
period_end   = s + d
```

The observation is eligible only when:

```text
period_end <= information_cutoff(t)
```

Exact equality is eligible.

## 4. OMNI Hourly Measurements

The audited OMNI timestamp is the start of an hourly measurement period:

```text
period = [s, s + 1h)
```

Therefore:

```text
s + 1h <= t - 1h
```

The latest eligible OMNI period start is normally `t - 2h`.

## 5. Source Gaps

Source gaps remain gaps. The cutoff layer performs no forward fill, backward fill, interpolation, or timeline reconstruction.

## 6. Timestamp Integrity

Input timestamps must be valid, contain no `NaT`, contain no duplicates, and be monotonically increasing. Continuity is not required.

## 7. Source-Specific Semantics

Kp predictor history is handled separately by canonical Kp interval logic and `kp_asof()`:

```text
eligible iff canonical Kp interval_end <= query_time
```

Retrospective Kp used for event and target ground truth is not predictor-side information and is not filtered through this generic cutoff.

## 8. Canonical Implementation

Implemented in:

```text
src/temporal/cutoff.py
```

Feature-building code must reuse these temporal semantics rather than implement independent cutoff rules.

## 9. Frozen Rule

```text
prediction time = t
information cutoff = t - 1h
eligible iff period_end <= information_cutoff
```
