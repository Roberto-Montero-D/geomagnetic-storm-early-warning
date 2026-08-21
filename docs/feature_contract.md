# Phase 0.7 — Causal Feature Contract

**Status:** Frozen and implemented  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Primary feature count:** 93

## 1. Purpose

This document freezes the implementation semantics of the primary causal
feature universe used by the geomagnetic-storm early-warning system.

All predictor features must satisfy the project information cutoff:

```text
information_cutoff(t) = t - 1h
maximum_feature_information_time <= information_cutoff(t)
```

No feature may use an observation, interval, value, or derived quantity whose
information time exceeds that cutoff.

The primary source universe remains:

- OMNI solar-wind measurements;
- causal historical Kp values through the canonical `kp_asof` mapping.

CDAW/LASCO CME information, AE, and Dst are not part of the primary feature
matrix under Protocol v1.3.

---

## 2. Frozen feature manifest

The primary matrix contains exactly 93 features in deterministic family order:

```text
raw            10
rolling        60
persistence     5
dynamics       15
interactions    3
-----------------
total          93
```

The family order is:

```text
raw
rolling
persistence
dynamics
interactions
```

Column order is part of the implementation contract and is defined by
`PRIMARY_FEATURE_COLUMNS` in `src/features/integrated.py`.

Duplicate feature names are prohibited.

---

## 3. Raw features — 10

The raw primary features are:

```text
bz_gsm
bt
speed
density
flow_pressure
kp_lag_1h
kp_lag_3h
kp_lag_6h
kp_lag_12h
kp_lag_24h
```

### 3.1 OMNI timestamp semantics

OMNI source timestamps represent hourly interval starts.

For prediction time `t`:

```text
information_cutoff = t - 1h
latest eligible hourly OMNI interval = [t - 2h, t - 1h)
latest eligible OMNI source timestamp = t - 2h
```

The exact latest eligible timestamp is required for raw OMNI state features.

If that timestamp is absent, the raw OMNI features are missing. The builder
must not fall back to an older row.

### 3.2 Kp semantics

Predictor-side Kp is obtained only through the canonical causal mapping:

```text
kp_lag_Lh(t) = kp_asof(t - Lh)
```

where `kp_asof(q)` returns the most recent canonical Kp interval satisfying:

```text
interval_end <= q
```

Raw repeated hourly Kp values must not be used as an independent lagging
mechanism.

---

## 4. Rolling features — 60

Rolling features are built for each of:

```text
bz_gsm
bt
speed
density
flow_pressure
```

using windows:

```text
3h
6h
12h
24h
```

and statistics:

```text
mean
min
std
```

This produces:

```text
5 variables × 4 windows × 3 statistics = 60 features
```

### 4.1 Physical-time window semantics

Rolling windows are physical-time windows, not N-row windows.

For prediction time `t`, cutoff `c = t - 1h`, and window length `W`:

```text
c - W < interval_end <= c
```

A missing source timestamp does not cause the window to extend farther into
the past to collect a fixed number of rows.

### 4.2 Missing values

Missing timestamps and missing values are not imputed.

Statistics use valid observations physically present inside the fixed window.
If no valid values exist for a variable/window, the statistic is `NaN`.

Standard deviation uses sample standard deviation:

```text
ddof = 1
```

and is therefore `NaN` when fewer than two valid observations exist.

---

## 5. Persistence features — 5

The frozen persistence conditions are:

```text
bz_gsm < -5 nT
bz_gsm < -10 nT
bz_gsm < -15 nT
speed > 500 km/s
speed > 600 km/s
```

Persistence is the number of consecutive valid hourly intervals satisfying
the condition and ending at the exact latest causally eligible OMNI interval.

The thresholds are strict inequalities.

A missing timestamp or missing value breaks an existing run.

If the latest eligible state itself is unavailable, persistence is `NaN`, not
zero, because the current condition is unknown.

No older timestamp may substitute for a missing latest state.

---

## 6. Dynamic features — 15

For each primary OMNI variable:

```text
bz_gsm
bt
speed
density
flow_pressure
```

the frozen dynamics are:

```text
delta_1h
delta_3h
slope_3h
```

This produces:

```text
5 variables × 3 dynamics = 15 features
```

### 6.1 Exact deltas

For latest eligible source time `s`:

```text
delta_1h = value(s) - value(s - 1h)
delta_3h = value(s) - value(s - 3h)
```

Both timestamps must exist exactly and both values must be valid.

Nearby rows must not substitute for missing required timestamps.

### 6.2 Frozen 3-hour slope convention

The previously generic protocol term `3h slope` is implemented as an
ordinary-least-squares slope in variable units per hour using the four exact
hourly samples:

```text
s - 3h
s - 2h
s - 1h
s
```

These four observations span three elapsed hours.

This convention intentionally makes `slope_3h` a trend estimate rather than
an alias for `delta_3h / 3`.

If any required timestamp or value is missing, `slope_3h` is `NaN`.

---

## 7. Interaction features — 3

Define:

```text
Bz_neg = max(-bz_gsm, 0)
```

The frozen interactions are:

```text
bz_neg_x_speed        = Bz_neg × speed
bz_neg_x_density      = Bz_neg × density
flow_pressure_x_speed = flow_pressure × speed
```

Interactions consume the already-constructed causal raw feature frame. They
must not independently select OMNI observations.

Missing raw inputs propagate to the affected interaction feature.

---

## 8. Missing-data contract

Phase 0.7 performs no imputation.

The following are prohibited:

- replacing a missing exact raw timestamp with an older observation;
- extending a rolling window backward because an expected timestamp is absent;
- substituting nearby timestamps for exact dynamic-feature timestamps;
- interpreting an unavailable latest persistence state as condition false;
- dropping a prediction row merely because one or more features are missing.

OMNI source fill sentinels for primary variables are converted to `NaN` at
the feature-construction boundary.

Missing-data handling for model training is a later modeling decision and
must not alter the causal feature definitions in this document.

---

## 9. Integrated feature frame

The canonical integrated builder is:

```text
build_primary_feature_frame()
```

in:

```text
src/features/integrated.py
```

It combines all five feature families using a shared prediction-time index.

The integrated builder requires:

- identical prediction indexes across feature families;
- deterministic manifest order;
- unique feature names;
- preservation of prediction rows with missing feature values.

---

## 10. Provenance and leakage invariant

Each causal feature family exposes information-time provenance.

The integrated audit consolidates these into:

```text
raw_information_time
rolling_information_time
persistence_information_time
dynamics_information_time
interaction_information_time
maximum_feature_information_time
information_cutoff
```

For every prediction row with available provenance:

```text
maximum_feature_information_time <= information_cutoff
```

and:

```text
information_cutoff = prediction_time - 1h
```

Violation of this invariant is an implementation error.

Interactions inherit the information time of the raw OMNI state from which
they are computed.

---

## 11. Phase 0.7 verification

The Phase 0.7 test suite verifies, among other cases:

- exact latest eligible OMNI selection;
- canonical Kp `asof` semantics;
- source fill-value handling;
- physical-time rolling-window boundaries;
- no rolling-window row-count backfill;
- strict persistence thresholds;
- missing timestamps breaking persistence runs;
- exact 1h and 3h dynamic timestamps;
- the frozen four-sample OLS 3h slope;
- interaction definitions and NaN propagation;
- deterministic 93-feature manifest order and uniqueness;
- future OMNI mutation cannot alter past feature vectors;
- future Kp mutation cannot alter past feature vectors;
- missing latest OMNI timestamps are not silently substituted;
- integrated provenance never exceeds the `t - 1h` cutoff.

Phase 0.7 is considered closed when these tests and the complete repository
test suite pass.

---

## 12. Frozen status

The definitions in this document are methodological decisions.

They must not be changed in response to validation or final-test performance
without an explicit protocol amendment.

Any future extension of the source universe or feature definitions must be
identified as a protocol extension rather than silently added to the primary
93-feature matrix.
