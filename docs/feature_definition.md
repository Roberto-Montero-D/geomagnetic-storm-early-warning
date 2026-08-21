# Primary Causal Feature Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implemented, tested, and frozen

## 1. Purpose

This document defines the complete primary causal feature universe implemented
in Phase 0.7.

The feature set is constructed only from:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

AE, Dst, and CDAW/LASCO CME-derived predictors remain excluded from the
primary feature universe.

## 2. Global Causal Rule

For prediction time `t`:

```text
information_cutoff = t - 1h
```

For hourly OMNI observations whose raw timestamp `s` is the start of the
represented interval:

```text
period = [s, s + 1h)
period_end = s + 1h
```

An OMNI observation is eligible only when:

```text
period_end <= information_cutoff
```

Therefore, the latest eligible raw OMNI interval start is normally:

```text
t - 2h
```

All feature families must satisfy:

```text
maximum_feature_information_time <= prediction_time - 1h
```

No feature family may independently weaken this rule.

## 3. Missingness and Source Gaps

Missing timestamps and missing values remain missing.

The feature pipeline performs no:

```text
forward fill
backward fill
nearest-row substitution
implicit interpolation
timeline reconstruction
```

If an exact timestamp required by a feature is unavailable, that feature must
remain `NaN` unless its definition explicitly permits computation from other
valid observations inside a fixed physical-time window.

Prediction rows are not dropped merely because one or more features are
missing.

OMNI source fill values are converted to `NaN` at the feature boundary before
feature calculations.

## 4. Feature Family Order

The complete primary feature frame is assembled in the frozen order:

```text
raw
rolling
persistence
dynamics
interactions
```

The canonical feature manifest contains exactly:

```text
raw           10
rolling       60
persistence    5
dynamics      15
interactions   3
----------------
total         93
```

Column names and ordering are deterministic and unique.

## 5. Raw Features

The raw family contains:

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

The five OMNI physical variables use the exact latest causally eligible hourly
interval.

If that exact OMNI timestamp is absent, the raw OMNI values remain `NaN`;
the implementation must not fall back to an older measurement.

Predictor-side Kp uses only the canonical Kp mapping:

```text
kp_lag_h(t) = kp_asof(t - h)
```

where `kp_asof(q)` returns the most recent canonical 3-hour Kp interval whose:

```text
interval_end <= q
```

Retrospective event or target Kp never enters the predictor-side raw feature
builder.

## 6. Rolling Features

Rolling features are calculated for each primary OMNI physical variable:

```text
bz_gsm
bt
speed
density
flow_pressure
```

Frozen windows:

```text
3h
6h
12h
24h
```

Frozen statistics:

```text
mean
min
std
```

For a W-hour window at prediction time `t`:

```text
cutoff = t - 1h
cutoff - W < period_end <= cutoff
```

Rolling windows are defined in physical time, not by row count.

A missing source timestamp inside a window does not cause the window to reach
farther backward to collect an additional row.

Statistics use the valid observations physically present inside the fixed
window. Missing values are ignored without imputation.

If no valid values exist for a variable/window, the result is `NaN`.

Rolling standard deviation uses sample standard deviation:

```text
ddof = 1
```

Therefore, fewer than two valid values produce `NaN` for `std`.

## 7. Persistence Features

Persistence features measure the duration, in consecutive valid hourly
intervals ending at the latest causally eligible OMNI interval, for which an
adverse condition remains satisfied.

Frozen Bz conditions:

```text
bz_gsm < -5 nT
bz_gsm < -10 nT
bz_gsm < -15 nT
```

Frozen speed conditions:

```text
speed > 500 km/s
speed > 600 km/s
```

Canonical persistence columns:

```text
bz_gsm_persist_lt_m5h
bz_gsm_persist_lt_m10h
bz_gsm_persist_lt_m15h
speed_persist_gt_500h
speed_persist_gt_600h
```

The threshold comparisons are strict.

A missing timestamp or missing value breaks an active persistence run.

If the latest eligible state itself is unavailable, the persistence value is
`NaN`, not zero, because the current condition is unknown.

## 8. Dynamic Features

Dynamic features are calculated for all five primary OMNI physical variables.

Frozen dynamic features:

```text
delta_1h
delta_3h
slope_3h
```

For the latest causally eligible source state `x(t0)`:

```text
delta_1h = x(t0) - x(t0 - 1h)
delta_3h = x(t0) - x(t0 - 3h)
```

Deltas require the exact physical timestamps. Missing required timestamps or
values produce `NaN`.

### 8.1 Frozen 3-Hour Slope Convention

The protocol originally specified a 3-hour slope without defining the exact
numerical estimator. Phase 0.7 freezes the implementation convention as an
ordinary-least-squares slope, in variable units per hour, using the four exact
hourly samples spanning three elapsed hours:

```text
t0 - 3h
t0 - 2h
t0 - 1h
t0
```

This convention makes `slope_3h` a trend estimate distinct from the
endpoint-only `delta_3h`.

All four exact timestamps and values must be present. Otherwise, the slope is
`NaN`.

This is an implementation clarification, not a performance-driven feature
change.

## 9. Interaction Features

Frozen interactions:

```text
Bz_neg * speed
Bz_neg * density
flow_pressure * speed
```

where:

```text
Bz_neg = max(-bz_gsm, 0)
```

Canonical interaction columns:

```text
bz_neg_x_speed
bz_neg_x_density
flow_pressure_x_speed
```

Interactions consume the already-causal raw feature frame. They do not
perform independent source selection.

Missing raw inputs propagate naturally to the affected interaction features.

## 10. Canonical Manifest

The integrated primary feature frame contains 93 features.

### 10.1 Raw — 10

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

### 10.2 Rolling — 60

For each of:

```text
bz_gsm
bt
speed
density
flow_pressure
```

construct:

```text
roll_mean_3h
roll_min_3h
roll_std_3h

roll_mean_6h
roll_min_6h
roll_std_6h

roll_mean_12h
roll_min_12h
roll_std_12h

roll_mean_24h
roll_min_24h
roll_std_24h
```

### 10.3 Persistence — 5

```text
bz_gsm_persist_lt_m5h
bz_gsm_persist_lt_m10h
bz_gsm_persist_lt_m15h
speed_persist_gt_500h
speed_persist_gt_600h
```

### 10.4 Dynamics — 15

For each primary OMNI physical variable:

```text
delta_1h
delta_3h
slope_3h
```

### 10.5 Interactions — 3

```text
bz_neg_x_speed
bz_neg_x_density
flow_pressure_x_speed
```

## 11. Provenance Audit

Each feature family exposes or contributes an information-time audit.

The integrated builder consolidates:

```text
raw_information_time
rolling_information_time
persistence_information_time
dynamics_information_time
interaction_information_time
maximum_feature_information_time
information_cutoff
```

Interactions inherit the information time of their raw OMNI inputs.

The complete integrated feature frame must satisfy:

```text
maximum_feature_information_time <= information_cutoff
```

for every prediction row with available provenance.

## 12. Leakage and Mutation Invariance

The Phase 0.7 test suite verifies that:

```text
future OMNI mutations cannot change a past feature vector
future Kp mutations cannot change a past feature vector
missing exact OMNI timestamps are not silently substituted
sparse prediction grids do not change physical-time feature definitions
missing feature values do not delete prediction rows
feature names are unique
feature count is exactly 93
feature order is deterministic
```

These tests establish family-level and integrated causal invariants.

The broader project-wide leakage suite remains a separate Phase 0 checkpoint
and is not marked complete solely by Phase 0.7.

## 13. Canonical Implementation

Raw:

```text
src/features/raw.py
```

Rolling:

```text
src/features/rolling.py
```

Persistence:

```text
src/features/persistence.py
```

Dynamics:

```text
src/features/dynamics.py
```

Interactions:

```text
src/features/interactions.py
```

Integrated assembly and manifest:

```text
src/features/integrated.py
```

Package exports:

```text
src/features/__init__.py
```

Primary tests:

```text
tests/test_features_raw.py
tests/test_features_rolling.py
tests/test_features_persistence.py
tests/test_features_dynamics.py
tests/test_features_interactions.py
tests/test_features_integrated.py
```

Feature construction elsewhere in the repository must reuse these canonical
implementations rather than reconstructing equivalent features independently.

## 14. Frozen Phase 0.7 Rule

The primary feature universe is frozen at 93 features across five families:

```text
raw
rolling
persistence
dynamics
interactions
```

No AE, Dst, CME-derived predictor, additional threshold, additional rolling
statistic, additional interaction, or alternative dynamic estimator may enter
the primary feature universe without a separately documented protocol
amendment made for methodological reasons rather than model performance.
