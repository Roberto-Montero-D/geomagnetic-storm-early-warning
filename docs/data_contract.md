# Data Contract

This document describes the availability and usage rules for each feature.

## Timestamp Convention

For a prediction at time `t`, the latest allowed observation is `t - 1h`.

## Feature Availability

| Feature | Source | Usable at t? |
|---------|--------|--------------|
| Bz | OMNI | Yes, if timestamp <= t-1h |
| V | OMNI | Yes, if timestamp <= t-1h |
| Kp | Kp index | Yes, if timestamp <= t-1h |
| CME | Catalog | Only if cme_available_at_t == True |

## CME Rule

Only CMEs with `cme_available_at_t == True` are used.