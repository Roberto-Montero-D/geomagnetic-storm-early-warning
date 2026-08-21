# Phase 1 Temporal Split Contract

## Atomic periods

Each prediction timestamp receives exactly one non-overlapping calendar period:

| Period label | Half-open interval |
|---|---|
| `initial_train` | 1996-01-01 <= t < 2017-01-01 |
| `validation_1` | 2017-01-01 <= t < 2019-01-01 |
| `validation_2` | 2019-01-01 <= t < 2021-01-01 |
| `validation_3` | 2021-01-01 <= t < 2022-01-01 |
| `final_test` | 2022-01-01 <= t < 2026-01-01 |

Timestamps outside the frozen primary calendar coverage receive `outside_primary` when the split utility is called on such timestamps.

## Development folds

Expanding training windows are derived from the atomic periods:

```text
screening
    train:      1996–2016
    validation: 2017–2018

walk_forward_1
    train:      1996–2018
    validation: 2019–2020

walk_forward_2
    train:      1996–2020
    validation: 2021
```

This representation avoids assigning a timestamp simultaneously to multiple atomic "train" labels while still producing the frozen expanding-window experiments.

## Final Test isolation

The protected Final Test is:

```text
2022-01-01 00:00 <= t < 2026-01-01 00:00
```

No Final Test row may enter any development training or validation mask. Intersecting a development mask with `supervised_eligible` does not change that rule.

Final Test membership depends only on prediction time and cannot be changed by missing features, unknown targets, source availability, or row status.

## Relationship to Phase 8

Phase 1 establishes and tests the split boundary but does not evaluate Final Test outcomes. The Final Test remains single-use until the complete model, feature set, imbalance strategy, alert rules, OOF threshold procedure, and operational threshold are frozen as required by the master protocol.
