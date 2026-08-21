"""Deterministic temporal split assignment for Phase 1.

This module maps prediction timestamps to the frozen chronological development
periods. It does not filter rows by supervised eligibility and does not expose
the protected Final Test as a development validation fold.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PERIOD_INITIAL_TRAIN = "initial_train"
PERIOD_VALIDATION_1 = "validation_1"
PERIOD_VALIDATION_2 = "validation_2"
PERIOD_VALIDATION_3 = "validation_3"
PERIOD_FINAL_TEST = "final_test"
PERIOD_OUTSIDE_PRIMARY = "outside_primary"

PERIOD_VALUES = (
    PERIOD_INITIAL_TRAIN,
    PERIOD_VALIDATION_1,
    PERIOD_VALIDATION_2,
    PERIOD_VALIDATION_3,
    PERIOD_FINAL_TEST,
    PERIOD_OUTSIDE_PRIMARY,
)

SPLIT_COLUMNS = (
    "period",
    "in_primary_coverage",
    "is_final_test",
)

# Atomic, non-overlapping calendar periods. Expanding training windows are
# derived from these atoms rather than assigning one timestamp to multiple
# "train" labels.
_PERIOD_BOUNDS = (
    (
        PERIOD_INITIAL_TRAIN,
        pd.Timestamp("1996-01-01 00:00"),
        pd.Timestamp("2017-01-01 00:00"),
    ),
    (
        PERIOD_VALIDATION_1,
        pd.Timestamp("2017-01-01 00:00"),
        pd.Timestamp("2019-01-01 00:00"),
    ),
    (
        PERIOD_VALIDATION_2,
        pd.Timestamp("2019-01-01 00:00"),
        pd.Timestamp("2021-01-01 00:00"),
    ),
    (
        PERIOD_VALIDATION_3,
        pd.Timestamp("2021-01-01 00:00"),
        pd.Timestamp("2022-01-01 00:00"),
    ),
    (
        PERIOD_FINAL_TEST,
        pd.Timestamp("2022-01-01 00:00"),
        pd.Timestamp("2026-01-01 00:00"),
    ),
)

PRIMARY_START = pd.Timestamp("1996-01-01 00:00")
PRIMARY_END_EXCLUSIVE = pd.Timestamp("2026-01-01 00:00")


def _validate_times(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(
        prediction_times,
        name="prediction_time",
    )

    if index.hasnans:
        raise ValueError("prediction_times must not contain NaT.")
    if index.has_duplicates:
        raise ValueError("prediction_times must be unique.")
    if not index.is_monotonic_increasing:
        raise ValueError(
            "prediction_times must be monotonically increasing."
        )
    if index.tz is not None:
        raise ValueError(
            "prediction_times must be timezone-naive."
        )
    if len(index) and (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError(
            "prediction_times must be aligned to whole hours."
        )

    return index


def assign_temporal_periods(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assign each timestamp to one atomic frozen calendar period."""

    index = _validate_times(prediction_times)
    period = pd.Series(
        PERIOD_OUTSIDE_PRIMARY,
        index=index,
        dtype="string",
    )

    for name, start, end_exclusive in _PERIOD_BOUNDS:
        mask = (index >= start) & (index < end_exclusive)
        period.loc[mask] = name

    in_primary = (
        (index >= PRIMARY_START)
        & (index < PRIMARY_END_EXCLUSIVE)
    )

    result = pd.DataFrame(
        {
            "period": period,
            "in_primary_coverage": in_primary.astype(bool),
            "is_final_test": period.eq(PERIOD_FINAL_TEST).astype(bool),
        },
        index=index,
    )

    if tuple(result.columns) != SPLIT_COLUMNS:
        raise AssertionError("Split columns are not deterministic.")
    if not result.index.equals(index):
        raise AssertionError("Split assignment changed the input index.")

    return result


def development_fold_masks(
    split_frame: pd.DataFrame,
) -> dict[str, dict[str, pd.Series]]:
    """Return frozen expanding-window train/validation masks.

    Fold definitions:
      screening:          train 1996-2016, validate 2017-2018
      walk_forward_1:     train 1996-2018, validate 2019-2020
      walk_forward_2:     train 1996-2020, validate 2021

    Final Test rows are excluded from every returned train/validation mask.
    """

    if "period" not in split_frame.columns:
        raise ValueError("split_frame must contain a 'period' column.")

    period = split_frame["period"]

    folds = {
        "screening": {
            "train": period.eq(PERIOD_INITIAL_TRAIN),
            "validation": period.eq(PERIOD_VALIDATION_1),
        },
        "walk_forward_1": {
            "train": period.isin(
                [PERIOD_INITIAL_TRAIN, PERIOD_VALIDATION_1]
            ),
            "validation": period.eq(PERIOD_VALIDATION_2),
        },
        "walk_forward_2": {
            "train": period.isin(
                [
                    PERIOD_INITIAL_TRAIN,
                    PERIOD_VALIDATION_1,
                    PERIOD_VALIDATION_2,
                ]
            ),
            "validation": period.eq(PERIOD_VALIDATION_3),
        },
    }

    final_test = period.eq(PERIOD_FINAL_TEST)
    for fold in folds.values():
        if (fold["train"] & final_test).any():
            raise AssertionError("Final Test entered a training mask.")
        if (fold["validation"] & final_test).any():
            raise AssertionError("Final Test entered a validation mask.")

    return folds
