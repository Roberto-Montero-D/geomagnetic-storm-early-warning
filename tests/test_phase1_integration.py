"""Phase 1.5 integration and isolation tests.

These tests compose the already-audited Phase 1 components without adding new
scientific definitions:

prediction grid -> canonical dataset -> row status -> temporal periods/folds

The suite is intentionally adversarial around the protected 2022 Final Test
boundary and around missing/unknown rows.
"""

import numpy as np
import pandas as pd

from src.dataset.builder import build_canonical_dataset
from src.dataset.prediction_grid import build_prediction_grid
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    PERIOD_VALIDATION_3,
    assign_temporal_periods,
    development_fold_masks,
)
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def _omni(start="2021-12-20 00:00", periods=600):
    index = pd.date_range(start, periods=periods, freq="h")
    x = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "bz_gsm": -3.0 - (x % 9),
            "bt": 6.0 + x * 0.01,
            "speed": 390.0 + x * 0.5,
            "density": 4.0 + x * 0.01,
            "flow_pressure": 1.0 + x * 0.005,
        },
        index=index,
    ).rename_axis("timestamp")


def _kp(start="2021-12-20 00:00", periods=200, value=2.0):
    starts = pd.date_range(start, periods=periods, freq="3h")
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": float(value),
        }
    )


def _pipeline(times, omni=None, kp=None):
    if omni is None:
        omni = _omni()
    if kp is None:
        kp = _kp()

    dataset, audit = build_canonical_dataset(
        omni,
        kp,
        times,
        return_audit=True,
    )
    status = build_row_status(dataset)
    splits = assign_temporal_periods(times)
    folds = development_fold_masks(splits)
    return dataset, audit, status, splits, folds


def test_all_phase1_outputs_remain_exactly_index_aligned():
    times = build_prediction_grid(
        start="2021-12-30 00:00",
        end_exclusive="2022-01-03 00:00",
    )

    dataset, audit, status, splits, folds = _pipeline(times)

    for frame in (dataset, audit, status, splits):
        pd.testing.assert_index_equal(frame.index, times)

    for fold in folds.values():
        pd.testing.assert_index_equal(fold["train"].index, times)
        pd.testing.assert_index_equal(fold["validation"].index, times)


def test_eligibility_never_changes_calendar_period_membership():
    times = pd.DatetimeIndex(
        [
            "2021-12-31 22:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ],
        name="prediction_time",
    )
    omni = _omni()
    kp = _kp()

    # Force one genuine feature gap on each side of the boundary.
    omni.loc[pd.Timestamp("2021-12-31 21:00"), "speed"] = np.nan
    omni.loc[pd.Timestamp("2021-12-31 23:00"), "speed"] = np.nan

    _, _, status, splits, _ = _pipeline(times, omni=omni, kp=kp)

    assert splits.loc[
        pd.Timestamp("2021-12-31 23:00"), "period"
    ] == PERIOD_VALIDATION_3
    assert splits.loc[
        pd.Timestamp("2022-01-01 00:00"), "period"
    ] == PERIOD_FINAL_TEST

    # Regardless of eligibility, the calendar split remains fixed.
    assert len(status) == len(splits)
    assert splits["period"].notna().all()


def test_missing_or_unknown_rows_survive_final_test_boundary():
    times = pd.DatetimeIndex(
        [
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ],
        name="prediction_time",
    )

    # End Kp coverage so at least later targets become unknown.
    kp = _kp(periods=97)
    dataset, _, status, splits, _ = _pipeline(times, kp=kp)

    pd.testing.assert_index_equal(dataset.index, times)
    pd.testing.assert_index_equal(status.index, times)
    assert len(dataset) == 3
    assert splits.loc[times[1], "period"] == PERIOD_FINAL_TEST
    assert splits.loc[times[2], "period"] == PERIOD_FINAL_TEST


def test_no_final_test_row_enters_any_development_mask_end_to_end():
    times = build_prediction_grid(
        start="2021-12-31 18:00",
        end_exclusive="2022-01-01 07:00",
    )

    _, _, _, splits, folds = _pipeline(times)
    final_test = splits["period"].eq(PERIOD_FINAL_TEST)

    assert final_test.any()

    for fold in folds.values():
        assert not (fold["train"] & final_test).any()
        assert not (fold["validation"] & final_test).any()


def test_supervised_filtering_cannot_pull_final_test_into_development():
    times = build_prediction_grid(
        start="2021-12-30 00:00",
        end_exclusive="2022-01-03 00:00",
    )

    _, _, status, splits, folds = _pipeline(times)
    eligible = status["supervised_eligible"]
    final_test = splits["is_final_test"]

    for fold in folds.values():
        train_eligible = fold["train"] & eligible
        validation_eligible = fold["validation"] & eligible

        assert not (train_eligible & final_test).any()
        assert not (validation_eligible & final_test).any()


def test_future_kp_mutation_changes_y_not_x_status_or_split():
    times = pd.DatetimeIndex(
        ["2021-12-31 21:00"],
        name="prediction_time",
    )
    omni = _omni()
    kp = _kp(value=1.0)

    before, _, status_before, split_before, _ = _pipeline(
        times, omni=omni, kp=kp
    )

    mutated = kp.copy()
    future = (
        mutated["interval_start"]
        == pd.Timestamp("2022-01-01 00:00")
    )
    assert future.sum() == 1
    mutated.loc[future, "kp"] = 7.0

    after, _, status_after, split_after, _ = _pipeline(
        times, omni=omni, kp=mutated
    )

    pd.testing.assert_series_equal(
        before.iloc[0][list(PRIMARY_FEATURE_COLUMNS)],
        after.iloc[0][list(PRIMARY_FEATURE_COLUMNS)],
    )
    assert before.iloc[0]["target"] == 0.0
    assert after.iloc[0]["target"] == 1.0
    pd.testing.assert_frame_equal(status_before, status_after)
    pd.testing.assert_frame_equal(split_before, split_after)


def test_final_test_membership_depends_only_on_prediction_time():
    times = pd.DatetimeIndex(
        [
            "2021-12-31 23:00",
            "2022-01-01 00:00",
        ],
        name="prediction_time",
    )

    split_a = assign_temporal_periods(times)

    omni = _omni()
    omni.loc[:, :] = np.nan
    kp = _kp()
    kp.loc[:, "kp"] = np.nan

    dataset, _ = build_canonical_dataset(
        omni, kp, times, return_audit=True
    )
    status = build_row_status(dataset)
    split_b = assign_temporal_periods(dataset.index)

    pd.testing.assert_frame_equal(split_a, split_b)
    assert not status["supervised_eligible"].all()
    assert split_b["period"].tolist() == [
        PERIOD_VALIDATION_3,
        PERIOD_FINAL_TEST,
    ]


def test_canonical_schema_is_unchanged_after_full_phase1_composition():
    times = build_prediction_grid(
        start="2021-12-31 12:00",
        end_exclusive="2022-01-01 12:00",
    )

    dataset, _, _, _, _ = _pipeline(times)

    assert tuple(dataset.columns[:-1]) == tuple(
        PRIMARY_FEATURE_COLUMNS
    )
    assert dataset.columns[-1] == "target"
    assert dataset.shape[1] == 94
