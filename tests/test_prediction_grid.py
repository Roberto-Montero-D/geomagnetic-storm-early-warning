import pandas as pd
import pytest

from src.dataset.prediction_grid import (
    DEFAULT_GRID_END_EXCLUSIVE,
    DEFAULT_GRID_START,
    build_prediction_grid,
)


def test_default_grid_uses_frozen_1996_through_2025_coverage():
    grid = build_prediction_grid()

    assert grid[0] == pd.Timestamp("1996-01-01 00:00")
    assert grid[-1] == pd.Timestamp("2025-12-31 23:00")
    assert DEFAULT_GRID_START == pd.Timestamp(
        "1996-01-01 00:00"
    )
    assert DEFAULT_GRID_END_EXCLUSIVE == pd.Timestamp(
        "2026-01-01 00:00"
    )


def test_default_grid_has_exact_expected_number_of_hours():
    grid = build_prediction_grid()

    expected = int(
        (
            pd.Timestamp("2026-01-01 00:00")
            - pd.Timestamp("1996-01-01 00:00")
        )
        / pd.Timedelta(hours=1)
    )

    assert len(grid) == expected


def test_grid_is_unique_sorted_and_exactly_hourly():
    grid = build_prediction_grid(
        start="2020-02-28 22:00",
        end_exclusive="2020-03-01 03:00",
    )

    assert grid.name == "prediction_time"
    assert grid.is_monotonic_increasing
    assert not grid.has_duplicates
    assert not grid.hasnans

    deltas = grid[1:] - grid[:-1]
    assert (deltas == pd.Timedelta(hours=1)).all()


def test_leap_day_is_preserved_as_calendar_time():
    grid = build_prediction_grid(
        start="2020-02-28 23:00",
        end_exclusive="2020-03-01 01:00",
    )

    assert pd.Timestamp("2020-02-29 00:00") in grid
    assert pd.Timestamp("2020-02-29 23:00") in grid
    assert len(grid) == 26


def test_half_open_end_boundary_is_excluded():
    grid = build_prediction_grid(
        start="2021-12-31 22:00",
        end_exclusive="2022-01-01 02:00",
    )

    expected = pd.DatetimeIndex(
        [
            "2021-12-31 22:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ],
        name="prediction_time",
    )

    pd.testing.assert_index_equal(grid, expected)
    assert pd.Timestamp("2022-01-01 02:00") not in grid


def test_split_boundary_timestamp_exists_exactly_once():
    grid = build_prediction_grid(
        start="2021-12-31 23:00",
        end_exclusive="2022-01-01 02:00",
    )

    assert (
        grid == pd.Timestamp("2022-01-01 00:00")
    ).sum() == 1


def test_custom_grid_does_not_require_source_data():
    # Phase 1.1 is intentionally independent of OMNI/Kp availability.
    grid = build_prediction_grid(
        start="1996-01-01 00:00",
        end_exclusive="1996-01-01 04:00",
    )

    expected = pd.DatetimeIndex(
        [
            "1996-01-01 00:00",
            "1996-01-01 01:00",
            "1996-01-01 02:00",
            "1996-01-01 03:00",
        ],
        name="prediction_time",
    )
    pd.testing.assert_index_equal(grid, expected)


@pytest.mark.parametrize(
    ("start", "end_exclusive"),
    [
        ("2020-01-01 00:30", "2020-01-02 00:00"),
        ("2020-01-01 00:00", "2020-01-02 00:30"),
        ("2020-01-01 00:00:01", "2020-01-02 00:00"),
    ],
)
def test_non_hour_aligned_boundaries_raise(
    start,
    end_exclusive,
):
    with pytest.raises(ValueError, match="whole hour"):
        build_prediction_grid(
            start=start,
            end_exclusive=end_exclusive,
        )


@pytest.mark.parametrize(
    ("start", "end_exclusive"),
    [
        ("2020-01-01 00:00", "2020-01-01 00:00"),
        ("2020-01-02 00:00", "2020-01-01 00:00"),
    ],
)
def test_empty_or_reversed_interval_raises(
    start,
    end_exclusive,
):
    with pytest.raises(ValueError, match="strictly earlier"):
        build_prediction_grid(
            start=start,
            end_exclusive=end_exclusive,
        )


def test_nat_boundary_raises():
    with pytest.raises(ValueError, match="must not be NaT"):
        build_prediction_grid(
            start=pd.NaT,
            end_exclusive="2020-01-02 00:00",
        )


def test_timezone_aware_boundary_raises():
    with pytest.raises(ValueError, match="timezone-naive"):
        build_prediction_grid(
            start=pd.Timestamp(
                "2020-01-01 00:00",
                tz="UTC",
            ),
            end_exclusive="2020-01-02 00:00",
        )
