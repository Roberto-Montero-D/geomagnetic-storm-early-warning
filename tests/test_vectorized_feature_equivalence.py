"""Reference-equivalence tests for optimized feature engines."""

import numpy as np
import pandas as pd
from pandas.testing import (
    assert_frame_equal,
    assert_series_equal,
)

from src.features.dynamics import build_dynamic_features
from src.features.persistence import build_persistence_features
from src.features.raw import (
    PRIMARY_OMNI_COLUMNS,
    PRIMARY_OMNI_FILL_VALUES,
)
from src.features.rolling import build_rolling_features
from src.temporal.cutoff import (
    information_cutoff,
    interval_end_times,
)


def _omni():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="h",
    )
    rng = np.random.default_rng(123456)

    frame = pd.DataFrame(
        {
            "bz_gsm": rng.normal(-2, 6, len(index)),
            "bt": rng.uniform(1, 15, len(index)),
            "speed": rng.normal(500, 120, len(index)),
            "density": rng.uniform(1, 20, len(index)),
            "flow_pressure": rng.uniform(0.1, 8, len(index)),
        },
        index=index,
    )

    frame.loc[index[12], "bz_gsm"] = 999.9
    frame.loc[index[20], "speed"] = 9999.0
    frame.loc[index[45], "density"] = np.nan

    return frame.drop(
        index=[
            index[30],
            index[31],
            index[70],
        ]
    )


def _normalized(omni):
    work = omni.loc[:, PRIMARY_OMNI_COLUMNS].copy()

    for column in PRIMARY_OMNI_COLUMNS:
        work[column] = pd.to_numeric(
            work[column],
            errors="raise",
        )
        work[column] = work[column].mask(
            work[column]
            == PRIMARY_OMNI_FILL_VALUES[column]
        )

    return work.astype(float)


def _reference_rolling(
    omni,
    prediction_index,
    windows=(3, 6, 12, 24),
):
    work = _normalized(omni)
    ends = pd.DatetimeIndex(
        interval_end_times(work.index)
    )
    values = work.to_numpy(dtype=float)

    names = tuple(
        f"{column}_roll_{stat}_{window}h"
        for column in PRIMARY_OMNI_COLUMNS
        for window in windows
        for stat in ("mean", "min", "std")
    )

    output = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=names,
        dtype=float,
    )

    maximum_information = []

    for t in prediction_index:
        cutoff = information_cutoff(t)
        row_max = pd.NaT

        for window in windows:
            lower = (
                cutoff
                - pd.Timedelta(hours=window)
            )

            mask = (
                (ends > lower)
                & (ends <= cutoff)
            )

            if not mask.any():
                continue

            window_values = values[mask]
            window_ends = ends[mask]

            for col_i, column in enumerate(
                PRIMARY_OMNI_COLUMNS
            ):
                series = window_values[:, col_i]
                valid = ~np.isnan(series)

                if not valid.any():
                    continue

                valid_values = series[valid]

                output.loc[
                    t,
                    f"{column}_roll_mean_{window}h",
                ] = float(
                    np.mean(valid_values)
                )

                output.loc[
                    t,
                    f"{column}_roll_min_{window}h",
                ] = float(
                    np.min(valid_values)
                )

                if valid_values.size >= 2:
                    output.loc[
                        t,
                        f"{column}_roll_std_{window}h",
                    ] = float(
                        np.std(
                            valid_values,
                            ddof=1,
                        )
                    )

                latest = window_ends[valid][-1]
                if (
                    pd.isna(row_max)
                    or latest > row_max
                ):
                    row_max = latest

        maximum_information.append(row_max)

    output.index.name = "prediction_time"

    audit = pd.DataFrame(
        index=prediction_index
    )
    audit.index.name = "prediction_time"

    # Build this exactly like the production raw-feature audit contract,
    # rather than coercing datetime resolution with pd.to_datetime().
    audit["information_cutoff"] = pd.Series(
        [
            information_cutoff(t)
            for t in prediction_index
        ],
        index=prediction_index,
        name="information_cutoff",
    )

    audit["maximum_rolling_information_time"] = pd.Series(
        maximum_information,
        index=prediction_index,
    )

    return output, audit


def _reference_persistence(
    omni,
    prediction_index,
):
    work = _normalized(omni)[
        ["bz_gsm", "speed"]
    ]

    definitions = (
        ("bz_gsm", lambda x: x < -5.0),
        ("bz_gsm", lambda x: x < -10.0),
        ("bz_gsm", lambda x: x < -15.0),
        ("speed", lambda x: x > 500.0),
        ("speed", lambda x: x > 600.0),
    )

    names = (
        "bz_gsm_persist_lt_m5h",
        "bz_gsm_persist_lt_m10h",
        "bz_gsm_persist_lt_m15h",
        "speed_persist_gt_500h",
        "speed_persist_gt_600h",
    )

    output = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=names,
        dtype=float,
    )

    for row_i, t in enumerate(
        prediction_index
    ):
        latest_start = (
            information_cutoff(t)
            - pd.Timedelta(hours=1)
        )

        for col_i, (
            column,
            predicate,
        ) in enumerate(definitions):
            if latest_start not in work.index:
                continue

            latest = work.loc[
                latest_start,
                column,
            ]

            if pd.isna(latest):
                continue

            if not predicate(float(latest)):
                output.iat[
                    row_i,
                    col_i,
                ] = 0.0
                continue

            duration = 0
            cursor = latest_start

            while cursor in work.index:
                value = work.loc[
                    cursor,
                    column,
                ]

                if (
                    pd.isna(value)
                    or not predicate(float(value))
                ):
                    break

                duration += 1
                cursor -= pd.Timedelta(hours=1)

            output.iat[
                row_i,
                col_i,
            ] = float(duration)

    output.index.name = "prediction_time"
    return output


def _reference_dynamics(
    omni,
    prediction_index,
):
    work = _normalized(omni)

    names = tuple(
        feature
        for column in PRIMARY_OMNI_COLUMNS
        for feature in (
            f"{column}_delta_1h",
            f"{column}_delta_3h",
            f"{column}_slope_3h",
        )
    )

    output = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=names,
        dtype=float,
    )

    weights = (
        np.array(
            [-1.5, -0.5, 0.5, 1.5]
        )
        / 5.0
    )

    for t in prediction_index:
        latest_start = (
            information_cutoff(t)
            - pd.Timedelta(hours=1)
        )

        for column in PRIMARY_OMNI_COLUMNS:
            latest = work[
                column
            ].get(
                latest_start,
                np.nan,
            )

            for lag in (1, 3):
                older = work[
                    column
                ].get(
                    latest_start
                    - pd.Timedelta(hours=lag),
                    np.nan,
                )

                if (
                    pd.notna(latest)
                    and pd.notna(older)
                ):
                    output.loc[
                        t,
                        f"{column}_delta_{lag}h",
                    ] = (
                        latest
                        - older
                    )

            times = pd.date_range(
                latest_start
                - pd.Timedelta(hours=3),
                latest_start,
                freq="h",
            )

            slope_values = (
                work[column]
                .reindex(times)
                .to_numpy(dtype=float)
            )

            if not np.isnan(
                slope_values
            ).any():
                output.loc[
                    t,
                    f"{column}_slope_3h",
                ] = (
                    slope_values
                    @ weights
                )

    output.index.name = "prediction_time"
    return output


def test_vectorized_rolling_matches_reference():
    omni = _omni()
    prediction_index = pd.date_range(
        "2020-01-01 05:00",
        periods=90,
        freq="h",
        name="prediction_time",
    )

    expected, expected_audit = (
        _reference_rolling(
            omni,
            prediction_index,
        )
    )

    actual, actual_audit = (
        build_rolling_features(
            omni,
            prediction_index,
            return_audit=True,
        )
    )

    assert_frame_equal(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert_series_equal(
        actual_audit[
            "information_cutoff"
        ],
        expected_audit[
            "information_cutoff"
        ],
        check_names=False,
    )

    assert_series_equal(
        actual_audit[
            "maximum_rolling_information_time"
        ],
        expected_audit[
            "maximum_rolling_information_time"
        ],
        check_names=False,
    )


def test_vectorized_persistence_matches_reference():
    omni = _omni()
    prediction_index = pd.date_range(
        "2020-01-01 05:00",
        periods=90,
        freq="h",
        name="prediction_time",
    )

    expected = _reference_persistence(
        omni,
        prediction_index,
    )

    actual = build_persistence_features(
        omni,
        prediction_index,
    )

    assert_frame_equal(
        actual,
        expected,
    )


def test_vectorized_dynamics_matches_reference():
    omni = _omni()
    prediction_index = pd.date_range(
        "2020-01-01 05:00",
        periods=90,
        freq="h",
        name="prediction_time",
    )

    expected = _reference_dynamics(
        omni,
        prediction_index,
    )

    actual = build_dynamic_features(
        omni,
        prediction_index,
    )

    assert_frame_equal(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
