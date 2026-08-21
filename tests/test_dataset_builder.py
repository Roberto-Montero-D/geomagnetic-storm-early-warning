import numpy as np
import pandas as pd
import pytest

from src.dataset.builder import (
    DATASET_AUDIT_COLUMNS,
    build_canonical_dataset,
)
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def _omni(
    *,
    start="2020-01-01 00:00",
    periods=120,
):
    index = pd.date_range(
        start,
        periods=periods,
        freq="h",
    )
    x = np.arange(periods, dtype=float)

    return pd.DataFrame(
        {
            "bz_gsm": -4.0 - (x % 8),
            "bt": 6.0 + x * 0.05,
            "speed": 400.0 + x * 2.0,
            "density": 5.0 + x * 0.02,
            "flow_pressure": 1.0 + x * 0.01,
        },
        index=index,
    ).rename_axis("timestamp")


def _kp_intervals(
    *,
    start="2020-01-01 00:00",
    periods=40,
    value=2.0,
):
    starts = pd.date_range(
        start,
        periods=periods,
        freq="3h",
    )
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": (
                starts + pd.Timedelta(hours=3)
            ),
            "kp": float(value),
        }
    )


def test_dataset_contains_exactly_93_features_plus_target():
    times = pd.DatetimeIndex(
        [
            "2020-01-02 12:00",
            "2020-01-02 13:00",
        ]
    )

    dataset = build_canonical_dataset(
        _omni(),
        _kp_intervals(),
        times,
    )

    assert tuple(dataset.columns[:-1]) == tuple(
        PRIMARY_FEATURE_COLUMNS
    )
    assert dataset.columns[-1] == "target"
    assert dataset.shape == (2, 94)


def test_requested_prediction_rows_are_preserved_exactly():
    times = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 01:00",
            "2020-01-02 12:00",
            "2020-01-05 23:00",
        ],
        name="prediction_time",
    )

    dataset, audit = build_canonical_dataset(
        _omni(),
        _kp_intervals(),
        times,
        return_audit=True,
    )

    pd.testing.assert_index_equal(
        dataset.index,
        times,
    )
    pd.testing.assert_index_equal(
        audit.index,
        times,
    )
    assert len(dataset) == len(times)
    assert len(audit) == len(times)


def test_builder_preserves_feature_missingness_instead_of_dropping():
    times = pd.DatetimeIndex(
        ["2020-01-01 00:00"]
    )

    dataset = build_canonical_dataset(
        _omni(),
        _kp_intervals(),
        times,
    )

    assert len(dataset) == 1
    assert dataset.loc[
        pd.Timestamp("2020-01-01 00:00"),
        list(PRIMARY_FEATURE_COLUMNS),
    ].isna().any()


def test_builder_preserves_unknown_target_instead_of_dropping():
    omni = _omni(periods=24)
    kp = _kp_intervals(periods=8)
    times = pd.DatetimeIndex(
        ["2020-01-01 22:00"]
    )

    dataset, audit = build_canonical_dataset(
        omni,
        kp,
        times,
        return_audit=True,
    )

    assert len(dataset) == 1
    assert pd.isna(dataset.iloc[0]["target"])
    assert audit.iloc[0]["target_status"] == "unknown"


def test_audit_metadata_is_separate_from_dataset_columns():
    times = pd.DatetimeIndex(
        ["2020-01-02 12:00"]
    )

    dataset, audit = build_canonical_dataset(
        _omni(),
        _kp_intervals(),
        times,
        return_audit=True,
    )

    assert tuple(audit.columns) == DATASET_AUDIT_COLUMNS
    assert not set(audit.columns).intersection(
        dataset.columns
    )
    assert "target_status" not in dataset.columns
    assert "information_cutoff" not in dataset.columns


def test_target_parameters_are_forwarded_to_canonical_target():
    kp = _kp_intervals(value=2.0)
    kp.loc[
        kp["interval_start"]
        == pd.Timestamp("2020-01-02 15:00"),
        "kp",
    ] = 4.0

    times = pd.DatetimeIndex(
        ["2020-01-02 12:00"]
    )

    primary = build_canonical_dataset(
        _omni(),
        kp,
        times,
        threshold=5.0,
        horizon_hours=6,
    )
    alternate_threshold = build_canonical_dataset(
        _omni(),
        kp,
        times,
        threshold=4.0,
        horizon_hours=6,
    )

    assert primary.iloc[0]["target"] == 0.0
    assert alternate_threshold.iloc[0]["target"] == 1.0


def test_feature_provenance_remains_causal_after_assembly():
    times = pd.date_range(
        "2020-01-02 12:00",
        periods=12,
        freq="h",
    )

    _, audit = build_canonical_dataset(
        _omni(),
        _kp_intervals(),
        times,
        return_audit=True,
    )

    valid = audit[
        "maximum_feature_information_time"
    ].notna()

    assert (
        audit.loc[
            valid,
            "maximum_feature_information_time",
        ]
        <= audit.loc[
            valid,
            "information_cutoff",
        ]
    ).all()


def test_future_kp_changes_target_but_not_feature_columns():
    omni = _omni()
    kp = _kp_intervals(value=1.0)
    times = pd.DatetimeIndex(
        ["2020-01-02 12:00"]
    )

    before = build_canonical_dataset(
        omni,
        kp,
        times,
    )

    mutated = kp.copy()
    mutated.loc[
        mutated["interval_start"]
        == pd.Timestamp("2020-01-02 15:00"),
        "kp",
    ] = 6.0

    after = build_canonical_dataset(
        omni,
        mutated,
        times,
    )

    pd.testing.assert_series_equal(
        before.iloc[0][list(PRIMARY_FEATURE_COLUMNS)],
        after.iloc[0][list(PRIMARY_FEATURE_COLUMNS)],
    )
    assert before.iloc[0]["target"] == 0.0
    assert after.iloc[0]["target"] == 1.0


def test_builder_does_not_impute_missing_omni_values():
    omni = _omni()
    kp = _kp_intervals()
    omni.loc[
        pd.Timestamp("2020-01-02 10:00"),
        "speed",
    ] = np.nan

    times = pd.DatetimeIndex(
        ["2020-01-02 12:00"]
    )

    dataset = build_canonical_dataset(
        omni,
        kp,
        times,
    )

    # The builder must expose whatever missingness the canonical feature
    # pipeline produces; it must not fill NaNs after feature construction.
    assert dataset.isna().sum().sum() > 0


@pytest.mark.parametrize(
    "times",
    [
        ["2020-01-01 00:30"],
        [
            "2020-01-01 01:00",
            "2020-01-01 00:00",
        ],
        [
            "2020-01-01 00:00",
            "2020-01-01 00:00",
        ],
    ],
)
def test_invalid_prediction_time_universe_raises(times):
    with pytest.raises(ValueError):
        build_canonical_dataset(
            _omni(),
            _kp_intervals(),
            pd.DatetimeIndex(times),
        )


def test_timezone_aware_prediction_times_raise():
    times = pd.date_range(
        "2020-01-01 00:00",
        periods=2,
        freq="h",
        tz="UTC",
    )

    with pytest.raises(
        ValueError,
        match="timezone-naive",
    ):
        build_canonical_dataset(
            _omni(),
            _kp_intervals(),
            times,
        )
