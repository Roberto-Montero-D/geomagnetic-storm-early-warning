import numpy as np
import pandas as pd
import pytest

from src.dataset.row_status import (
    ROW_STATUS_COLUMNS,
    ROW_STATUS_ELIGIBLE,
    ROW_STATUS_FEATURE_INCOMPLETE,
    ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET,
    ROW_STATUS_UNKNOWN_TARGET,
    build_row_status,
)
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def _dataset(
    *,
    rows=4,
):
    index = pd.date_range(
        "2020-01-01 00:00",
        periods=rows,
        freq="h",
        name="prediction_time",
    )
    frame = pd.DataFrame(
        1.0,
        index=index,
        columns=list(PRIMARY_FEATURE_COLUMNS),
    )
    frame["target"] = 0.0
    return frame


def test_complete_known_row_is_supervised_eligible():
    dataset = _dataset(rows=1)

    status = build_row_status(dataset)

    row = status.iloc[0]
    assert row["target_known"]
    assert row["features_complete"]
    assert row["n_missing_features"] == 0
    assert row["supervised_eligible"]
    assert row["row_status"] == ROW_STATUS_ELIGIBLE


def test_known_target_with_missing_feature_is_feature_incomplete():
    dataset = _dataset(rows=1)
    dataset.iloc[
        0,
        dataset.columns.get_loc(
            PRIMARY_FEATURE_COLUMNS[0]
        ),
    ] = np.nan

    status = build_row_status(dataset)
    row = status.iloc[0]

    assert row["target_known"]
    assert not row["features_complete"]
    assert row["n_missing_features"] == 1
    assert not row["supervised_eligible"]
    assert (
        row["row_status"]
        == ROW_STATUS_FEATURE_INCOMPLETE
    )


def test_complete_features_with_unknown_target_is_unknown_target():
    dataset = _dataset(rows=1)
    dataset.iloc[
        0,
        dataset.columns.get_loc("target"),
    ] = np.nan

    status = build_row_status(dataset)
    row = status.iloc[0]

    assert not row["target_known"]
    assert row["features_complete"]
    assert row["n_missing_features"] == 0
    assert not row["supervised_eligible"]
    assert (
        row["row_status"]
        == ROW_STATUS_UNKNOWN_TARGET
    )


def test_missing_features_and_unknown_target_get_combined_status():
    dataset = _dataset(rows=1)
    dataset.iloc[
        0,
        dataset.columns.get_loc(
            PRIMARY_FEATURE_COLUMNS[3]
        ),
    ] = np.nan
    dataset.iloc[
        0,
        dataset.columns.get_loc("target"),
    ] = np.nan

    status = build_row_status(dataset)
    row = status.iloc[0]

    assert not row["target_known"]
    assert not row["features_complete"]
    assert row["n_missing_features"] == 1
    assert not row["supervised_eligible"]
    assert (
        row["row_status"]
        == ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET
    )


def test_missing_feature_count_is_exact():
    dataset = _dataset(rows=1)

    for column in PRIMARY_FEATURE_COLUMNS[:7]:
        dataset.loc[dataset.index[0], column] = np.nan

    status = build_row_status(dataset)

    assert status.iloc[0]["n_missing_features"] == 7


def test_target_nan_does_not_count_as_missing_feature():
    dataset = _dataset(rows=1)
    dataset.loc[dataset.index[0], "target"] = np.nan

    status = build_row_status(dataset)

    assert status.iloc[0]["n_missing_features"] == 0


def test_extra_nonpredictive_columns_do_not_affect_feature_status():
    dataset = _dataset(rows=1)
    dataset["audit_note"] = np.nan

    status = build_row_status(dataset)

    assert status.iloc[0]["features_complete"]
    assert status.iloc[0]["supervised_eligible"]


def test_status_preserves_every_row_and_index_order():
    dataset = _dataset(rows=5)

    status = build_row_status(dataset)

    pd.testing.assert_index_equal(
        status.index,
        dataset.index,
    )
    assert len(status) == len(dataset)
    assert tuple(status.columns) == ROW_STATUS_COLUMNS


def test_status_function_does_not_modify_dataset():
    dataset = _dataset(rows=2)
    dataset.iloc[
        0,
        dataset.columns.get_loc(
            PRIMARY_FEATURE_COLUMNS[0]
        ),
    ] = np.nan
    before = dataset.copy(deep=True)

    build_row_status(dataset)

    pd.testing.assert_frame_equal(dataset, before)


def test_all_four_status_classes_are_reachable():
    dataset = _dataset(rows=4)

    dataset.loc[
        dataset.index[1],
        PRIMARY_FEATURE_COLUMNS[0],
    ] = np.nan

    dataset.loc[
        dataset.index[2],
        "target",
    ] = np.nan

    dataset.loc[
        dataset.index[3],
        PRIMARY_FEATURE_COLUMNS[0],
    ] = np.nan
    dataset.loc[
        dataset.index[3],
        "target",
    ] = np.nan

    status = build_row_status(dataset)

    assert status["row_status"].tolist() == [
        ROW_STATUS_ELIGIBLE,
        ROW_STATUS_FEATURE_INCOMPLETE,
        ROW_STATUS_UNKNOWN_TARGET,
        ROW_STATUS_FEATURE_INCOMPLETE_UNKNOWN_TARGET,
    ]


def test_missing_required_feature_column_raises():
    dataset = _dataset(rows=1).drop(
        columns=[PRIMARY_FEATURE_COLUMNS[0]]
    )

    with pytest.raises(
        ValueError,
        match="missing required canonical columns",
    ):
        build_row_status(dataset)


def test_missing_target_column_raises():
    dataset = _dataset(rows=1).drop(
        columns=["target"]
    )

    with pytest.raises(
        ValueError,
        match="missing required canonical columns",
    ):
        build_row_status(dataset)


def test_duplicate_index_raises():
    dataset = _dataset(rows=2)
    dataset.index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:00",
        ],
        name="prediction_time",
    )

    with pytest.raises(
        ValueError,
        match="index must be unique",
    ):
        build_row_status(dataset)
