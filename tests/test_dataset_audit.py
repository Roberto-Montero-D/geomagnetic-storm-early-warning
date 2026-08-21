import numpy as np
import pandas as pd
import pytest

from src.dataset.dataset_audit import (
    FINAL_TEST_FORBIDDEN_OUTCOME_COLUMNS,
    audit_dataset_by_period,
    audit_feature_missingness_by_period,
)
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import assign_temporal_periods
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def _dataset():
    times = pd.DatetimeIndex(
        [
            "2016-12-31 23:00",
            "2017-01-01 00:00",
            "2019-01-01 00:00",
            "2021-01-01 00:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ],
        name="prediction_time",
    )
    data = pd.DataFrame(
        1.0,
        index=times,
        columns=list(PRIMARY_FEATURE_COLUMNS),
    )
    data["target"] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    return data


def test_development_periods_report_target_prevalence():
    dataset = _dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)

    audit = audit_dataset_by_period(dataset, status, splits)

    assert audit.loc["initial_train", "n_positive"] == 0
    assert audit.loc["validation_1", "n_positive"] == 1
    assert audit.loc["validation_2", "target_prevalence"] == 0.0
    assert audit.loc["validation_3", "target_prevalence"] == 1.0


def test_final_test_outcome_fields_are_redacted():
    dataset = _dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)

    audit = audit_dataset_by_period(dataset, status, splits)

    for column in FINAL_TEST_FORBIDDEN_OUTCOME_COLUMNS:
        assert pd.isna(audit.loc["final_test", column])


def test_final_test_structural_feature_fields_are_available():
    dataset = _dataset()
    dataset.loc[
        pd.Timestamp("2022-01-01 01:00"),
        PRIMARY_FEATURE_COLUMNS[0],
    ] = np.nan
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)

    audit = audit_dataset_by_period(dataset, status, splits)

    assert audit.loc["final_test", "n_rows"] == 2
    assert audit.loc["final_test", "n_feature_complete"] == 1
    assert audit.loc["final_test", "n_feature_incomplete"] == 1
    assert audit.loc["final_test", "fraction_feature_complete"] == 0.5


def test_changing_final_test_targets_cannot_change_audit_output():
    dataset = _dataset()
    splits = assign_temporal_periods(dataset.index)

    status_a = build_row_status(dataset)
    audit_a = audit_dataset_by_period(dataset, status_a, splits)

    mutated = dataset.copy()
    final = splits["period"].eq("final_test")
    mutated.loc[final, "target"] = [0.0, np.nan]

    status_b = build_row_status(mutated)
    audit_b = audit_dataset_by_period(mutated, status_b, splits)

    pd.testing.assert_series_equal(
        audit_a.loc["final_test"],
        audit_b.loc["final_test"],
    )


def test_development_unknown_target_is_reported():
    dataset = _dataset()
    dataset.loc[pd.Timestamp("2017-01-01 00:00"), "target"] = np.nan
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)

    audit = audit_dataset_by_period(dataset, status, splits)

    assert audit.loc["validation_1", "n_target_known"] == 0
    assert audit.loc["validation_1", "n_unknown_target"] == 1
    assert pd.isna(audit.loc["validation_1", "target_prevalence"])


def test_feature_missingness_audit_has_93_rows_per_period():
    dataset = _dataset()
    splits = assign_temporal_periods(dataset.index)

    audit = audit_feature_missingness_by_period(dataset, splits)

    periods = splits["period"].nunique()
    assert len(audit) == periods * len(PRIMARY_FEATURE_COLUMNS)


def test_feature_missingness_audit_is_safe_for_final_test():
    dataset = _dataset()
    splits = assign_temporal_periods(dataset.index)

    before = audit_feature_missingness_by_period(dataset, splits)

    mutated = dataset.copy()
    final = splits["period"].eq("final_test")
    mutated.loc[final, "target"] = [np.nan, 1.0]

    after = audit_feature_missingness_by_period(mutated, splits)

    pd.testing.assert_frame_equal(before, after)


def test_feature_missingness_counts_exactly():
    dataset = _dataset()
    feature = PRIMARY_FEATURE_COLUMNS[0]
    dataset.loc[pd.Timestamp("2022-01-01 00:00"), feature] = np.nan
    splits = assign_temporal_periods(dataset.index)

    audit = audit_feature_missingness_by_period(dataset, splits)

    row = audit.loc[("final_test", feature)]
    assert row["n_rows"] == 2
    assert row["n_missing"] == 1
    assert row["fraction_missing"] == 0.5


def test_misaligned_status_raises():
    dataset = _dataset()
    status = build_row_status(dataset).iloc[:-1]
    splits = assign_temporal_periods(dataset.index)

    with pytest.raises(ValueError, match="indices must match"):
        audit_dataset_by_period(dataset, status, splits)


def test_misaligned_splits_raise():
    dataset = _dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index).iloc[:-1]

    with pytest.raises(ValueError, match="indices must match"):
        audit_dataset_by_period(dataset, status, splits)
