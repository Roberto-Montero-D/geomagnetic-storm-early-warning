import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import PERIOD_FINAL_TEST
from src.evaluation.oof_predictions import PHASE6_OOF_COLUMNS
from src.features.integrated import PRIMARY_FEATURE_COLUMNS
from src.phase7.contract import (
    PHASE7_FEATURES,
    PHASE7_MODEL_CONFIG_ID,
    Phase7Experiment,
    get_phase7_experiment,
)
from src.phase7.oof import (
    PHASE7_OOF_COLUMNS,
    Phase7OOFPredictions,
    _generate_phase7_oof_from_dataset,
    assert_phase7_oof_is_development_only,
    assert_phase7_primary_control_dataset,
    build_phase7_experiment_dataset,
)


def _canonical_kp_fixture() -> pd.DataFrame:
    starts = pd.date_range(
        "2020-01-01 00:00",
        periods=16,
        freq="3h",
    )
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": [
                2.0, 2.0, 5.0, 5.0,
                2.0, 2.0, 6.0, 6.0,
                2.0, 2.0, 7.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
            ],
        }
    )


def _base_dataset_for_target_tests() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01 00:00",
        periods=24,
        freq="h",
        name="prediction_time",
    )
    frame = pd.DataFrame(
        {
            column: (
                np.arange(len(index), dtype=float) + position
            )
            for position, column in enumerate(
                PRIMARY_FEATURE_COLUMNS
            )
        },
        index=index,
    )
    frame["target"] = 0.0
    return frame


def _synthetic_oof_inputs():
    initial = pd.to_datetime(
        [
            "2016-01-01 00:00",
            "2016-01-01 01:00",
            "2017-01-01 00:00",
            "2018-01-01 00:00",
        ]
    )
    validation_2 = pd.to_datetime(
        [
            "2019-01-01 00:00",
            "2020-01-01 00:00",
        ]
    )
    validation_3 = pd.to_datetime(
        [
            "2021-01-01 00:00",
            "2021-01-01 01:00",
        ]
    )
    final_test = pd.to_datetime(
        ["2022-01-01 00:00"]
    )

    index = pd.DatetimeIndex(
        [
            *initial,
            *validation_2,
            *validation_3,
            *final_test,
        ],
        name="prediction_time",
    )

    frame = pd.DataFrame(
        {
            column: (
                np.linspace(0.0, 1.0, len(index)) + position
            )
            for position, column in enumerate(
                PRIMARY_FEATURE_COLUMNS
            )
        },
        index=index,
    )
    frame["target"] = [
        0, 1, 0, 1,
        0, 1,
        0, 1,
        0,
    ]

    splits = pd.DataFrame(
        {
            "period": [
                "initial_train",
                "initial_train",
                "validation_1",
                "validation_1",
                "validation_2",
                "validation_2",
                "validation_3",
                "validation_3",
                PERIOD_FINAL_TEST,
            ]
        },
        index=index,
    )

    folds = {
        "walk_forward_1": DevelopmentFold(
            name="walk_forward_1",
            train_index=pd.DatetimeIndex(
                initial,
                name="prediction_time",
            ),
            validation_index=pd.DatetimeIndex(
                validation_2,
                name="prediction_time",
            ),
        ),
        "walk_forward_2": DevelopmentFold(
            name="walk_forward_2",
            train_index=pd.DatetimeIndex(
                [*initial, *validation_2],
                name="prediction_time",
            ),
            validation_index=pd.DatetimeIndex(
                validation_3,
                name="prediction_time",
            ),
        ),
    }

    events = pd.DataFrame(
        {
            "event_id": [1, 2],
            "start_time": pd.to_datetime(
                [
                    "2019-01-01 00:00",
                    "2021-01-01 01:00",
                ]
            ),
            "end_time": pd.to_datetime(
                [
                    "2019-01-01 00:00",
                    "2021-01-01 01:00",
                ]
            ),
        }
    )

    return frame, splits, folds, events


class _DeterministicModel:
    def fit(self, x, y):
        self._offset = float(y.mean())
        return self

    def predict_proba(self, x):
        raw = x.loc[:, PHASE7_FEATURES[0]].to_numpy(dtype=float)
        scaled = raw - raw.min() if len(raw) else raw
        denominator = (
            scaled.max()
            if len(scaled) and scaled.max() > 0
            else 1.0
        )
        p = np.clip(
            0.1
            + 0.5 * (scaled / denominator)
            + 0.1 * self._offset,
            0.0,
            1.0,
        )
        return np.column_stack([1.0 - p, p])


def test_phase7_oof_contract_reuses_phase6_columns():
    assert PHASE7_OOF_COLUMNS == PHASE6_OOF_COLUMNS
    assert PHASE7_OOF_COLUMNS == (
        "probability",
        "target",
        "storm_id",
        "fold",
    )


def test_experiment_dataset_changes_only_target():
    base = _base_dataset_for_target_tests()
    kp = _canonical_kp_fixture()

    actual = build_phase7_experiment_dataset(
        base,
        kp,
        "t5_h3",
    )

    assert_frame_equal(
        actual.loc[:, list(PRIMARY_FEATURE_COLUMNS)],
        base.loc[:, list(PRIMARY_FEATURE_COLUMNS)],
    )
    assert tuple(actual.columns) == (
        *PRIMARY_FEATURE_COLUMNS,
        "target",
    )
    assert not actual["target"].equals(base["target"])


def test_different_experiments_preserve_identical_predictors():
    base = _base_dataset_for_target_tests()
    kp = _canonical_kp_fixture()

    datasets = [
        build_phase7_experiment_dataset(
            base,
            kp,
            experiment_id,
        )
        for experiment_id in (
            "t5_h3",
            "t5_h24",
            "t7_h6",
        )
    ]

    expected = datasets[0].loc[
        :, list(PRIMARY_FEATURE_COLUMNS)
    ]

    for dataset in datasets[1:]:
        assert_frame_equal(
            dataset.loc[:, list(PRIMARY_FEATURE_COLUMNS)],
            expected,
        )


def test_primary_control_dataset_requires_exact_t5_h6_target():
    base = _base_dataset_for_target_tests()
    kp = _canonical_kp_fixture()

    control = build_phase7_experiment_dataset(
        base,
        kp,
        "t5_h6",
    )
    base["target"] = control["target"]

    assert_phase7_primary_control_dataset(base, kp)

    known = base["target"].dropna().index[0]
    base.loc[known, "target"] = (
        1.0 - float(base.loc[known, "target"])
    )

    with pytest.raises(
        AssertionError,
        match="does not reproduce",
    ):
        assert_phase7_primary_control_dataset(base, kp)


def test_phase7_oof_core_uses_frozen_model_and_exact_folds(
    monkeypatch,
):
    dataset, splits, folds, events = _synthetic_oof_inputs()
    called = []

    def fake_factory(config_id):
        called.append(config_id)
        return _DeterministicModel()

    monkeypatch.setattr(
        "src.phase7.oof.make_phase5_model_by_id",
        fake_factory,
    )

    result = _generate_phase7_oof_from_dataset(
        dataset,
        folds,
        events,
        splits,
        get_phase7_experiment("t5_h6"),
    )

    assert isinstance(result, Phase7OOFPredictions)
    assert result.experiment_id == "t5_h6"
    assert result.config_id == PHASE7_MODEL_CONFIG_ID
    assert called == [
        PHASE7_MODEL_CONFIG_ID,
        PHASE7_MODEL_CONFIG_ID,
    ]

    expected_index = pd.DatetimeIndex(
        [
            "2019-01-01 00:00",
            "2020-01-01 00:00",
            "2021-01-01 00:00",
            "2021-01-01 01:00",
        ],
        name="timestamp",
    )
    assert result.table.index.equals(expected_index)
    assert result.table["fold"].tolist() == [
        "walk_forward_1",
        "walk_forward_1",
        "walk_forward_2",
        "walk_forward_2",
    ]
    assert result.table["target"].tolist() == [0, 1, 0, 1]


def test_phase7_oof_is_explicitly_development_only(
    monkeypatch,
):
    dataset, splits, folds, events = _synthetic_oof_inputs()

    monkeypatch.setattr(
        "src.phase7.oof.make_phase5_model_by_id",
        lambda _: _DeterministicModel(),
    )

    result = _generate_phase7_oof_from_dataset(
        dataset,
        folds,
        events,
        splits,
        get_phase7_experiment("t6_h6"),
    )

    assert_phase7_oof_is_development_only(result, splits)

    assert (
        splits.loc[result.table.index, "period"]
        != PERIOD_FINAL_TEST
    ).all()


def test_unregistered_phase7_specification_is_rejected():
    base = _base_dataset_for_target_tests()
    kp = _canonical_kp_fixture()

    drifted = Phase7Experiment(
        "t5_h6",
        5.0,
        12,
        True,
    )

    with pytest.raises(
        ValueError,
        match="differs from the frozen registry",
    ):
        build_phase7_experiment_dataset(
            base,
            kp,
            drifted,
        )
