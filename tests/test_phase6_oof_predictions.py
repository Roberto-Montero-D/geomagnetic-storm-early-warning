"""Tests for the frozen Phase 6 OOF prediction contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluation.oof_predictions as oof_module
from src.baselines.framework import DevelopmentFold


def _folds():
    return {
        "walk_forward_1": DevelopmentFold(
            name="walk_forward_1",
            train_index=pd.date_range(
                "2017-01-01",
                periods=3,
                freq="h",
            ),
            validation_index=pd.date_range(
                "2019-01-01",
                periods=2,
                freq="h",
            ),
        ),
        "walk_forward_2": DevelopmentFold(
            name="walk_forward_2",
            train_index=pd.date_range(
                "2017-01-01",
                periods=5,
                freq="h",
            ),
            validation_index=pd.date_range(
                "2021-01-01",
                periods=2,
                freq="h",
            ),
        ),
    }


def _splits(folds):
    index = pd.DatetimeIndex(
        sorted(
            set().union(
                *[
                    set(fold.train_index)
                    | set(fold.validation_index)
                    for fold in folds.values()
                ]
            )
        )
    )

    period = []

    for timestamp in index:
        if timestamp < pd.Timestamp(
            "2019-01-01"
        ):
            period.append(
                "initial_train"
            )
        elif timestamp < pd.Timestamp(
            "2021-01-01"
        ):
            period.append(
                "validation_2"
            )
        else:
            period.append(
                "validation_3"
            )

    return pd.DataFrame(
        {
            "period": period,
        },
        index=index,
    )


def _events():
    return pd.DataFrame(
        {
            "event_id": [10, 11],
            "start_time": pd.to_datetime(
                [
                    "2019-01-01 01:00",
                    "2021-01-01 00:00",
                ]
            ),
            "end_time": pd.to_datetime(
                [
                    "2019-01-01 01:00",
                    "2021-01-01 00:00",
                ]
            ),
        }
    )


def test_storm_id_maps_only_active_event_hours():
    index = pd.DatetimeIndex(
        [
            "2019-01-01 00:00",
            "2019-01-01 01:00",
            "2019-01-01 02:00",
        ]
    )

    result = (
        oof_module
        ._storm_id_for_predictions(
            index,
            _events(),
        )
    )

    assert result.dtype == "Int64"
    assert pd.isna(
        result.iloc[0]
    )
    assert result.iloc[1] == 10
    assert pd.isna(
        result.iloc[2]
    )


def test_oof_validator_rejects_final_test():
    folds = _folds()
    splits = _splits(
        folds
    )

    table = pd.DataFrame(
        {
            "probability": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "target": [
                0,
                1,
                0,
                1,
            ],
            "storm_id": pd.array(
                [
                    pd.NA,
                    10,
                    11,
                    pd.NA,
                ],
                dtype="Int64",
            ),
            "fold": [
                "walk_forward_1",
                "walk_forward_1",
                "walk_forward_2",
                "walk_forward_2",
            ],
        },
        index=pd.DatetimeIndex(
            list(
                folds[
                    "walk_forward_1"
                ].validation_index
            )
            + list(
                folds[
                    "walk_forward_2"
                ].validation_index
            ),
            name="timestamp",
        ),
    )

    bad_time = pd.Timestamp(
        "2022-01-01"
    )

    bad_row = table.iloc[
        [-1]
    ].copy()
    bad_row.index = pd.DatetimeIndex(
        [bad_time],
        name="timestamp",
    )

    table = pd.concat(
        [
            table.iloc[:-1],
            bad_row,
        ]
    ).sort_index()

    splits.loc[
        bad_time,
        "period",
    ] = "final_test"

    with pytest.raises(
        AssertionError,
        match="Final Test",
    ):
        oof_module._validate_oof_table(
            table,
            folds,
            splits,
        )


def test_generator_uses_only_frozen_winner_and_validation_rows(
    monkeypatch,
):
    folds = _folds()
    splits = _splits(
        folds
    )

    feature = (
        oof_module
        .PHASE5_FEATURES[0]
    )

    all_index = splits.index

    dataset = pd.DataFrame(
        {
            feature: np.arange(
                len(all_index),
                dtype=float,
            ),
            "target": (
                np.arange(
                    len(all_index)
                )
                % 2
            ),
        },
        index=all_index,
    )

    monkeypatch.setattr(
        oof_module,
        "PHASE5_FEATURES",
        (feature,),
    )

    monkeypatch.setattr(
        oof_module,
        "_validate_confirmation_fold",
        lambda fold, split_table, fold_name: None,
    )

    fit_sizes = []

    class FakeModel:
        def fit(
            self,
            x,
            y,
        ):
            fit_sizes.append(
                len(x)
            )
            return self

        def predict_proba(
            self,
            x,
        ):
            probability = np.linspace(
                0.2,
                0.8,
                len(x),
            )
            return np.column_stack(
                [
                    1.0 - probability,
                    probability,
                ]
            )

    requested = []

    def fake_factory(
        config_id,
    ):
        requested.append(
            config_id
        )
        return FakeModel()

    monkeypatch.setattr(
        oof_module,
        "make_phase5_model_by_id",
        fake_factory,
    )

    result = (
        oof_module
        .generate_phase6_oof_predictions(
            dataset,
            folds,
            _events(),
            splits,
        )
    )

    expected_index = (
        folds[
            "walk_forward_1"
        ].validation_index
        .append(
            folds[
                "walk_forward_2"
            ].validation_index
        )
        .sort_values()
        .rename(
            "timestamp"
        )
    )

    assert result.config_id == (
        oof_module
        .PHASE6_SELECTED_CONFIG_ID
    )

    assert result.table.index.equals(
        expected_index
    )

    assert tuple(
        result.table.columns
    ) == (
        oof_module
        .PHASE6_OOF_COLUMNS
    )

    assert requested == [
        oof_module
        .PHASE6_SELECTED_CONFIG_ID,
        oof_module
        .PHASE6_SELECTED_CONFIG_ID,
    ]

    assert fit_sizes == [
        len(
            folds[
                "walk_forward_1"
            ].train_index
        ),
        len(
            folds[
                "walk_forward_2"
            ].train_index
        ),
    ]

    assert set(
        result.table["fold"]
    ) == {
        "walk_forward_1",
        "walk_forward_2",
    }


def test_generator_rejects_any_other_configuration():
    folds = _folds()

    with pytest.raises(
        ValueError,
        match="frozen Phase 5 winner",
    ):
        (
            oof_module
            .generate_phase6_oof_predictions(
                pd.DataFrame(),
                folds,
                _events(),
                _splits(
                    folds
                ),
                config_id=(
                    "extratrees_n100_dnone"
                ),
            )
        )
