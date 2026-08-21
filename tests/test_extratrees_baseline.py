import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.baselines.extratrees import (
    DEFAULT_RANDOM_STATE,
    EXTRATREES_FEATURES,
    fit_extratrees_fold,
    make_extratrees_model,
)
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


def _frame():
    index = pd.date_range(
        "2010-01-01",
        periods=20,
        freq="h",
        name="prediction_time",
    )
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(size=(20, len(EXTRATREES_FEATURES))),
        index=index,
        columns=list(EXTRATREES_FEATURES),
    )
    frame["target"] = [0, 1] * 10
    frame["engineered_future_like"] = np.arange(20)
    return frame


def _fold():
    index = pd.date_range(
        "2010-01-01",
        periods=20,
        freq="h",
        name="prediction_time",
    )
    return DevelopmentFold(
        name="screening",
        train_index=index[:14],
        validation_index=index[14:],
    )


def _tree_signature(model):
    signature = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        signature.append(
            (
                tree.feature.copy(),
                tree.threshold.copy(),
                tree.children_left.copy(),
                tree.children_right.copy(),
            )
        )
    return signature


def test_b3_uses_exact_raw_primary_manifest():
    assert EXTRATREES_FEATURES == tuple(PRIMARY_RAW_FEATURE_COLUMNS)
    assert len(EXTRATREES_FEATURES) == 10


def test_model_is_unbalanced_and_reproducible():
    model = make_extratrees_model(
        n_estimators=100,
        max_depth=10,
    )

    assert model.class_weight is None
    assert model.random_state == DEFAULT_RANDOM_STATE
    assert model.n_estimators == 100
    assert model.max_depth == 10


def test_frozen_hyperparameters_are_used_by_default():
    model = make_extratrees_model()

    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.class_weight is None


@pytest.mark.parametrize(
    ("n_estimators", "max_depth"),
    [
        (0, 10),
        (-1, 10),
        (100, 0),
        (100, -2),
    ],
)
def test_invalid_numeric_hyperparameters_raise(
    n_estimators,
    max_depth,
):
    with pytest.raises(ValueError):
        make_extratrees_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )


def test_fit_returns_validation_probabilities_on_exact_index():
    result = fit_extratrees_fold(
        _frame(),
        _fold(),
        n_estimators=50,
        max_depth=10,
    )

    pd.testing.assert_index_equal(
        result.validation_probability.index,
        _fold().validation_index,
    )
    assert result.validation_probability.name == "probability"
    assert result.validation_probability.between(0, 1).all()


def test_unrequested_engineered_column_cannot_affect_b3():
    frame = _frame()
    before = fit_extratrees_fold(
        frame,
        _fold(),
        n_estimators=50,
        max_depth=10,
    ).validation_probability

    mutated = frame.copy()
    mutated["engineered_future_like"] = np.linspace(
        -1e12, 1e12, len(mutated)
    )
    after = fit_extratrees_fold(
        mutated,
        _fold(),
        n_estimators=50,
        max_depth=10,
    ).validation_probability

    pd.testing.assert_series_equal(before, after)


def test_validation_target_cannot_affect_fit_or_probability():
    frame = _frame()
    before = fit_extratrees_fold(
        frame,
        _fold(),
        n_estimators=50,
        max_depth=10,
    ).validation_probability

    mutated = frame.copy()
    mutated.loc[_fold().validation_index, "target"] = 1
    after = fit_extratrees_fold(
        mutated,
        _fold(),
        n_estimators=50,
        max_depth=10,
    ).validation_probability

    pd.testing.assert_series_equal(before, after)


def test_validation_features_do_not_affect_fitted_tree_state():
    frame = _frame()
    result_a = fit_extratrees_fold(
        frame,
        _fold(),
        n_estimators=25,
        max_depth=6,
    )

    mutated = frame.copy()
    mutated.loc[
        _fold().validation_index,
        EXTRATREES_FEATURES,
    ] += 1e6

    result_b = fit_extratrees_fold(
        mutated,
        _fold(),
        n_estimators=25,
        max_depth=6,
    )

    sig_a = _tree_signature(result_a.model)
    sig_b = _tree_signature(result_b.model)

    assert len(sig_a) == len(sig_b)
    for tree_a, tree_b in zip(sig_a, sig_b):
        for arr_a, arr_b in zip(tree_a, tree_b):
            np.testing.assert_array_equal(arr_a, arr_b)


def test_training_features_do_affect_fitted_tree_state():
    frame = _frame()
    result_a = fit_extratrees_fold(
        frame,
        _fold(),
        n_estimators=25,
        max_depth=6,
    )

    mutated = frame.copy()
    feature = EXTRATREES_FEATURES[0]
    mutated.loc[_fold().train_index, feature] *= 100.0

    result_b = fit_extratrees_fold(
        mutated,
        _fold(),
        n_estimators=25,
        max_depth=6,
    )

    sig_a = _tree_signature(result_a.model)
    sig_b = _tree_signature(result_b.model)

    any_difference = False
    for tree_a, tree_b in zip(sig_a, sig_b):
        for arr_a, arr_b in zip(tree_a, tree_b):
            if not np.array_equal(arr_a, arr_b):
                any_difference = True
                break
        if any_difference:
            break

    assert any_difference


def test_missing_training_predictor_is_rejected():
    frame = _frame()
    frame.loc[_fold().train_index[0], EXTRATREES_FEATURES[0]] = np.nan

    with pytest.raises(AssertionError, match="training data"):
        fit_extratrees_fold(
            frame,
            _fold(),
            n_estimators=25,
            max_depth=6,
        )


def test_missing_validation_predictor_is_rejected():
    frame = _frame()
    frame.loc[
        _fold().validation_index[0],
        EXTRATREES_FEATURES[0],
    ] = np.nan

    with pytest.raises(AssertionError, match="validation data"):
        fit_extratrees_fold(
            frame,
            _fold(),
            n_estimators=25,
            max_depth=6,
        )


def test_single_class_training_target_is_rejected():
    frame = _frame()
    frame.loc[_fold().train_index, "target"] = 0

    with pytest.raises(ValueError, match="both classes"):
        fit_extratrees_fold(
            frame,
            _fold(),
            n_estimators=25,
            max_depth=6,
        )


def test_target_is_not_in_extratrees_feature_manifest():
    assert "target" not in EXTRATREES_FEATURES
