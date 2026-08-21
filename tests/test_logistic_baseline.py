import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.baselines.logistic import (
    LOGISTIC_FEATURES,
    fit_logistic_fold,
    make_logistic_pipeline,
)
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


def _frame():
    index = pd.date_range(
        "2010-01-01",
        periods=12,
        freq="h",
        name="prediction_time",
    )
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(12, len(LOGISTIC_FEATURES))),
        index=index,
        columns=list(LOGISTIC_FEATURES),
    )
    frame["target"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    frame["engineered_future_like"] = np.arange(12)
    return frame


def _fold():
    index = pd.date_range(
        "2010-01-01",
        periods=12,
        freq="h",
        name="prediction_time",
    )
    return DevelopmentFold(
        name="screening",
        train_index=index[:8],
        validation_index=index[8:],
    )


def test_b2_uses_exact_raw_primary_manifest():
    assert LOGISTIC_FEATURES == tuple(PRIMARY_RAW_FEATURE_COLUMNS)
    assert len(LOGISTIC_FEATURES) == 10


def test_pipeline_is_unbalanced_and_scaled():
    pipeline = make_logistic_pipeline()

    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert pipeline.named_steps["classifier"].class_weight is None


def test_fit_returns_validation_probabilities_on_exact_index():
    result = fit_logistic_fold(_frame(), _fold())

    pd.testing.assert_index_equal(
        result.validation_probability.index,
        _fold().validation_index,
    )
    assert result.validation_probability.name == "probability"
    assert result.validation_probability.between(0, 1).all()


def test_unrequested_engineered_column_cannot_affect_b2():
    frame = _frame()
    before = fit_logistic_fold(frame, _fold()).validation_probability

    mutated = frame.copy()
    mutated["engineered_future_like"] = np.linspace(
        -1e12, 1e12, len(mutated)
    )
    after = fit_logistic_fold(
        mutated, _fold()
    ).validation_probability

    pd.testing.assert_series_equal(before, after)


def test_validation_target_cannot_affect_fit_or_probability():
    frame = _frame()
    before = fit_logistic_fold(frame, _fold()).validation_probability

    mutated = frame.copy()
    mutated.loc[_fold().validation_index, "target"] = [1, 1, 1, 1]
    after = fit_logistic_fold(
        mutated, _fold()
    ).validation_probability

    pd.testing.assert_series_equal(before, after)


def test_validation_features_do_not_affect_fitted_scaler_state():
    frame = _frame()
    result_a = fit_logistic_fold(frame, _fold())

    mutated = frame.copy()
    mutated.loc[_fold().validation_index, LOGISTIC_FEATURES] += 1e6
    result_b = fit_logistic_fold(mutated, _fold())

    scaler_a = result_a.model.named_steps["scaler"]
    scaler_b = result_b.model.named_steps["scaler"]

    np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_)


def test_validation_features_do_not_affect_fitted_classifier_state():
    frame = _frame()
    result_a = fit_logistic_fold(frame, _fold())

    mutated = frame.copy()
    mutated.loc[_fold().validation_index, LOGISTIC_FEATURES] -= 1e6
    result_b = fit_logistic_fold(mutated, _fold())

    clf_a = result_a.model.named_steps["classifier"]
    clf_b = result_b.model.named_steps["classifier"]

    np.testing.assert_allclose(clf_a.coef_, clf_b.coef_)
    np.testing.assert_allclose(clf_a.intercept_, clf_b.intercept_)


def test_training_features_do_affect_fitted_state():
    frame = _frame()
    result_a = fit_logistic_fold(frame, _fold())

    mutated = frame.copy()
    feature = LOGISTIC_FEATURES[0]
    mutated.loc[_fold().train_index, feature] *= 10.0
    result_b = fit_logistic_fold(mutated, _fold())

    scaler_a = result_a.model.named_steps["scaler"]
    scaler_b = result_b.model.named_steps["scaler"]

    assert not np.allclose(scaler_a.scale_, scaler_b.scale_)


def test_missing_training_predictor_is_rejected():
    frame = _frame()
    frame.loc[_fold().train_index[0], LOGISTIC_FEATURES[0]] = np.nan

    with pytest.raises(AssertionError, match="training data"):
        fit_logistic_fold(frame, _fold())


def test_missing_validation_predictor_is_rejected():
    frame = _frame()
    frame.loc[_fold().validation_index[0], LOGISTIC_FEATURES[0]] = np.nan

    with pytest.raises(AssertionError, match="validation data"):
        fit_logistic_fold(frame, _fold())


def test_single_class_training_target_is_rejected():
    frame = _frame()
    frame.loc[_fold().train_index, "target"] = 0

    with pytest.raises(ValueError, match="both classes"):
        fit_logistic_fold(frame, _fold())


def test_target_is_not_in_logistic_feature_manifest():
    assert "target" not in LOGISTIC_FEATURES
