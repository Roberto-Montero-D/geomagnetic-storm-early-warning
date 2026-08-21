import numpy as np
import pandas as pd

from src.baselines.framework import DevelopmentFold
from src.evaluation.development_predictions import (
    BASELINE_NAMES,
    generate_fold_baseline_predictions,
)
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


def _dataset():
    index=pd.date_range("2016-12-31 00:00",periods=72,freq="h",name="prediction_time")
    rng=np.random.default_rng(31)
    frame=pd.DataFrame(
        rng.normal(size=(len(index),len(PRIMARY_RAW_FEATURE_COLUMNS))),
        index=index,columns=list(PRIMARY_RAW_FEATURE_COLUMNS),
    )
    # Put physical variables on plausible positive/negative scales.
    frame["bz_gsm"]=rng.normal(0,6,len(index))
    frame["bt"]=np.abs(rng.normal(7,2,len(index)))
    frame["speed"]=rng.normal(500,80,len(index))
    frame["density"]=np.abs(rng.normal(6,2,len(index)))
    frame["flow_pressure"]=np.abs(rng.normal(2,0.5,len(index)))
    for c in ["kp_lag_1h","kp_lag_3h","kp_lag_6h","kp_lag_12h","kp_lag_24h"]:
        frame[c]=rng.uniform(0,8,len(index))
    frame["target"]=([0,1]*36)
    return frame


def _fold(index):
    return DevelopmentFold(
        name="synthetic",
        train_index=index[:48],
        validation_index=index[48:],
    )


def test_generates_all_four_baselines_on_exact_validation_index():
    dataset=_dataset()
    fold=_fold(dataset.index)
    result=generate_fold_baseline_predictions(dataset,fold)

    assert tuple(result.probabilities)==BASELINE_NAMES
    assert result.fold_name=="synthetic"
    for p in result.probabilities.values():
        pd.testing.assert_index_equal(p.index,fold.validation_index)
        assert p.name=="probability"


def test_deterministic_baselines_are_binary_probabilities():
    dataset=_dataset()
    fold=_fold(dataset.index)
    result=generate_fold_baseline_predictions(dataset,fold)

    for name in ("B0_persistence","B1_physical"):
        values=result.probabilities[name].dropna()
        assert values.isin([0.0,1.0]).all()


def test_probabilistic_baselines_are_bounded():
    dataset=_dataset()
    fold=_fold(dataset.index)
    result=generate_fold_baseline_predictions(dataset,fold)

    for name in ("B2_logistic","B3_extratrees"):
        assert result.probabilities[name].between(0,1).all()


def test_validation_target_mutation_cannot_change_any_prediction():
    dataset=_dataset()
    fold=_fold(dataset.index)
    before=generate_fold_baseline_predictions(dataset,fold)

    mutated=dataset.copy()
    mutated.loc[fold.validation_index,"target"]=1-mutated.loc[fold.validation_index,"target"]
    after=generate_fold_baseline_predictions(mutated,fold)

    for name in BASELINE_NAMES:
        pd.testing.assert_series_equal(
            before.probabilities[name],
            after.probabilities[name],
        )


def test_rows_outside_fold_cannot_change_predictions():
    dataset=_dataset()
    fold=_fold(dataset.index)
    before=generate_fold_baseline_predictions(dataset,fold)

    extra_time=pd.Timestamp("2022-01-01 00:00")
    extra=dataset.iloc[[0]].copy()
    extra.index=pd.DatetimeIndex([extra_time],name="prediction_time")
    extra.loc[:,list(PRIMARY_RAW_FEATURE_COLUMNS)]=1e9
    extra["target"]=1
    augmented=pd.concat([dataset,extra])

    after=generate_fold_baseline_predictions(augmented,fold)

    for name in BASELINE_NAMES:
        pd.testing.assert_series_equal(
            before.probabilities[name],
            after.probabilities[name],
            check_freq=False,
        )
