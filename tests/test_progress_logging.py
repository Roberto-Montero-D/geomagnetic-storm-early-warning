"""Instrumentation-only regression tests for progress logging."""
from __future__ import annotations

import pandas as pd
import pytest

from src.dataset import builder as dataset_builder
from src.features import integrated as integrated_features


def test_integrated_progress_callback_does_not_change_result(monkeypatch):
    index=pd.date_range(
        "2020-01-01",periods=3,freq="h",name="prediction_time"
    )
    raw=pd.DataFrame({"raw_x":[1.,2.,3.]},index=index)
    raw_audit=pd.DataFrame({
        "maximum_feature_information_time":index-pd.Timedelta(hours=2),
        "omni_information_time":index-pd.Timedelta(hours=2),
        "information_cutoff":index-pd.Timedelta(hours=1),
    },index=index)
    rolling=pd.DataFrame({"roll_x":[1.,2.,3.]},index=index)
    rolling_audit=pd.DataFrame({
        "maximum_rolling_information_time":index-pd.Timedelta(hours=2),
        "information_cutoff":index-pd.Timedelta(hours=1),
    },index=index)
    persistence=pd.DataFrame({"persist_x":[1.,2.,3.]},index=index)
    persistence_audit=pd.DataFrame({
        "persistence_information_time":index-pd.Timedelta(hours=2),
        "information_cutoff":index-pd.Timedelta(hours=1),
    },index=index)
    dynamics=pd.DataFrame({"dyn_x":[1.,2.,3.]},index=index)
    dynamics_audit=pd.DataFrame({
        "dynamics_information_time":index-pd.Timedelta(hours=2),
        "information_cutoff":index-pd.Timedelta(hours=1),
    },index=index)
    interactions=pd.DataFrame({"int_x":[1.,2.,3.]},index=index)

    monkeypatch.setattr(
        integrated_features,"PRIMARY_RAW_FEATURE_COLUMNS",("raw_x",)
    )
    monkeypatch.setattr(
        integrated_features,"PRIMARY_FEATURE_COLUMNS",
        ("raw_x","roll_x","persist_x","dyn_x","int_x")
    )
    monkeypatch.setattr(
        integrated_features,"build_raw_features",
        lambda *a,**k:(raw.copy(),raw_audit.copy())
    )
    monkeypatch.setattr(
        integrated_features,"build_rolling_features",
        lambda *a,**k:(rolling.copy(),rolling_audit.copy())
    )
    monkeypatch.setattr(
        integrated_features,"build_persistence_features",
        lambda *a,**k:(persistence.copy(),persistence_audit.copy())
    )
    monkeypatch.setattr(
        integrated_features,"build_dynamic_features",
        lambda *a,**k:(dynamics.copy(),dynamics_audit.copy())
    )
    monkeypatch.setattr(
        integrated_features,"build_interaction_features",
        lambda *a,**k:interactions.copy()
    )

    no_log=integrated_features.build_primary_feature_frame(
        pd.DataFrame(),pd.DataFrame(),index,return_audit=True
    )
    messages=[]
    with_log=integrated_features.build_primary_feature_frame(
        pd.DataFrame(),pd.DataFrame(),index,return_audit=True,
        progress=messages.append
    )

    pd.testing.assert_frame_equal(no_log[0],with_log[0])
    pd.testing.assert_frame_equal(no_log[1],with_log[1])
    assert any("[features 1/5]" in m for m in messages)
    assert any("[features 5/5]" in m for m in messages)


def test_dataset_progress_callback_is_optional_and_reports_completion(
    monkeypatch,
):
    index=pd.date_range(
        "2020-01-01",periods=3,freq="h",name="prediction_time"
    )
    features=pd.DataFrame({"f":[1.,2.,3.]},index=index)
    fa=pd.DataFrame({
        c: index-pd.Timedelta(hours=1)
        for c in dataset_builder.FEATURE_AUDIT_COLUMNS
    },index=index)
    target=pd.Series([0,1,0],index=index,name="target")
    ta=pd.DataFrame(index=index)
    for c in dataset_builder.TARGET_AUDIT_COLUMNS:
        ta[c]=0

    monkeypatch.setattr(
        dataset_builder,"PRIMARY_FEATURE_COLUMNS",("f",)
    )
    monkeypatch.setattr(
        dataset_builder,"build_primary_feature_frame",
        lambda *a,**k:(features.copy(),fa.copy())
    )
    monkeypatch.setattr(
        dataset_builder,"build_event_window_target",
        lambda *a,**k:(target.copy(),ta.copy())
    )

    baseline=dataset_builder.build_canonical_dataset(
        pd.DataFrame(),pd.DataFrame(),index
    )
    messages=[]
    logged=dataset_builder.build_canonical_dataset(
        pd.DataFrame(),pd.DataFrame(),index,progress=messages.append
    )

    pd.testing.assert_frame_equal(baseline,logged)
    assert any("[dataset 1/3]" in m for m in messages)
    assert any("[dataset 2/3]" in m for m in messages)
    assert any("Canonical dataset complete" in m for m in messages)
