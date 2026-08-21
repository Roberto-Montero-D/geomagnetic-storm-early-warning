"""B1 Physical baseline with predeclared configuration."""

from __future__ import annotations
import math
import pandas as pd

PHYSICAL_FEATURES = ("bz_gsm", "speed")
DEFAULT_BZ_MAGNITUDE_NT = 5.0
DEFAULT_SPEED_THRESHOLD_KM_S = 500.0

def _validate_thresholds(bz_magnitude_nt: float, speed_threshold_km_s: float):
    try:
        bz_threshold=float(bz_magnitude_nt); speed_threshold=float(speed_threshold_km_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("Physical thresholds must be numeric.") from exc
    if not math.isfinite(bz_threshold) or bz_threshold <= 0:
        raise ValueError("bz_magnitude_nt must be finite and positive.")
    if not math.isfinite(speed_threshold) or speed_threshold <= 0:
        raise ValueError("speed_threshold_km_s must be finite and positive.")
    return bz_threshold, speed_threshold

def predict_physical(features: pd.DataFrame, *,
                     bz_magnitude_nt: float = DEFAULT_BZ_MAGNITUDE_NT,
                     speed_threshold_km_s: float = DEFAULT_SPEED_THRESHOLD_KM_S) -> pd.Series:
    missing=[c for c in PHYSICAL_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"features is missing required physical columns: {missing}")
    bz_threshold,speed_threshold=_validate_thresholds(bz_magnitude_nt,speed_threshold_km_s)
    bz=pd.to_numeric(features["bz_gsm"],errors="raise")
    speed=pd.to_numeric(features["speed"],errors="raise")
    prediction=pd.Series(pd.NA,index=features.index,dtype="Int8",name="prediction")
    known=bz.notna() & speed.notna()
    prediction.loc[known]=((bz.loc[known] < -bz_threshold) &
                           (speed.loc[known] > speed_threshold)).astype("int8")
    return prediction

def predict_physical_for_index(dataset: pd.DataFrame, prediction_index: pd.DatetimeIndex, *,
                               bz_magnitude_nt: float = DEFAULT_BZ_MAGNITUDE_NT,
                               speed_threshold_km_s: float = DEFAULT_SPEED_THRESHOLD_KM_S) -> pd.Series:
    missing=prediction_index.difference(dataset.index)
    if len(missing):
        raise ValueError("prediction_index contains timestamps absent from dataset.")
    return predict_physical(dataset.loc[prediction_index,list(PHYSICAL_FEATURES)],
                            bz_magnitude_nt=bz_magnitude_nt,
                            speed_threshold_km_s=speed_threshold_km_s)
