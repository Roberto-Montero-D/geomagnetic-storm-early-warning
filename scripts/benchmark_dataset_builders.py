"""Synthetic benchmark for optimized canonical feature builders."""
from time import perf_counter
import numpy as np
import pandas as pd
from src.features.rolling import build_rolling_features
from src.features.persistence import build_persistence_features
from src.features.dynamics import build_dynamic_features


def main():
    hours = 24 * 365 * 3
    index = pd.date_range("2015-01-01", periods=hours, freq="h")
    rng = np.random.default_rng(7)
    omni = pd.DataFrame({
        "bz_gsm": rng.normal(-2, 6, hours),
        "bt": rng.uniform(1, 15, hours),
        "speed": rng.normal(500, 120, hours),
        "density": rng.uniform(1, 20, hours),
        "flow_pressure": rng.uniform(0.1, 8, hours),
    }, index=index)
    prediction_index = pd.date_range(
        index[30], index[-1] + pd.Timedelta(hours=2),
        freq="h", name="prediction_time"
    )
    for name, function in (
        ("rolling", build_rolling_features),
        ("persistence", build_persistence_features),
        ("dynamics", build_dynamic_features),
    ):
        start = perf_counter()
        result = function(omni, prediction_index)
        print(f"{name:12s} {perf_counter()-start:9.3f} s  {result.shape}")


if __name__ == "__main__":
    main()
