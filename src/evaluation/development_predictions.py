"""Generate protected validation predictions for B0-B3.

This checkpoint performs model/rule execution only. Operational threshold
selection and metric aggregation remain separate evaluation steps.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from src.baselines.framework import DevelopmentFold
from src.baselines.persistence import predict_persistence_for_index
from src.baselines.physical import predict_physical_for_index
from src.baselines.logistic import fit_logistic_fold
from src.baselines.extratrees import fit_extratrees_fold
from src.evaluation.operational import binary_predictions_as_probabilities

BASELINE_NAMES=("B0_persistence","B1_physical","B2_logistic","B3_extratrees")

@dataclass(frozen=True)
class FoldBaselinePredictions:
    fold_name: str
    probabilities: dict[str,pd.Series]

def generate_fold_baseline_predictions(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
) -> FoldBaselinePredictions:
    """Generate validation probabilities using only one protected fold."""
    b0=binary_predictions_as_probabilities(
        predict_persistence_for_index(dataset,fold.validation_index)
    )
    b1=binary_predictions_as_probabilities(
        predict_physical_for_index(dataset,fold.validation_index)
    )
    b2=fit_logistic_fold(dataset,fold).validation_probability
    b3=fit_extratrees_fold(dataset,fold).validation_probability

    probabilities={
        "B0_persistence":b0,
        "B1_physical":b1,
        "B2_logistic":b2,
        "B3_extratrees":b3,
    }

    for name,series in probabilities.items():
        if not series.index.equals(fold.validation_index):
            raise AssertionError(f"{name} prediction index does not match validation index.")
        if series.name != "probability":
            raise AssertionError(f"{name} output must be named probability.")

    return FoldBaselinePredictions(fold.name,probabilities)
