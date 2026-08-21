"""Phase 1.6 descriptive dataset audit.

Development periods may report target-derived statistics.
The protected 2022-2025 Final Test is audited structurally only: no target
prevalence, positive count, negative count, or other outcome-derived summary is
computed or returned for Final Test rows.
"""

from __future__ import annotations

import pandas as pd

from src.features.integrated import PRIMARY_FEATURE_COLUMNS
from .temporal_splits import PERIOD_FINAL_TEST


AUDIT_COLUMNS = (
    "n_rows",
    "n_supervised_eligible",
    "fraction_supervised_eligible",
    "n_feature_complete",
    "fraction_feature_complete",
    "n_target_known",
    "fraction_target_known",
    "n_feature_incomplete",
    "n_unknown_target",
    "n_feature_incomplete_unknown_target",
    "n_positive",
    "n_negative",
    "target_prevalence",
)

FINAL_TEST_FORBIDDEN_OUTCOME_COLUMNS = (
    "n_target_known",
    "fraction_target_known",
    "n_unknown_target",
    "n_feature_incomplete_unknown_target",
    "n_supervised_eligible",
    "fraction_supervised_eligible",
    "n_positive",
    "n_negative",
    "target_prevalence",
)


def _require_aligned(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    if not dataset.index.equals(status.index):
        raise ValueError("dataset and status indices must match exactly.")
    if not dataset.index.equals(splits.index):
        raise ValueError("dataset and splits indices must match exactly.")

    required_status = {
        "target_known",
        "features_complete",
        "n_missing_features",
        "supervised_eligible",
        "row_status",
    }
    missing_status = required_status.difference(status.columns)
    if missing_status:
        raise ValueError(
            f"status is missing required columns: {sorted(missing_status)}"
        )

    if "period" not in splits.columns:
        raise ValueError("splits must contain a 'period' column.")

    missing_features = set(PRIMARY_FEATURE_COLUMNS).difference(dataset.columns)
    if missing_features:
        raise ValueError(
            "dataset is missing frozen primary feature columns."
        )
    if "target" not in dataset.columns:
        raise ValueError("dataset must contain target.")


def audit_dataset_by_period(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Return one descriptive audit row per temporal period.

    For development periods, target-derived summaries are allowed.

    For ``final_test``, only structural feature-side summaries are populated.
    All target/eligibility/outcome-derived fields are returned as ``pd.NA``.
    This prevents the Phase 1 audit API from becoming a convenient path for
    inspecting protected Final Test outcomes.
    """

    _require_aligned(dataset, status, splits)

    rows = []
    period_order = list(dict.fromkeys(splits["period"].astype(str).tolist()))

    for period in period_order:
        mask = splits["period"].astype(str).eq(period)
        ds = dataset.loc[mask]
        st = status.loc[mask]

        n_rows = len(ds)
        n_feature_complete = int(st["features_complete"].sum())
        n_feature_incomplete = int((~st["features_complete"]).sum())

        record = {
            "period": period,
            "n_rows": n_rows,
            "n_supervised_eligible": pd.NA,
            "fraction_supervised_eligible": pd.NA,
            "n_feature_complete": n_feature_complete,
            "fraction_feature_complete": (
                n_feature_complete / n_rows if n_rows else pd.NA
            ),
            "n_target_known": pd.NA,
            "fraction_target_known": pd.NA,
            "n_feature_incomplete": n_feature_incomplete,
            "n_unknown_target": pd.NA,
            "n_feature_incomplete_unknown_target": pd.NA,
            "n_positive": pd.NA,
            "n_negative": pd.NA,
            "target_prevalence": pd.NA,
        }

        if period != PERIOD_FINAL_TEST:
            known = st["target_known"]
            eligible = st["supervised_eligible"]
            n_known = int(known.sum())
            n_eligible = int(eligible.sum())

            y = ds.loc[known, "target"]
            n_positive = int((y == 1).sum())
            n_negative = int((y == 0).sum())

            record.update(
                {
                    "n_supervised_eligible": n_eligible,
                    "fraction_supervised_eligible": (
                        n_eligible / n_rows if n_rows else pd.NA
                    ),
                    "n_target_known": n_known,
                    "fraction_target_known": (
                        n_known / n_rows if n_rows else pd.NA
                    ),
                    "n_unknown_target": int(
                        (st["row_status"] == "unknown_target").sum()
                    ),
                    "n_feature_incomplete_unknown_target": int(
                        (
                            st["row_status"]
                            == "feature_incomplete_unknown_target"
                        ).sum()
                    ),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "target_prevalence": (
                        n_positive / n_known if n_known else pd.NA
                    ),
                }
            )

        rows.append(record)

    result = pd.DataFrame(rows).set_index("period")
    result = result.loc[:, list(AUDIT_COLUMNS)]
    return result


def audit_feature_missingness_by_period(
    dataset: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Return structural predictor missingness by period.

    This function never reads the target column and is therefore safe for
    structural auditing of the protected Final Test.
    """

    if not dataset.index.equals(splits.index):
        raise ValueError("dataset and splits indices must match exactly.")
    if "period" not in splits.columns:
        raise ValueError("splits must contain a 'period' column.")

    missing_features = set(PRIMARY_FEATURE_COLUMNS).difference(dataset.columns)
    if missing_features:
        raise ValueError(
            "dataset is missing frozen primary feature columns."
        )

    x = dataset.loc[:, list(PRIMARY_FEATURE_COLUMNS)]
    period = splits["period"].astype(str)

    records = []
    for name in dict.fromkeys(period.tolist()):
        mask = period.eq(name)
        block = x.loc[mask]
        n_rows = len(block)

        for feature in PRIMARY_FEATURE_COLUMNS:
            n_missing = int(block[feature].isna().sum())
            records.append(
                {
                    "period": name,
                    "feature": feature,
                    "n_rows": n_rows,
                    "n_missing": n_missing,
                    "fraction_missing": (
                        n_missing / n_rows if n_rows else pd.NA
                    ),
                }
            )

    return (
        pd.DataFrame(records)
        .set_index(["period", "feature"])
        .sort_index()
    )
