"""Integrated causal feature-frame construction.

This module is the single assembly point for the frozen Phase 0.7 primary
feature universe:

    raw
    rolling
    persistence
    dynamics
    interactions

Each family remains independently testable. This module combines them in a
deterministic order and consolidates provenance into a row-level causal audit.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.features.dynamics import (
    DYNAMIC_FEATURE_COLUMNS,
    build_dynamic_features,
)
from src.features.interactions import (
    INTERACTION_FEATURE_COLUMNS,
    build_interaction_features,
)
from src.features.persistence import (
    PERSISTENCE_FEATURE_COLUMNS,
    build_persistence_features,
)
from src.features.raw import (
    PRIMARY_RAW_FEATURE_COLUMNS,
    build_raw_features,
)
from src.features.rolling import (
    build_rolling_features,
    rolling_feature_names,
)

FEATURE_FAMILY_ORDER = (
    "raw",
    "rolling",
    "persistence",
    "dynamics",
    "interactions",
)

PRIMARY_FEATURE_COLUMNS = (
    *PRIMARY_RAW_FEATURE_COLUMNS,
    *rolling_feature_names(),
    *PERSISTENCE_FEATURE_COLUMNS,
    *DYNAMIC_FEATURE_COLUMNS,
    *INTERACTION_FEATURE_COLUMNS,
)


def _validate_manifest() -> None:
    if len(PRIMARY_FEATURE_COLUMNS) != len(set(PRIMARY_FEATURE_COLUMNS)):
        raise AssertionError(
            "Primary feature manifest contains duplicate column names."
        )


_validate_manifest()


def build_primary_feature_frame(
    omni: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete frozen Phase 0.7 causal feature frame."""

    raw, raw_audit = build_raw_features(
        omni,
        kp_intervals,
        prediction_times,
        return_audit=True,
    )
    rolling, rolling_audit = build_rolling_features(
        omni,
        raw.index,
        return_audit=True,
    )
    persistence, persistence_audit = build_persistence_features(
        omni,
        raw.index,
        return_audit=True,
    )
    dynamics, dynamics_audit = build_dynamic_features(
        omni,
        raw.index,
        return_audit=True,
    )
    interactions = build_interaction_features(raw)

    families = (
        raw,
        rolling,
        persistence,
        dynamics,
        interactions,
    )

    for family in families[1:]:
        if not family.index.equals(raw.index):
            raise AssertionError(
                "Feature-family prediction indexes are not identical."
            )

    features = pd.concat(families, axis=1)

    if features.columns.has_duplicates:
        duplicates = features.columns[
            features.columns.duplicated()
        ].tolist()
        raise AssertionError(
            f"Duplicate integrated feature columns: {duplicates}"
        )

    features = features.loc[:, PRIMARY_FEATURE_COLUMNS]
    features.index.name = "prediction_time"

    if not return_audit:
        return features

    audit = pd.DataFrame(index=features.index)
    audit.index.name = "prediction_time"

    audit["raw_information_time"] = raw_audit[
        "maximum_feature_information_time"
    ]
    audit["rolling_information_time"] = rolling_audit[
        "maximum_rolling_information_time"
    ]
    audit["persistence_information_time"] = persistence_audit[
        "persistence_information_time"
    ]
    audit["dynamics_information_time"] = dynamics_audit[
        "dynamics_information_time"
    ]

    # Interactions are deterministic transformations of the raw state, so
    # their provenance is exactly the raw OMNI information time.
    audit["interaction_information_time"] = raw_audit[
        "omni_information_time"
    ]

    audit["maximum_feature_information_time"] = audit[
        [
            "raw_information_time",
            "rolling_information_time",
            "persistence_information_time",
            "dynamics_information_time",
            "interaction_information_time",
        ]
    ].max(axis=1)

    audit["information_cutoff"] = raw_audit["information_cutoff"]

    cutoff_sources = (
        rolling_audit["information_cutoff"],
        persistence_audit["information_cutoff"],
        dynamics_audit["information_cutoff"],
    )
    if any(
        not source.equals(audit["information_cutoff"])
        for source in cutoff_sources
    ):
        raise AssertionError(
            "Feature families disagree on the information cutoff."
        )

    violation = (
        audit["maximum_feature_information_time"].notna()
        & (
            audit["maximum_feature_information_time"]
            > audit["information_cutoff"]
        )
    )
    if violation.any():
        raise AssertionError(
            "Integrated feature provenance exceeds the information cutoff."
        )

    return features, audit
