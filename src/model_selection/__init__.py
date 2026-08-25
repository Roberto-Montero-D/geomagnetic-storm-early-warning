"""Phase 5 model-selection contracts."""

from .contract import (
    EXTRATREES_GRID,
    LIGHTGBM_GRID,
    XGBOOST_GRID,
    ModelConfiguration,
    PHASE5_ADVANCE_PER_FAMILY,
    PHASE5_CONFIGURATIONS,
    PHASE5_CONFIG_IDS,
    PHASE5_CONFIRMATION_FOLDS,
    PHASE5_CONFIRMATION_RANKING_RULE,
    PHASE5_FEATURES,
    PHASE5_IMBALANCE_EXPERIMENT,
    PHASE5_MAX_FAR_PER_DAY,
    PHASE5_RANDOM_STATE,
    PHASE5_RANKING_RULE,
    PHASE5_SCREENING_FOLD,
    PHASE5_STACKING_STATUS,
    validate_phase5_contract,
)

from .factories import (
    FAMILY_FIXED_PARAMS,
    SUPPORTED_FAMILIES,
    assert_phase5_dependencies,
    configuration_by_id,
    dependency_versions,
    make_phase5_model,
    make_phase5_model_by_id,
)

__all__ = [
    "EXTRATREES_GRID","LIGHTGBM_GRID","XGBOOST_GRID","ModelConfiguration",
    "PHASE5_ADVANCE_PER_FAMILY","PHASE5_CONFIGURATIONS","PHASE5_CONFIG_IDS",
    "PHASE5_CONFIRMATION_FOLDS","PHASE5_CONFIRMATION_RANKING_RULE",
    "PHASE5_FEATURES","PHASE5_IMBALANCE_EXPERIMENT","PHASE5_MAX_FAR_PER_DAY",
    "PHASE5_RANDOM_STATE","PHASE5_RANKING_RULE","PHASE5_SCREENING_FOLD",
    "PHASE5_STACKING_STATUS","validate_phase5_contract",
    "FAMILY_FIXED_PARAMS",
    "SUPPORTED_FAMILIES",
    "assert_phase5_dependencies",
    "configuration_by_id",
    "dependency_versions",
    "make_phase5_model",
    "make_phase5_model_by_id",
]
