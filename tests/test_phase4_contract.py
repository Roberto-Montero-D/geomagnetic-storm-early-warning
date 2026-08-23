from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES
from src.imbalance.contract import *

def test_features_are_phase3_freeze():
    assert PHASE4_FEATURES == PHASE3_SELECTED_FEATURES
    assert len(PHASE4_FEATURES)==10

def test_exact_grid():
    assert len(PHASE4_EXPERIMENTS)==17
    assert len(set(PHASE4_EXPERIMENT_NAMES))==17
    assert [x.parameter for x in PHASE4_EXPERIMENTS if x.strategy=="class_weighting"] == [1,3,5,10,20,50]
    assert [x.parameter for x in PHASE4_EXPERIMENTS if x.strategy=="random_undersampling"] == ["10:1","5:1","2:1"]
    assert [x.parameter for x in PHASE4_EXPERIMENTS if x.strategy=="smote"] == [3,5,7]
    assert [x.parameter for x in PHASE4_EXPERIMENTS if x.strategy=="borderline_smote"] == [3,5,7]

def test_fixed_model_and_evaluation_contract():
    assert PHASE4_EXTRATREES_PARAMS == {"n_estimators":100,"max_depth":10,"random_state":42}
    assert PHASE4_SCREENING_FOLD=="initial_screening"
    assert PHASE4_CONFIRMATION_FOLDS==("walk_forward_1","walk_forward_2")
    assert PHASE4_MAX_FAR_PER_DAY==0.2
    assert PHASE4_ADVANCE_COUNT==3
    assert PHASE4_CONTRACT_STATUS=="frozen"
