import pandas as pd
from src.baselines.physical import DEFAULT_BZ_MAGNITUDE_NT, DEFAULT_SPEED_THRESHOLD_KM_S, predict_physical
from src.baselines.extratrees import DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH, make_extratrees_model

def test_b1_configuration_is_frozen():
    assert DEFAULT_BZ_MAGNITUDE_NT == 5.0
    assert DEFAULT_SPEED_THRESHOLD_KM_S == 500.0
    x=pd.DataFrame({"bz_gsm":[-5.0,-5.01],"speed":[600.0,500.01]})
    assert predict_physical(x).tolist() == [0,1]

def test_b3_configuration_is_frozen():
    assert DEFAULT_N_ESTIMATORS == 100
    assert DEFAULT_MAX_DEPTH == 10
    model=make_extratrees_model()
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.class_weight is None
