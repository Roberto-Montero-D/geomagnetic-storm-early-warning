import pytest
from src.model_selection.contract import PHASE5_CONFIGURATIONS,PHASE5_RANDOM_STATE
from src.model_selection.factories import FAMILY_FIXED_PARAMS,assert_phase5_dependencies,configuration_by_id,dependency_versions,make_phase5_model,make_phase5_model_by_id

def test_dependencies():
    assert set(dependency_versions())=={"scikit-learn","lightgbm","xgboost"}
    assert_phase5_dependencies()

def test_all_27_instantiate_and_preserve_frozen_params():
    assert len(PHASE5_CONFIGURATIONS)==27
    for c in PHASE5_CONFIGURATIONS:
        m=make_phase5_model(c); p=m.get_params()
        assert hasattr(m,"fit") and hasattr(m,"predict_proba")
        assert p["random_state"]==PHASE5_RANDOM_STATE
        for k,v in c.as_dict().items(): assert p[k]==v

def test_no_class_weight_or_positive_weight():
    for c in PHASE5_CONFIGURATIONS:
        p=make_phase5_model(c).get_params()
        if c.family in ("extratrees","lightgbm"): assert p["class_weight"] is None
        if c.family == "xgboost":
            assert p.get("scale_pos_weight") in (None, 1)

def test_fixed_params_do_not_overlap_grid():
    for c in PHASE5_CONFIGURATIONS:
        assert not set(c.as_dict()) & set(FAMILY_FIXED_PARAMS[c.family])

def test_lookup_roundtrip():
    for c in PHASE5_CONFIGURATIONS:
        assert configuration_by_id(c.config_id)==c
        assert make_phase5_model_by_id(c.config_id).get_params()["random_state"]==42

def test_unknown_rejected():
    with pytest.raises(KeyError): configuration_by_id("not_a_phase5_configuration")

def test_non_frozen_rejected():
    from src.model_selection.contract import ModelConfiguration
    rogue=ModelConfiguration("rogue","extratrees",(("n_estimators",999),("max_depth",1),("random_state",42)))
    with pytest.raises(ValueError,match="only be created from frozen"): make_phase5_model(rogue)
