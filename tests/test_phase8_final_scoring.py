import numpy as np
import pandas as pd
import pytest
from src.final_test.scoring import score_phase8_final_test
from src.final_test.contract import PHASE8_OPERATIONAL_THRESHOLD

def test_phase8_threshold_remains_frozen():
    assert PHASE8_OPERATIONAL_THRESHOLD == 0.10

def test_scoring_rejects_misaligned_targets():
    idx=pd.date_range("2022-01-01",periods=4,freq="h")
    p=pd.Series([.1,.2,.3,.4],index=idx)
    y=pd.Series([0,0,1,1],index=idx+pd.Timedelta(hours=1))
    events=pd.DataFrame(columns=["storm_id","storm_start","storm_end"])
    with pytest.raises(ValueError,match="identical indices"):
        score_phase8_final_test(p,y,events)

def test_scoring_has_no_threshold_argument():
    import inspect
    sig=inspect.signature(score_phase8_final_test)
    assert tuple(sig.parameters)==("probabilities","targets","events")

def test_runner_requires_explicit_execution_flag():
    from scripts.run_phase8_final_test import parse_args
    # Structural test: parser implementation contains the explicit guard option.
    import inspect
    source=inspect.getsource(parse_args)
    assert "--execute-protected-final-test" in source
