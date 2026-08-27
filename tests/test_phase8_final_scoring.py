import pandas as pd
import pytest

from src.final_test.contract import (
    PHASE8_OPERATIONAL_THRESHOLD,
)
from src.final_test.scoring import (
    _scope_phase8_events,
    score_phase8_final_test,
)


def _events():
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "start_time": pd.to_datetime(
                [
                    "2021-12-31 23:00",
                    "2022-01-01 02:00",
                    "2025-12-31 22:00",
                    "2026-01-01 00:00",
                ]
            ),
            "end_time": pd.to_datetime(
                [
                    "2022-01-01 01:00",
                    "2022-01-01 05:00",
                    "2025-12-31 23:00",
                    "2026-01-01 03:00",
                ]
            ),
            "boundary_status": [
                "complete",
                "complete",
                "complete",
                "complete",
            ],
        }
    )


def test_phase8_threshold_remains_frozen():
    assert PHASE8_OPERATIONAL_THRESHOLD == 0.10


def test_scoring_rejects_misaligned_targets():
    idx = pd.date_range(
        "2022-01-01",
        periods=4,
        freq="h",
    )
    p = pd.Series(
        [0.1, 0.2, 0.3, 0.4],
        index=idx,
    )
    y = pd.Series(
        [0, 0, 1, 1],
        index=idx + pd.Timedelta(hours=1),
    )

    with pytest.raises(
        ValueError,
        match="identical indices",
    ):
        score_phase8_final_test(
            p,
            y,
            _events(),
        )


def test_scoring_has_no_threshold_argument():
    import inspect

    signature = inspect.signature(
        score_phase8_final_test
    )

    assert tuple(signature.parameters) == (
        "probabilities",
        "targets",
        "events",
    )


def test_event_scope_uses_canonical_phase8_bounds():
    scoped = _scope_phase8_events(
        _events()
    )

    assert scoped["event_id"].tolist() == [
        2,
        3,
    ]


def test_event_scope_uses_canonical_event_column_names():
    bad = _events().rename(
        columns={
            "start_time": "storm_start",
        }
    )

    with pytest.raises(
        ValueError,
        match="canonical columns",
    ):
        _scope_phase8_events(bad)


def test_runner_requires_explicit_execution_flag():
    from scripts.run_phase8_final_test import parse_args
    import inspect

    source = inspect.getsource(parse_args)

    assert "--execute-protected-final-test" in source
