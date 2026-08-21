"""Canonical retrospective target construction."""

from .event_window import (
    DEFAULT_TARGET_HORIZON_HOURS,
    DEFAULT_TARGET_THRESHOLD,
    build_event_window_target,
)

__all__ = [
    "DEFAULT_TARGET_THRESHOLD",
    "DEFAULT_TARGET_HORIZON_HOURS",
    "build_event_window_target",
]
