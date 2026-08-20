"""Runtime logging gates for TriAttention diagnostics."""

from __future__ import annotations

import os

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def runtime_logging_enabled(default: bool = True) -> bool:
    """Master switch for TriAttention runtime diagnostic logging."""

    return _env_bool("TRIATTN_RUNTIME_LOGGING", default)


def runtime_profile_logging_enabled(name: str, default: str = "0") -> bool:
    """Return whether a profile-specific log stream may emit records."""

    default_enabled = str(default).strip().lower() in _TRUE_VALUES
    return runtime_logging_enabled() and _env_bool(name, default_enabled)
