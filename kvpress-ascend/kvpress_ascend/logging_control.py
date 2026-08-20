"""Logging control for kvpress-ascend.

Two switches:
- ``KVPRESS_RUNTIME_LOGGING`` (master): startup / decision / event logs.
- ``KVPRESS_PROBE`` (per-inference verification): logs on EVERY model step
  whether the patch entered its core code and prints the core parameters.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from vllm.logger import logger as _vllm_logger
except Exception:  # pragma: no cover - lightweight/test environments
    _vllm_logger = logging.getLogger("kvpress_ascend")

_MARKER = "KVPRESS-ASCEND"


def get_logger() -> Any:
    return _vllm_logger


def _enabled() -> bool:
    from .envs import env_bool

    return env_bool("KVPRESS_RUNTIME_LOGGING", True)


def _probe_enabled() -> bool:
    from .envs import env_bool

    return env_bool("KVPRESS_RUNTIME_LOGGING", True) and env_bool("KVPRESS_PROBE", True)


def log_info(message: str, *args: Any, **kwargs: Any) -> None:
    if _enabled():
        _vllm_logger.info(f"[{_MARKER}] " + message, *args, **kwargs)


def log_debug(message: str, *args: Any, **kwargs: Any) -> None:
    if _enabled():
        _vllm_logger.debug(f"[{_MARKER}] " + message, *args, **kwargs)


def log_warning(message: str, *args: Any, **kwargs: Any) -> None:
    _vllm_logger.warning(f"[{_MARKER}] " + message, *args, **kwargs)


def log_error(message: str, *args: Any, **kwargs: Any) -> None:
    _vllm_logger.error(f"[{_MARKER}] " + message, *args, **kwargs)


def probe(message: str, *args: Any) -> None:
    """Per-inference core-entry probe (the user-visible 'is it really live' signal)."""
    if _probe_enabled():
        _vllm_logger.info(f"[{_MARKER}][PROBE] " + message, *args)


def attention_hook_log(message: str, *args: Any) -> None:
    from .envs import env_bool

    if env_bool("KVPRESS_LOG_ATTENTION_HOOK", False) and _enabled():
        _vllm_logger.debug(f"[{_MARKER}][HOOK] " + message, *args)
