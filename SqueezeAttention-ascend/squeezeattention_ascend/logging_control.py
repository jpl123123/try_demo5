"""Logging control for SqueezeAttention-ascend.

- ``SQUEEZE_RUNTIME_LOGGING`` (master): startup / decision / event logs.
- ``SQUEEZE_PROBE`` (per-inference verification): per-step core-entry probe.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from vllm.logger import logger as _vllm_logger
except Exception:  # pragma: no cover - lightweight/test environments
    _vllm_logger = logging.getLogger("squeezeattention_ascend")

_MARKER = "SQUEEZE-ASCEND"


def get_logger() -> Any:
    return _vllm_logger


def _enabled() -> bool:
    from .envs import env_bool

    return env_bool("SQUEEZE_RUNTIME_LOGGING", True)


def _probe_enabled() -> bool:
    from .envs import env_bool

    return env_bool("SQUEEZE_RUNTIME_LOGGING", True) and env_bool("SQUEEZE_PROBE", True)


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
    if _probe_enabled():
        _vllm_logger.info(f"[{_MARKER}][PROBE] " + message, *args)


def cluster_log(message: str, *args: Any) -> None:
    from .envs import env_bool

    if env_bool("SQUEEZE_LOG_BUDGETS", True) and _enabled():
        _vllm_logger.info(f"[{_MARKER}][CLUSTER] " + message, *args)
