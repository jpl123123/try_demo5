"""vLLM plugin entry point for SqueezeAttention-ascend.

Activation (the user's ``export squeeze`` style):
    export SQUEEZE_ENABLE=1      # alias: export SQUEEZE=1

With the switch off the plugin is a no-op, so both tools can be installed in
the same environment.
"""

from __future__ import annotations

import os

from .envs import env_bool
from .logging_control import get_logger

logger = get_logger()
from .logging_control import log_info, log_warning


def register_squeezeattention_backend() -> None:
    """Install SqueezeAttention-ascend integration when loaded by vLLM."""
    if not (env_bool("SQUEEZE_ENABLE", False) or env_bool("SQUEEZE", False)):
        log_info("disabled (set SQUEEZE_ENABLE=1 to activate)")
        return
    try:
        from .runtime.monkeypatch import install_squeezeattention_integration_monkeypatches

        install_squeezeattention_integration_monkeypatches(
            patch_scheduler=True,
            patch_worker=True,
        )
        log_info(
            "plugin activated: SQUEEZE_ENABLE=1 mode=%s build=%s",
            os.environ.get("SQUEEZE_MODE", "uniform"),
            "squeezeattention-ascend-v1-20260820",
        )
    except Exception as exc:  # pragma: no cover - safety guard
        log_warning(
            "plugin activation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        raise
