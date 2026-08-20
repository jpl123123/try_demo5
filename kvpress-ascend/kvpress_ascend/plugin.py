"""vLLM plugin entry point for kvpress-ascend.

Activation (the user's ``export kvpress`` style):
    export KVPRESS_ENABLE=1      # alias: export KVPRESS=1

With the switch off the plugin is a no-op, so both tools can be installed in
the same environment.
"""

from __future__ import annotations

import os

from .envs import env_bool
from .logging_control import get_logger

logger = get_logger()
from .logging_control import log_info, log_warning


def register_kvpress_backend() -> None:
    """Install kvpress-ascend integration when the plugin is loaded by vLLM."""
    if not (env_bool("KVPRESS_ENABLE", False) or env_bool("KVPRESS", False)):
        log_info("disabled (set KVPRESS_ENABLE=1 to activate)")
        return
    try:
        from .runtime.monkeypatch import install_kvpress_integration_monkeypatches

        install_kvpress_integration_monkeypatches(
            patch_scheduler=True,
            patch_worker=True,
        )
        log_info(
            "plugin activated: KVPRESS_ENABLE=1 press=%s build=%s",
            os.environ.get("KVPRESS_PRESS", "KnormPress"),
            "kvpress-ascend-v1-20260820",
        )
    except Exception as exc:  # pragma: no cover - safety guard
        log_warning(
            "plugin activation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        raise
