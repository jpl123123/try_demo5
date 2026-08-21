"""Simulated-debug environment: install vllm / vllm_ascend stubs and drive a
fake scheduler + NPUWorker loop on CPU (no NPU, no real vLLM on this machine).

Usage:
    import sim_env
    sim_env.install_stubs()          # puts tests/stubs on sys.path
    sim_env.reset()                  # clears patch state + env
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

_STUBS = Path(__file__).parent / "stubs"

# Original stub class attributes, captured once so reset_patch_state can
# restore them in place (reloading the stub modules would recreate the classes
# and orphan references held by imported test modules).
_ORIGINALS: dict[Any, dict[str, Any]] = {}


def install_stubs() -> None:
    if str(_STUBS) not in sys.path:
        sys.path.insert(0, str(_STUBS))


def _capture_originals() -> None:
    from vllm.v1.core import StubScheduler, StubKVCacheManager
    from vllm_ascend.worker.worker import NPUWorker
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    targets = {
        StubScheduler: ("__init__", "schedule", "update_from_output"),
        StubKVCacheManager: ("allocate_slots",),
        NPUWorker: ("init_device", "execute_model"),
        NPUModelRunner: ("_prepare_inputs",),
    }
    for cls, attrs in targets.items():
        _ORIGINALS.setdefault(cls, {})
        for attr in attrs:
            if attr not in _ORIGINALS[cls]:
                _ORIGINALS[cls][attr] = getattr(cls, attr)


def restore_stub_classes() -> None:
    for cls, attrs in _ORIGINALS.items():
        for attr, original in attrs.items():
            try:
                setattr(cls, attr, original)
            except Exception:
                pass


def capture_logger() -> io.StringIO:
    """Attach a StringIO capture handler to the stub vllm logger."""
    from vllm.logger import logger as stub_logger

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    stub_logger.handlers = [handler]
    stub_logger.setLevel(logging.DEBUG)
    stub_logger.propagate = False
    return stream


def reset_patch_state() -> None:
    """Reset patch-install state so each test starts clean."""
    install_stubs()
    _capture_originals()
    restore_stub_classes()
    for name in (
        "kvpress_ascend.runtime.combo",
        "kvpress_ascend.runtime.monkeypatch",
        "kvpress_ascend.runtime.worker_hooks",
        "kvpress_ascend.runtime.input_patch_v1",
        "squeezeattention_ascend.runtime.monkeypatch",
        "squeezeattention_ascend.runtime.worker_hooks",
        "squeezeattention_ascend.runtime.input_patch_v1",
    ):
        try:
            module = importlib.import_module(name)
            importlib.reload(module)
        except Exception:
            pass
    try:
        from kvpress_ascend.runtime import input_patch_state as kps

        kps.reset()
    except Exception:
        pass
    try:
        from squeezeattention_ascend.runtime import input_patch_state as sps

        sps.reset()
    except Exception:
        pass


def clear_env(prefix: str = "KVPRESS") -> None:
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            os.environ.pop(key, None)


def clear_all_env() -> None:
    clear_env("KVPRESS")
    clear_env("SQUEEZE")
