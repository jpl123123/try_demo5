"""Shared simulated-debug fixtures for kvpress-ascend tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

# Install the vllm / vllm_ascend stubs BEFORE any kvpress_ascend import so the
# runtime modules bind the stub logger instead of the fallback logger.
sim_env.install_stubs()


@pytest.fixture(autouse=True)
def clean_env():
    sim_env.clear_all_env()
    sim_env.reset_patch_state()
    yield
    sim_env.clear_all_env()


@pytest.fixture()
def logs():
    sim_env.install_stubs()
    return sim_env.capture_logger()


def install_squeeze_plugin(env: dict[str, str] | None = None) -> None:
    """Install stubs and activate the SqueezeAttention plugin with the given env."""
    sim_env.install_stubs()
    for key, value in (env or {}).items():
        os.environ[key] = value
    from squeezeattention_ascend.plugin import register_squeezeattention_backend

    register_squeezeattention_backend()


@pytest.fixture()
def loop(logs):
    sim_env.install_stubs()
    install_squeeze_plugin(
        {
            "SQUEEZE_ENABLE": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_START_SIZE": "2",
            "SQUEEZE_MIN_RECLAIM_BLOCKS": "1",
            "SQUEEZE_KMEANS_SEED": "42",
            "SQUEEZE_RUNTIME_LOGGING": "1",
            "SQUEEZE_PROBE": "1",
        }
    )
    from sim_loop import SimLoop

    sim = SimLoop(block_size=8, num_blocks=512)
    return sim, logs
