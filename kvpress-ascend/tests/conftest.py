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

# Coexistence tests also import the sibling SqueezeAttention-ascend package.
_SIBLING = Path(__file__).parent.parent.parent / "SqueezeAttention-ascend"
if _SIBLING.exists() and str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))


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


def install_kvpress_plugin(env: dict[str, str] | None = None) -> None:
    """Install stubs and activate the kvpress plugin with the given env."""
    sim_env.install_stubs()
    for key, value in (env or {}).items():
        os.environ[key] = value
    from kvpress_ascend.plugin import register_kvpress_backend

    register_kvpress_backend()


def install_squeeze_plugin(env: dict[str, str] | None = None) -> None:
    """Install stubs and activate the SqueezeAttention plugin with the given env."""
    sim_env.install_stubs()
    for key, value in (env or {}).items():
        os.environ[key] = value
    from squeezeattention_ascend.plugin import register_squeezeattention_backend

    register_squeezeattention_backend()
