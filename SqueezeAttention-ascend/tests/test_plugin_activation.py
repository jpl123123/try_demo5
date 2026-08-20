"""Plugin activation + patch installation tests for SqueezeAttention-ascend."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_squeeze_plugin  # noqa: E402


def test_disabled_by_default_is_noop(logs):
    sim_env.install_stubs()
    install_squeeze_plugin({})
    text = logs.getvalue()
    assert "disabled (set SQUEEZE_ENABLE=1 to activate)" in text
    import vllm.v1.core.sched.scheduler as sched_mod

    assert "squeeze_config" not in vars(sched_mod.Scheduler)


def test_activated_installs_patches(logs):
    sim_env.install_stubs()
    install_squeeze_plugin({"SQUEEZE_ENABLE": "1", "SQUEEZE_MODE": "uniform"})
    text = logs.getvalue()
    assert "plugin activated: SQUEEZE_ENABLE=1" in text
    assert "Installed SqueezeAttention monkeypatch integration" in text
    assert "Installed SqueezeAttention runtime worker patches for Ascend" in text
    assert "Installed SqueezeAttention input patches" in text
    assert "squeezeattention-ascend-v1-20260820" in text

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm_ascend.worker.worker as ascend_worker_mod

    assert sched_mod.Scheduler.__init__.__name__ == "_patched_scheduler_init"
    assert ascend_worker_mod.NPUWorker.init_device.__name__ == "_patched_ascend_worker_init_device"


def test_alias_env_squeeze_activates(logs):
    sim_env.install_stubs()
    install_squeeze_plugin({"SQUEEZE": "1"})
    assert "plugin activated: SQUEEZE_ENABLE=1" in logs.getvalue()


def test_invalid_mode_fails_fast():
    sim_env.install_stubs()
    import pytest

    with pytest.raises(ValueError, match="SQUEEZE_MODE"):
        install_squeeze_plugin({"SQUEEZE_ENABLE": "1", "SQUEEZE_MODE": "bogus"})


def test_scheduler_init_attaches_config(logs):
    sim_env.install_stubs()
    install_squeeze_plugin({"SQUEEZE_ENABLE": "1"})
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler()
    assert scheduler.squeeze_config is not None
    assert scheduler.squeeze_config.mode == "uniform"
    assert scheduler._squeeze_effective_len_tracker is not None
    assert "Scheduler initialized" in logs.getvalue()
