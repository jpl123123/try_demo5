"""Plugin activation + patch installation tests (simulated debug)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin  # noqa: E402


def test_disabled_by_default_is_noop(logs):
    sim_env.install_stubs()
    install_kvpress_plugin({})  # no KVPRESS_ENABLE
    text = logs.getvalue()
    assert "disabled (set KVPRESS_ENABLE=1 to activate)" in text
    # Scheduler must remain unpatched.
    import vllm.v1.core.sched.scheduler as sched_mod

    assert "kvpress_config" not in vars(sched_mod.Scheduler)


def test_activated_installs_scheduler_and_worker_patches(logs):
    sim_env.install_stubs()
    install_kvpress_plugin({"KVPRESS_ENABLE": "1", "KVPRESS_PRESS": "KnormPress"})
    text = logs.getvalue()
    assert "plugin activated: KVPRESS_ENABLE=1" in text
    assert "Installed kvpress monkeypatch integration" in text
    assert "Installed kvpress runtime worker patches for Ascend: vllm_ascend.worker.worker.NPUWorker" in text
    assert "Installed kvpress input patches" in text
    assert "kvpress-ascend-v1-20260820" in text

    import vllm.v1.core.sched.scheduler as sched_mod
    import vllm_ascend.worker.worker as ascend_worker_mod

    assert sched_mod.Scheduler.__init__.__name__ == "_patched_scheduler_init"
    assert ascend_worker_mod.NPUWorker.init_device.__name__ == "_patched_ascend_worker_init_device"
    assert ascend_worker_mod.NPUWorker.execute_model.__name__ == "_patched_ascend_worker_execute_model"


def test_alias_env_kvpress_activates(logs):
    sim_env.install_stubs()
    install_kvpress_plugin({"KVPRESS": "1"})
    assert "plugin activated: KVPRESS_ENABLE=1" in logs.getvalue()


def test_unsupported_press_fails_fast():
    sim_env.install_stubs()
    import pytest

    with pytest.raises(ValueError, match="mask-based press"):
        install_kvpress_plugin({"KVPRESS_ENABLE": "1", "KVPRESS_PRESS": "AdaKVPress"})


def test_unknown_press_fails_fast():
    sim_env.install_stubs()
    import pytest

    with pytest.raises(ValueError, match="unsupported"):
        install_kvpress_plugin({"KVPRESS_ENABLE": "1", "KVPRESS_PRESS": "NoSuchPress"})


def test_scheduler_init_attaches_config_and_tracker(logs):
    sim_env.install_stubs()
    install_kvpress_plugin({"KVPRESS_ENABLE": "1"})

    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler()
    assert scheduler.kvpress_config is not None
    assert scheduler.kvpress_config.press_name == "KnormPress"
    assert scheduler._kvpress_effective_len_tracker is not None
    assert "Scheduler initialized" in logs.getvalue()
