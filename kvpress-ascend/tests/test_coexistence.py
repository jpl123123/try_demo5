"""Coexistence test: both plugins installed in the same environment; each
activates independently through its own env switch (export kvpress / export
squeeze), and neither interferes with the other."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin  # noqa: E402
from conftest import install_squeeze_plugin  # noqa: E402


def test_both_installed_kvpress_only_activates(logs):
    sim_env.install_stubs()
    # Install both plugins; only KVPRESS_ENABLE set.
    install_kvpress_plugin({"KVPRESS_ENABLE": "1", "KVPRESS_PRESS": "KnormPress"})
    install_squeeze_plugin({})
    text = logs.getvalue()
    assert "plugin activated: KVPRESS_ENABLE=1" in text
    assert "disabled (set SQUEEZE_ENABLE=1 to activate)" in text
    # Both patch sets installed, but only kvpress runtime state attached.
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler()
    assert scheduler.kvpress_config is not None
    assert not hasattr(scheduler, "squeeze_config")


def test_both_installed_squeeze_only_activates(logs):
    sim_env.install_stubs()
    install_squeeze_plugin({"SQUEEZE_ENABLE": "1"})
    install_kvpress_plugin({})
    text = logs.getvalue()
    assert "plugin activated: SQUEEZE_ENABLE=1" in text
    assert "disabled (set KVPRESS_ENABLE=1 to activate)" in text
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler()
    assert scheduler.squeeze_config is not None
    assert not hasattr(scheduler, "kvpress_config")


def test_both_activated_together(logs):
    sim_env.install_stubs()
    install_kvpress_plugin({"KVPRESS_ENABLE": "1", "KVPRESS_PRESS": "KnormPress"})
    install_squeeze_plugin({"SQUEEZE_ENABLE": "1"})
    text = logs.getvalue()
    assert "plugin activated: KVPRESS_ENABLE=1" in text
    assert "plugin activated: SQUEEZE_ENABLE=1" in text
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler()
    # Both configs attach (the second install is additive; the schedule wrapper
    # serves both signal channels).
    assert scheduler.kvpress_config is not None
    assert scheduler.squeeze_config is not None


def test_both_plugins_run_together_end_to_end(logs):
    """KVPRESS and SQUEEZE active in one engine: both proxies are installed on
    the worker; the scheduler emits both signal channels; the run completes
    with compression events from the active kvpress press."""
    from sim_loop import SimLoop

    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "StreamingLLMPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_SINK_TOKENS": "4",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
        }
    )
    install_squeeze_plugin(
        {
            "SQUEEZE_ENABLE": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_START_SIZE": "2",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=3)
    text = logs.getvalue()
    assert "[KVPRESS-ASCEND][PROBE] step=" in text
    assert "[SQUEEZE-ASCEND][PROBE] step=" in text
    # The worker ends up wrapped by the last-installed proxy; the run must be
    # stable either way (compression + tracker from the active path).
    kp_tracker = getattr(
        sim.scheduler, "_kvpress_effective_len_tracker", None
    )
    sq_tracker = getattr(
        sim.scheduler, "_squeeze_effective_len_tracker", None
    )
    assert (kp_tracker is not None and kp_tracker.get("req-1") is not None) or (
        sq_tracker is not None and sq_tracker.get("req-1") is not None
    )
