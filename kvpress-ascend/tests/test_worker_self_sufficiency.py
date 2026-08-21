"""TriAttention-philosophy regression: the worker compresses on its own.

The user's real-machine log showed ``seq_len=0`` + no eviction even though the
probe fired. TriAttention's scheduling philosophy is that the WORKER derives
the length itself (block-table capacity) and self-triggers, so compression must
keep working even when:

- the engine-core scheduler sends no signals (unpatched/lagging engine core),
- ``scheduled_new_reqs`` registration fails or is absent,
- attention hooks find 0 layers (compiled prefill).

These tests drive the full loop with all of those channels stripped and assert
that compression still fires and blocks are still reclaimed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402


def _strip_scheduler_channels(scheduler_output):
    """Worst case: no signals, no new-req entries (only num_scheduled_tokens)."""
    scheduler_output.kvpress_signals = None
    scheduler_output.squeeze_signals = None
    scheduler_output.scheduled_new_reqs = []
    return scheduler_output


def test_worker_self_triggers_without_scheduler_signals(logs):
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "KnormPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
            "KVPRESS_DEFER_PREFILL_COMPRESSION": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    freed_before = len(sim.freed_blocks())

    # Strip all scheduler-side channels before the worker sees the output.
    for step in range(6):
        sim.scheduler._sim_num_tokens = {"req-1": 1}
        scheduler_output = sim.scheduler.schedule()
        _strip_scheduler_channels(scheduler_output)
        sim.scheduler._sim_num_tokens = None
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            req = sim.scheduler.requests[rid]
            req.num_computed_tokens += n
            needed = (req.num_computed_tokens + n + sim.scheduler.block_size - 1) // sim.scheduler.block_size
            sim._grow_request_blocks(req, needed)
        model_runner_output = sim.worker.execute_model(scheduler_output)
        sim.scheduler.update_from_output(scheduler_output, model_runner_output)

    text = logs.getvalue()
    assert "worker self-trigger req=req-1" in text or "worker_length_threshold" in text
    assert "COMPRESS req=req-1" in text or "compression applied req=req-1" in text
    assert len(sim.freed_blocks()) > freed_before
    # Effective length tracked even though the scheduler never signalled.
    assert sim.scheduler_tracker_len("req-1") is not None


def test_worker_self_triggers_without_registration(logs):
    """Even with no new-req registration, the lazy state backfill + block
    capacity self-trigger must compress."""
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "StreamingLLMPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_SINK_TOKENS": "4",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
            "KVPRESS_DEFER_PREFILL_COMPRESSION": "1",
            "KVPRESS_LOG_DECISIONS": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    freed_before = 0

    # Strip channels from the VERY FIRST step: no registration ever happens.
    for step, n in enumerate([64, 1, 1, 1, 1, 1, 1]):
        sim.scheduler._sim_num_tokens = {"req-1": n}
        scheduler_output = sim.scheduler.schedule()
        _strip_scheduler_channels(scheduler_output)
        sim.scheduler._sim_num_tokens = None
        for rid, num in scheduler_output.num_scheduled_tokens.items():
            req = sim.scheduler.requests[rid]
            req.num_computed_tokens += num
            needed = (req.num_computed_tokens + num + sim.scheduler.block_size - 1) // sim.scheduler.block_size
            sim._grow_request_blocks(req, needed)
        model_runner_output = sim.worker.execute_model(scheduler_output)
        sim.scheduler.update_from_output(scheduler_output, model_runner_output)

    assert "backfilled runtime state for scheduled request" in logs.getvalue()
    assert len(sim.freed_blocks()) > freed_before
    assert sim.scheduler_tracker_len("req-1") is not None


def test_combo_self_triggers_without_scheduler_signals(logs):
    """Combo mode compresses from the worker side alone too."""
    from conftest import install_squeeze_plugin

    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "KnormPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
            "KVPRESS_COMBO": "1",
        }
    )
    install_squeeze_plugin(
        {
            "SQUEEZE_ENABLE": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_START_SIZE": "2",
            "SQUEEZE_KMEANS_SEED": "42",
        }
    )
    from kvpress_ascend.runtime.combo import KVPressSqueezeComboRunner

    sim = SimLoop(block_size=8, num_blocks=512)
    assert isinstance(sim.worker.model_runner, KVPressSqueezeComboRunner)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    freed_before = len(sim.freed_blocks())

    for step in range(6):
        sim.scheduler._sim_num_tokens = {"req-1": 1}
        scheduler_output = sim.scheduler.schedule()
        _strip_scheduler_channels(scheduler_output)
        sim.scheduler._sim_num_tokens = None
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            req = sim.scheduler.requests[rid]
            req.num_computed_tokens += n
            needed = (req.num_computed_tokens + n + sim.scheduler.block_size - 1) // sim.scheduler.block_size
            sim._grow_request_blocks(req, needed)
        model_runner_output = sim.worker.execute_model(scheduler_output)
        sim.scheduler.update_from_output(scheduler_output, model_runner_output)

    text = logs.getvalue()
    assert "COMPRESS req=req-1 mode=combo" in text
    assert len(sim.freed_blocks()) > freed_before
    assert sim.scheduler_tracker_len("req-1") is not None
