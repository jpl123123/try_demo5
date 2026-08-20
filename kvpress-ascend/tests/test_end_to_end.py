"""End-to-end simulated run: plugin + scheduler + worker + compaction + input
patch + probe logs, driven on CPU (simulated debug, no NPU)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402


@pytest.fixture()
def loop(logs):
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "StreamingLLMPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_SINK_TOKENS": "4",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
            "KVPRESS_DEFER_PREFILL_COMPRESSION": "1",
            "KVPRESS_RUNTIME_LOGGING": "1",
            "KVPRESS_PROBE": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    return sim, logs


def test_prefill_then_decode_compresses_and_reclaims(loop):
    sim, logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=32)  # two prefill chunks
    sim.decode("req-1", steps=4)

    text = logs.getvalue()
    # Probe lines must show the patch entering its core on every step.
    assert "[KVPRESS-ASCEND][PROBE] step=" in text
    assert "core_entered=1" in text
    assert "hook_entered=1" in text
    assert "press=StreamingLLMPress" in text

    # Compression event applied after prefill (deferred) with reclaim.
    assert "COMPRESS req=req-1 press=StreamingLLMPress" in text
    assert "reclaimed_blocks=" in text

    # Scheduler side: effective length tracked and blocks freed.
    assert sim.scheduler_tracker_len("req-1") is not None
    assert sim.freed_blocks(), "no blocks were reclaimed on the scheduler side"

    # Worker side: block row shrunk below the prefill row.
    assert sim.row_blocks("req-1") < 8


def test_compacted_cache_content_is_correct(loop):
    """After compression the row prefix must hold exactly the kept tokens."""
    sim, _logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    before_blocks = sim.row_blocks("req-1")

    # Snapshot the row values written by the (deterministic) stub attention
    # before compression happens on the first decode step.
    k_cache = sim.kv_tensor(0)[0]
    block_size = sim.worker.model_runner.block_size
    row = sim.row_block_ids("req-1")
    values_before = [
        float(k_cache[row[t // block_size], t % block_size, 0, 0].item())
        for t in range(len(row) * block_size)
    ]

    sim.decode("req-1", steps=2)

    assert sim.freed_blocks()
    row = sim.row_block_ids("req-1")
    assert len(row) < before_blocks

    prefix_values = []
    for i in range(len(row) * block_size):
        b, s = i // block_size, i % block_size
        prefix_values.append(float(k_cache[row[b], s, 0, 0].item()))
    # Kept tokens: sink 0..3 + recent 36..63, in original order (sorted
    # per-head keep indices -> ordered prefix).
    kept = list(range(4)) + list(range(36, 64))
    assert prefix_values[:32] == [values_before[t] for t in kept], prefix_values[:32]


def test_seq_lens_and_positions_patched_after_compression(loop):
    sim, _logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=2)

    runner = sim.worker.model_runner
    # vLLM's bookkeeping (rewritten to the effective view by the scheduler
    # sync) stays near the compressed length...
    req = sim.scheduler.requests["req-1"]
    assert req.num_computed_tokens <= 34
    # ...and the effective seq_lens buffer used by attention metadata must
    # reflect the compressed history (32 kept + 2 decoded).
    effective = sim.scheduler_tracker_len("req-1")
    assert effective is not None and effective <= 34
    seq_lens_np = runner.seq_lens.np
    # seq_lens = effective history (32 + 2 decoded) + this step's token.
    assert int(seq_lens_np[0]) <= 35
    # Positions buffer (active slice) must be within the compacted row capacity.
    row_capacity = sim.row_blocks("req-1") * runner.block_size
    active_slots = int(runner.block_table.num_slots)
    active_positions = runner._positions_np_buf[:active_slots]
    assert int(active_positions.max()) < row_capacity


def test_second_compression_cycle_keeps_working(loop):
    sim, logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    # Decode long enough to cross the budget again (budget 32 + min 1 block).
    sim.decode("req-1", steps=40)
    text = logs.getvalue()
    compress_count = text.count("status=applied") + text.count("COMPRESS req=req-1")
    assert compress_count >= 2, f"expected >=2 compressions, saw {compress_count}"
    # Effective length stays bounded near the budget.
    effective = sim.scheduler_tracker_len("req-1")
    assert effective is not None and effective <= 40


def test_no_prefix_caching_required(loop):
    """With prefix caching disabled the flow must not touch hash logic at all."""
    sim, logs = loop
    sim.add_request("req-1", num_prompt_tokens=48)
    sim.prefill("req-1", 48, chunk=48)
    sim.decode("req-1", steps=2)
    assert "prefix" not in logs.getvalue().lower() or "no-prefix" in logs.getvalue().lower()
