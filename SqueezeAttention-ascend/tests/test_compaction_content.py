"""SqueezeAttention compaction content correctness (simulated debug)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_squeeze_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402


def test_uniform_mode_compaction_content(loop):
    """After compression every layer's row prefix holds the recency keep set
    [0, start_size) ∪ last(K - start_size) in original order."""
    sim, _logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)

    # Snapshot layer-0 row values written by the stub attention (per-token).
    runner = sim.worker.model_runner
    k_cache = runner.kv_caches[0][0]
    block_size = runner.block_size
    row = sim.row_block_ids("req-1")
    values_before = [
        float(k_cache[row[t // block_size], t % block_size, 0, 0].item())
        for t in range(len(row) * block_size)
    ]
    before_blocks = len(row)

    sim.decode("req-1", steps=2)
    assert sim.freed_blocks()
    row = sim.row_block_ids("req-1")
    assert len(row) < before_blocks

    state = sim.worker.model_runner.state_store.get("req-1")
    budgets = state.extra["sliding_windows"]
    start = state.extra.get("budget_diagnostics", {}).get("start_size", 2) or 2
    start = 2  # SQUEEZE_START_SIZE in the fixture
    k = max(budgets)
    kept = list(range(start)) + list(range(64 - (k - start), 64))

    prefix_values = []
    for i in range(len(row) * block_size):
        b, s = i // block_size, i % block_size
        prefix_values.append(float(k_cache[row[b], s, 0, 0].item()))
    assert prefix_values[:k] == [values_before[t] for t in kept], (
        prefix_values[:k],
        [values_before[t] for t in kept],
    )


def test_all_layers_compacted_identically(loop):
    """Uniform mode: every layer's cache row prefix matches (same keep set)."""
    sim, _logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=2)

    runner = sim.worker.model_runner
    row = sim.row_block_ids("req-1")
    block_size = runner.block_size
    layer0 = [
        float(runner.kv_caches[0][0][row[i // block_size], i % block_size, 0, 0].item())
        for i in range(32)
    ]
    for layer in range(1, runner.model.num_layers):
        layer_i = [
            float(runner.kv_caches[layer][0][row[i // block_size], i % block_size, 0, 0].item())
            for i in range(32)
        ]
        # Values differ per layer but the compacted prefix is aligned: the
        # stub attention writes the same token value for every layer at
        # position (token) — so prefixes must match exactly here.
        assert layer_i == layer0
