"""End-to-end simulated run for SqueezeAttention-ascend.

Flow under test (the SqueezeAttention mechanism, converted):
prefill -> per-layer cosine-similarity capture (hidd_data) -> KMeans layer-wise
budgets ([CLUSTER] log) -> streaming recency compaction at the shared keep
count K -> block reclaim + effective input overrides -> per-step probe logs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_squeeze_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402


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
    sim = SimLoop(block_size=8, num_blocks=512)
    return sim, logs


def test_prefill_clusters_then_compresses(loop):
    sim, logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=4)

    text = logs.getvalue()
    # Per-step core-entry probes.
    assert "[SQUEEZE-ASCEND][PROBE] step=" in text
    assert "core_entered=1" in text
    assert "hook_entered=1" in text
    assert "mode=uniform" in text
    # Layer importance captured + budgets finalized after prefill.
    assert "[SQUEEZE-ASCEND][CLUSTER] budgets" in text
    assert "budgets_ready=1" in text
    # Compression applied with reclaim.
    assert "COMPRESS req=req-1 mode=uniform" in text
    assert "reclaimed_blocks=" in text

    # Worker row shrunk; scheduler side freed blocks + tracked effective len.
    assert sim.row_blocks("req-1") < 8
    assert sim.freed_blocks()
    assert sim.scheduler_tracker_len("req-1") is not None


def test_budgets_ready_only_after_prefill(loop):
    sim, logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 32, chunk=32)  # half prefill
    text = logs.getvalue()
    assert "budgets_ready=0" in text or "budgets_ready=1" not in text
    assert "[SQUEEZE-ASCEND][CLUSTER]" not in text
    sim.prefill("req-1", 32, chunk=32)  # complete prefill
    assert "[SQUEEZE-ASCEND][CLUSTER]" in logs.getvalue()


def test_class_weighted_mode_runs(logs):
    sim_env.install_stubs()
    install_squeeze_plugin(
        {
            "SQUEEZE_ENABLE": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_START_SIZE": "2",
            "SQUEEZE_MODE": "class_weighted",
            "SQUEEZE_FAKE_KEY_PADDING": "1",
            "SQUEEZE_MIN_RECLAIM_BLOCKS": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=3)
    text = logs.getvalue()
    assert "mode=class_weighted" in text
    assert "COMPRESS req=req-1 mode=class_weighted" in text
    assert sim.scheduler_tracker_len("req-1") is not None


def test_probe_off_suppresses_per_step_lines(logs):
    sim_env.install_stubs()
    install_squeeze_plugin(
        {
            "SQUEEZE_ENABLE": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_PROBE": "0",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", 64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=2)
    assert "[PROBE]" not in logs.getvalue()
    # CLUSTER (budget) log still available.
    assert "[SQUEEZE-ASCEND][CLUSTER]" in logs.getvalue()


def test_uniform_mode_shared_k_is_max_budget(loop):
    """In uniform mode all layers share K = max per-layer budget, so the
    compressed row capacity matches the budgeted tokens."""
    sim, _logs = loop
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=3)
    state = sim.worker.model_runner.state_store.get("req-1")
    assert state is not None
    budgets = state.extra.get("sliding_windows")
    assert budgets and len(budgets) == 4  # one per layer
    keep = sim.scheduler_tracker_len("req-1")
    # Effective history length stays near the max budget.
    assert keep is not None and keep <= max(budgets) + 4
