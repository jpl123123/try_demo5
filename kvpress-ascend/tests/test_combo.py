"""Combo mode tests: layer budgets x token press, ONE eviction per boundary.

Covers the user's design question: with both tools enabled, the combo runner
performs a single physical compaction per compression boundary (single event,
single reclaim, single probe line), instead of the naive double-proxy setup
that would evict twice and double-free blocks.
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin, install_squeeze_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402

# Real vLLM V1 shape for scheduled_new_reqs entries.
NewScheduledRequest = namedtuple(
    "NewScheduledRequest",
    ["req_id", "request", "num_computed_tokens", "num_prompt_tokens", "num_scheduled_tokens"],
)


def _install_combo(logs):
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "StreamingLLMPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_SINK_TOKENS": "4",
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
    return sim_env, logs


def test_combo_installs_single_proxy_and_skips_standalone(logs):
    _install_combo(logs)
    text = logs.getvalue()
    assert "COMBO=1" in text
    assert "combo mode active (KVPRESS_COMBO=1): SqueezeAttention standalone install skipped" in text
    assert "Installed combo worker patches for Ascend" in text

    sim = SimLoop(block_size=8, num_blocks=512)
    from kvpress_ascend.runtime.combo import KVPressSqueezeComboRunner

    assert isinstance(sim.worker.model_runner, KVPressSqueezeComboRunner)
    # Single proxy: unwrapping ends at the raw runner immediately.
    assert sim.worker.model_runner._base_runner.__class__.__name__ == "NPUModelRunner"


def test_combo_one_eviction_per_boundary(logs):
    _install_combo(logs)
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    reclaim_batches_before = len(sim.manager.block_pool.freed)
    sim.decode("req-1", steps=8)
    text = logs.getvalue()

    # Probe lines use the combo marker with both dimensions.
    probe_lines = [l for l in text.splitlines() if "[PROBE]" in l and "mode=combo" in l]
    assert probe_lines
    assert "core_entered=1" in probe_lines[0]
    assert "press=StreamingLLMPress" in probe_lines[0]
    assert "budgets_ready=1" in text

    # One applied compression event for the first boundary (single eviction).
    applied = [l for l in text.splitlines() if "COMPRESS req=req-1 mode=combo" in l]
    assert applied, "no combo compression applied"
    assert len(applied) == 1, f"expected ONE eviction at the boundary, saw {len(applied)}"

    # One reclaim pass on the scheduler side, blocks freed exactly once.
    batches = sim.manager.block_pool.freed[reclaim_batches_before:]
    flat = [b for batch in batches for b in batch]
    assert flat, "no blocks reclaimed"
    assert len(flat) == len(set(flat)), "blocks freed more than once (double reclaim)"

    # Effective length tracked (scheduler side applied the single event).
    assert sim.scheduler_tracker_len("req-1") is not None


def test_combo_class_weighted_with_fake_keys(logs):
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
            "SQUEEZE_MODE": "class_weighted",
            "SQUEEZE_FAKE_KEY_PADDING": "1",
            "SQUEEZE_INI_SIZE": "0.5",
            "SQUEEZE_CLASS3_SIZE": "0.25",
            "SQUEEZE_START_SIZE": "2",
            "SQUEEZE_KMEANS_SEED": "42",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=3)
    text = logs.getvalue()
    assert "mode=combo" in text
    assert sim.scheduler_tracker_len("req-1") is not None


def test_namedtuple_registration_parsing():
    """Real vLLM NewScheduledRequest namedtuples must register correctly."""
    from kvpress_ascend.runtime.request_key_compat import iter_scheduled_new_requests

    request = type("R", (), {"req_id": "req-1", "num_prompt_tokens": 123})()
    item = NewScheduledRequest("req-1", request, 0, 123, 1)
    so = type("SO", (), {"scheduled_new_reqs": [item]})()
    parsed = list(iter_scheduled_new_requests(so))
    assert len(parsed) == 1
    req_id, req, num_prompt = parsed[0]
    assert req_id == "req-1"
    assert req is request
    assert num_prompt == 123

    # Plain (None, request) tuples from the stub scheduler still work.
    so2 = type("SO", (), {"scheduled_new_reqs": [(None, request)]})()
    parsed2 = list(iter_scheduled_new_requests(so2))
    assert parsed2[0][0] == "req-1"
    assert parsed2[0][2] == 123


def test_registration_via_namedtuple_end_to_end(logs):
    """The full loop registers requests from real-VLLM-shaped new-req entries."""
    _install_combo(logs)
    sim = SimLoop(block_size=8, num_blocks=512)

    # Bypass SimLoop.add_request to inject a namedtuple-style new-req entry.
    from vllm.v1.core import StubRequest

    req = StubRequest(req_id="req-1", num_prompt_tokens=64)
    req.block_ids = []
    sim.scheduler.add_request(req)

    # Patch the stub scheduler to emit namedtuple entries.
    from vllm.v1.core import StubScheduler

    original_schedule = StubScheduler.schedule

    def _schedule_with_namedtuples(self):
        out = original_schedule(self)
        out.scheduled_new_reqs = [
            NewScheduledRequest(r.req_id, r, 0, r.num_prompt_tokens, 1)
            for r in self.requests.values()
        ]
        return out

    StubScheduler.schedule = _schedule_with_namedtuples
    try:
        sim.prefill("req-1", 64, chunk=64)
        sim.decode("req-1", steps=2)
    finally:
        StubScheduler.schedule = original_schedule

    text = logs.getvalue()
    assert "combo registered request req=req-1 prefill_len=64" in text
    assert "budgets_ready=1" in text
    assert sim.scheduler_tracker_len("req-1") is not None
