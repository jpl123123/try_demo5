"""MTP spec-decode regression: scheduled_tokens > 1 must NOT be treated as
prefill when the step is validating draft tokens (qwen3_5_mtp, 3 spec tokens).

This was the exact trap that kept the worker from ever compressing on the
user's real deployment: MTP decode schedules 1 target + N draft tokens, so
``num_scheduled_tokens[req]`` is 4, and the naive ``scheduled_tokens > 1 =>
prefill`` gate skipped every decode step forever (seq_len=0, no eviction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from kvpress_ascend.runtime.request_key_compat import (  # noqa: E402
    iter_scheduled_token_items,
    req_id_from_scheduled_key,
)
from kvpress_ascend.runtime.thresholds import (  # noqa: E402
    is_prefill_phase_for_limit,
    is_request_scheduled_as_spec_decode,
)


def _make_output(num_scheduled, spec_tokens=None, new_reqs=None):
    return type(
        "SO",
        (),
        {
            "num_scheduled_tokens": num_scheduled,
            "scheduled_spec_decode_tokens": spec_tokens or {},
            "scheduled_new_reqs": new_reqs or [],
        },
    )()


def test_spec_decode_step_is_not_prefill():
    so = _make_output(
        {"req-1": 4},
        spec_tokens={"req-1": [11, 22, 33]},
    )
    assert is_request_scheduled_as_spec_decode(so, "req-1") is True
    assert (
        is_prefill_phase_for_limit(
            scheduler_output=so,
            req_id="req-1",
            scheduled_tokens=4,
            prefill_len=32768,
            num_computed_tokens=33000,
        )
        is False
    ), "MTP decode (4 tokens incl. drafts) must not be classified as prefill"


def test_chunked_prefill_still_prefill():
    so = _make_output({"req-1": 4096})
    assert (
        is_prefill_phase_for_limit(
            scheduler_output=so,
            req_id="req-1",
            scheduled_tokens=4096,
            prefill_len=32768,
            num_computed_tokens=8192,
        )
        is True
    )


def test_new_req_membership_wins():
    request = type("R", (), {"req_id": "req-1", "num_prompt_tokens": 32768})()
    so = _make_output({"req-1": 4}, spec_tokens={"req-1": [1, 2, 3]}, new_reqs=[request])
    assert (
        is_prefill_phase_for_limit(
            scheduler_output=so,
            req_id="req-1",
            scheduled_tokens=4,
            prefill_len=32768,
            num_computed_tokens=4000,
        )
        is True
    ), "request still in scheduled_new_reqs => prefill"


def test_plain_decode_with_known_prompt_is_not_prefill():
    so = _make_output({"req-1": 1})
    assert (
        is_prefill_phase_for_limit(
            scheduler_output=so,
            req_id="req-1",
            scheduled_tokens=1,
            prefill_len=32768,
            num_computed_tokens=32769,
        )
        is False
    )


def test_object_keys_normalize_to_req_id():
    """Real vLLM may key num_scheduled_tokens by request objects."""
    request = type("R", (), {"req_id": "req-1", "num_prompt_tokens": 64})()
    assert req_id_from_scheduled_key("req-1") == "req-1"
    assert req_id_from_scheduled_key(request) == "req-1"
    items = list(iter_scheduled_token_items(_make_output({request: 4})))
    assert items == [("req-1", 4)]


def test_mtp_decode_loop_compresses(logs):
    """End-to-end: MTP-style decode steps (4 scheduled tokens + spec dict)
    still trigger compression after prefill."""
    from conftest import install_kvpress_plugin
    from sim_loop import SimLoop

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

    from vllm.v1.core import StubScheduler

    original_schedule = StubScheduler.schedule

    def _mtp_schedule(self):
        out = original_schedule(self)
        # MTP decode: 1 target + 3 draft tokens per request per step.
        out.num_scheduled_tokens = {rid: 4 for rid in out.num_scheduled_tokens}
        out.total_num_scheduled_tokens = 4 * len(out.num_scheduled_tokens)
        out.scheduled_spec_decode_tokens = {
            rid: [1, 2, 3] for rid in out.num_scheduled_tokens
        }
        return out

    StubScheduler.schedule = _mtp_schedule
    try:
        sim.decode("req-1", steps=8)
    finally:
        StubScheduler.schedule = original_schedule

    assert len(sim.freed_blocks()) > freed_before, "MTP decode must compress"
    assert sim.scheduler_tracker_len("req-1") is not None
