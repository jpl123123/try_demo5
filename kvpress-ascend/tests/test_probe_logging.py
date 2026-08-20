"""Probe / logging switch tests (the per-inference verification switch)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from conftest import install_kvpress_plugin  # noqa: E402
from sim_loop import SimLoop  # noqa: E402


def _run(env: dict) -> str:
    sim_env.install_stubs()
    install_kvpress_plugin(env)
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    sim.decode("req-1", steps=3)
    return sim


def test_probe_off_suppresses_per_step_lines(logs):
    env = {
        "KVPRESS_ENABLE": "1",
        "KVPRESS_PRESS": "StreamingLLMPress",
        "KVPRESS_COMPRESSION_RATIO": "0.5",
        "KVPRESS_PROBE": "0",
    }
    sim = _run(env)
    text = logs.getvalue()
    assert "[PROBE]" not in text
    # Startup/install logs still present.
    assert "plugin activated" in text
    assert sim.scheduler_tracker_len("req-1") is not None


def test_master_logging_off_suppresses_all_but_probe_also_off(logs):
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "KnormPress",
            "KVPRESS_RUNTIME_LOGGING": "0",
            "KVPRESS_PROBE": "1",
        }
    )
    text = logs.getvalue()
    assert "[KVPRESS-ASCEND]" not in text


def test_probe_lines_carry_core_parameters(logs):
    env = {
        "KVPRESS_ENABLE": "1",
        "KVPRESS_PRESS": "StreamingLLMPress",
        "KVPRESS_COMPRESSION_RATIO": "0.5",
        "KVPRESS_SINK_TOKENS": "4",
        "KVPRESS_PROBE": "1",
    }
    _run(env)
    text = logs.getvalue()
    probe_lines = [l for l in text.splitlines() if "[PROBE]" in l]
    assert probe_lines, "no probe lines"
    first = probe_lines[0]
    for token in (
        "core_entered=1",
        "hook_entered=1",
        "press=StreamingLLMPress",
        "ratio=0.500",
        "seq_len=",
        "budget=",
        "keep=",
        "last_event=",
    ):
        assert token in first, f"missing {token!r} in {first}"
    # A COMPRESS probe with core parameters exists too.
    compress_lines = [l for l in text.splitlines() if "COMPRESS req=req-1" in l]
    assert compress_lines


def test_hook_log_gated_by_env(logs):
    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "KnormPress",
            "KVPRESS_LOG_ATTENTION_HOOK": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=512)
    sim.add_request("r", 8)
    sim.prefill("r", 8, chunk=8)
    text = logs.getvalue()
    assert "attention capture" in text or "[HOOK]" in text
