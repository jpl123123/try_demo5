"""Regression: hooks must be transparent to torch.compile (dynamo tracing).

If a hook performs tensor ops (or feeds values to Python) inside a traced
region, the compiled graph gains extra outputs and desyncs the npugraph_ex /
AOT artifacts (``ValueError: too many values to unpack (expected 24)``).
While ``torch.compiler.is_compiling()`` is true the hooks must short-circuit
to the original forward without capturing anything.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

sim_env.install_stubs()

from kvpress_ascend.runtime.attention_hooks import AttentionHooks  # noqa: E402


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, positions, query, key, value, kv_cache, attn_metadata):
        self.calls += 1
        return query + 1


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = _FakeAttention()

    def forward(self, hidden_states, positions=None, attn_metadata=None, kv_cache=None):
        out = self.self_attn(positions, hidden_states, hidden_states, hidden_states, kv_cache, attn_metadata)
        return hidden_states + out


class _FakeModel(torch.nn.Module):
    """Container exposing ``.layers`` like a vLLM decoder stack."""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers


def _install_fake_compiler(compiling: bool):
    real_compiler = getattr(torch, "compiler", None)
    torch.compiler = SimpleNamespace(is_compiling=lambda: compiling)
    return real_compiler


def test_kvpress_hooks_skip_capture_while_compiling():
    layer = _FakeLayer(0)
    hooks = AttentionHooks()
    assert hooks.install(_FakeModel(torch.nn.ModuleList([layer]))) == 1

    real_compiler = _install_fake_compiler(True)
    try:
        hidden = torch.randn(4, 8)
        layer(hidden, torch.arange(4), None, None)
    finally:
        if real_compiler is None:
            del torch.compiler
        else:
            torch.compiler = real_compiler

    # No capture while compiling; the original forward still ran.
    assert hooks.get_query(0) is None
    assert hooks.captured_this_step is False
    assert layer.self_attn.calls == 1


def test_kvpress_hooks_capture_when_not_compiling():
    layer = _FakeLayer(0)
    hooks = AttentionHooks()
    hooks.install(_FakeModel(torch.nn.ModuleList([layer])))

    hidden = torch.randn(4, 8)
    layer(hidden, torch.arange(4), None, None)

    assert hooks.get_query(0) is not None
    assert hooks.captured_this_step is True
