"""Regression: SqueezeAttention hooks must be transparent to torch.compile.

The hooks only store tensor references inside the forward (no tensor ops), and
skip capture entirely while ``torch.compiler.is_compiling()`` is true. The
cosine-similarity math runs in the runner proxy after the forward, outside any
compiled region.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

sim_env.install_stubs()

from squeezeattention_ascend.runtime.attention_hooks import SqueezeAttentionHooks  # noqa: E402


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


def test_squeeze_hooks_skip_capture_while_compiling():
    layer = _FakeLayer(0)
    hooks = SqueezeAttentionHooks()
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

    assert hooks.get_capture_pair(0) is None
    assert hooks.captured_this_step is False
    assert layer.self_attn.calls == 1


def test_squeeze_hooks_store_references_only_when_not_compiling():
    layer = _FakeLayer(0)
    hooks = SqueezeAttentionHooks()
    hooks.install(_FakeModel(torch.nn.ModuleList([layer])))

    hidden = torch.randn(4, 8)
    layer(hidden, torch.arange(4), None, None)

    pair = hooks.get_capture_pair(0)
    assert pair is not None
    layer_input, attn_out = pair
    # References only: the captured tensors are the exact forward tensors
    # (no extra graph ops were inserted), and the cosine similarity has NOT
    # been computed inside the forward.
    assert layer_input is hidden
    assert attn_out is not None
    assert hooks._layer_similarities == {}
    hooks.clear_capture_pair(0)
    assert hooks.get_capture_pair(0) is None
