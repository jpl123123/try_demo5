"""KV layout + compression engine unit tests (simulated debug on CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

sim_env.install_stubs()

from kvpress_ascend.core.kv_layout import (  # noqa: E402
    compact_request_kv_in_place_per_head,
    gather_request_k_dense,
    gather_request_kv_dense,
    split_kv_axes,
)
from kvpress_ascend.core.press_bridge import build_press, select_keep_indices  # noqa: E402
from kvpress_ascend.runtime.compression_engine import compress_request  # noqa: E402
from kvpress_ascend.runtime.attention_hooks import AttentionHooks  # noqa: E402


def _make_cache(num_blocks=16, block_size=8, heads=4, dim=4, layer=0):
    """Cache with identifiable values: token t of head h has K value
    ``layer * 1000 + h * 100 + t`` at dim 0."""
    k = torch.zeros(num_blocks, block_size, heads, dim)
    v = torch.zeros(num_blocks, block_size, heads, dim)
    for b in range(num_blocks):
        for s in range(block_size):
            t = b * block_size + s
            for h in range(heads):
                k[b, s, h, 0] = layer * 1000 + h * 100 + t
                v[b, s, h, 1] = layer * 1000 + h * 100 + t
    return (k, v)


def _write_request_row(cache, block_ids, total_tokens):
    k, v = cache
    for t in range(total_tokens):
        b = block_ids[t // k.shape[1]]
        s = t % k.shape[1]
        for h in range(k.shape[2]):
            k[b, s, h, 0] = 5000 + h * 100 + t
            v[b, s, h, 1] = 5000 + h * 100 + t


def test_split_kv_axes_supported_layouts():
    k = torch.zeros(4, 8, 2, 4)
    v = torch.zeros(4, 8, 2, 4)
    kk, vv = split_kv_axes((k, v))
    assert kk is k and vv is v

    combined0 = torch.zeros(2, 4, 8, 2, 4)
    kk, vv = split_kv_axes(combined0)
    assert kk.shape == (4, 8, 2, 4)

    combined1 = torch.zeros(4, 2, 8, 2, 4)
    kk, vv = split_kv_axes(combined1)
    assert kk.shape == (4, 8, 2, 4)


def test_gather_and_per_head_compact_content():
    cache = _make_cache()
    block_ids = [2, 5, 9]  # non-consecutive -> gather path
    block_size = 8
    total = 24
    _write_request_row(cache, block_ids, total)

    keys, values = gather_request_kv_dense(cache, block_ids, block_size, total)
    assert keys.shape == (1, 4, total, 4)
    # Token 10, head 2 -> 5000 + 200 + 10
    assert keys[0, 2, 10, 0].item() == 5210.0
    assert values[0, 2, 10, 1].item() == 5210.0

    # Keep tokens [2, 5, 8, 11] for every head (sorted per-head sets).
    keep = torch.tensor([[2, 5, 8, 11]] * 4)
    kept = compact_request_kv_in_place_per_head(
        cache, block_ids, block_size, keep, total, prefix_only=True
    )
    assert kept == 4
    # Row prefix slots now hold the kept tokens, in order.
    for i, t in enumerate([2, 5, 8, 11]):
        b = block_ids[i // block_size]
        s = i % block_size
        assert cache[0][b, s, 2, 0].item() == 5000 + 200 + t


def test_press_selection_math():
    torch.manual_seed(0)
    keys = torch.randn(1, 4, 64, 8)
    # Knorm: score = -||k||; topk keeps the smallest-norm keys (kvpress
    # ScorerPress semantics).
    press = build_press("KnormPress", 0.5)
    keep = select_keep_indices(press, keys, 32)
    assert keep.shape == (4, 32)
    norms = keys[0].norm(dim=-1)
    top = norms.topk(32, largest=False).indices
    assert torch.equal(keep[0], top[0].sort().values)

    # StreamingLLM: sink + recent.
    press_s = build_press("StreamingLLMPress", 0.5, n_sink=4)
    keep_s = select_keep_indices(press_s, keys, 32)
    expected = list(range(4)) + list(range(36, 64))
    assert keep_s[0].tolist() == expected

    # Random with seed is deterministic.
    press_r = build_press("RandomPress", 0.5, seed=7)
    keep_r1 = select_keep_indices(press_r, keys, 32)
    keep_r2 = select_keep_indices(press_r, keys, 32)
    assert torch.equal(keep_r1, keep_r2)


def test_snapkv_window_attention_scoring():
    torch.manual_seed(1)
    keys = torch.randn(1, 4, 64, 8)
    queries = torch.randn(1, 8, 16, 8)  # 16 query heads -> 4 kv heads
    press = build_press("SnapKVPress", 0.5, window_size=16)
    keep = select_keep_indices(press, keys, 32, queries=queries)
    assert keep.shape == (4, 32)
    # Window tokens (last 16) must be kept (padded with max score).
    assert set(range(48, 64)).issubset(set(keep[0].tolist()))


def test_compress_request_applies_and_reports_reclaim():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner(num_layers=4, block_size=8)
    req_id = "req-1"
    runner.requests[req_id] = type(
        "R", (), {"req_id": req_id, "num_computed_tokens": 64}
    )()
    # Give the request a row of 8 blocks (64 tokens).
    block_ids = list(range(8))
    runner.input_batch = type(
        "B",
        (),
        {
            "req_id_to_index": {req_id: 0},
            "num_computed_tokens_cpu": __import__("numpy").zeros(16, dtype="int64"),
            "block_table": type(
                "T",
                (),
                {
                    "block_tables": None,
                    "num_blocks_per_row": __import__("numpy").zeros(16, dtype="int64"),
                },
            )(),
        },
    )()
    runner.input_batch.block_table.num_blocks_per_row[0] = 8
    runner.input_batch.block_table.block_table = type(
        "BT", (), {"np": __import__("numpy").zeros((16, 64), dtype="int64")}
    )()
    runner.input_batch.block_table.block_table.np[0, :8] = block_ids

    # Write identifiable values into the row (all 4 layers).
    for layer in range(4):
        k, v = runner.kv_caches[layer]
        for t in range(64):
            b, s = t // 8, t % 8
            k[b, s, :, 0] = layer * 1000 + t
            v[b, s, :, 1] = layer * 1000 + t

    press = build_press("StreamingLLMPress", 0.5, n_sink=4)
    hooks = AttentionHooks()
    event = compress_request(
        base_runner=runner,
        req_id=req_id,
        keep_count=32,
        total_tokens=64,
        block_size=8,
        press=press,
        hooks=hooks,
        scheduled_tokens=1,
    )
    assert event["status"] == "applied"
    assert event["cache_len_after"] == 32
    retained = event["retained_cache_len"]
    required_blocks = (retained + 8 - 1) // 8
    assert required_blocks == 5
    # Kept prefix content: sink tokens 0..3 + recent 28 tokens (36..63).
    kept = list(range(4)) + list(range(36, 64))
    for layer in range(4):
        k, _ = runner.kv_caches[layer]
        for i, t in enumerate(kept):
            b, s = i // 8, i % 8
            assert k[b, s, 0, 0].item() == layer * 1000 + t, (layer, i, t)
    details = event["details"]
    assert details["reclaimed_block_count"] == 3  # 8 -> 5 blocks
    groups = details["block_reclaim"]["groups"]
    assert groups[0]["block_ids_after"] == block_ids[:5]


def test_gather_consecutive_fast_path():
    cache = _make_cache()
    block_ids = [3, 4, 5]  # consecutive -> dense view fast path
    _write_request_row(cache, block_ids, 24)
    keys = gather_request_k_dense(cache, block_ids, 8, 24)
    assert keys[0, 1, 7, 0].item() == 5107.0
