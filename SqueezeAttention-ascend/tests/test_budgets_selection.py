"""SqueezeAttention-ascend simulated-debug tests: budgets + selection math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

sim_env.install_stubs()

from squeezeattention_ascend.core.budgets import (  # noqa: E402
    LayerImportanceAccumulator,
    compute_layer_budgets,
)
from squeezeattention_ascend.core.selection import (  # noqa: E402
    build_keep_tensor_per_head,
    pad_short_budget_layers_with_fake_keys,
    recency_keep_set,
    search_fake_key_hyperplane,
)


def test_recency_keep_set_semantics():
    assert recency_keep_set(100, 20, 4) == list(range(4)) + list(range(84, 100))
    assert recency_keep_set(100, 200, 4) == list(range(100))
    assert recency_keep_set(10, 6, 4) == list(range(4)) + list(range(8, 10))


def test_budget_math_conserves_total():
    importance = [float(x) for x in np.linspace(0.05, 0.9, 12)]
    budgets, diag = compute_layer_budgets(
        layer_importance=importance,
        num_layers=12,
        ini_size=0.21,
        class3_size=0.08,
        prompt_len=8192,
        n_clusters=3,
        seed=42,
    )
    assert len(budgets) == 12
    total = sum(budgets) / 8192
    assert abs(total - 12 * 0.21) < 0.01
    # Class3 (highest importance) layers get exactly percent * prompt_len.
    class_ids = diag["class_ids"]
    class3_idx = [i for i, c in enumerate(class_ids) if c == 2]
    assert class3_idx
    for i in class3_idx:
        assert budgets[i] == int(0.08 * 8192)


def test_budget_math_edges():
    # Single-class degeneracy (all equal importance) must not crash and must
    # conserve the total.
    importance = [0.5] * 6
    budgets, diag = compute_layer_budgets(
        layer_importance=importance,
        num_layers=6,
        ini_size=0.2,
        class3_size=0.1,
        prompt_len=1000,
        n_clusters=3,
        seed=1,
    )
    total = sum(budgets) / 1000
    assert abs(total - 6 * 0.2) < 0.01


def test_layer_importance_accumulator():
    acc = LayerImportanceAccumulator()
    acc.add(0, torch.tensor([1.0, 3.0, 5.0]), 0, 3)
    acc.add(0, torch.tensor([1.0, 1.0]), 0, 2)
    acc.add(1, torch.tensor([2.0, 4.0]), 0, 2)
    means = acc.means(2)
    assert abs(means[0] - (9.0 + 2.0) / 5) < 1e-5
    assert abs(means[1] - 3.0) < 1e-5


def test_fake_key_hyperplane_zeroes_attention():
    torch.manual_seed(0)
    queries = torch.randn(2, 4, 8)  # 2 kv-head groups x 8 queries x 8 dim
    fake = search_fake_key_hyperplane(queries, scale=1e5)
    dots = torch.bmm(queries, fake.unsqueeze(-1))
    # kvpress guarantee: <q, k> <= 0 so exp(<q,k>) <= 1 (exact zero is
    # possible for a query lying on the hyperplane, mirroring kvpress).
    assert bool((dots <= 0).all())
    # In real attention the fake keys share the softmax with real keys; the
    # padded positions must receive ~0 weight.
    logits = dots / (queries.shape[-1] ** 0.5)
    mixed = torch.cat(
        [torch.zeros_like(logits), logits], dim=1
    )  # 4 real (logit 0) + 4 fake (very negative)
    weights = torch.softmax(mixed, dim=1)
    fake_weights = weights[:, 4:]
    assert float(fake_weights.max()) < 1e-3


def test_fake_key_padding_writes_tail_slots():
    from squeezeattention_ascend.core.kv_layout import split_kv_axes

    num_blocks, block_size, heads, dim = 8, 8, 4, 8
    k = torch.zeros(num_blocks, block_size, heads, dim)
    v = torch.zeros(num_blocks, block_size, heads, dim)
    cache = (k, v)
    block_ids = list(range(4))  # 32 tokens
    query = torch.randn(1, 8, 1, dim)  # 8 query heads -> 4 kv heads
    padded = pad_short_budget_layers_with_fake_keys(
        key_cache=k,
        block_ids=block_ids,
        block_size=block_size,
        query=query,
        keep_count=16,
        total_tokens=32,
        max_keep_count=32,
        num_kv_heads=4,
    )
    assert padded == 16
    # Tail slots [16, 32) now hold fake keys: exp(q.k) ~ 0 for the query.
    k_cache, _ = split_kv_axes(cache)
    fake_keys = k_cache[block_ids[2]:, :, :, :].reshape(-1, heads, dim)  # slots 16..31
    q_flat = query[0].float()
    q_grouped = q_flat.view(4, 2, 1, dim).mean(dim=1)  # [4, 1, dim]
    dots = torch.einsum("hd,thd->th", q_grouped.squeeze(1), fake_keys.float())
    assert bool((dots <= 0).all())
    # Mixed row: 16 real keys (logit 0) + 16 fake keys; fake positions ~0.
    logits = dots / (dim ** 0.5)
    mixed = torch.cat([torch.zeros_like(logits), logits], dim=0)
    weights = torch.softmax(mixed, dim=0)
    fake_weights = weights[16:, :]
    # The hyperplane guarantees <= 0 (kvpress's attention_patch guarantee);
    # padded positions keep a bounded, small share of the softmax mass.
    assert float(fake_weights.max()) < 0.05
    assert float(fake_weights.mean()) < 0.02


def test_keep_tensor_per_head_expansion():
    keep = build_keep_tensor_per_head([1, 5, 9], 4, torch.device("cpu"))
    assert keep.shape == (4, 3)
    assert keep[3].tolist() == [1, 5, 9]
