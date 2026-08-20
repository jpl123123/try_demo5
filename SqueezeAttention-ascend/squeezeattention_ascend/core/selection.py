"""SqueezeAttention token-level selection for the Ascend block cache.

The paper's per-layer streaming drop keeps, for layer ``L`` with budget
``sliding_windows[L]``:

    [0, start_size)  ∪  [T - (sliding_windows[L] - start_size), T)

i.e. the sink tokens plus the most recent ``budget - start_size`` tokens.

Two modes:

- ``uniform`` (default): all layers compact to the same block-aligned keep
  count ``K`` (the max per-layer budget). This is the only count physically
  expressible with vLLM-Ascend's shared block-table row and uniform seq_len.
  Each layer's keep *set* is the recency set of size ``K`` (identical across
  layers since the sets are nested as the budget grows).
- ``class_weighted`` (experimental): each layer keeps its own count
  ``budget_L``; short-budget layers' tail slots ``[budget_L, K)`` are padded
  with fake keys (hyperplane from the latest query, kvpress-style) so
  ``exp(q.k) ≈ 0`` during the next decode step.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def recency_keep_set(total_tokens: int, keep_count: int, start_size: int) -> list[int]:
    """Sink tokens + most recent tokens (StreamingLLM-style)."""
    total_tokens = max(0, int(total_tokens))
    keep_count = max(0, min(int(keep_count), total_tokens))
    start_size = max(0, min(int(start_size), total_tokens))
    if keep_count <= 0:
        return []
    if total_tokens <= keep_count:
        return list(range(total_tokens))
    sink = list(range(min(start_size, total_tokens)))
    recent = list(range(total_tokens - (keep_count - start_size), total_tokens))
    return sink + recent


def build_keep_tensor_per_head(
    keep_set: list[int],
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand one keep set to per-head indices [num_kv_heads, len(keep_set)]."""
    indices = torch.as_tensor(keep_set, device=device, dtype=torch.long)
    return indices.unsqueeze(0).expand(num_kv_heads, -1).contiguous()


def search_fake_key_hyperplane(
    X: torch.Tensor,
    max_iter: int = 1000,
    scale: float = 1e5,
) -> torch.Tensor:
    """kvpress ``search_hyperplane``: find Y with <X_i, Y> <= 0 for all rows.

    Returns ``-scale * Y / ||Y||^2`` so that ``exp(<X, Y>) ≈ 0``.
    """
    Y = X.mean(1)
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -scale * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError(
        "Could not find fake keys such that for every query q, exp(<q, k>) = 0"
    )


def pad_short_budget_layers_with_fake_keys(
    *,
    key_cache: torch.Tensor,
    block_ids: list[int],
    block_size: int,
    query: torch.Tensor,
    keep_count: int,
    total_tokens: int,
    max_keep_count: int,
    num_kv_heads: int,
) -> int:
    """Write fake keys into K-cache slots [keep_count, max_keep_count).

    ``query``: post-RoPE query ``[1, H, q_len, D]`` of the current step (the
    hyperplane nullifies attention for it). Only K needs padding: with
    ``exp(q.k) ≈ 0`` the attention weight is ~0 regardless of V.
    """
    if max_keep_count <= keep_count or max_keep_count > total_tokens:
        return 0
    device = key_cache.device
    from .kv_layout import _resolve_token_slots

    q = query.float()
    bsz, num_heads, q_len, head_dim = q.shape
    groups = max(1, num_heads // max(1, num_kv_heads))
    # Aggregate queries per KV-head group (average) -> [bsz, num_kv_heads, q_len, D].
    q_grouped = q.view(bsz, num_kv_heads, groups, q_len, head_dim).mean(dim=2)
    q_flat = q_grouped.reshape(bsz * num_kv_heads, q_len, head_dim)
    fake = search_fake_key_hyperplane(q_flat)
    fake = fake.view(bsz, num_kv_heads, head_dim).to(key_cache.dtype)

    tail = torch.arange(keep_count, max_keep_count, device=device, dtype=torch.long)
    if tail.numel() == 0:
        return 0
    src_blocks, src_off = _resolve_token_slots(
        block_ids, block_size, tail, device=device
    )
    head_idx = torch.arange(num_kv_heads, device=device, dtype=torch.long)
    # key_cache[src_blocks, src_off, head_idx] with broadcast [T, 1] x [1, H].
    key_cache[src_blocks[:, None], src_off[:, None], head_idx[None, :]] = fake
    return int(tail.numel())
