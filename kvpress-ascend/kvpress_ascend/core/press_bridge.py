"""kvpress -> Ascend scoring bridge.

Converts kvpress's HF-coupled scoring mechanism (``press.score(module,
hidden_states, keys, values, attentions, kwargs)`` over dense DynamicCache
tensors) into Ascend-native scoring over:

- ``keys`` / ``values``: dense ``[1, H, T, D]`` gathered from the Ascend block
  cache (``kv_layout.gather_request_kv_dense``);
- ``queries``: post-RoPE query tensor ``[1, H, q_len, D]`` captured by
  ``attention_hooks`` from the vLLM-Ascend attention layer (kvpress would
  re-derive pre-RoPE queries via ``get_prerope_query_states`` + RoPE — the
  post-RoPE capture is mathematically equivalent for attention scoring);
- ``attentions``: computed on demand with torch (kvpress would read them from
  the HF eager attention output).

The output is per-head keep indices with a **uniform keep count** — the only
count layout physically expressible in the shared Ascend block table.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .vendored_presses import (
    DecodingPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
    VENDORED_PRESS_REGISTRY,
)


class PressSource:
    """Resolve a press instance from installed kvpress or the vendored fallback."""

    def __init__(self, prefer_installed: bool = True):
        self.prefer_installed = prefer_installed
        self._installed_available: Optional[bool] = None

    @property
    def installed_available(self) -> bool:
        if self._installed_available is None:
            try:
                import kvpress  # noqa: F401

                self._installed_available = True
            except Exception:
                self._installed_available = False
        return self._installed_available

    def build(self, press_name: str, **params) -> Any:
        if self.prefer_installed and self.installed_available:
            try:
                import kvpress

                cls = getattr(kvpress, press_name)
                return cls(**params)
            except Exception:
                pass
        cls = VENDORED_PRESS_REGISTRY[press_name]
        return cls(**params)


def _head_groups(num_heads: int, num_kv_heads: int) -> int:
    return max(1, num_heads // max(1, num_kv_heads))


def compute_window_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    *,
    num_kv_heads: int,
) -> torch.Tensor:
    """SnapKV-style window attention: last ``window_size`` queries vs all keys.

    kvpress computes this from pre-RoPE queries + position embeddings; here we
    use the post-RoPE captured queries directly (identical scoring math).
    """
    bsz, num_heads, q_len, head_dim = queries.shape
    keys_b = keys
    if q_len <= window_size:
        q_window = queries
    else:
        q_window = queries[:, :, -window_size:, :]
    # Expand KV heads to query heads for the matmul.
    groups = _head_groups(num_heads, num_kv_heads)
    if groups > 1:
        k_expanded = keys_b.repeat_interleave(groups, dim=1)
    else:
        k_expanded = keys_b
    k_len = keys_b.shape[2]
    attn_weights = torch.matmul(q_window, k_expanded.transpose(2, 3)) / math.sqrt(head_dim)
    # Causal mask within the window (kvpress's triu mask semantics).
    mask = torch.ones_like(attn_weights) * float("-inf")
    mask = torch.triu(mask, diagonal=k_len - window_size + 1)
    attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
    return attn_weights


def _avg_pool1d(scores: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)


def score_knorm(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    return -keys.norm(dim=-1)


def score_random(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    generator = None
    seed = getattr(press, "seed", None)
    if seed is not None:
        generator = torch.Generator(device=keys.device)
        generator.manual_seed(int(seed))
    return torch.rand(*keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype)


def score_streaming_llm(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    k_len = keys.shape[2]
    n_sink = int(getattr(press, "n_sink", 4))
    assert k_len > n_sink, f"Input should contain more tokens than n_sink={n_sink}"
    compression_ratio = float(getattr(press, "compression_ratio", 0.0))
    n_pruned = k_len - int(k_len * (1 - compression_ratio))
    scores = torch.ones_like(keys[..., 0])
    scores[:, :, n_sink : n_sink + n_pruned] = 0
    return scores


def score_snapkv(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    queries = ctx.get("queries")
    num_kv_heads = keys.shape[1]
    if queries is None:
        raise ValueError("SnapKVPress requires captured queries on Ascend")
    window_size = int(getattr(press, "window_size", 64))
    kernel_size = int(getattr(press, "kernel_size", 5))
    num_heads = queries.shape[1]
    attn_weights = compute_window_attention(
        queries, keys, window_size, num_kv_heads=num_kv_heads
    )
    # Score the first k_len - window_size keys with the window queries.
    k_len = keys.shape[2]
    if k_len <= window_size:
        attn_slice = attn_weights
    else:
        attn_slice = attn_weights[..., : k_len - window_size]
    scores = attn_slice.mean(dim=-2)
    scores = _avg_pool1d(scores, kernel_size)
    groups = _head_groups(num_heads, num_kv_heads)
    scores = scores.view(1, num_kv_heads, groups, scores.shape[-1])
    scores = scores.mean(2)
    pad_value = float(scores.max().item()) + 1.0
    scores = F.pad(scores, (0, min(window_size, k_len)), value=pad_value)
    return scores


def score_tova(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    queries = ctx.get("queries")
    if queries is None:
        raise ValueError("TOVAPress requires captured queries on Ascend")
    num_kv_heads = keys.shape[1]
    attn_weights = compute_window_attention(queries, keys, 1, num_kv_heads=num_kv_heads)
    scores = attn_weights.mean(1)
    scores = scores.repeat(1, num_kv_heads, 1)
    pad_value = float(scores.max().item()) + 1.0
    scores = F.pad(scores, (0, 1), value=pad_value)
    return scores


def score_observed_attention(press: Any, keys: torch.Tensor, **ctx: Any) -> torch.Tensor:
    """Observed-attention scoring from the captured queries of the latest
    prefill/decode step(s): score = mean over the query window of softmax
    attention weights, normalized by the number of attending queries (kvpress
    semantics, computed natively instead of read from HF eager attention)."""
    queries = ctx.get("queries")
    if queries is None:
        raise ValueError("ObservedAttentionPress requires captured queries on Ascend")
    num_kv_heads = keys.shape[1]
    num_heads = queries.shape[1]
    k_len = keys.shape[2]
    q_len = queries.shape[2]
    attn_weights = compute_window_attention(queries, keys, q_len, num_kv_heads=num_kv_heads)
    # kvpress: scores = attentions.sum(2) / arange(n_tokens, 0, -1)
    scores = attn_weights.sum(dim=-2)
    n_tokens_in_sum = torch.arange(k_len, 0, -1, device=attn_weights.device, dtype=attn_weights.dtype)
    scores = scores / n_tokens_in_sum
    groups = _head_groups(num_heads, num_kv_heads)
    scores = scores.view(1, num_kv_heads, groups, k_len).mean(2)
    return scores


_PRESS_SCORERS = {
    "KnormPress": score_knorm,
    "RandomPress": score_random,
    "StreamingLLMPress": score_streaming_llm,
    "SnapKVPress": score_snapkv,
    "TOVAPress": score_tova,
    "ObservedAttentionPress": score_observed_attention,
}


def score_press(
    press: Any,
    keys: torch.Tensor,
    *,
    queries: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the press scoring against dense gathered keys.

    Returns scores of shape [1, num_kv_heads, T].
    """
    name = press.__class__.__name__
    scorer = _PRESS_SCORERS.get(name)
    if scorer is None:
        # Installed kvpress press may carry a different class name for the
        # same semantics (e.g. subclasses); fall back to the registered type.
        for registered_name, scorer_fn in _PRESS_SCORERS.items():
            if registered_name in str(type(press)):
                scorer = scorer_fn
                break
    if scorer is None:
        raise ValueError(
            f"Press {name!r} is not supported by the kvpress-ascend scoring "
            f"bridge; supported: {sorted(_PRESS_SCORERS)}"
        )
    return scorer(press, keys, queries=queries)


def select_keep_indices(
    press: Any,
    keys: torch.Tensor,
    n_kept: int,
    *,
    queries: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-head keep indices (uniform count ``n_kept``) from press scores.

    kvpress's ``ScorerPress.compress`` does ``scores.topk(n_kept, dim=-1)``
    and gathers in topk order; the Ascend block cache instead needs the kept
    token indices **sorted ascending** so the compacted prefix preserves the
    original causal token order (mathematically equivalent for attention, and
    deterministic for debugging/reclaim).
    """
    scores = score_press(press, keys, queries=queries)
    k_len = keys.shape[2]
    if n_kept >= k_len:
        indices = torch.arange(k_len, device=keys.device, dtype=torch.long)
        return indices.unsqueeze(0).expand(keys.shape[1], -1)
    indices = scores.topk(n_kept, dim=-1).indices  # [1, H, n_kept]
    return indices[0].sort(dim=-1).values.contiguous()


def build_press(
    press_name: str,
    compression_ratio: float,
    *,
    window_size: int = 64,
    n_sink: int = 4,
    seed: Optional[int] = None,
    target_size: int = 0,
    compression_interval: int = 512,
    prefer_installed: bool = True,
) -> Any:
    """Build a press instance from the vendored (or installed) registry."""
    source = PressSource(prefer_installed=prefer_installed)
    if press_name == "DecodingPress":
        base_name = "KnormPress"
        base_params: dict[str, Any] = {"compression_ratio": compression_ratio}
        if seed is not None:
            base_params["seed"] = seed
        base = source.build(base_name, **base_params)
        return DecodingPress(
            base_press=base,
            compression_interval=compression_interval,
            target_size=target_size if target_size > 0 else 2048,
        )
    params: dict[str, Any] = {"compression_ratio": compression_ratio}
    if press_name == "RandomPress":
        params["seed"] = seed
    if press_name == "StreamingLLMPress":
        params["n_sink"] = n_sink
    if press_name == "SnapKVPress":
        params["window_size"] = window_size
    return source.build(press_name, **params)
