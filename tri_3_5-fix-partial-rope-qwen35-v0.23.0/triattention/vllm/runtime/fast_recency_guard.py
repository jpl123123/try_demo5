"""Fast-recency safety guards."""

from __future__ import annotations

from pathlib import Path

from .config import TriAttentionRuntimeConfig


def has_available_sparse_stats(config: TriAttentionRuntimeConfig) -> bool:
    """Return whether sparse-stat scoring has a usable stats file."""
    stats_path = getattr(config, "sparse_stats_path", None)
    if stats_path is None:
        return False
    try:
        return Path(stats_path).expanduser().exists()
    except (OSError, TypeError, ValueError):
        return False


def uses_pure_fast_recency(config: TriAttentionRuntimeConfig) -> bool:
    """Return whether the current config will use pure recency selection."""
    if not bool(getattr(config, "fast_recency_only", False)):
        return False
    return not (
        bool(getattr(config, "fast_recency_accuracy_guard", True))
        and has_available_sparse_stats(config)
    )


def should_guard_fast_recency_long_context(
    *,
    config: TriAttentionRuntimeConfig,
    effective_tokens: int,
    prefill_len: int,
) -> bool:
    if not uses_pure_fast_recency(config):
        return False
    if not bool(getattr(config, "fast_recency_long_context_guard", False)):
        return False
    threshold = int(getattr(config, "fast_recency_long_context_guard_tokens", 0) or 0)
    if threshold <= 0:
        return False
    context_len = max(int(effective_tokens), int(prefill_len))
    return context_len >= threshold
