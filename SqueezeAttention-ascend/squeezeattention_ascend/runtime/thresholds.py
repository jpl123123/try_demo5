"""Compression thresholds and prefill-phase helpers for kvpress-ascend.

Mirrors the TriAttention scheduling philosophy (tri_3_5 thresholds.py +
prefill_phase.py):

- the trigger threshold is ``budget + max(min_reclaim_blocks * block_size)``;
- prefill-phase detection uses LOGICAL scheduler progress (the request being
  in ``scheduled_new_reqs``, or ``num_scheduled_tokens > 1`` for chunked
  prefill), NOT the compressed effective KV length — effective length stays
  small by design after compaction, so using it would stick decode steps
  behind prefill-only gates;
- the worker self-triggers from the ACTUAL block-table capacity, so the
  pipeline compresses even when engine-core scheduler signals are absent or
  lag behind.
"""

from __future__ import annotations

from typing import Any


def is_request_scheduled_as_prefill(scheduler_output: Any, req_id: str) -> bool:
    """The request is still in ``scheduled_new_reqs`` (vLLM's most reliable
    in-prefill signal; chunked prefill keeps a request there across chunks)."""
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if not isinstance(scheduled_new_reqs, (list, tuple)):
        return False
    for new_req in scheduled_new_reqs:
        candidate = getattr(new_req, "req_id", None)
        if candidate is None:
            candidate = getattr(new_req, "request_id", None)
        if candidate == req_id:
            return True
    return False


def is_prefill_phase_for_limit(
    *,
    scheduler_output: Any,
    req_id: str,
    scheduled_tokens: int,
    prefill_len: int,
    num_computed_tokens: Any,
) -> bool:
    """Classify prefill for prefill-only policy gates (logical progress)."""
    if is_request_scheduled_as_prefill(scheduler_output, str(req_id)):
        return True
    if int(scheduled_tokens) > 1:
        return True
    if int(prefill_len) <= 0 or num_computed_tokens is None:
        return False
    try:
        return int(num_computed_tokens) < int(prefill_len)
    except (TypeError, ValueError):
        return False


def compression_length_threshold(
    *,
    kv_budget: int,
    min_reclaim_blocks: int,
    block_size: int,
    prefill_len: int = 0,
    protect_prefill: bool = True,
    include_prefill_in_budget: bool = True,
) -> int:
    """Trigger threshold: budget + reclaim interval (+ protected prefill)."""
    threshold = max(1, int(kv_budget)) + max(
        1, max(0, int(min_reclaim_blocks)) * max(1, int(block_size))
    )
    if protect_prefill and not include_prefill_in_budget:
        threshold += max(0, int(prefill_len))
    return threshold


def resolve_request_prefill_len(request_like: Any) -> int:
    """Best-effort prompt length from a request-like object (TriAttention's
    ``_resolve_full_prefill_len_from_request_like``)."""
    candidates: list[int] = []
    prompt_token_ids = getattr(request_like, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            candidates.append(len(prompt_token_ids))
        except Exception:
            pass
    for attr_name in ("prompt_token_ids_len", "num_prompt_tokens"):
        raw = getattr(request_like, attr_name, None)
        if raw is None:
            continue
        try:
            candidates.append(int(raw))
        except (TypeError, ValueError):
            continue
    prefill_token_ids = getattr(request_like, "prefill_token_ids", None)
    if prefill_token_ids is not None:
        try:
            candidates.append(len(prefill_token_ids))
        except Exception:
            pass
    return max(candidates, default=0)
