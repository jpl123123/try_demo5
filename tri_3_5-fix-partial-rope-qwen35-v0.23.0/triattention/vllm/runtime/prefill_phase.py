"""Helpers for identifying scheduler prefill phase."""

from __future__ import annotations

from typing import Any

from .request_key_compat import req_id_from_scheduled_key


def is_request_scheduled_as_prefill(scheduler_output: Any, req_id: str) -> bool:
    """Return whether the scheduler reports this request as a new/prefill item.

    vLLM's chunked-prefill path can keep a request in ``scheduled_new_reqs``
    across multiple chunks.  That is the most reliable runtime signal that the
    request is still in prompt processing; compressed effective KV length is not,
    because it intentionally stays below the full prompt length after compaction.
    """
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


def is_request_scheduled_as_spec_decode(scheduler_output: Any, req_id: str) -> bool:
    """Return whether this request is validating speculative draft tokens."""
    scheduled_spec_decode_tokens = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        None,
    )
    if not isinstance(scheduled_spec_decode_tokens, dict):
        return False
    for raw_key, draft_token_ids in scheduled_spec_decode_tokens.items():
        candidate = req_id_from_scheduled_key(raw_key)
        if candidate != req_id:
            continue
        try:
            return len(draft_token_ids) > 0
        except Exception:
            return bool(draft_token_ids)
    return False


def is_prefill_phase_for_limit(
    *,
    scheduler_output: Any,
    req_id: str,
    scheduled_tokens: int,
    prefill_len: int,
    num_computed_tokens: int | None,
) -> bool:
    """Classify prefill for prefill-only policy gates.

    This intentionally uses logical scheduler/request progress rather than
    compressed effective KV length.  Effective length remains small by design
    after TriAttention compaction, so using it here would keep decode steps stuck
    behind prefill-only limits.
    """
    if is_request_scheduled_as_prefill(scheduler_output, req_id):
        return True
    is_spec_decode_step = is_request_scheduled_as_spec_decode(
        scheduler_output,
        req_id,
    )
    if is_spec_decode_step:
        return False
    if int(scheduled_tokens) > 1 and not is_spec_decode_step:
        return True
    if prefill_len <= 0 or num_computed_tokens is None:
        return False
    return int(num_computed_tokens) < int(prefill_len)
