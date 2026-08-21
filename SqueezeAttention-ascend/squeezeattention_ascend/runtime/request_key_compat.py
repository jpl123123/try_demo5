"""vLLM scheduler-output compatibility helpers for kvpress-ascend.

Mirrors TriAttention's request_key_compat: vLLM may key
``num_scheduled_tokens`` / ``scheduled_spec_decode_tokens`` by request-id
strings OR by request objects / wrappers carrying ``request_id`` / ``req_id``
(and MTP spec-decode keys can be tuples). All helpers normalize to req_id
strings so the state store and gating never miss a request.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


def req_id_from_scheduled_key(key: Any) -> Optional[str]:
    """Normalize a scheduler key to a req_id string."""
    if isinstance(key, str):
        return key
    if isinstance(key, (int,)):
        return str(key)
    for attr in ("request_id", "req_id"):
        req_id = getattr(key, attr, None)
        if isinstance(req_id, str):
            return req_id
        if isinstance(req_id, int):
            return str(req_id)
    return None


def iter_scheduled_token_items(scheduler_output: Any) -> Iterator[tuple[str, int]]:
    """Yield ``(req_id, num_scheduled_tokens)`` for the scheduled batch."""
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return
    for raw_key, raw_value in scheduled.items():
        req_id = req_id_from_scheduled_key(raw_key)
        if req_id is None:
            continue
        try:
            yield req_id, max(1, int(raw_value))
        except (TypeError, ValueError):
            continue


def iter_scheduled_new_requests(
    scheduler_output: Any,
) -> Iterator[tuple[Optional[str], Any, int]]:
    """Yield ``(req_id, request, num_prompt_tokens)`` for newly scheduled reqs.

    Parses both real vLLM ``NewScheduledRequest`` namedtuples (fields
    ``req_id, request, num_computed_tokens, num_prompt_tokens,
    num_scheduled_tokens``) and plain ``(None, request)`` tuples.
    """
    new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if not isinstance(new_reqs, (list, tuple)):
        return
    for item in new_reqs:
        if item is None:
            continue
        req_id = getattr(item, "req_id", None)
        request = getattr(item, "request", None)
        num_prompt = getattr(item, "num_prompt_tokens", None)
        if req_id is None and isinstance(item, tuple):
            for field in item:
                if field is None:
                    continue
                if isinstance(field, str) and req_id is None:
                    req_id = field
                    continue
                if hasattr(field, "req_id"):
                    request = field
                    if req_id is None:
                        req_id = getattr(field, "req_id", None)
                    if num_prompt is None:
                        num_prompt = getattr(field, "num_prompt_tokens", None)
                    break
        if req_id is None and request is not None:
            req_id = getattr(request, "req_id", None)
        if req_id is None:
            continue
        if num_prompt is None and request is not None:
            num_prompt = getattr(request, "num_prompt_tokens", None)
        if num_prompt is None:
            num_prompt = getattr(request, "prefill_len", None)
        yield str(req_id), request, int(num_prompt or 0)


def is_request_scheduled_as_spec_decode(scheduler_output: Any, req_id: str) -> bool:
    """Whether this request is validating speculative (MTP) draft tokens.

    Mirrors TriAttention: ``scheduled_spec_decode_tokens`` keys may be
    req_id strings, request objects, or MTP tuple keys — normalize them.
    """
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
