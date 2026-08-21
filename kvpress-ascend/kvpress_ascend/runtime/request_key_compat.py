"""vLLM scheduler-output compatibility helpers for kvpress-ascend.

Real vLLM V1 ``SchedulerOutput.scheduled_new_reqs`` entries are
``NewScheduledRequest`` namedtuples with fields like ``(req_id, request,
num_computed_tokens, num_prompt_tokens, num_scheduled_tokens)`` — NOT plain
``(None, request)`` tuples. These helpers parse both shapes robustly.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


def iter_scheduled_new_requests(
    scheduler_output: Any,
) -> Iterator[tuple[Optional[str], Any, int]]:
    """Yield ``(req_id, request, num_prompt_tokens)`` for newly scheduled reqs."""
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
            # Positional fallback for plain tuples / unknown namedtuples:
            # try to locate the request object by its req_id attribute.
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


def iter_scheduled_token_items(scheduler_output: Any) -> Iterator[tuple[str, int]]:
    """Yield ``(req_id, num_scheduled_tokens)`` for the scheduled batch."""
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return
    for req_id, num in scheduled.items():
        try:
            yield str(req_id), max(1, int(num))
        except (TypeError, ValueError):
            continue
