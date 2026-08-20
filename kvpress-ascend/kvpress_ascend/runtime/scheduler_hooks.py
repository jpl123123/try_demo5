"""Scheduler-side hooks for kvpress-ascend.

kvpress's HF flow compresses at boundaries tracked by ``cache_position`` and
keeps the model's ``past_key_values`` length consistent. The Ascend equivalent
lives here:

- an effective-length tracker (the compressed history length per request);
- pre-schedule sync that feeds the effective length into vLLM's
  ``num_computed_tokens`` (so block allocation and sampling bookkeeping use the
  compressed view);
- compression candidate signals on ``schedule()``;
- compression-event application + physical block reclaim on
  ``update_from_output()``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..logging_control import log_debug, log_info
from .block_sync import free_reclaimed_blocks
from .signals import CompressionSignal


class EffectiveLenTracker:
    """Compressed history length per request (scheduler side)."""

    def __init__(self) -> None:
        self._lens: dict[str, int] = {}

    def apply_compression(self, req_id: str, cache_len_after: int) -> None:
        self._lens[str(req_id)] = int(cache_len_after)

    def advance(self, req_id: str, scheduled_tokens: int) -> int:
        current = self._lens.get(str(req_id))
        if current is None:
            return 0
        next_len = current + max(1, int(scheduled_tokens))
        self._lens[str(req_id)] = next_len
        return next_len

    def get(self, req_id: str) -> Optional[int]:
        return self._lens.get(str(req_id))

    def remove(self, req_id: str) -> None:
        self._lens.pop(str(req_id), None)

    def __len__(self) -> int:
        return len(self._lens)


def update_request_effective_kv_offset(request: Any, cache_len_after: int) -> None:
    try:
        setattr(request, "_kvpress_effective_kv_offset", int(cache_len_after))
    except Exception:
        pass


def sync_effective_kv_offsets_before_schedule(scheduler: Any, tracker: EffectiveLenTracker) -> None:
    """Feed the compressed history length into vLLM's num_computed_tokens."""
    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, dict):
        return
    for req_id, request in requests.items():
        effective = tracker.get(str(req_id))
        if effective is None:
            continue
        num_computed = getattr(request, "num_computed_tokens", None)
        if not isinstance(num_computed, int) or num_computed <= int(effective):
            continue
        try:
            setattr(request, "_kvpress_orig_num_computed", num_computed)
            setattr(request, "num_computed_tokens", int(effective))
        except Exception:
            pass


def build_signals(
    scheduler: Any,
    scheduler_output: Any,
    tracker: EffectiveLenTracker,
    *,
    budget: int,
    min_reclaim_blocks: int,
    block_size: int,
    protect_prefill: bool,
    defer_prefill_compression: bool,
    step: int,
    log_decisions: bool,
) -> dict[str, CompressionSignal]:
    """Compute compression candidate signals for the scheduled batch.

    kvpress's ``is_prefilling`` phase gate becomes ``defer_prefill_compression``:
    during chunked prefill the candidate is suppressed unless compression is
    explicitly allowed mid-prefill.
    """
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    signals: dict[str, CompressionSignal] = {}
    if not isinstance(scheduled, dict) or not scheduled:
        return signals

    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, dict):
        return signals
    prefill_lens = getattr(scheduler, "kvpress_prefill_lens", None)
    if not isinstance(prefill_lens, dict):
        prefill_lens = {}

    threshold = max(1, budget + max(0, min_reclaim_blocks) * max(1, block_size))

    for req_id, num_scheduled_tokens in scheduled.items():
        try:
            scheduled_tokens = max(1, int(num_scheduled_tokens))
        except (TypeError, ValueError):
            continue
        request = requests.get(req_id)
        if request is None:
            continue
        num_computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        effective = tracker.get(str(req_id))
        if effective is None:
            effective = num_computed
        estimated_cache_len = int(effective)
        prefill_len = int(prefill_lens.get(str(req_id), 0) or 0)
        is_prefill_step = 0 < num_computed < prefill_len
        if defer_prefill_compression and is_prefill_step:
            continue
        if estimated_cache_len + scheduled_tokens <= threshold:
            continue
        signals[str(req_id)] = CompressionSignal(
            req_id=str(req_id),
            should_compress=True,
            reason="length_threshold",
            step=int(step),
            estimated_cache_len=estimated_cache_len,
            prefill_len=prefill_len,
            scheduled_tokens=scheduled_tokens,
            protect_prefill=bool(protect_prefill),
            is_prefill_step=is_prefill_step,
        )
        if log_decisions:
            log_debug(
                "signal req=%s effective=%d scheduled=%d threshold=%d",
                req_id, estimated_cache_len, scheduled_tokens, threshold,
            )
    return signals


def _read_compression_events(
    scheduler_output: Any,
    model_runner_output: Any,
) -> list[dict[str, Any]]:
    from .output_bridge import read_events_from_output

    events = read_events_from_output(model_runner_output)
    if events is not None:
        return events
    raw = getattr(scheduler_output, "kvpress_compression_events", None)
    if isinstance(raw, list):
        return raw
    return []


def _num_required_blocks(token_len: int, block_size: int) -> int:
    if token_len <= 0:
        return 0
    return (token_len + block_size - 1) // block_size


def apply_compression_events(
    scheduler: Any,
    tracker: EffectiveLenTracker,
    compression_events: list[dict[str, Any]],
    *,
    block_size: int,
    enable_block_reclaim: bool,
    log_decisions: bool,
) -> None:
    """Apply worker-reported compression events: tracker update + block reclaim."""
    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, dict):
        requests = {}
    for event in compression_events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if req_id is None:
            continue
        cache_len_after = event.get("cache_len_after")
        if not isinstance(cache_len_after, int):
            continue
        retained_cache_len = event.get("retained_cache_len")
        if not isinstance(retained_cache_len, int) or retained_cache_len <= 0:
            retained_cache_len = cache_len_after
        tracker.apply_compression(str(req_id), int(cache_len_after))
        request = requests.get(req_id)
        if request is not None:
            update_request_effective_kv_offset(request, int(cache_len_after))
        if not enable_block_reclaim:
            continue
        _reclaim_scheduler_blocks(
            scheduler=scheduler,
            req_id=req_id,
            retained_cache_len=int(retained_cache_len),
            block_size=block_size,
            event=event,
            log_decisions=log_decisions,
        )


def _reclaim_scheduler_blocks(
    *,
    scheduler: Any,
    req_id: str,
    retained_cache_len: int,
    block_size: int,
    event: dict[str, Any],
    log_decisions: bool,
) -> None:
    required_blocks = _num_required_blocks(retained_cache_len, block_size)
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is None:
        return
    details = event.get("details")
    reclaim_groups = (
        details.get("block_reclaim", {}).get("groups")
        if isinstance(details, dict) and isinstance(details.get("block_reclaim"), dict)
        else None
    )
    if isinstance(reclaim_groups, list) and reclaim_groups:
        for group in reclaim_groups:
            gid = int(group.get("gid", 0))
            block_ids_after = group.get("block_ids_after")
            if not isinstance(block_ids_after, list):
                continue
            single_managers = _resolve_single_type_managers(manager)
            if gid < len(single_managers):
                sub_manager = single_managers[gid]
                blocks = getattr(sub_manager, "req_to_blocks", None)
            else:
                sub_manager = manager
                blocks = getattr(manager, "req_to_blocks", None)
            if not isinstance(blocks, dict):
                continue
            current = blocks.get(req_id)
            if not isinstance(current, (list, tuple)):
                continue
            removed = list(current[len(block_ids_after):])
            blocks[req_id] = list(block_ids_after)
            if removed and free_reclaimed_blocks(sub_manager, removed):
                log_info(
                    "scheduler reclaim req=%s gid=%d blocks %d -> %d freed=%d",
                    req_id, gid, len(current), len(block_ids_after), len(removed),
                )
        return

    # Fallback: truncate the first manager's row.
    blocks = getattr(manager, "req_to_blocks", None)
    if isinstance(blocks, dict):
        current = blocks.get(req_id)
        if isinstance(current, (list, tuple)) and len(current) > required_blocks:
            removed = list(current[required_blocks:])
            blocks[req_id] = list(current[:required_blocks])
            if removed and free_reclaimed_blocks(manager, removed):
                log_info(
                    "scheduler reclaim req=%s blocks %d -> %d freed=%d",
                    req_id, len(current), required_blocks, len(removed),
                )


def _resolve_single_type_managers(manager: Any) -> list[Any]:
    coordinator = getattr(manager, "coordinator", None)
    single = getattr(coordinator, "single_type_managers", None) if coordinator else None
    if isinstance(single, (list, tuple)):
        return list(single)
    return []


def cleanup_finished_requests(scheduler: Any, tracker: EffectiveLenTracker) -> None:
    finished = getattr(scheduler, "finished_req_ids", None)
    if not isinstance(finished, (list, tuple, set)):
        return
    for req_id in finished:
        tracker.remove(str(req_id))
    prefill_lens = getattr(scheduler, "kvpress_prefill_lens", None)
    if isinstance(prefill_lens, dict):
        for req_id in finished:
            prefill_lens.pop(str(req_id), None)
