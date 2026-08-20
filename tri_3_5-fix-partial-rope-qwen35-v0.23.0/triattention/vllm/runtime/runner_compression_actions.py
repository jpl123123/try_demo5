"""Compression action execution for TriAttentionModelRunner."""
from __future__ import annotations

from typing import Any

from .constants import TRITON_SCORING_REQUIRED_MARKER
from .prefill_phase import is_prefill_phase_for_limit
from .signals import CompressionSignal


def _limited_signal_items(
    signals: dict[str, CompressionSignal],
    max_compressions_per_step: int,
) -> tuple[list[tuple[str, CompressionSignal]], list[str]]:
    compressing: list[tuple[str, CompressionSignal]] = []
    delayed: list[str] = []
    remaining = max(0, int(max_compressions_per_step))
    for req_id, signal in signals.items():
        if not signal.should_compress:
            continue
        if bool(getattr(signal, "force", False)):
            compressing.append((req_id, signal))
            continue
        if max_compressions_per_step > 0 and remaining <= 0:
            delayed.append(req_id)
            continue
        compressing.append((req_id, signal))
        if max_compressions_per_step > 0:
            remaining -= 1
    return compressing, delayed


def execute_runner_compression_actions(
    *,
    executor: Any,
    state_store: Any,
    scheduler_output: Any,
    signals: dict[str, CompressionSignal],
    strict_no_downgrade: bool,
    allowed_strict_skip_reasons: set[str],
    logger: Any,
    log_decisions: bool,
    log_worker_events: bool = True,
    logging_enabled: bool = True,
    log_execution_path: bool = False,
    log_execution_path_core_only: bool = False,
    log_selector_debug: bool = False,
    max_compressions_per_step: int = 0,
) -> list[dict[str, Any]]:
    """Execute compression for triggered requests and emit scheduler-side events."""
    events: list[dict[str, Any]] = []
    noisy_skip_reasons = {
        "under_budget",
        "defer_recompress",
        "batch_queue_dedup",
        "prefill_incomplete",
        "prefill_compression_limit",
        "zero_copy_recency_not_ready",
        "fast_recency_long_context_guard",
        "initial_decode_grace",
        "compression_step_limited",
    }
    signal_items, delayed_req_ids = _limited_signal_items(
        signals,
        max_compressions_per_step=max_compressions_per_step,
    )

    def _event_common(signal: CompressionSignal) -> dict[str, int | bool]:
        scheduled_tokens = int(getattr(signal, "scheduled_tokens", 1))
        prefill_len = int(getattr(signal, "prefill_len", 0))
        is_prefill_step = is_prefill_phase_for_limit(
            scheduler_output=scheduler_output,
            req_id=str(getattr(signal, "req_id", "")),
            scheduled_tokens=scheduled_tokens,
            prefill_len=prefill_len,
            num_computed_tokens=None,
        )
        return {
            "scheduled_tokens": scheduled_tokens,
            "estimated_cache_len": int(getattr(signal, "estimated_cache_len", 0)),
            "prefill_len": prefill_len,
            "is_prefill_step": is_prefill_step,
        }

    for req_id in delayed_req_ids:
        signal = signals[req_id]
        if hasattr(state_store, "mark_compression_skipped"):
            state_store.mark_compression_skipped(
                req_id=req_id,
                reason="compression_step_limited",
                step=signal.step,
            )
        events.append(
            {
                "req_id": req_id,
                "step": signal.step,
                "status": "skipped",
                "reason": "compression_step_limited",
                "cache_len_after": None,
                "details": {"max_compressions_per_step": max_compressions_per_step},
                **_event_common(signal),
            }
        )
    for req_id, signal in signal_items:
        # Guard against V1 batch-queue race: scheduler may emit consecutive
        # compression signals for the same request before update_from_output
        # runs.  The worker-side block table was already shrunk by the first
        # step, so executing again would desync scheduler/worker block counts.
        # Exception: during chunked prefill (scheduled_tokens > 1), each step
        # adds up to 2048 tokens, so consecutive compression is expected and
        # necessary to avoid excessive accumulation.
        req_state = state_store.get(req_id) if hasattr(state_store, "get") else None
        if req_state is not None:
            last_step = getattr(req_state, "last_compression_step", -1)
            compression_count = int(getattr(req_state, "compression_count", 0) or 0)
            sched_tokens = int(getattr(signal, "scheduled_tokens", 1))
            if (
                compression_count > 0
                and last_step >= 0
                and signal.step - last_step <= 1
                and sched_tokens <= 1
                and not bool(getattr(signal, "force", False))
            ):
                if log_decisions:
                    logger.debug(
                        "TriAttention compression skipped (batch-queue dedup) "
                        "req=%s step=%d last_compression_step=%d",
                        req_id, signal.step, last_step,
                    )
                if hasattr(state_store, "mark_compression_skipped"):
                    state_store.mark_compression_skipped(
                        req_id=req_id,
                        reason="batch_queue_dedup",
                        step=signal.step,
                    )
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "skipped",
                        "reason": "batch_queue_dedup",
                        "cache_len_after": None,
                        "details": {"last_compression_step": last_step},
                        **_event_common(signal),
                    }
                )
                continue
        if log_execution_path and not log_execution_path_core_only:
            logger.info(
                "TRIATTN_EXEC_PATH runner_executor_enter req=%s step=%d "
                "signal_reason=%s scheduled_tokens=%d estimated_cache_len=%d "
                "prefill_len=%d strict=%s",
                req_id,
                signal.step,
                getattr(signal, "reason", None),
                int(getattr(signal, "scheduled_tokens", 1)),
                int(getattr(signal, "estimated_cache_len", 0)),
                int(getattr(signal, "prefill_len", 0)),
                strict_no_downgrade,
            )
        try:
            result = executor.execute(
                req_id=req_id,
                signal=signal,
                scheduler_output=scheduler_output,
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            if strict_no_downgrade:
                logger.exception(
                    "TriAttention strict mode fatal: compression executor exception "
                    "req=%s step=%d",
                    req_id,
                    signal.step,
                )
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:executor_exception:"
                    f"req={req_id}:step={signal.step}:type={type(exc).__name__}"
                ) from exc
            if TRITON_SCORING_REQUIRED_MARKER in str(exc):
                logger.exception(
                    "TriAttention fatal: Triton scoring is required. "
                    "req=%s step=%d",
                    req_id,
                    signal.step,
                )
                raise
            state_store.mark_compression_skipped(
                req_id=req_id,
                reason=f"executor_exception:{type(exc).__name__}",
                step=signal.step,
            )
            logger.exception(
                "TriAttention compression executor failed req=%s step=%d",
                req_id,
                signal.step,
            )
            events.append(
                {
                    "req_id": req_id,
                    "step": signal.step,
                    "status": "error",
                    "reason": f"executor_exception:{type(exc).__name__}",
                    "cache_len_after": None,
                    **_event_common(signal),
                }
            )
            continue

        if log_execution_path and (
            not log_execution_path_core_only
            or result.applied
            or result.reason not in noisy_skip_reasons
        ):
            result_details = result.details if isinstance(result.details, dict) else {}
            selector_debug = result_details.get("selector_debug")
            execution_path = (
                selector_debug.get("execution_path")
                if isinstance(selector_debug, dict)
                else None
            )
            logger.info(
                "TRIATTN_EXEC_PATH runner_executor_result req=%s step=%d "
                "applied=%s reason=%s cache_len_after=%s path=%s",
                req_id,
                signal.step,
                result.applied,
                result.reason,
                result.cache_len_after,
                execution_path,
            )

        if (
            strict_no_downgrade
            and not result.applied
            and result.reason not in allowed_strict_skip_reasons
        ):
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:unexpected_skip:"
                f"req={req_id}:step={signal.step}:reason={result.reason}"
            )

        if result.applied:
            cache_len_after = (
                signal.estimated_cache_len
                if result.cache_len_after is None
                else result.cache_len_after
            )
            details = result.details if isinstance(result.details, dict) else {}
            before_len = details.get("effective_tokens_before")
            budget_total = details.get("budget_total")
            reclaimed_block_count = details.get("reclaimed_block_count")
            recent_unabsorbed_tokens = details.get("recent_unabsorbed_tokens")
            selector_status = details.get("selector_status")
            selector_debug = details.get("selector_debug") if log_selector_debug else None
            block_reclaim = details.get("block_reclaim")
            reclaim_mode = (
                block_reclaim.get("mode")
                if isinstance(block_reclaim, dict)
                else None
            )
            if log_worker_events:
                logger.info(
                    "TriAttention compression applied req=%s step=%d reason=%s "
                    "before=%s after=%d reclaimed_blocks=%s selector=%s reclaim=%s "
                    "selector_debug=%s",
                    req_id, signal.step, result.reason,
                    before_len, cache_len_after, reclaimed_block_count,
                    selector_status, reclaim_mode, selector_debug,
                )
            elif log_decisions:
                logger.debug(
                    "TriAttention compression applied req=%s step=%d reason=%s "
                    "before=%s after=%d reclaimed_blocks=%s selector=%s reclaim=%s "
                    "selector_debug=%s",
                    req_id, signal.step, result.reason,
                    before_len, cache_len_after, reclaimed_block_count,
                    selector_status, reclaim_mode, selector_debug,
                )
            # Resolve scheduler_nct for this request so state can record
            # the num_computed_tokens at compression time (used by
            # build_effective_sparse_overrides for stable delta).
            _sched_nct = None
            _cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if _cached_reqs is not None:
                _cr_ids = getattr(_cached_reqs, "req_ids", None)
                _cr_nct = getattr(_cached_reqs, "num_computed_tokens", None)
                if isinstance(_cr_ids, list) and isinstance(_cr_nct, list):
                    try:
                        _idx = _cr_ids.index(req_id)
                        _sched_nct = int(_cr_nct[_idx])
                    except (ValueError, IndexError):
                        pass
            state_store.mark_compressed(
                req_id=req_id,
                step=signal.step,
                cache_len=cache_len_after,
                scheduled_tokens=int(getattr(signal, "scheduled_tokens", 1)),
                scheduler_nct=_sched_nct,
            )
            if log_decisions:
                logger.debug(
                    "TriAttention compression applied req=%s step=%d reason=%s",
                    req_id,
                    signal.step,
                    result.reason,
                )
            if log_decisions and isinstance(before_len, int):
                logger.debug(
                    "TriAttention compression summary req=%s step=%d before=%d after=%d "
                    "budget=%s reclaimed_blocks=%s recent_unabsorbed=%s "
                    "scheduled_tokens=%s estimated_cache_len=%s reason=%s",
                    req_id,
                    signal.step,
                    before_len,
                    cache_len_after,
                    budget_total,
                    reclaimed_block_count,
                    recent_unabsorbed_tokens,
                    int(getattr(signal, "scheduled_tokens", 1)),
                    int(getattr(signal, "estimated_cache_len", 0)),
                    result.reason,
                )
            events.append(
                {
                    "req_id": req_id,
                    "step": signal.step,
                    "status": "applied",
                    "reason": result.reason,
                    "cache_len_after": cache_len_after,
                    "scheduler_nct": _sched_nct,
                    "details": result.details,
                    **_event_common(signal),
                    "block_reclaim": (
                        result.details.get("block_reclaim")
                        if isinstance(result.details, dict)
                        else None
                    ),
                }
            )
            continue

        state_store.mark_compression_skipped(
            req_id=req_id,
            reason=result.reason,
            step=signal.step,
        )
        details = result.details if isinstance(result.details, dict) else {}
        skip_logger = logger.debug if result.reason in noisy_skip_reasons else logger.info
        if logging_enabled and not (
            log_execution_path_core_only and result.reason in noisy_skip_reasons
        ):
            skip_logger(
                "TriAttention compression skipped req=%s step=%d reason=%s "
                "cache_len_after=%s details=%s",
                req_id,
                signal.step,
                result.reason,
                result.cache_len_after,
                details,
            )
        events.append(
            {
                "req_id": req_id,
                "step": signal.step,
                "status": "skipped",
                "reason": result.reason,
                "cache_len_after": result.cache_len_after,
                "details": details,
                **_event_common(signal),
            }
        )
    return events
