"""Runtime length/guard context preparation for TriAttention runtime hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .prefill_phase import is_prefill_phase_for_limit
from .request_key_compat import get_scheduled_token_items
from .signals import CompressionSignal
from .thresholds import (
    compression_reclaim_interval_tokens,
    should_defer_initial_decode_compression,
)


def effective_budget_for_signal(
    config: TriAttentionRuntimeConfig,
    signal: CompressionSignal,
    total_tokens: int,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return min(total_tokens, budget)


def _resolve_estimated_effective_tokens(
    *,
    signal: CompressionSignal,
    req_runtime_state: Any,
) -> int:
    if req_runtime_state is not None:
        compression_count = getattr(req_runtime_state, "compression_count", None)
        current_cache_len = getattr(req_runtime_state, "current_cache_len", None)
        if (
            isinstance(compression_count, int)
            and compression_count > 0
            and isinstance(current_cache_len, int)
            and current_cache_len > 0
        ):
            return max(0, int(current_cache_len))
    return max(0, int(getattr(signal, "estimated_cache_len", 0)))


def effective_len_guard_upper(
    config: TriAttentionRuntimeConfig,
    signal: CompressionSignal,
) -> int:
    budget = config.kv_budget
    if signal.protect_prefill and not config.include_prefill_in_budget:
        budget += max(signal.prefill_len, 0)
    return budget + max(1, config.effective_len_guard_divide_multiples) * max(
        1,
        config.divide_length,
    )


def scheduled_tokens_for_req(scheduler_output: Any, req_id: str) -> int:
    try:
        cached = getattr(scheduler_output, "_triattention_cached_scheduled_tokens_by_req_id", None)
    except Exception:
        cached = None
    if isinstance(cached, dict):
        value = cached.get(req_id)
        if isinstance(value, int):
            return max(1, value)
    by_req_id: dict[str, int] = {}
    for _raw_key, key_req_id, scheduled_tokens in get_scheduled_token_items(scheduler_output):
        by_req_id[key_req_id] = int(scheduled_tokens)
    try:
        setattr(scheduler_output, "_triattention_cached_scheduled_tokens_by_req_id", by_req_id)
    except Exception:
        pass
    if req_id in by_req_id:
        return max(1, int(by_req_id[req_id]))
    return 1


def min_block_capacity_tokens(
    block_ids_by_group: Any,
    block_size: int,
    group_ids: set[int] | None = None,
) -> int | None:
    if block_size <= 0:
        return None
    if not isinstance(block_ids_by_group, (list, tuple)):
        return None
    capacities: list[int] = []
    for gid, group_block_ids in enumerate(block_ids_by_group):
        if group_ids is not None and int(gid) not in group_ids:
            continue
        if not isinstance(group_block_ids, (list, tuple)):
            continue
        capacities.append(len(group_block_ids) * block_size)
    if not capacities:
        return None
    return min(capacities)


def _is_ascend_runner(base_runner: Any) -> bool:
    module_name = type(base_runner).__module__
    if isinstance(module_name, str) and module_name.startswith("vllm_ascend."):
        return True
    return "vllm_ascend" in repr(type(base_runner))


@dataclass(frozen=True)
class HookRuntimeContext:
    scheduled_tokens: int
    num_computed_tokens: int
    estimated_effective_tokens: int
    effective_tokens: int
    budget_total: int
    recent_unabsorbed_tokens: int | None
    should_defer_recompress: bool
    defer_reason: str | None = None


def build_hook_runtime_context(
    *,
    base_runner: Any,
    config: TriAttentionRuntimeConfig,
    req_id: str,
    req_state: Any,
    req_runtime_state: Any,
    signal: CompressionSignal,
    scheduler_output: Any,
    compressed_once: set[str],
    original_block_ids_by_group: Any,
    block_size_hint: int,
    compressible_group_ids: set[int] | None = None,
) -> HookRuntimeContext:
    block_capacity_hint = min_block_capacity_tokens(
        block_ids_by_group=original_block_ids_by_group,
        block_size=block_size_hint,
        group_ids=compressible_group_ids,
    )

    scheduled_tokens = scheduled_tokens_for_req(
        scheduler_output=scheduler_output,
        req_id=req_id,
    )
    signal_scheduled_tokens = max(
        1,
        int(getattr(signal, "scheduled_tokens", 1) or 1),
    )
    scheduled_tokens = max(scheduled_tokens, signal_scheduled_tokens)
    num_computed_tokens = int(getattr(req_state, "num_computed_tokens", 0))
    estimated_effective_tokens = _resolve_estimated_effective_tokens(
        signal=signal,
        req_runtime_state=req_runtime_state,
    )

    _post_forward = getattr(signal, '_post_forward', False)
    if _post_forward:
        # Post-forward mode (prefill): KV cache already contains the
        # current chunk's tokens, so use the full estimate directly.
        effective_tokens = max(0, estimated_effective_tokens)
    else:
        # Pre-forward mode (decode): subtract tokens not yet in KV.
        effective_tokens = max(0, estimated_effective_tokens - max(0, scheduled_tokens))
    # Cap at the actual number of tokens in KV.
    kv_upper = num_computed_tokens + (scheduled_tokens if _post_forward else 0)
    if effective_tokens > kv_upper:
        effective_tokens = kv_upper

    if isinstance(block_capacity_hint, int):
        physical_upper = block_capacity_hint + block_size_hint
        if effective_tokens > physical_upper:
            effective_tokens = physical_upper
    physical_capacity_boundary_hit = (
        isinstance(block_capacity_hint, int)
        and block_size_hint > 0
        and not _post_forward
        and bool(getattr(signal, "force", False))
        and (effective_tokens + max(1, scheduled_tokens)) >= block_capacity_hint
    )

    recent_unabsorbed_tokens: int | None = None
    if req_runtime_state is not None:
        baseline = int(getattr(req_runtime_state, "last_absorbed_cache_len", 0))
        recent_unabsorbed_tokens = max(0, effective_tokens - baseline)
        setattr(
            base_runner,
            "_triattention_active_recent_unabsorbed_tokens",
            recent_unabsorbed_tokens,
        )
    else:
        setattr(base_runner, "_triattention_active_recent_unabsorbed_tokens", None)

    prefill_len = int(getattr(signal, "prefill_len", 0) or 0)
    is_prefill_step = is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id=req_id,
        scheduled_tokens=scheduled_tokens,
        prefill_len=prefill_len,
        num_computed_tokens=num_computed_tokens,
    )
    prefill_tokens_after_step = num_computed_tokens + (
        scheduled_tokens if _post_forward else 0
    )
    if _post_forward:
        prefill_tokens_after_step = max(
            prefill_tokens_after_step,
            estimated_effective_tokens,
        )
    prefill_incomplete = (
        prefill_len > 0
        and is_prefill_step
        and prefill_tokens_after_step < prefill_len
    )

    defer_for_prefill = (
        config.enable_experimental_kv_compaction
        and prefill_incomplete
        and (
            bool(getattr(config, "defer_prefill_compression", False))
            or (
                bool(getattr(config, "defer_prefill_compression_on_ascend", False))
                and _is_ascend_runner(base_runner)
            )
        )
    )
    is_ascend = _is_ascend_runner(base_runner)
    prefill_compression_count = (
        int(getattr(req_runtime_state, "compression_count", 0) or 0)
        if req_runtime_state is not None
        else 0
    )
    prefill_limit_hit = (
        config.enable_experimental_kv_compaction
        and is_prefill_step
        and is_ascend
        and prefill_compression_count
        >= int(getattr(config, "prefill_max_compressions_on_ascend", 1) or 0)
    )
    initial_decode_grace_hit = (
        config.enable_experimental_kv_compaction
        and should_defer_initial_decode_compression(
            config=config,
            effective_tokens=estimated_effective_tokens,
            prefill_len=prefill_len,
            is_ascend=is_ascend,
            is_prefill_step=is_prefill_step,
            compressed_once=(
                bool(prefill_compression_count)
                or req_id in compressed_once
            ),
        )
    )
    if (
        config.fail_on_effective_len_regression
        and config.enable_experimental_block_reclaim
        and req_id in compressed_once
        and not prefill_incomplete
    ):
        guard_upper = effective_len_guard_upper(config, signal)
        estimated_slack = max(1, int(getattr(signal, "estimated_cache_len", 0)) - num_computed_tokens)
        regression_slack = block_size_hint + estimated_slack + max(1, scheduled_tokens)
        if (
            effective_tokens > (guard_upper + regression_slack)
            and num_computed_tokens > (guard_upper + regression_slack)
            and effective_tokens >= int(config.effective_len_regression_ratio * num_computed_tokens)
        ):
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:effective_len_regressed:"
                f"req={req_id}:effective_tokens={effective_tokens}:"
                f"num_computed_tokens={num_computed_tokens}:guard_upper={guard_upper}"
            )

    budget_total = effective_budget_for_signal(config, signal, effective_tokens)
    local_length_threshold = budget_total + compression_reclaim_interval_tokens(
        config,
        block_size=block_size_hint,
        is_ascend=is_ascend,
        is_prefill_step=is_prefill_step,
    )
    length_gate_hit = estimated_effective_tokens >= local_length_threshold
    kv_override = str(getattr(signal, "reason", "")) == "kv_usage_threshold"
    should_defer_recompress = (
        not physical_capacity_boundary_hit
        and (
            defer_for_prefill
            or prefill_limit_hit
            or initial_decode_grace_hit
            or (
                config.enable_experimental_kv_compaction
                and req_id in compressed_once
                and not kv_override
                and not length_gate_hit
            )
        )
    )
    defer_reason = None
    if defer_for_prefill:
        defer_reason = "prefill_incomplete"
    elif prefill_limit_hit:
        defer_reason = "prefill_compression_limit"
    elif initial_decode_grace_hit:
        defer_reason = "initial_decode_grace"
    return HookRuntimeContext(
        scheduled_tokens=int(scheduled_tokens),
        num_computed_tokens=int(num_computed_tokens),
        estimated_effective_tokens=int(estimated_effective_tokens),
        effective_tokens=int(effective_tokens),
        budget_total=int(budget_total),
        recent_unabsorbed_tokens=(
            int(recent_unabsorbed_tokens)
            if isinstance(recent_unabsorbed_tokens, int)
            else None
        ),
        should_defer_recompress=bool(should_defer_recompress),
        defer_reason=defer_reason,
    )
