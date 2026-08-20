"""Group-level compaction pipeline orchestration for TriAttention runtime hook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .fast_recency_guard import uses_pure_fast_recency
from .layout_engine import execute_group_compaction, num_required_blocks
from .plan_models import PlacementPlan, ReclaimEvent, ReclaimGroup
from .selection_planner import prepare_group_layer_compactions
from .signals import CompressionSignal

try:
    from vllm.logger import logger as _runtime_logger
except Exception:  # pragma: no cover - fallback for lightweight tests
    _runtime_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupPipelineOutcome:
    cache_len_after: int
    selection_mode: str
    block_reclaim_groups: list[ReclaimGroup]
    mutable_block_ids_by_group: list[list[int] | None]
    reclaim_mode: str = "truncate_tail"
    selector_debug: dict[str, Any] | None = None


def _core_trace_enabled(config: TriAttentionRuntimeConfig) -> bool:
    return bool(
        getattr(config, "logging_enabled", True)
        and getattr(config, "log_execution_path", True)
        and getattr(config, "log_core_trace", False)
    )


def _selector_debug_enabled(config: TriAttentionRuntimeConfig) -> bool:
    return bool(
        getattr(config, "logging_enabled", True)
        and getattr(config, "log_execution_path", True)
        and getattr(config, "log_selector_debug", False)
    )


def _core_trace(
    config: TriAttentionRuntimeConfig,
    message: str,
    *args: Any,
) -> None:
    if not _core_trace_enabled(config):
        return
    _runtime_logger.info("TRIATTN_CORE_TRACE " + message, *args)


def normalize_mutable_block_ids_by_group(
    original_block_ids_by_group: Any,
) -> list[list[int] | None] | None:
    if not original_block_ids_by_group:
        return None
    if not isinstance(original_block_ids_by_group, (list, tuple)):
        return None
    mutable_block_ids_by_group: list[list[int] | None] = []
    for group_block_ids in original_block_ids_by_group:
        if not isinstance(group_block_ids, (list, tuple)):
            mutable_block_ids_by_group.append(None)
            continue
        mutable_block_ids_by_group.append([int(block_id) for block_id in group_block_ids])
    return mutable_block_ids_by_group


def try_build_recency_tail_block_remap(
    *,
    config: TriAttentionRuntimeConfig,
    mutable_block_ids_by_group: list[list[int] | None],
    effective_tokens: int,
    budget_total: int,
    block_size: int,
    compressible_group_ids: set[int] | None = None,
    retained_token_padding: int = 0,
) -> GroupPipelineOutcome | None:
    """Zero-copy fast path for recency-only, block-aligned tail retention.

    FAST_RECENCY_ONLY keeps the newest tokens. When the budget is an integral
    number of KV blocks, we can remap the request's block table to the newest
    physical blocks instead of copying KV tensors into the old prefix blocks.
    The effective length may be up to one block smaller than the nominal budget
    when the tail starts in the middle of a block, which is the price of keeping
    this path copy-free.
    """
    if not uses_pure_fast_recency(config):
        return None
    if not bool(getattr(config, "enable_zero_copy_recency", True)):
        return None
    if not bool(getattr(config, "enable_experimental_block_reclaim", False)):
        return None
    if bool(getattr(config, "protect_prefill", False)):
        return None
    if block_size <= 0 or budget_total <= 0:
        return None
    if budget_total % block_size != 0:
        return None

    budget_blocks = budget_total // block_size
    if budget_blocks <= 0:
        return None

    remapped_block_ids_by_group: list[list[int] | None] = []
    reclaim_groups: list[ReclaimGroup] = []
    cache_len_after: int | None = None
    remapped_any = False

    for gid, normalized_block_ids in enumerate(mutable_block_ids_by_group):
        if compressible_group_ids is not None and int(gid) not in compressible_group_ids:
            remapped_block_ids_by_group.append(normalized_block_ids)
            continue
        if not normalized_block_ids:
            remapped_block_ids_by_group.append(normalized_block_ids)
            continue
        group_capacity_tokens = len(normalized_block_ids) * block_size
        group_total_tokens = min(int(effective_tokens), group_capacity_tokens)
        if group_total_tokens <= int(budget_total):
            return None
        before_required = num_required_blocks(group_total_tokens, block_size)
        if before_required <= budget_blocks:
            return None
        if len(normalized_block_ids) < before_required:
            return None
        start_block = before_required - budget_blocks
        kept_tail_block_ids = list(normalized_block_ids[start_block:before_required])
        if len(kept_tail_block_ids) != budget_blocks:
            return None
        # Decode may allocate the current token's destination block before the
        # token is written into KV. Preserve those trailing blocks while still
        # remapping the already-computed KV tail, otherwise the first decode
        # compression opportunity degenerates into zero_copy_recency_not_ready.
        group_cache_len_after = group_total_tokens - start_block * block_size
        if group_cache_len_after <= 0 or group_cache_len_after > budget_total:
            return None
        retained_tokens = int(group_cache_len_after) + max(
            0,
            int(retained_token_padding),
        )
        retained_blocks = num_required_blocks(retained_tokens, block_size)
        trailing_block_ids = list(
            normalized_block_ids[
                before_required:min(len(normalized_block_ids), start_block + retained_blocks)
            ]
        )
        kept_block_ids = kept_tail_block_ids + trailing_block_ids
        if len(kept_block_ids) < retained_blocks:
            missing_blocks = retained_blocks - len(kept_block_ids)
            borrow_start = max(0, start_block - missing_blocks)
            borrowed_slack_block_ids = list(normalized_block_ids[borrow_start:start_block])
            kept_block_ids += borrowed_slack_block_ids
        kept_block_id_set = set(kept_block_ids)
        removed_block_ids = [
            block_id for block_id in normalized_block_ids
            if block_id not in kept_block_id_set
        ]
        if cache_len_after is None:
            cache_len_after = int(group_cache_len_after)
        elif cache_len_after != int(group_cache_len_after):
            return None
        remapped_block_ids_by_group.append(kept_block_ids)
        if removed_block_ids:
            reclaim_groups.append(
                ReclaimGroup(
                    gid=gid,
                    block_ids_before=list(normalized_block_ids),
                    block_ids_after=kept_block_ids,
                    block_ids_removed=removed_block_ids,
                )
            )
        remapped_any = True

    if not remapped_any or cache_len_after is None or not reclaim_groups:
        return None

    return GroupPipelineOutcome(
        cache_len_after=int(cache_len_after),
        selection_mode="zero_copy_tail",
        block_reclaim_groups=reclaim_groups,
        mutable_block_ids_by_group=remapped_block_ids_by_group,
        reclaim_mode="remap_tail",
        selector_debug=(
            {
                "execution_path": "worker_hook>zero_copy_recency_tail",
                "score_backend": "recency_only",
            }
            if _selector_debug_enabled(config)
            else None
        ),
    )


def run_group_compaction_pipeline(
    *,
    req_id: str,
    signal: CompressionSignal,
    config: TriAttentionRuntimeConfig,
    strict_triton_required: bool,
    num_computed_tokens: int,
    effective_tokens: int,
    budget_total: int,
    block_size: int,
    mutable_block_ids_by_group: list[list[int] | None],
    group_tensors: dict[int, list[tuple[int, Any]]],
    select_keep_indices: Callable[..., dict[str, Any] | None] | None,
    select_keep_indices_for_group: Callable[..., dict[str, Any] | None] | None,
    shared_compact_fn: Callable[..., Any],
    per_head_compact_fn: Callable[..., Any],
    compressible_group_ids: set[int] | None = None,
    gather_dense_fn: Callable[..., torch.Tensor] | None = None,
    retained_token_padding: int = 0,
) -> GroupPipelineOutcome | dict[str, Any]:
    compacted_any_group = False
    cache_len_after: int | None = None
    expected_cache_len_after: int | None = None
    selection_mode = "fallback"
    block_reclaim_groups: list[ReclaimGroup] = []
    selector_debug_enabled = _selector_debug_enabled(config)
    selector_debug_by_group: list[dict[str, Any]] = []
    step = int(getattr(signal, "step", 0))
    _core_trace(
        config,
        "enter run_group_compaction_pipeline req=%s step=%d groups=%d "
        "group_tensor_groups=%d effective_tokens=%d budget_total=%d block_size=%d "
        "num_computed_tokens=%d strict_triton_required=%s",
        req_id,
        step,
        len(mutable_block_ids_by_group),
        len(group_tensors),
        int(effective_tokens),
        int(budget_total),
        int(block_size),
        int(num_computed_tokens),
        bool(strict_triton_required),
    )

    for gid, normalized_block_ids in enumerate(mutable_block_ids_by_group):
        if compressible_group_ids is not None and int(gid) not in compressible_group_ids:
            _core_trace(
                config,
                "skip run_group_compaction_pipeline_group req=%s step=%d gid=%d "
                "reason=noncompressible_group",
                req_id,
                step,
                int(gid),
            )
            continue
        if not normalized_block_ids:
            _core_trace(
                config,
                "skip run_group_compaction_pipeline_group req=%s step=%d gid=%d "
                "reason=empty_block_ids",
                req_id,
                step,
                int(gid),
            )
            continue
        layer_tensors = group_tensors.get(gid)
        if not layer_tensors:
            _core_trace(
                config,
                "skip run_group_compaction_pipeline_group req=%s step=%d gid=%d "
                "reason=no_layer_tensors block_ids=%d",
                req_id,
                step,
                int(gid),
                len(normalized_block_ids),
            )
            continue
        group_capacity_tokens = len(normalized_block_ids) * block_size
        group_total_tokens = min(effective_tokens, group_capacity_tokens)
        if group_total_tokens <= 0:
            _core_trace(
                config,
                "skip run_group_compaction_pipeline_group req=%s step=%d gid=%d "
                "reason=non_positive_tokens capacity_tokens=%d effective_tokens=%d",
                req_id,
                step,
                int(gid),
                int(group_capacity_tokens),
                int(effective_tokens),
            )
            continue
        group_prefill_len = min(int(signal.prefill_len), group_total_tokens)
        group_budget_total = min(budget_total, group_total_tokens)
        round_start = int(max(0, num_computed_tokens))
        group_cache_len_after: int | None = None
        _core_trace(
            config,
            "enter prepare_group_layer_compactions req=%s step=%d gid=%d "
            "layers=%d total_tokens=%d budget_total=%d prefill_len=%d "
            "round_start=%d",
            req_id,
            step,
            int(gid),
            len(layer_tensors),
            int(group_total_tokens),
            int(group_budget_total),
            int(group_prefill_len),
            int(round_start),
        )
        try:
            group_selection = prepare_group_layer_compactions(
                req_id=req_id,
                gid=gid,
                layer_tensors=layer_tensors,
                normalized_block_ids=normalized_block_ids,
                block_size=block_size,
                group_total_tokens=group_total_tokens,
                group_prefill_len=group_prefill_len,
                protect_prefill=signal.protect_prefill,
                round_start=round_start,
                group_budget_total=group_budget_total,
                config=config,
                strict_triton_required=strict_triton_required,
                select_keep_indices=select_keep_indices,
                select_keep_indices_for_group=select_keep_indices_for_group,
                gather_dense_fn=gather_dense_fn,
            )
        except ValueError as exc:
            if str(exc) == "prefill_exceeds_budget":
                _core_trace(
                    config,
                    "exit prepare_group_layer_compactions req=%s step=%d gid=%d "
                    "status=error reason=prefill_exceeds_budget",
                    req_id,
                    step,
                    int(gid),
                )
                _core_trace(
                    config,
                    "exit run_group_compaction_pipeline req=%s step=%d "
                    "applied=False reason=prefill_exceeds_budget",
                    req_id,
                    step,
                )
                return {"applied": False, "reason": "prefill_exceeds_budget"}
            _core_trace(
                config,
                "exit prepare_group_layer_compactions req=%s step=%d gid=%d "
                "status=error error=%s",
                req_id,
                step,
                int(gid),
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            _core_trace(
                config,
                "exit prepare_group_layer_compactions req=%s step=%d gid=%d "
                "status=error error=%s",
                req_id,
                step,
                int(gid),
                type(exc).__name__,
            )
            raise
        prepared_layer_compactions = group_selection.tasks
        selection_mode = group_selection.selection_mode
        if selector_debug_enabled:
            _core_trace(
                config,
                "exit prepare_group_layer_compactions req=%s step=%d gid=%d "
                "tasks=%d selection_mode=%s selector_debug=%s",
                req_id,
                step,
                int(gid),
                len(prepared_layer_compactions),
                group_selection.selection_mode,
                group_selection.selector_debug,
            )
            selector_debug_by_group.append(
                {
                    "gid": gid,
                    "selection_mode": str(group_selection.selection_mode),
                    "group_total_tokens": int(group_total_tokens),
                    "group_budget_total": int(group_budget_total),
                    "layer_count": len(prepared_layer_compactions),
                    "selector": group_selection.selector_debug,
                }
            )
        else:
            _core_trace(
                config,
                "exit prepare_group_layer_compactions req=%s step=%d gid=%d "
                "tasks=%d selection_mode=%s",
                req_id,
                step,
                int(gid),
                len(prepared_layer_compactions),
                group_selection.selection_mode,
            )

        _core_trace(
            config,
            "enter execute_group_compaction req=%s step=%d gid=%d tasks=%d "
            "block_ids=%d total_tokens=%d block_reclaim=%s require_physical=%s",
            req_id,
            step,
            int(gid),
            len(prepared_layer_compactions),
            len(normalized_block_ids),
            int(group_total_tokens),
            bool(config.enable_experimental_block_reclaim),
            bool(config.require_physical_reclaim),
        )
        try:
            group_outcome = execute_group_compaction(
                req_id=req_id,
                gid=gid,
                normalized_block_ids=normalized_block_ids,
                tasks=prepared_layer_compactions,
                block_size=block_size,
                total_tokens=group_total_tokens,
                retained_token_padding=retained_token_padding,
                enable_experimental_block_reclaim=config.enable_experimental_block_reclaim,
                require_physical_reclaim=config.require_physical_reclaim,
                shared_compact_fn=shared_compact_fn,
                per_head_compact_fn=per_head_compact_fn,
            )
        except Exception as exc:
            first_layer_idx = (
                prepared_layer_compactions[0].layer_idx if prepared_layer_compactions else -1
            )
            _core_trace(
                config,
                "exit execute_group_compaction req=%s step=%d gid=%d "
                "status=error first_layer=%s error=%s",
                req_id,
                step,
                int(gid),
                first_layer_idx,
                type(exc).__name__,
            )
            _core_trace(
                config,
                "exit run_group_compaction_pipeline req=%s step=%d "
                "applied=False reason=compaction_failed:g%d:l%s:%s",
                req_id,
                step,
                int(gid),
                first_layer_idx,
                type(exc).__name__,
            )
            return {
                "applied": False,
                "reason": f"compaction_failed:g{gid}:l{first_layer_idx}:{type(exc).__name__}",
            }
        _core_trace(
            config,
            "exit execute_group_compaction req=%s step=%d gid=%d "
            "layer_results=%d kept_blocks=%d reclaim_group=%s",
            req_id,
            step,
            int(gid),
            len(group_outcome.layer_results),
            len(group_outcome.kept_block_ids),
            group_outcome.reclaim_group is not None,
        )

        for layer_idx, layer_compaction in group_outcome.layer_results:
            layer_cache_len_after = int(layer_compaction.cache_len_after)
            if expected_cache_len_after is None:
                expected_cache_len_after = layer_cache_len_after
            elif layer_cache_len_after != expected_cache_len_after:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:inconsistent_cache_len_after:"
                    f"req={req_id}:gid={gid}:layer={layer_idx}:"
                    f"expected={expected_cache_len_after}:actual={layer_cache_len_after}"
                )
            cache_len_after = layer_cache_len_after
            if group_cache_len_after is None:
                group_cache_len_after = layer_cache_len_after
            compacted_any_group = True

        if config.enable_experimental_block_reclaim and group_cache_len_after is not None:
            mutable_block_ids_by_group[gid] = list(group_outcome.kept_block_ids)
            if group_outcome.reclaim_group is not None:
                block_reclaim_groups.append(group_outcome.reclaim_group)

    if not compacted_any_group or cache_len_after is None:
        _core_trace(
            config,
            "exit run_group_compaction_pipeline req=%s step=%d "
            "applied=False reason=no_compactable_groups",
            req_id,
            step,
        )
        return {"applied": False, "reason": "no_compactable_groups"}

    selector_debug: dict[str, Any] | None = None
    if selector_debug_enabled:
        selector_debug = {
            "execution_path": "worker_hook>group_pipeline>selector_hf>layout_compaction",
            "groups": selector_debug_by_group,
        }

    outcome = GroupPipelineOutcome(
        cache_len_after=int(cache_len_after),
        selection_mode=str(selection_mode),
        block_reclaim_groups=block_reclaim_groups,
        mutable_block_ids_by_group=mutable_block_ids_by_group,
        selector_debug=selector_debug,
    )
    _core_trace(
        config,
        "exit run_group_compaction_pipeline req=%s step=%d applied=True "
        "selection_mode=%s cache_len_after=%d reclaim_groups=%d",
        req_id,
        step,
        outcome.selection_mode,
        int(outcome.cache_len_after),
        len(outcome.block_reclaim_groups),
    )
    return outcome


def finalize_hook_placement_result(
    *,
    req_state: Any,
    original_block_ids_by_group: Any,
    config: TriAttentionRuntimeConfig,
    selector_status: str,
    outcome: GroupPipelineOutcome,
    effective_tokens: int,
    budget_total: int,
    recent_unabsorbed_tokens: int | None,
    retained_cache_len: int | None = None,
) -> dict[str, Any]:
    block_reclaim_payload: ReclaimEvent | None = None
    if config.enable_experimental_block_reclaim and outcome.block_reclaim_groups:
        reassigned_block_ids = []
        for idx, group_block_ids in enumerate(outcome.mutable_block_ids_by_group):
            if group_block_ids is None:
                reassigned_block_ids.append(original_block_ids_by_group[idx])
            else:
                reassigned_block_ids.append(group_block_ids)
        req_state.block_ids = (
            tuple(reassigned_block_ids)
            if isinstance(original_block_ids_by_group, tuple)
            else reassigned_block_ids
        )
        block_reclaim_payload = ReclaimEvent(
            mode=outcome.reclaim_mode,
            groups=outcome.block_reclaim_groups,
        )

    placement_plan = PlacementPlan(
        cache_len_after=int(outcome.cache_len_after),
        selector_status=str(selector_status),
        selection_mode=str(outcome.selection_mode),
        effective_tokens_before=int(effective_tokens),
        budget_total=int(budget_total),
        retained_cache_len=(
            int(retained_cache_len)
            if isinstance(retained_cache_len, int)
            else None
        ),
        recent_unabsorbed_tokens=(
            int(recent_unabsorbed_tokens)
            if isinstance(recent_unabsorbed_tokens, int)
            else None
        ),
        block_reclaim=block_reclaim_payload,
        selector_debug=outcome.selector_debug,
    )
    return placement_plan.to_hook_result_dict()
