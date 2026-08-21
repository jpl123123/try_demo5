"""Per-request compression engine for kvpress-ascend.

The Ascend conversion of kvpress's ``BasePress.forward_hook`` + 
``ScorerPress.compress`` flow:

1. gather dense K (per layer) from the Ascend block cache (HF DynamicCache
   read equivalent);
2. run the press scoring bridge with the latest captured queries (HF attention
   weights equivalent);
3. compact every compressible layer in place (per-head keep sets, uniform
   count) — the physical "new compressed cache tensor" step;
4. shrink the block row and report the reclaim plan for the scheduler.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from ..core.kv_layout import (
    compact_request_kv_in_place_per_head,
    gather_request_k_dense,
    gather_request_kv_dense,
)
from ..core.press_bridge import select_keep_indices
from ..logging_control import log_debug, log_info, log_warning, probe
from .group_resolver import resolve_group_tensors


def _debug_disable_compaction() -> bool:
    return os.environ.get("KVPRESS_DEBUG_DISABLE_COMPACTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _table_block_size(base_runner: Any) -> int:
    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    if block_size <= 0:
        block_size = int(os.environ.get("KVPRESS_BLOCK_SIZE_HINT", "0") or 0)
    return block_size if block_size > 0 else 128


def _request_block_ids(base_runner: Any, req_id: str, gid: int = 0) -> Optional[list[int]]:
    """Resolve the request's block-id row from the worker block table."""
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        return None
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        return None
    req_index = req_id_to_index.get(req_id)
    if not isinstance(req_index, int):
        return None

    inner = getattr(block_table_obj, "block_tables", None)
    tables = list(inner) if isinstance(inner, (list, tuple)) and inner else [block_table_obj]
    if gid < 0 or gid >= len(tables):
        gid = 0
    table = tables[gid]
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if isinstance(block_table_np, np.ndarray) and block_table_np.ndim == 2:
        row = block_table_np[req_index]
        num_blocks = getattr(table, "num_blocks_per_row", None)
        if isinstance(num_blocks, np.ndarray) and 0 <= req_index < num_blocks.shape[0]:
            row = row[: int(num_blocks[req_index])]
        return [int(x) for x in row if int(x) >= 0]
    return None


def _request_block_capacity(base_runner: Any, req_id: str) -> Optional[int]:
    """Authoritative worker-side KV length: the request's block-table row
    capacity (TriAttention's ``_get_actual_kv_from_model_runner``)."""
    input_batch = getattr(base_runner, "input_batch", None)
    if input_batch is None:
        return None
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        return None
    req_index = req_id_to_index.get(req_id)
    if not isinstance(req_index, int):
        return None
    block_table_obj = getattr(input_batch, "block_table", None)
    if block_table_obj is None:
        return None
    inner = getattr(block_table_obj, "block_tables", None)
    tables = list(inner) if isinstance(inner, (list, tuple)) and inner else [block_table_obj]
    capacities: list[int] = []
    for table in tables:
        try:
            n_blocks = int(table.num_blocks_per_row[req_index])
        except Exception:
            continue
        try:
            bs = int(table.block_size)
        except Exception:
            continue
        if n_blocks > 0 and bs > 0:
            capacities.append(n_blocks * bs)
    return max(capacities) if capacities else None


def _request_token_count(base_runner: Any, req_id: str, block_size: int) -> Optional[int]:
    """Current KV token count: block capacity first (authoritative), then the
    input-batch num_computed mirror, then the request state (TriAttention
    philosophy: the worker self-derives the length)."""
    capacity = _request_block_capacity(base_runner, req_id)
    if capacity is not None and capacity > 0:
        return capacity
    input_batch = getattr(base_runner, "input_batch", None)
    if input_batch is not None:
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if isinstance(req_id_to_index, dict):
            req_index = req_id_to_index.get(req_id)
            if isinstance(req_index, int):
                num_computed = getattr(input_batch, "num_computed_tokens_cpu", None)
                if num_computed is not None:
                    try:
                        value = int(num_computed[req_index])
                        if value > 0:
                            return value
                    except Exception:
                        pass
    requests_dict = getattr(base_runner, "requests", None)
    if isinstance(requests_dict, dict):
        req_state = requests_dict.get(req_id)
        if req_state is not None:
            try:
                value = int(getattr(req_state, "num_computed_tokens", 0) or 0)
                if value > 0:
                    return value
            except Exception:
                pass
    block_ids = _request_block_ids(base_runner, req_id)
    if block_ids:
        return len(block_ids) * block_size
    return None


def compress_request(
    *,
    base_runner: Any,
    req_id: str,
    keep_count: int,
    total_tokens: int,
    block_size: int,
    press: Any,
    hooks: Any,
    max_layers_to_score: int = 0,
    min_reclaim_blocks: int = 1,
    scheduled_tokens: int = 1,
    prefer_installed_kvpress: bool = True,
) -> dict[str, Any]:
    """Run one kvpress compression for a request.

    Returns an event dict (``status`` applied/skipped) consumed by the runner
    proxy and forwarded to the scheduler for block reclaim.
    """
    if _debug_disable_compaction():
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "compaction_disabled_debug",
            "cache_len_after": total_tokens,
        }
    if keep_count <= 0 or keep_count >= total_tokens:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "under_budget" if keep_count >= total_tokens else "empty_keep",
            "cache_len_after": total_tokens,
        }

    group_tensors = resolve_group_tensors(base_runner)
    if not group_tensors:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "no_kv_tensors",
            "cache_len_after": total_tokens,
        }

    # retained physical length: kept prefix + room for this step's tokens.
    retained_cache_len = keep_count + max(1, int(scheduled_tokens))
    required_blocks = (retained_cache_len + block_size - 1) // block_size

    reclaimed_by_group: dict[int, dict[str, Any]] = {}
    total_reclaimed = 0
    compacted_layers = 0
    applied_any = False
    selector_debug: dict[str, Any] = {}

    for gid, layer_tensors in group_tensors.items():
        block_ids = _request_block_ids(base_runner, req_id, gid)
        if not block_ids:
            continue
        current_blocks = len(block_ids)
        reclaimable = current_blocks - required_blocks
        reclaimed_by_group[gid] = {
            "gid": gid,
            "block_ids_before": list(block_ids),
            "block_ids_after": list(block_ids[:required_blocks]),
            "required_blocks": required_blocks,
            "reclaimable_blocks": max(0, reclaimable),
            "scored_layers": 0,
            "total_layers": len(layer_tensors),
        }
        if reclaimable <= 0:
            continue

        # Score a sampled subset of layers when the request is long; every
        # layer must still be COMPACTED (shared block row), so un-scored
        # layers reuse the keep set of the nearest scored layer.
        layer_items = list(layer_tensors)
        scored_items = list(layer_items)
        if max_layers_to_score > 0 and len(scored_items) > max_layers_to_score:
            scored_items = _sample_layers(scored_items, max_layers_to_score)
        keep_by_scored_idx: dict[int, torch.Tensor] = {}
        for layer_idx, kv_cache in scored_items:
            try:
                keys, _values = gather_request_kv_dense(
                    kv_cache, block_ids, block_size, total_tokens
                )
                queries = hooks.get_query(layer_idx) if hooks is not None else None
                keep_by_scored_idx[int(layer_idx)] = select_keep_indices(
                    press,
                    keys,
                    keep_count,
                    queries=queries,
                )
            except Exception as exc:  # pragma: no cover - per-layer safety
                log_warning(
                    "compression scoring failed req=%s gid=%d layer=%d: %s: %s",
                    req_id, gid, layer_idx, type(exc).__name__, exc,
                )
        if not keep_by_scored_idx:
            continue

        def _keep_for_layer(layer_idx: int) -> torch.Tensor:
            keep = keep_by_scored_idx.get(int(layer_idx))
            if keep is not None:
                return keep
            nearest = min(
                keep_by_scored_idx.keys(),
                key=lambda scored_idx: abs(int(scored_idx) - int(layer_idx)),
            )
            return keep_by_scored_idx[nearest]

        scored_layers = 0
        for layer_idx, kv_cache in layer_items:
            try:
                keep_tensor = _keep_for_layer(layer_idx)
                compact_request_kv_in_place_per_head(
                    kv_cache,
                    block_ids,
                    block_size,
                    keep_tensor,
                    total_tokens,
                    preserve_dropped_tokens=False,
                    prefix_only=True,
                )
                compacted_layers += 1
                scored_layers += 1
                selector_debug[f"layer_{layer_idx}"] = {
                    "scores_from": (
                        "keys_only" if hooks is None or hooks.get_query(layer_idx) is None else "keys+queries"
                    ),
                    "keep": int(keep_count),
                }
            except Exception as exc:  # pragma: no cover - per-layer safety
                log_warning(
                    "compression layer failed req=%s gid=%d layer=%d: %s: %s",
                    req_id, gid, layer_idx, type(exc).__name__, exc,
                )
        if scored_layers > 0:
            applied_any = True
        reclaimed_by_group[gid]["scored_layers"] = scored_layers
        total_reclaimed += max(0, reclaimable)

    if not applied_any:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "no_layers_scored",
            "cache_len_after": total_tokens,
            "details": {"group_reclaim": reclaimed_by_group},
        }
    if min_reclaim_blocks > 0 and total_reclaimed < min_reclaim_blocks:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "below_min_reclaim",
            "cache_len_after": total_tokens,
            "details": {
                "group_reclaim": reclaimed_by_group,
                "reclaimed_blocks": total_reclaimed,
                "min_reclaim_blocks": min_reclaim_blocks,
            },
        }

    log_info(
        "compression applied req=%s press=%s before=%d kept=%d retained=%d "
        "reclaimed_blocks=%d groups=%d layers_compacted=%d",
        req_id,
        press.__class__.__name__,
        total_tokens,
        keep_count,
        retained_cache_len,
        total_reclaimed,
        len(group_tensors),
        compacted_layers,
    )
    probe(
        "COMPRESS req=%s press=%s before=%d after=%d retained=%d "
        "reclaimed_blocks=%d groups=%d layers_compacted=%d",
        req_id,
        press.__class__.__name__,
        total_tokens,
        keep_count,
        retained_cache_len,
        total_reclaimed,
        len(group_tensors),
        compacted_layers,
    )
    reclaim_groups = [
        dict(reclaimed_by_group[gid])
        for gid in sorted(reclaimed_by_group)
        if int(reclaimed_by_group[gid].get("scored_layers", 0) or 0) > 0
    ]
    return {
        "req_id": req_id,
        "status": "applied",
        "reason": "kvpress_compaction",
        "cache_len_after": keep_count,
        "effective_cache_len_after": keep_count,
        "retained_cache_len": retained_cache_len,
        "details": {
            "press": press.__class__.__name__,
            "selector": "ascend_native",
            "effective_tokens_before": total_tokens,
            "keep_count": keep_count,
            "retained_cache_len": retained_cache_len,
            "reclaimed_block_count": total_reclaimed,
            "block_reclaim": {
                "mode": "truncate_tail",
                "groups": reclaim_groups,
            },
            "selector_debug": selector_debug,
        },
    }


def _sample_layers(
    layer_items: list[tuple[int, Any]],
    limit: int,
) -> list[tuple[int, Any]]:
    if len(layer_items) <= limit:
        return layer_items
    indices = {
        int(round(i * (len(layer_items) - 1) / (limit - 1)))
        for i in range(limit)
    }
    return [layer_items[i] for i in sorted(indices)]


def gather_layer_k_for_debug(
    base_runner: Any,
    req_id: str,
    layer_idx: int,
    gid: int = 0,
    total_tokens: int = 0,
) -> Optional[torch.Tensor]:
    """Debug helper: dense K of one layer for one request."""
    group_tensors = resolve_group_tensors(base_runner)
    layer_tensors = group_tensors.get(int(gid))
    if not layer_tensors:
        return None
    block_ids = _request_block_ids(base_runner, req_id, gid)
    block_size = _table_block_size(base_runner)
    if not block_ids:
        return None
    for lidx, kv_cache in layer_tensors:
        if int(lidx) == int(layer_idx):
            if total_tokens <= 0:
                total_tokens = len(block_ids) * block_size
            return gather_request_k_dense(kv_cache, block_ids, block_size, total_tokens)
    return None
