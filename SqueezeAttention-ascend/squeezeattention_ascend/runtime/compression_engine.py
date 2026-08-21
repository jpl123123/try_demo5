"""Per-request compression engine for SqueezeAttention-ascend.

The Ascend conversion of SqueezeAttention's per-layer streaming drop:

1. after prefill, finalize per-layer budgets (KMeans on captured layer
   importance) — ``budgets.py``;
2. build per-layer recency keep sets ``[0, start_size) ∪ last(budget_L -
   start_size)`` (``selection.py``);
3. compact every compressible layer in place (per-head sets, uniform count in
   ``uniform`` mode; per-layer counts + fake-key padding in the experimental
   ``class_weighted`` mode);
4. shrink the block row and report the reclaim plan to the scheduler.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from ..core.budgets import compute_layer_budgets
from ..core.kv_layout import compact_request_kv_in_place_per_head, gather_request_kv_dense
from ..core.selection import (
    build_keep_tensor_per_head,
    pad_short_budget_layers_with_fake_keys,
    recency_keep_set,
)
from ..logging_control import cluster_log, log_debug, log_warning, probe
from .group_resolver import resolve_group_tensors


def _debug_disable_compaction() -> bool:
    return os.environ.get("SQUEEZE_DEBUG_DISABLE_COMPACTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _table_block_size(base_runner: Any) -> int:
    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    if block_size <= 0:
        block_size = int(os.environ.get("SQUEEZE_BLOCK_SIZE_HINT", "0") or 0)
    return block_size if block_size > 0 else 128


def _request_block_ids(base_runner: Any, req_id: str, gid: int = 0) -> Optional[list[int]]:
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
    """Authoritative worker-side KV length: block-table row capacity."""
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
    """Block capacity first (authoritative), then input-batch mirror, then
    request state (TriAttention philosophy)."""
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


def finalize_budgets(
    *,
    base_runner: Any,
    req_id: str,
    layer_importance: list[float],
    num_layers: int,
    prompt_len: int,
    ini_size: float,
    class3_size: float,
    n_clusters: int,
    seed: Optional[int],
    log_budgets: bool,
) -> tuple[list[int], dict[str, Any]]:
    """Run the SqueezeAttention KMeans budget allocation for one request."""
    budgets, diagnostics = compute_layer_budgets(
        layer_importance=layer_importance,
        num_layers=num_layers,
        ini_size=ini_size,
        class3_size=class3_size,
        prompt_len=prompt_len,
        n_clusters=n_clusters,
        seed=seed,
    )
    if log_budgets:
        class_ids = diagnostics["class_ids"]
        cluster_log(
            "budgets req=%s layers=%d prompt_len=%d ini_size=%.3f class3=%.3f "
            "class_sizes=%s budgets=%s",
            req_id,
            num_layers,
            prompt_len,
            ini_size,
            class3_size,
            diagnostics["class_sizes"],
            budgets,
        )
    return budgets, diagnostics


def compress_request(
    *,
    base_runner: Any,
    req_id: str,
    keep_count: int,
    budgets: list[int],
    total_tokens: int,
    block_size: int,
    start_size: int,
    mode: str,
    hooks: Any,
    fake_key_padding: bool = False,
    min_reclaim_blocks: int = 1,
    scheduled_tokens: int = 1,
) -> dict[str, Any]:
    """Run one SqueezeAttention compression for a request.

    ``keep_count`` is the uniform K (``uniform`` mode); ``budgets`` carries the
    per-layer budgets used by ``class_weighted`` mode.
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

    retained_cache_len = keep_count + max(1, int(scheduled_tokens))
    required_blocks = (retained_cache_len + block_size - 1) // block_size

    total_reclaimed = 0
    compacted_layers = 0
    applied_any = False
    padded_slots = 0
    selector_debug: dict[str, Any] = {}

    for gid, layer_tensors in group_tensors.items():
        block_ids = _request_block_ids(base_runner, req_id, gid)
        if not block_ids:
            continue
        current_blocks = len(block_ids)
        reclaimable = current_blocks - required_blocks
        if reclaimable <= 0:
            continue
        for layer_idx, kv_cache in layer_tensors:
            try:
                layer_budget = budgets[layer_idx] if layer_idx < len(budgets) else keep_count
                if mode == "class_weighted":
                    layer_keep = min(int(layer_budget), int(total_tokens))
                    if layer_keep <= 0:
                        continue
                    keep_set = recency_keep_set(total_tokens, layer_keep, start_size)
                    max_keep = max(1, keep_count)
                    if len(keep_set) < max_keep:
                        # Pad this layer's set with the most recent dropped
                        # tokens so the keep count is uniform for the shared
                        # row; fake keys (below) nullify them for attention.
                        extra = recency_keep_set(
                            total_tokens, max_keep, start_size
                        )
                        seen = set(keep_set)
                        keep_set = keep_set + [t for t in extra if t not in seen][
                            : max_keep - len(keep_set)
                        ]
                    num_kv_heads = kv_cache[0].shape[2] if isinstance(kv_cache, (list, tuple)) else kv_cache.shape[2]
                    keep_tensor = build_keep_tensor_per_head(
                        keep_set, num_kv_heads, kv_cache[0].device if isinstance(kv_cache, (list, tuple)) else kv_cache.device
                    )
                else:
                    keep_set = recency_keep_set(total_tokens, keep_count, start_size)
                    num_kv_heads = kv_cache[0].shape[2] if isinstance(kv_cache, (list, tuple)) else kv_cache.shape[2]
                    keep_tensor = build_keep_tensor_per_head(
                        keep_set, num_kv_heads, kv_cache[0].device if isinstance(kv_cache, (list, tuple)) else kv_cache.device
                    )
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
                applied_any = True
                selector_debug[f"layer_{layer_idx}"] = {
                    "mode": mode,
                    "keep": int(keep_tensor.shape[1]),
                    "budget": int(layer_budget),
                }
                if mode == "class_weighted" and fake_key_padding and hooks is not None:
                    query = hooks.get_query(layer_idx)
                    if query is not None:
                        key_cache = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
                        try:
                            padded_slots += pad_short_budget_layers_with_fake_keys(
                                key_cache=key_cache,
                                block_ids=block_ids,
                                block_size=block_size,
                                query=query,
                                keep_count=int(keep_tensor.shape[1]),
                                total_tokens=total_tokens,
                                max_keep_count=keep_count,
                                num_kv_heads=int(num_kv_heads),
                            )
                        except Exception as exc:
                            log_warning(
                                "fake-key padding failed req=%s layer=%d: %s: %s",
                                req_id, layer_idx, type(exc).__name__, exc,
                            )
            except Exception as exc:  # pragma: no cover - per-layer safety
                log_warning(
                    "compression layer failed req=%s gid=%d layer=%d: %s: %s",
                    req_id, gid, layer_idx, type(exc).__name__, exc,
                )
        total_reclaimed += max(0, reclaimable)

    if not applied_any:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "no_layers_scored",
            "cache_len_after": total_tokens,
        }
    if min_reclaim_blocks > 0 and total_reclaimed < min_reclaim_blocks:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "below_min_reclaim",
            "cache_len_after": total_tokens,
            "details": {
                "reclaimed_blocks": total_reclaimed,
                "min_reclaim_blocks": min_reclaim_blocks,
            },
        }

    probe(
        "COMPRESS req=%s mode=%s before=%d after=%d retained=%d "
        "reclaimed_blocks=%d layers_compacted=%d fake_key_slots=%d",
        req_id,
        mode,
        total_tokens,
        keep_count,
        retained_cache_len,
        total_reclaimed,
        compacted_layers,
        padded_slots,
    )
    return {
        "req_id": req_id,
        "status": "applied",
        "reason": "squeeze_compaction",
        "cache_len_after": keep_count,
        "effective_cache_len_after": keep_count,
        "retained_cache_len": retained_cache_len,
        "details": {
            "mode": mode,
            "start_size": int(start_size),
            "effective_tokens_before": total_tokens,
            "keep_count": keep_count,
            "retained_cache_len": retained_cache_len,
            "reclaimed_block_count": total_reclaimed,
            "fake_key_slots": padded_slots,
            "block_reclaim": {
                "mode": "truncate_tail",
                "groups": [
                    {
                        "gid": gid,
                        "block_ids_before": list(block_ids),
                        "block_ids_after": list(block_ids[:required_blocks]),
                        "required_blocks": required_blocks,
                        "reclaimable_blocks": max(0, len(block_ids) - required_blocks),
                    }
                    for gid, block_ids in [
                        (g, _request_block_ids(base_runner, req_id, g) or [])
                        for g in sorted(group_tensors)
                    ]
                ],
            },
            "selector_debug": selector_debug,
        },
    }
