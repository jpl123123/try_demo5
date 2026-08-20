"""Worker/scheduler block-table synchronization for SqueezeAttention-ascend.

After the SqueezeAttention press has selected keep indices and the KV cache has been
compacted in place (kept entries in the first blocks of the request row), both
sides of the vLLM-Ascend block table must be shrunk:

- worker side: ``input_batch.block_table`` ``num_blocks_per_row`` and the
  request state's ``block_ids`` (so the next ``append_row`` starts at the
  right offset);
- scheduler side: the freed blocks returned to the KV cache manager
  (``req_to_blocks`` / ``block_pool.free_blocks``).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..logging_control import log_debug, log_warning


def _debug_disabled() -> bool:
    return os.environ.get("SQUEEZE_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _row_block_count(table: Any, req_index: int) -> Optional[int]:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if isinstance(num_blocks_per_row, np.ndarray):
        if 0 <= req_index < int(num_blocks_per_row.shape[0]):
            return int(num_blocks_per_row[req_index])
    return None


def _clear_table_row_tail(table: Any, req_index: int, used_blocks: int) -> bool:
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if not isinstance(block_table_np, np.ndarray) or block_table_np.ndim != 2:
        return False
    if req_index < 0 or req_index >= int(block_table_np.shape[0]):
        return False
    start = max(0, min(int(used_blocks), int(block_table_np.shape[1])))
    if start < int(block_table_np.shape[1]):
        block_table_np[req_index, start:] = 0
    return True


def _rewrite_table_row(table: Any, req_index: int, block_ids: list[int]) -> bool:
    add_row = getattr(table, "add_row", None)
    if callable(add_row):
        add_row(block_ids, req_index)
        _clear_table_row_tail(
            table,
            req_index,
            _row_block_count(table, req_index) or len(block_ids),
        )
        return True
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if not isinstance(num_blocks_per_row, np.ndarray):
        return False
    if not isinstance(block_table_np, np.ndarray):
        return False
    if len(block_ids) > block_table_np.shape[1]:
        return False
    block_table_np[req_index, :] = 0
    block_table_np[req_index, : len(block_ids)] = block_ids
    num_blocks_per_row[req_index] = len(block_ids)
    return True


def _inner_tables(block_table_obj: Any) -> list[Any]:
    inner = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner, (list, tuple)) and inner:
        return list(inner)
    return [block_table_obj]


def apply_worker_block_reclaim(
    *,
    base_runner: Any,
    req_id: str,
    retained_cache_len: int,
    block_size: int,
    block_ids_after: Optional[list[int]] = None,
    gid: int = 0,
) -> bool:
    """Shrink the worker-side block-table row(s) for one compressed request."""
    if _debug_disabled():
        return False
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        log_warning("worker reclaim: block table not found (req=%s)", req_id)
        return False
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        log_warning("worker reclaim: req_id_to_index not found (req=%s)", req_id)
        return False
    req_index = req_id_to_index.get(req_id)
    if not isinstance(req_index, int):
        return False

    tables = _inner_tables(block_table_obj)
    if gid < 0 or gid >= len(tables):
        gid = 0
    table = tables[gid]
    required_blocks = max(1, (retained_cache_len + block_size - 1) // block_size)
    current = _row_block_count(table, req_index)
    if current is None:
        return False
    if block_ids_after is not None:
        if _rewrite_table_row(table, req_index, block_ids_after):
            log_debug(
                "worker reclaim remap: req=%s gid=%d num_blocks %d -> %d",
                req_id, gid, current, len(block_ids_after),
            )
            return True
    if current > required_blocks:
        num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
        if isinstance(num_blocks_per_row, np.ndarray):
            num_blocks_per_row[req_index] = required_blocks
            _clear_table_row_tail(table, req_index, required_blocks)
            log_debug(
                "worker reclaim: req=%s num_blocks %d -> %d (retained=%d block_size=%d)",
                req_id, current, required_blocks, retained_cache_len, block_size,
            )
            return True
    return False


def truncate_request_state_block_ids(
    *,
    base_runner: Any,
    req_id: str,
    required_blocks: int,
    gid: int = 0,
) -> None:
    """Truncate ``req_state.block_ids`` (CPU-side block tracking) after reclaim."""
    requests_dict = getattr(base_runner, "requests", None)
    if not isinstance(requests_dict, dict):
        return
    req_state = requests_dict.get(req_id)
    if req_state is None:
        return
    block_ids_attr = getattr(req_state, "block_ids", None)
    if isinstance(block_ids_attr, (list, tuple)) and block_ids_attr:
        first = block_ids_attr[0]
        if isinstance(first, (list, tuple)):
            # Nested per-group layout: [group0_blocks, group1_blocks, ...]
            try:
                group = block_ids_attr[gid]
            except (IndexError, TypeError):
                group = block_ids_attr
            if isinstance(group, (list, tuple)) and len(group) > required_blocks:
                trimmed = list(group[:required_blocks])
                if isinstance(group, tuple):
                    trimmed = tuple(trimmed)
                rewritten = list(block_ids_attr)
                rewritten[gid] = trimmed
                setattr(
                    req_state,
                    "block_ids",
                    tuple(rewritten) if isinstance(block_ids_attr, tuple) else rewritten,
                )
        elif gid == 0 and len(block_ids_attr) > required_blocks:
            # Flat single-group layout.
            trimmed = list(block_ids_attr[:required_blocks])
            setattr(
                req_state,
                "block_ids",
                tuple(trimmed) if isinstance(block_ids_attr, tuple) else trimmed,
            )


def free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
    """Return reclaimed blocks to the vLLM KV cache manager (scheduler side)."""
    if not removed_blocks:
        return False
    block_pool = getattr(manager, "block_pool", None)
    if block_pool is None:
        # Fallback: manager-level free API.
        free_fn = getattr(manager, "free_blocks", None)
        if callable(free_fn):
            free_fn(removed_blocks)
            return True
        return False
    free_fn = getattr(block_pool, "free_blocks", None)
    if callable(free_fn):
        free_fn(reversed(removed_blocks))
        return True
    return False


def resolve_scheduler_manager_blocks(
    scheduler: Any,
    req_id: str,
) -> tuple[Any, list[Any]]:
    """Resolve (manager, req_to_blocks) for the request on the scheduler side."""
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is None:
        return None, []
    req_to_blocks = getattr(manager, "req_to_blocks", None)
    if req_to_blocks is None:
        coordinator = getattr(manager, "coordinator", None)
        single_managers = getattr(coordinator, "single_type_managers", None) if coordinator else None
        if isinstance(single_managers, (list, tuple)) and single_managers:
            manager = single_managers[0]
            req_to_blocks = getattr(manager, "req_to_blocks", None)
    blocks = req_to_blocks.get(req_id) if isinstance(req_to_blocks, dict) else None
    return manager, (list(blocks) if isinstance(blocks, (list, tuple)) else [])
