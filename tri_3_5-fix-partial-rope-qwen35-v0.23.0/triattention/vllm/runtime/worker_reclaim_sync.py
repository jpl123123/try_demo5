"""Worker-side block-table reclaim synchronization helpers for TriAttention runtime."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from vllm.logger import logger

from .logging_control import runtime_logging_enabled

_DEBUG_DISABLE_LOGGED = False


def _event_reclaim_groups(event: dict[str, Any]) -> tuple[str, dict[int, dict[str, Any]]]:
    block_reclaim = event.get("block_reclaim")
    if not isinstance(block_reclaim, dict):
        return "truncate_tail", {}
    mode = block_reclaim.get("mode")
    if mode not in {"truncate_tail", "remap_tail"}:
        mode = "truncate_tail"
    groups = block_reclaim.get("groups")
    if not isinstance(groups, list):
        return str(mode), {}
    by_gid: dict[int, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = group.get("gid")
        if isinstance(gid, int):
            by_gid[gid] = group
    return str(mode), by_gid


def _block_ids_after(group: dict[str, Any] | None) -> list[int] | None:
    if not isinstance(group, dict):
        return None
    block_ids_after = group.get("block_ids_after")
    if not isinstance(block_ids_after, list):
        return None
    if not all(isinstance(block_id, int) for block_id in block_ids_after):
        return None
    if len(set(block_ids_after)) != len(block_ids_after):
        return None
    return list(block_ids_after)


def _clear_table_row_tail(table: Any, req_index: int, used_blocks: int) -> bool:
    block_table = getattr(table, "block_table", None)
    block_table_np = getattr(block_table, "np", None)
    if not isinstance(block_table_np, np.ndarray):
        return False
    if block_table_np.ndim != 2:
        return False
    if req_index < 0 or req_index >= int(block_table_np.shape[0]):
        return False
    start = max(0, min(int(used_blocks), int(block_table_np.shape[1])))
    if start >= int(block_table_np.shape[1]):
        return True
    block_table_np[req_index, start:] = 0
    return True


def _row_block_count(table: Any, req_index: int, fallback: int) -> int:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if isinstance(num_blocks_per_row, np.ndarray):
        if 0 <= req_index < int(num_blocks_per_row.shape[0]):
            return int(num_blocks_per_row[req_index])
    return int(fallback)


def _rewrite_table_row(table: Any, req_index: int, block_ids: list[int]) -> bool:
    add_row = getattr(table, "add_row", None)
    if callable(add_row):
        add_row(block_ids, req_index)
        _clear_table_row_tail(
            table,
            req_index,
            _row_block_count(table, req_index, len(block_ids)),
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
    block_table_np[req_index, :len(block_ids)] = block_ids
    num_blocks_per_row[req_index] = len(block_ids)
    return True


def apply_worker_block_reclaim_events(
    *,
    base_runner: Any,
    events: list[dict[str, Any]] | None,
) -> None:
    """Apply reclaim shrink to worker-side block tables after compression.

    In vLLM V1, the block table lives at ``base_runner.input_batch.block_table``
    and tracks per-request block counts in ``num_blocks_per_row``.  After
    compression compacts KV cache data into fewer blocks, we must update these
    counters so that subsequent ``append_row()`` calls start from the correct
    offset and don't overflow the max-blocks-per-request limit.
    """
    global _DEBUG_DISABLE_LOGGED
    if os.environ.get("TRIATTN_DEBUG_DISABLE_WORKER_RECLAIM_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        if not _DEBUG_DISABLE_LOGGED and runtime_logging_enabled():
            logger.info("TriAttention worker reclaim sync disabled by debug env")
            _DEBUG_DISABLE_LOGGED = True
        return

    if not isinstance(events, list) or not events:
        return

    # Resolve the vLLM V1 block table.
    input_batch = getattr(base_runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        if getattr(base_runner, "block_tables", None) is not None:
            # Formal V2 runner manages block tables directly on base_runner
            # rather than on input_batch. In that path, hook-side compaction
            # already updates the canonical tables, so there is nothing for the
            # old V1 reclaim-sync helper to do here.
            return
        logger.warning(
            "TriAttention worker reclaim: block table not found. "
            "input_batch=%s block_table=%s",
            type(input_batch).__name__ if input_batch else None,
            type(block_table_obj).__name__ if block_table_obj else None,
        )
        return

    # Resolve request-id → row-index mapping.
    # In vLLM V1, req_id_to_index lives on input_batch, and request states
    # (with block_ids) live in base_runner.requests.
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        logger.warning(
            "TriAttention worker reclaim: req_id_to_index not found on input_batch. "
            "input_batch=%s",
            type(input_batch).__name__ if input_batch else None,
        )
        return

    # The block table may be a single BlockTable (with num_blocks_per_row) or
    # a MultiGroupBlockTable (with .block_tables list of per-group BlockTables).
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, list):
        # MultiGroupBlockTable
        tables = inner_tables
    else:
        # Single BlockTable
        tables = [block_table_obj]

    cache_config = getattr(base_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 16))
    if block_size <= 0:
        block_size = 16

    for event in events:
        if not isinstance(event, dict) or event.get("status") != "applied":
            continue
        req_id = event.get("req_id")
        if req_id is None:
            continue
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            continue
        cache_len_after = event.get("cache_len_after")
        if not isinstance(cache_len_after, int) or cache_len_after <= 0:
            continue

        details = event.get("details")
        retained_cache_len = (
            details.get("retained_cache_len")
            if isinstance(details, dict)
            else None
        )
        if not isinstance(retained_cache_len, int) or retained_cache_len <= 0:
            retained_cache_len = cache_len_after
        required_blocks = (retained_cache_len + block_size - 1) // block_size
        reclaim_mode, groups_by_gid = _event_reclaim_groups(event)
        has_explicit_reclaim_groups = bool(groups_by_gid)

        for gid, table in enumerate(tables):
            group_payload = groups_by_gid.get(gid)
            if has_explicit_reclaim_groups and group_payload is None:
                continue
            num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
            if num_blocks_per_row is None:
                continue
            if not isinstance(num_blocks_per_row, np.ndarray):
                continue
            current = int(num_blocks_per_row[req_index])
            block_ids_after = _block_ids_after(group_payload)
            if reclaim_mode == "remap_tail":
                if block_ids_after is not None:
                    if _rewrite_table_row(table, req_index, block_ids_after):
                        if runtime_logging_enabled():
                            logger.debug(
                                "TriAttention worker remap: req=%s gid=%d "
                                "num_blocks %d -> %d",
                                req_id, gid, current, len(block_ids_after),
                            )
                    else:
                        logger.warning(
                            "TriAttention worker remap failed: req=%s gid=%d "
                            "table=%s",
                            req_id, gid, type(table).__name__,
                        )
                    continue
            target_blocks = (
                len(block_ids_after)
                if block_ids_after is not None
                else required_blocks
            )
            if current > target_blocks:
                num_blocks_per_row[req_index] = target_blocks
                if runtime_logging_enabled():
                    logger.debug(
                        "TriAttention worker reclaim: req=%s num_blocks %d -> %d "
                        "(cache_len_after=%d block_size=%d)",
                        req_id, current, target_blocks, cache_len_after, block_size,
                    )
            _clear_table_row_tail(
                table,
                req_index,
                _row_block_count(table, req_index, min(current, target_blocks)),
            )

        # Also truncate req_state.block_ids (CPU-side block tracking).
        # In vLLM V1, per-request state lives in base_runner.requests dict.
        requests_dict = getattr(base_runner, "requests", None)
        if isinstance(requests_dict, dict):
            req_state = requests_dict.get(req_id)
            if req_state is not None:
                block_ids_attr = getattr(req_state, "block_ids", None)
                if isinstance(block_ids_attr, (list, tuple)):
                    if reclaim_mode == "remap_tail":
                        rewritten_groups: list[Any] = []
                        changed = False
                        for gid, group_blocks in enumerate(block_ids_attr):
                            block_ids_after = _block_ids_after(groups_by_gid.get(gid))
                            if block_ids_after is None:
                                rewritten_groups.append(group_blocks)
                                continue
                            if isinstance(group_blocks, tuple):
                                rewritten_groups.append(tuple(block_ids_after))
                            else:
                                rewritten_groups.append(list(block_ids_after))
                            changed = True
                        if changed:
                            if isinstance(block_ids_attr, tuple):
                                setattr(req_state, "block_ids", tuple(rewritten_groups))
                            else:
                                setattr(req_state, "block_ids", rewritten_groups)
                    else:
                        rewritten_groups: list[Any] = []
                        changed_tuple_group = False
                        for gid, group_blocks in enumerate(block_ids_attr):
                            group_payload = groups_by_gid.get(gid)
                            if has_explicit_reclaim_groups and group_payload is None:
                                rewritten_groups.append(group_blocks)
                                continue
                            block_ids_after = _block_ids_after(group_payload)
                            if block_ids_after is not None:
                                if isinstance(group_blocks, list):
                                    group_blocks[:] = list(block_ids_after)
                                    rewritten_groups.append(group_blocks)
                                else:
                                    rewritten_groups.append(tuple(block_ids_after))
                                    changed_tuple_group = True
                                continue
                            if (
                                isinstance(group_blocks, list)
                                and len(group_blocks) > required_blocks
                            ):
                                del group_blocks[required_blocks:]
                                rewritten_groups.append(group_blocks)
                            else:
                                rewritten_groups.append(group_blocks)
                        if changed_tuple_group:
                            if isinstance(block_ids_attr, tuple):
                                setattr(req_state, "block_ids", tuple(rewritten_groups))
                            else:
                                setattr(req_state, "block_ids", rewritten_groups)
