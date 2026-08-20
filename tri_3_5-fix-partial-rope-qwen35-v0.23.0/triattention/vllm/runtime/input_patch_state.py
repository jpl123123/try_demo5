"""Active runtime override state for the vLLM input patch backend.

This module isolates mutable patch state from `gpu_seq_len_patch.py` so the
runtime adapter can depend on a small, patch-agnostic state interface.
"""

from __future__ import annotations

import torch

ACTIVE_EFFECTIVE_NUM_COMPUTED_TOKENS: torch.Tensor | None = None
ACTIVE_EFFECTIVE_POSITIONS: torch.Tensor | None = None
ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX: dict[int, int] | None = None
ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX: dict[int, int] | None = None
ACTIVE_EFFECTIVE_BASE_LOOKUP_KEYS_CPU: torch.Tensor | None = None
ACTIVE_EFFECTIVE_BASE_LOOKUP_VALS_CPU: torch.Tensor | None = None
ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_KEYS_CPU: torch.Tensor | None = None
ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_VALS_CPU: torch.Tensor | None = None
ACTIVE_EFFECTIVE_BASE_LOOKUP_DEVICE_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_DEVICE_CACHE: dict[
    tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]
] = {}
ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE: int | None = None
ACTIVE_SINGLE_EFFECTIVE_POS_DELTA: int = 0
ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU: torch.Tensor | None = None
ACTIVE_EXPECTED_REQ_ROW_INDICES_DEVICE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
ACTIVE_EXPECTED_REQ_IDS: tuple[object, ...] | None = None
ACTIVE_EXPECTED_QUERY_LENS_CPU: torch.Tensor | None = None
ACTIVE_EXPECTED_QUERY_LENS_DEVICE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
ACTIVE_PACKED_POS_DELTAS_CPU: torch.Tensor | None = None
ACTIVE_PACKED_POS_DELTAS_DEVICE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
ACTIVE_EFFECTIVE_MAX_SEQ_LEN: int | None = None
ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE: int | None = None
ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS: int | None = None
ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS: int | None = None
ACTIVE_EFFECTIVE_OVERRIDES_ENABLED: bool = False
ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED: bool = False
ACTIVE_EFFECTIVE_MAPPING_VALIDATED: bool = False


def set_active_effective_num_computed_tokens(tensor: torch.Tensor | None) -> None:
    global ACTIVE_EFFECTIVE_NUM_COMPUTED_TOKENS
    ACTIVE_EFFECTIVE_NUM_COMPUTED_TOKENS = tensor


def set_active_effective_positions(tensor: torch.Tensor | None) -> None:
    global ACTIVE_EFFECTIVE_POSITIONS
    ACTIVE_EFFECTIVE_POSITIONS = tensor


def set_active_effective_overrides_enabled(enabled: bool) -> None:
    global ACTIVE_EFFECTIVE_OVERRIDES_ENABLED, ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED
    global ACTIVE_EFFECTIVE_MAPPING_VALIDATED
    ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = bool(enabled)
    if not enabled:
        ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED = False
        ACTIVE_EFFECTIVE_MAPPING_VALIDATED = False
    else:
        ACTIVE_EFFECTIVE_MAPPING_VALIDATED = False


def set_active_effective_max_seq_len(value: int | None) -> None:
    global ACTIVE_EFFECTIVE_MAX_SEQ_LEN
    ACTIVE_EFFECTIVE_MAX_SEQ_LEN = None if value is None else max(0, int(value))


def set_active_block_table_trim_observation(
    *,
    block_size: int | None,
    original_cols: int | None,
    effective_cols: int | None,
) -> None:
    global ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE
    global ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS
    global ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS
    ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE = block_size
    ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS = original_cols
    ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS = effective_cols


def mark_active_effective_overrides_consumed() -> None:
    global ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED
    ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED = True


def active_effective_overrides_consumed() -> bool:
    return bool(ACTIVE_EFFECTIVE_OVERRIDES_CONSUMED)


def mark_active_effective_mapping_validated() -> None:
    global ACTIVE_EFFECTIVE_MAPPING_VALIDATED
    ACTIVE_EFFECTIVE_MAPPING_VALIDATED = True


def active_effective_mapping_validated() -> bool:
    return bool(ACTIVE_EFFECTIVE_MAPPING_VALIDATED)


def _build_sparse_lookup_cpu_tensors(
    sparse_values_by_req_idx: dict[int, int] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not sparse_values_by_req_idx:
        return None, None
    keys = torch.as_tensor([int(k) for k in sparse_values_by_req_idx.keys()], dtype=torch.long)
    if keys.numel() == 0:
        return None, None
    vals = torch.as_tensor([int(v) for v in sparse_values_by_req_idx.values()], dtype=torch.long)
    if keys.numel() > 1:
        order = torch.argsort(keys)
        keys = keys.index_select(0, order)
        vals = vals.index_select(0, order)
    return keys, vals


def _lookup_device_key(device: torch.device) -> tuple[str, int | None]:
    return (device.type, device.index)


def _resolve_sparse_lookup_tensors_for_device(
    *,
    keys_cpu: torch.Tensor | None,
    vals_cpu: torch.Tensor | None,
    device: torch.device,
    cache: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if keys_cpu is None or vals_cpu is None:
        return None
    dev_key = _lookup_device_key(device)
    cached = cache.get(dev_key)
    if cached is not None:
        return cached
    keys = keys_cpu.to(device=device, dtype=torch.long)
    vals = vals_cpu.to(device=device, dtype=torch.long)
    cache[dev_key] = (keys, vals)
    return keys, vals


def _resolve_index_vector_for_device(
    *,
    values_cpu: torch.Tensor | None,
    device: torch.device,
    cache: dict[tuple[str, int | None], torch.Tensor],
) -> torch.Tensor | None:
    if values_cpu is None:
        return None
    dev_key = _lookup_device_key(device)
    cached = cache.get(dev_key)
    if cached is not None:
        return cached
    values = values_cpu.to(device=device, dtype=torch.long)
    cache[dev_key] = values
    return values


def set_active_effective_sparse_overrides(
    *,
    effective_base_by_req_idx: dict[int, int] | None,
    effective_pos_delta_by_req_idx: dict[int, int] | None,
    single_effective_seq_base: int | None = None,
    single_effective_pos_delta: int = 0,
    expected_req_row_indices: tuple[int, ...] | None = None,
    expected_req_ids: tuple[object, ...] | None = None,
    expected_query_lens: tuple[int, ...] | None = None,
    packed_pos_deltas: tuple[int, ...] | None = None,
) -> None:
    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
    global ACTIVE_EFFECTIVE_BASE_LOOKUP_KEYS_CPU, ACTIVE_EFFECTIVE_BASE_LOOKUP_VALS_CPU
    global ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_KEYS_CPU, ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_VALS_CPU
    global ACTIVE_EFFECTIVE_BASE_LOOKUP_DEVICE_CACHE, ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_DEVICE_CACHE
    global ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE, ACTIVE_SINGLE_EFFECTIVE_POS_DELTA
    global ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU, ACTIVE_EXPECTED_REQ_ROW_INDICES_DEVICE_CACHE
    global ACTIVE_EXPECTED_REQ_IDS
    global ACTIVE_EXPECTED_QUERY_LENS_CPU, ACTIVE_EXPECTED_QUERY_LENS_DEVICE_CACHE
    global ACTIVE_PACKED_POS_DELTAS_CPU, ACTIVE_PACKED_POS_DELTAS_DEVICE_CACHE
    ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = effective_base_by_req_idx
    ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX = effective_pos_delta_by_req_idx
    (
        ACTIVE_EFFECTIVE_BASE_LOOKUP_KEYS_CPU,
        ACTIVE_EFFECTIVE_BASE_LOOKUP_VALS_CPU,
    ) = _build_sparse_lookup_cpu_tensors(effective_base_by_req_idx)
    (
        ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_KEYS_CPU,
        ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_VALS_CPU,
    ) = _build_sparse_lookup_cpu_tensors(effective_pos_delta_by_req_idx)
    ACTIVE_EFFECTIVE_BASE_LOOKUP_DEVICE_CACHE = {}
    ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_DEVICE_CACHE = {}
    ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = single_effective_seq_base
    ACTIVE_SINGLE_EFFECTIVE_POS_DELTA = int(single_effective_pos_delta)
    if expected_req_row_indices:
        ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU = torch.as_tensor(
            [int(v) for v in expected_req_row_indices],
            dtype=torch.long,
        )
    else:
        ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU = None
    ACTIVE_EXPECTED_REQ_ROW_INDICES_DEVICE_CACHE = {}
    ACTIVE_EXPECTED_REQ_IDS = tuple(expected_req_ids) if expected_req_ids else None
    if expected_query_lens:
        ACTIVE_EXPECTED_QUERY_LENS_CPU = torch.as_tensor(
            [int(v) for v in expected_query_lens],
            dtype=torch.long,
        )
    else:
        ACTIVE_EXPECTED_QUERY_LENS_CPU = None
    ACTIVE_EXPECTED_QUERY_LENS_DEVICE_CACHE = {}
    if packed_pos_deltas:
        ACTIVE_PACKED_POS_DELTAS_CPU = torch.as_tensor(
            [int(v) for v in packed_pos_deltas],
            dtype=torch.long,
        )
    else:
        ACTIVE_PACKED_POS_DELTAS_CPU = None
    ACTIVE_PACKED_POS_DELTAS_DEVICE_CACHE = {}


def get_active_effective_base_lookup_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return _resolve_sparse_lookup_tensors_for_device(
        keys_cpu=ACTIVE_EFFECTIVE_BASE_LOOKUP_KEYS_CPU,
        vals_cpu=ACTIVE_EFFECTIVE_BASE_LOOKUP_VALS_CPU,
        device=device,
        cache=ACTIVE_EFFECTIVE_BASE_LOOKUP_DEVICE_CACHE,
    )


def get_active_effective_pos_delta_lookup_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return _resolve_sparse_lookup_tensors_for_device(
        keys_cpu=ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_KEYS_CPU,
        vals_cpu=ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_VALS_CPU,
        device=device,
        cache=ACTIVE_EFFECTIVE_POS_DELTA_LOOKUP_DEVICE_CACHE,
    )


def get_active_expected_req_row_indices(device: torch.device) -> torch.Tensor | None:
    return _resolve_index_vector_for_device(
        values_cpu=ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU,
        device=device,
        cache=ACTIVE_EXPECTED_REQ_ROW_INDICES_DEVICE_CACHE,
    )


def get_active_expected_query_lens(device: torch.device) -> torch.Tensor | None:
    return _resolve_index_vector_for_device(
        values_cpu=ACTIVE_EXPECTED_QUERY_LENS_CPU,
        device=device,
        cache=ACTIVE_EXPECTED_QUERY_LENS_DEVICE_CACHE,
    )


def get_active_packed_pos_deltas(device: torch.device) -> torch.Tensor | None:
    return _resolve_index_vector_for_device(
        values_cpu=ACTIVE_PACKED_POS_DELTAS_CPU,
        device=device,
        cache=ACTIVE_PACKED_POS_DELTAS_DEVICE_CACHE,
    )
