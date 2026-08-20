"""v1 model-runner input patch for SqueezeAttention-ascend.

Patches ``vllm_ascend.worker.model_runner_v1.NPUModelRunner._prepare_inputs`` so
that, when the runner proxy has activated effective overrides for this step:

- ``seq_lens`` / ``optimistic_seq_lens_cpu`` are rewritten to
  ``effective_base + num_scheduled_tokens``;
- ``positions`` are rewritten to ``effective_base + [0..scheduled)``;
- the slot mapping is recomputed from the effective positions, so this step's
  tokens land in the slots right after the compacted prefix.

Adapted from tri_3_5's input_patch_vllm_v1_backend for vLLM-Ascend v0.23.0
(where ``positions`` is a plain GPU tensor with CPU mirror
``_positions_np_buf`` and ``seq_lens`` mirror is ``optimistic_seq_lens_cpu``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..logging_control import log_debug, log_warning
from . import input_patch_state as _patch_state


def _runner_positions_np(runner: Any, total: int) -> np.ndarray:
    positions = getattr(runner, "positions", None)
    np_attr = getattr(positions, "np", None)
    if np_attr is not None:
        return np_attr[:total]
    positions_np_buf = getattr(runner, "_positions_np_buf", None)
    if positions_np_buf is not None:
        return positions_np_buf[:total]
    if isinstance(positions, torch.Tensor):
        return positions[:total].detach().cpu().numpy()
    raise RuntimeError(
        "SQUEEZE_V1_POSITIONS_BUFFER_UNSUPPORTED: cannot read positions numpy view"
    )


def _runner_seq_lens_np(runner: Any) -> np.ndarray:
    seq_lens = getattr(runner, "seq_lens", None)
    np_attr = getattr(seq_lens, "np", None)
    if np_attr is not None:
        return np_attr
    optimistic = getattr(runner, "optimistic_seq_lens_cpu", None)
    if isinstance(optimistic, torch.Tensor):
        return optimistic.numpy()
    if isinstance(optimistic, np.ndarray):
        return optimistic
    if isinstance(seq_lens, torch.Tensor):
        return seq_lens.detach().cpu().numpy()
    raise RuntimeError(
        "SQUEEZE_V1_SEQ_LENS_BUFFER_UNSUPPORTED: cannot read seq_lens numpy view"
    )


def _runner_device(runner: Any) -> torch.device:
    device = getattr(runner, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cpu")


def _effective_block_table_capacity(runner: Any, row: int) -> int | None:
    input_batch = getattr(runner, "input_batch", None)
    block_table_obj = getattr(input_batch, "block_table", None) if input_batch else None
    if block_table_obj is None:
        return None
    inner = getattr(block_table_obj, "block_tables", None)
    tables = list(inner) if isinstance(inner, (list, tuple)) and inner else [block_table_obj]
    capacities: list[int] = []
    for table in tables:
        block_size = None
        for attr_name in ("block_size", "logical_block_size", "physical_block_size"):
            try:
                block_size = int(getattr(table, attr_name))
            except Exception:
                continue
            if block_size and block_size > 0:
                break
        if block_size is None:
            cache_config = getattr(runner, "cache_config", None)
            try:
                block_size = int(getattr(cache_config, "block_size"))
            except Exception:
                block_size = None
        num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
        row_blocks = None
        if isinstance(num_blocks_per_row, np.ndarray) and 0 <= row < num_blocks_per_row.shape[0]:
            row_blocks = int(num_blocks_per_row[row])
        if block_size and row_blocks:
            capacities.append(block_size * row_blocks)
    if capacities:
        return min(capacities)
    return None


def _apply_sparse_seq_len_overrides_in_place(
    *,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
    runner: Any,
) -> bool:
    if num_reqs <= 0:
        return False
    applied = False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        base = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE)
        if base >= int(num_computed_tokens_cpu[0]):
            return False
        new_seq_len = base + int(num_scheduled_tokens[0])
        capacity = _effective_block_table_capacity(runner, 0)
        if capacity is not None and capacity > 0 and new_seq_len > capacity:
            new_seq_len = int(capacity)
        seq_lens_np[0] = new_seq_len
        return True

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return False
    seq_lens_np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
    for req_idx, effective_base in sparse_bases.items():
        idx = int(req_idx)
        if 0 <= idx < num_reqs and int(effective_base) < int(num_computed_tokens_cpu[idx]):
            new_seq_len = int(effective_base) + int(num_scheduled_tokens[idx])
            capacity = _effective_block_table_capacity(runner, idx)
            if capacity is not None and capacity > 0 and new_seq_len > capacity:
                new_seq_len = int(capacity)
            seq_lens_np[idx] = new_seq_len
            applied = True
    return applied


def _build_effective_slot_positions(
    *,
    positions_np: np.ndarray,
    req_indices: np.ndarray,
    runner: Any,
) -> np.ndarray | None:
    if int(req_indices.size) == 0:
        return None
    max_row = int(req_indices.max(initial=-1))
    num_rows = max_row + 1

    if num_rows == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        base = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE)
        if base >= int(positions_np[0]):
            return None
        capacity = _effective_block_table_capacity(runner, 0)
        if capacity is not None and capacity > 0 and base + positions_np.size > capacity:
            base = max(0, capacity - int(positions_np.size))
        return base + np.arange(int(positions_np.size), dtype=positions_np.dtype)

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return None
    out = positions_np.copy()
    changed = False
    for req_idx, effective_base in sparse_bases.items():
        row = int(req_idx)
        if row < 0 or row >= num_rows:
            continue
        token_indices = np.nonzero(req_indices == row)[0]
        if token_indices.size == 0:
            continue
        if int(effective_base) >= int(positions_np[token_indices[0]]):
            continue
        capacity = _effective_block_table_capacity(runner, row)
        base = int(effective_base)
        if capacity is not None and capacity > 0 and base + token_indices.size > capacity:
            base = max(0, capacity - int(token_indices.size))
        out[token_indices] = base + np.arange(int(token_indices.size), dtype=positions_np.dtype)
        changed = True
    return out if changed else None


def _sync_seq_lens_to_runner_buffers(
    *,
    runner: Any,
    seq_lens_np: np.ndarray,
    num_reqs: int,
) -> bool:
    if num_reqs <= 0:
        return False
    synced = False
    seq_lens = getattr(runner, "seq_lens", None)
    copy_to_gpu = getattr(seq_lens, "copy_to_gpu", None)
    if callable(copy_to_gpu):
        copy_to_gpu()
        synced = True
    elif isinstance(seq_lens, torch.Tensor):
        values = torch.as_tensor(seq_lens_np[:num_reqs], device=seq_lens.device, dtype=seq_lens.dtype)
        seq_lens[:num_reqs].copy_(values)
        if int(seq_lens.numel()) > num_reqs:
            seq_lens[num_reqs:].fill_(0)
        synced = True

    optimistic = getattr(runner, "optimistic_seq_lens_cpu", None)
    if isinstance(optimistic, torch.Tensor):
        values = torch.as_tensor(seq_lens_np[:num_reqs], device=optimistic.device, dtype=optimistic.dtype)
        optimistic[:num_reqs].copy_(values)
        if int(optimistic.numel()) > num_reqs:
            optimistic[num_reqs:].fill_(0)
        synced = True
    elif isinstance(optimistic, np.ndarray):
        optimistic[:num_reqs] = seq_lens_np[:num_reqs]
        if optimistic.size > num_reqs:
            optimistic[num_reqs:].fill(0)
        synced = True
    return synced


def _apply_v1_effective_slot_mapping(
    *,
    runner: Any,
    block_table: Any,
    req_indices_np: np.ndarray,
    slot_positions_np: np.ndarray,
    num_reqs: int,
    total_num_scheduled_tokens: int,
) -> bool:
    if block_table is None:
        return False
    has_commit = callable(getattr(block_table, "commit_slot_mapping", None))
    if has_commit:
        block_table.compute_slot_mapping(req_indices_np, slot_positions_np)
        block_table.commit_slot_mapping(int(total_num_scheduled_tokens))
        return True
    device = _runner_device(runner)
    positions_gpu = torch.as_tensor(slot_positions_np, device=device, dtype=torch.int64)
    query_start_loc = getattr(runner, "query_start_loc", None)
    qsl_gpu_attr = getattr(query_start_loc, "gpu", None)
    if qsl_gpu_attr is not None:
        query_start_loc_gpu = qsl_gpu_attr[: num_reqs + 1]
    elif isinstance(query_start_loc, torch.Tensor):
        query_start_loc_gpu = query_start_loc[: num_reqs + 1]
    else:
        cu = np.zeros(int(num_reqs) + 1, dtype=np.int64)
        counts = np.bincount(req_indices_np.astype(np.int64, copy=False), minlength=int(num_reqs)).astype(np.int64)
        cu[1:] = np.cumsum(counts)
        query_start_loc_gpu = torch.as_tensor(cu, device=device, dtype=torch.int64)
    block_table.compute_slot_mapping(int(num_reqs), query_start_loc_gpu, positions_gpu)
    return True


def make_patched_v1_prepare_inputs(
    original_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``NPUModelRunner._prepare_inputs(scheduler_output,
    num_scheduled_tokens)`` to apply effective (compressed) input overrides."""

    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return out
        # Snapshot the override state BEFORE consuming it: mark_consumed clears
        # the active bases, and the application below needs them.
        sparse_bases = dict(_patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX)
        single_base = _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE
        _patch_state.mark_consumed()

        total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        input_batch = getattr(self, "input_batch", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0))
        if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
            return out

        try:
            req_indices = np.repeat(getattr(self, "arange_np", np.arange(num_reqs))[:num_reqs], num_scheduled_tokens)
            positions_np = _runner_positions_np(self, total_num_scheduled_tokens)
            seq_lens_np = _runner_seq_lens_np(self)

            _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = sparse_bases
            _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = single_base
            _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = bool(
                sparse_bases or single_base is not None
            )
            try:
                slot_positions_np = _build_effective_slot_positions(
                    positions_np=positions_np,
                    req_indices=req_indices,
                    runner=self,
                )
            finally:
                _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = {}
                _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = None
                _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = False
            slot_applied = False
            if slot_positions_np is not None:
                # Rewrite the positions buffer (used by RoPE and the model
                # forward) to the compressed view, then recompute slots.
                positions_np[: total_num_scheduled_tokens] = slot_positions_np[
                    : total_num_scheduled_tokens
                ]
                positions_t = getattr(self, "positions", None)
                if isinstance(positions_t, torch.Tensor) and positions_t.numel() >= total_num_scheduled_tokens:
                    positions_t[:total_num_scheduled_tokens].copy_(
                        torch.as_tensor(
                            slot_positions_np[: total_num_scheduled_tokens],
                            device=positions_t.device,
                            dtype=positions_t.dtype,
                        )
                    )
                slot_applied = _apply_v1_effective_slot_mapping(
                    runner=self,
                    block_table=getattr(input_batch, "block_table", None),
                    req_indices_np=req_indices,
                    slot_positions_np=slot_positions_np,
                    num_reqs=num_reqs,
                    total_num_scheduled_tokens=total_num_scheduled_tokens,
                )

            _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = sparse_bases
            _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = single_base
            _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = bool(
                sparse_bases or single_base is not None
            )
            try:
                seq_applied = _apply_sparse_seq_len_overrides_in_place(
                    seq_lens_np=seq_lens_np,
                    num_computed_tokens_cpu=getattr(input_batch, "num_computed_tokens_cpu", None),
                    num_scheduled_tokens=num_scheduled_tokens,
                    num_reqs=num_reqs,
                    runner=self,
                )
            finally:
                _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = {}
                _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = None
                _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = False
            if seq_applied:
                if seq_lens_np.size > num_reqs:
                    seq_lens_np[num_reqs:].fill(0)
                _sync_seq_lens_to_runner_buffers(
                    runner=self,
                    seq_lens_np=seq_lens_np,
                    num_reqs=num_reqs,
                )
                active = seq_lens_np[:num_reqs]
                _patch_state.set_active_effective_max_seq_len(
                    int(active.max(initial=0)) if active.size else None
                )
            else:
                _patch_state.set_active_effective_max_seq_len(None)
            log_debug(
                "prepare_inputs overrides applied: num_reqs=%d total=%d seq_applied=%s slot_applied=%s",
                num_reqs, total_num_scheduled_tokens, seq_applied, slot_applied,
            )
        except Exception:
            log_warning("prepare_inputs override failed; running with base inputs", exc_info=True)
            _patch_state.mark_consumed()
        return out

    setattr(_patched_prepare_inputs, "_squeeze_patched", True)
    return _patched_prepare_inputs
