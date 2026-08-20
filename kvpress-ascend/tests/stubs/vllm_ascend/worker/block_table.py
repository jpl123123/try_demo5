"""Stub: vllm_ascend.worker.block_table.BlockTable / MultiGroupBlockTable.

Mirrors the vllm-ascend v0.23.0 BlockTable surface used by the patches:

- ``block_table`` buffer exposing ``.np`` (numpy view of the 2D row table)
- ``num_blocks_per_row`` (numpy per-request block counts)
- ``block_size`` / ``logical_block_size`` / ``physical_block_size``
- ``append_row`` / ``add_row`` / ``clear_row`` / ``commit_block_table``
- ``compute_slot_mapping(num_reqs, query_start_loc, positions)`` writing
  ``slot_mapping`` (token slot ids = block_id * block_size + offset)
- ``get_device_tensor``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class CpuGpuBuffer:
    """Minimal CpuGpuBuffer stand-in with .np / .gpu / .cpu views."""

    def __init__(self, np_array: np.ndarray):
        self.np: np.ndarray = np_array
        self.gpu = torch.as_tensor(np_array)
        self.cpu = torch.as_tensor(np_array)

    def copy_to_gpu(self, n: int | None = None) -> None:
        self.gpu = torch.as_tensor(self.np)


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
    ):
        self.block_size = block_size
        self.logical_block_size = block_size
        self.physical_block_size = block_size
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.block_table = CpuGpuBuffer(
            np.zeros((max_num_reqs, max_num_blocks_per_req), dtype=np.int64)
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int64)
        self.slot_mapping = CpuGpuBuffer(
            np.full((max_num_reqs * max_num_blocks_per_req * block_size,), -1, dtype=np.int64)
        )
        self.num_slots = 0

    def add_row(self, block_ids, row_idx: int) -> None:
        ids = list(block_ids)
        self.block_table.np[row_idx, : len(ids)] = ids
        self.num_blocks_per_row[row_idx] = len(ids)

    def append_row(self, block_ids, row_idx: int) -> None:
        ids = list(block_ids)
        start = int(self.num_blocks_per_row[row_idx])
        self.block_table.np[row_idx, start : start + len(ids)] = ids
        self.num_blocks_per_row[row_idx] = start + len(ids)

    def clear_row(self, row_idx: int) -> None:
        self.block_table.np[row_idx, :] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        self.block_table.np[tgt] = self.block_table.np[src]
        self.num_blocks_per_row[tgt] = self.num_blocks_per_row[src]

    def swap_row(self, src: int, tgt: int) -> None:
        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]
        self.num_blocks_per_row[[src, tgt]] = self.num_blocks_per_row[[tgt, src]]

    def compute_slot_mapping(self, num_reqs: int, query_start_loc, positions) -> None:
        qsl = query_start_loc
        if hasattr(qsl, "gpu"):
            qsl = qsl.gpu
        qsl = qsl.detach().cpu().numpy() if torch.is_tensor(qsl) else np.asarray(qsl)
        pos = positions
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
        pos = np.asarray(pos)
        total = 0
        for row in range(int(num_reqs)):
            start, end = int(qsl[row]), int(qsl[row + 1])
            row_pos = pos[start:end]
            n_blocks = int(self.num_blocks_per_row[row])
            for i, p in enumerate(row_pos):
                block_idx = int(p) // self.block_size
                if block_idx >= n_blocks:
                    self.slot_mapping.np[total] = -1
                else:
                    block_id = int(self.block_table.np[row, block_idx])
                    self.slot_mapping.np[total] = block_id * self.block_size + int(p) % self.block_size
                total += 1
        self.num_slots = total

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu()

    def get_device_tensor(self, num_reqs: int | None = None) -> torch.Tensor:
        return self.block_table.gpu


class MultiGroupBlockTable:
    """Container of per-group BlockTables (hybrid/MTP models)."""

    def __init__(self, tables: list[BlockTable]):
        self.block_tables = tables

    def __getitem__(self, idx: int) -> BlockTable:
        return self.block_tables[idx]

    def append_row(self, block_ids, row_idx: int) -> None:
        if isinstance(block_ids, (list, tuple)) and len(block_ids) == len(self.block_tables) and isinstance(
            block_ids[0], (list, tuple)
        ):
            for gid, group_ids in enumerate(block_ids):
                self.block_tables[gid].append_row(group_ids, row_idx)
        else:
            self.block_tables[0].append_row(block_ids, row_idx)

    def add_row(self, block_ids, row_idx: int) -> None:
        if isinstance(block_ids, (list, tuple)) and len(block_ids) == len(self.block_tables) and isinstance(
            block_ids[0], (list, tuple)
        ):
            for gid, group_ids in enumerate(block_ids):
                self.block_tables[gid].add_row(group_ids, row_idx)
        else:
            self.block_tables[0].add_row(block_ids, row_idx)

    def compute_slot_mapping(self, num_reqs: int, query_start_loc, positions) -> None:
        for table in self.block_tables:
            table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for table in self.block_tables:
            table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for table in self.block_tables:
            table.clear_row(0)
