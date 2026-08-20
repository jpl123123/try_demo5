"""End-to-end simulated run driver: scheduler + worker + runner loop on CPU.

Drives the patched Scheduler / NPUWorker the same way vLLM V1 engine-core does:
    scheduler.schedule() -> worker.execute_model(scheduler_output) ->
    scheduler.update_from_output(scheduler_output, model_runner_output)

Requests live in the scheduler's ``requests`` dict; the worker mirrors them
into the NPUModelRunner ``requests``; block ids are tracked in request state
and the KV cache manager.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from vllm.v1.core import StubRequest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_manager import StubKVCacheManager
from vllm_ascend.worker.worker import NPUWorker


class SimLoop:
    def __init__(self, block_size: int = 8, num_blocks: int = 512):
        self.manager = StubKVCacheManager(num_blocks)
        self.scheduler = Scheduler(block_size=block_size, kv_cache_manager=self.manager)
        self.worker = NPUWorker()
        self.worker.init_device()
        # The proxy delegates reads to the base runner, so request state must
        # live on the base runner (as vLLM sets it).
        self._base_runner().requests = self.scheduler.requests
        self.step = 0

    def _base_runner(self):
        """Unwrap every proxy layer to the raw NPUModelRunner (coexistence
        tests install two proxies; request state must live on the raw runner)."""
        mr = self.worker.model_runner
        while hasattr(mr, "_base_runner"):
            mr = mr._base_runner
        return mr

    def add_request(self, req_id: str, num_prompt_tokens: int) -> StubRequest:
        req = StubRequest(req_id=req_id, num_prompt_tokens=num_prompt_tokens)
        req.block_ids = self._alloc_blocks(0)
        self.scheduler.add_request(req)
        return req

    def _alloc_blocks(self, n: int) -> list[int]:
        # Deterministic block allocator from the manager pool.
        used = set()
        for blocks in self.manager.req_to_blocks.values():
            used.update(int(b) for b in blocks)
        out = []
        nxt = 0
        while len(out) < n:
            if nxt not in used:
                out.append(nxt)
            nxt += 1
        return out

    def _grow_request_blocks(self, req: StubRequest, needed_blocks: int) -> None:
        current = len(req.block_ids)
        if needed_blocks <= current:
            return
        extra = self._alloc_blocks(needed_blocks - current)
        req.block_ids.extend(extra)
        self.manager.req_to_blocks[req.req_id] = list(req.block_ids)

    def step_schedule_and_execute(
        self,
        num_tokens_by_req: Optional[dict[str, int]] = None,
    ) -> tuple[Any, Any]:
        """One vLLM V1 step: schedule -> worker execute -> update_from_output.

        Returns (scheduler_output, model_runner_output).
        """
        self.step += 1
        self.scheduler._sim_num_tokens = num_tokens_by_req or {}
        scheduler_output = self.scheduler.schedule()
        self.scheduler._sim_num_tokens = None
        # Block growth happens after schedule (vLLM allocate_slots), using the
        # (possibly effective-rewritten) num_computed_tokens.
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            req = self.scheduler.requests[req_id]
            needed = (req.num_computed_tokens + n + self.scheduler.block_size - 1) // self.scheduler.block_size
            self._grow_request_blocks(req, needed)
        # Mirror scheduler output bookkeeping into worker request states.
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            req = self.scheduler.requests[rid]
            req.num_computed_tokens += n
        model_runner_output = self.worker.execute_model(scheduler_output)
        self.scheduler.update_from_output(scheduler_output, model_runner_output)
        return scheduler_output, model_runner_output

    def prefill(self, req_id: str, prompt_len: int, chunk: int = 16) -> None:
        remaining = prompt_len
        while remaining > 0:
            n = min(chunk, remaining)
            self.step_schedule_and_execute({req_id: n})
            remaining -= n

    def decode(self, req_id: str, steps: int = 1) -> None:
        for _ in range(steps):
            self.step_schedule_and_execute({req_id: 1})

    # -- introspection ------------------------------------------------------

    def kv_tensor(self, layer: int = 0) -> tuple:
        return self.worker.model_runner.kv_caches[layer]

    def row_blocks(self, req_id: str, gid: int = 0) -> int:
        batch = self.worker.model_runner.input_batch
        idx = batch.req_id_to_index[req_id]
        table = batch.block_table
        if hasattr(table, "block_tables"):
            table = table.block_tables[gid]
        return int(table.num_blocks_per_row[idx])

    def row_block_ids(self, req_id: str) -> list[int]:
        batch = self.worker.model_runner.input_batch
        idx = batch.req_id_to_index[req_id]
        table = batch.block_table
        n = int(table.num_blocks_per_row[idx])
        return [int(table.block_table.np[idx, i]) for i in range(n)]

    def cache_value_at(self, layer: int, token_idx: int, head: int = 0, dim: int = 0) -> float:
        """Read a token's K value from the block cache row (identifiable tokens
        written by the stub attention module)."""
        block_ids = self.row_block_ids_of_runner(layer)
        block_size = self.worker.model_runner.block_size
        block_id = block_ids[token_idx // block_size]
        offset = token_idx % block_size
        return float(self.kv_tensor(layer)[0][block_id, offset, head, dim])

    def row_block_ids_of_runner(self, _layer: int = 0) -> list[int]:
        return self.row_block_ids(list(self.scheduler.requests.keys())[0])

    def freed_blocks(self) -> list[int]:
        return [b for batch in self.manager.block_pool.freed for b in batch]

    def scheduler_tracker_len(self, req_id: str) -> Optional[int]:
        tracker = self.scheduler._squeeze_effective_len_tracker
        return tracker.get(req_id) if tracker is not None else None
