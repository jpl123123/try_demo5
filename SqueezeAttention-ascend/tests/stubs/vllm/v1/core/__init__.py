"""Stub vLLM V1 scheduler / KV cache manager / engine core surfaces.

Mirrors the vllm-ascend v0.23.0-era vLLM V1 interfaces that the kvpress-ascend
and SqueezeAttention-ascend monkeypatches touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class StubRequest:
    req_id: str
    num_computed_tokens: int = 0
    num_prompt_tokens: int = 0
    block_ids: Optional[list] = None
    state: Any = None

    def __post_init__(self):
        if self.block_ids is None:
            self.block_ids = []


class StubSchedulerOutput:
    """Mirrors the vLLM V1 SchedulerOutput fields used by the patches."""

    def __init__(self, num_scheduled_tokens: dict[str, int] | None = None):
        self.num_scheduled_tokens: dict[str, int] = num_scheduled_tokens or {}
        self.total_num_scheduled_tokens: int = sum(self.num_scheduled_tokens.values())
        self.scheduled_new_reqs: list[Any] = []
        self.finished_req_ids: list[str] = []
        self.scheduled_cached_reqs: Any = None
        self.kvpress_signals: Any = None
        self.kvpress_step: int = 0
        self.squeeze_signals: Any = None
        self.squeeze_step: int = 0


@dataclass
class StubCachedReqs:
    req_ids: list[str]
    new_block_ids: list[Any]
    num_computed_tokens: list[int]


class StubBlockPool:
    def __init__(self, num_gpu_blocks: int):
        self.num_gpu_blocks = num_gpu_blocks
        self.freed: list[list[int]] = []

    def free_blocks(self, blocks) -> None:
        self.freed.append([int(b) for b in blocks])


class StubKVCacheManager:
    """Mirrors vLLM V1 KVCacheManager (req_to_blocks + block_pool + usage)."""

    def __init__(self, num_blocks: int = 256):
        self.req_to_blocks: dict[str, list[int]] = {}
        self.block_pool = StubBlockPool(num_blocks)
        self.usage: float = 0.0
        self.coordinator: Any = None
        self.single_type_managers: list[Any] = []

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        blocks = self.req_to_blocks.setdefault(request.req_id, [])
        if hasattr(request, "block_ids") and isinstance(request.block_ids, list):
            blocks[:] = [int(b) for b in request.block_ids]
        return len(blocks)


class StubScheduler:
    """Minimal V1 Scheduler: requests dict + schedule/update bookkeeping."""

    def __init__(self, block_size: int = 128, kv_cache_manager: Any = None):
        self.requests: dict[str, StubRequest] = {}
        self.block_size = block_size
        self.kv_cache_manager = kv_cache_manager or StubKVCacheManager()
        self.finished_req_ids: list[str] = []
        self.max_num_scheduled_tokens: int = 4096
        self._kvpress_step = 0
        self._sim_num_tokens: Optional[dict[str, int]] = None
        self._sim_seen: set[str] = set()

    def add_request(self, req: StubRequest) -> None:
        self.requests[req.req_id] = req
        self.kv_cache_manager.req_to_blocks.setdefault(req.req_id, list(req.block_ids or []))

    def schedule(self) -> StubSchedulerOutput:
        tokens = self._sim_num_tokens or {}
        out = StubSchedulerOutput(
            num_scheduled_tokens={
                rid: int(tokens.get(rid, 1)) for rid in self.requests
            }
        )
        new_reqs = [
            (None, req)
            for rid, req in self.requests.items()
            if rid not in self._sim_seen
        ]
        self._sim_seen.update(rid for rid in self.requests)
        out.scheduled_new_reqs = new_reqs
        out.finished_req_ids = list(self.finished_req_ids)
        self.finished_req_ids = []
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        for rid in scheduler_output.finished_req_ids:
            self.requests.pop(rid, None)
            self.kv_cache_manager.req_to_blocks.pop(rid, None)
        return {rid: None for rid in list(self.requests)}


class StubEngineCore:
    def __init__(self):
        self.batch_queue: Any = None
        self.scheduler: Any = None
        self.model_executor: Any = None
        self.is_ec_producer: bool = False
        self.is_pooling_model: bool = False

    def step_with_batch_queue(self):
        raise NotImplementedError("stub")
