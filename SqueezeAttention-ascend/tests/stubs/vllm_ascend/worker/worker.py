"""Stub: vllm_ascend.worker.worker.NPUWorker."""

from __future__ import annotations

from typing import Any

from .model_runner_v1 import NPUModelRunner


class NPUWorker:
    def __init__(self, vllm_config=None, local_rank=0, rank=0, distributed_init_method=None, is_driver_worker=False, **kwargs):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.is_driver_worker = is_driver_worker
        self.model_runner: Any = None
        self.cache_config = None
        self.model_config = None
        self.parallel_config = None
        self.device_config = None
        self._kvpress_runner_proxy_installed = False
        self._squeeze_runner_proxy_installed = False

    def init_device(self):
        self.model_runner = NPUModelRunner()
        self.cache_config = self.model_runner.cache_config
        self.model_config = self.model_runner.model_config
        self.parallel_config = type(
            "P", (), {"tensor_parallel_rank": 0, "tensor_parallel_size": 1}
        )()

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output=None):
        return self.model_runner.sample_tokens(grammar_output)

    def determine_available_memory(self) -> int:
        return 1 << 34

    def load_model(self) -> None:
        pass
