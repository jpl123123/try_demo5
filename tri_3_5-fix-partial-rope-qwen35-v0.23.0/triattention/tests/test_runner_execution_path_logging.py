import sys
import types

import numpy as np


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    def debug(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.SimpleNamespace()
if "vllm.logger" not in sys.modules:
    sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        Tensor=object,
        is_tensor=lambda value: False,
    )

from triattention.vllm.runtime.runner import TriAttentionModelRunner
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.state import RequestStateStore
from triattention.vllm.runtime.signals import CompressionSignal


def test_runner_trigger_guard_marks_pre_core_skip():
    logger = _Logger()
    runner = object.__new__(TriAttentionModelRunner)
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._last_step = 11
    runner._logger = logger

    runner._log_execution_path_trigger_guard(
        req_id="req-1",
        reason="fast_recency_long_context_guard",
        hint="set_sparse_stats_path_or_disable_long_context_guard",
        prefill_len=19789,
    )
    runner._log_execution_path_trigger_guard(
        req_id="req-1",
        reason="fast_recency_long_context_guard",
        hint="set_sparse_stats_path_or_disable_long_context_guard",
        prefill_len=19789,
    )

    assert len(logger.lines) == 1
    line = logger.lines[0]
    assert "TRIATTN_EXEC_PATH runner_trigger_guard" in line
    assert "reason=fast_recency_long_context_guard" in line
    assert "core_entered=False" in line
    assert "hint=set_sparse_stats_path_or_disable_long_context_guard" in line


def test_runner_trigger_guard_suppressed_for_core_only_logging():
    logger = _Logger()
    runner = object.__new__(TriAttentionModelRunner)
    runner._log_execution_path = True
    runner._log_execution_path_core_only = True
    runner._logged_execution_path_trigger_guards = set()
    runner._last_step = 11
    runner._logger = logger

    runner._log_execution_path_trigger_guard(
        req_id="req-1",
        reason="fast_recency_long_context_guard",
        prefill_len=9863,
    )

    assert logger.lines == []


class _AscendRunner:
    pass


_AscendRunner.__module__ = "vllm_ascend.test"


class _StateStore:
    def __init__(self):
        self.state = types.SimpleNamespace(
            prefill_len=9863,
            compression_count=0,
            current_cache_len=0,
            current_cache_len_step=-1,
            current_cache_len_semantics="unknown",
        )
        self.skipped = None

    def get(self, req_id):
        return self.state

    def ensure(self, *, req_id, prefill_len, protect_prefill):
        self.state.prefill_len = max(
            int(getattr(self.state, "prefill_len", 0) or 0),
            int(prefill_len),
        )
        self.state.protect_prefill = bool(protect_prefill)
        return self.state

    def update_cache_len(self, req_id, cache_len, step=None):
        self.state.current_cache_len = max(0, int(cache_len))
        self.state.current_cache_len_semantics = "estimated_with_scheduled"
        if isinstance(step, int):
            self.state.current_cache_len_step = step

    def mark_compression_skipped(self, **kwargs):
        self.skipped = kwargs


def test_runner_keeps_existing_signal_on_first_decode_core_entry():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=128)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=9863),
    }
    state_store = _StateStore()
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        defer_prefill_compression_on_ascend=False,
        log_decisions=False,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 6
    runner._logger = logger
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 9864

    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=9864,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=9863,
        scheduled_tokens=1,
    )

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {"req-1": signal},
    )

    assert signals == {"req-1": signal}
    assert state_store.skipped is None
    assert logger.lines == []


def test_runner_first_compression_ignores_inflated_block_table_capacity():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=1536)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=20543),
    }
    state_store = _StateStore()
    state_store.state = types.SimpleNamespace(
        prefill_len=20543,
        compression_count=0,
        current_cache_len=0,
        current_cache_len_step=-1,
        current_cache_len_semantics="unknown",
    )
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        kv_budget=1536,
        divide_length=128,
        min_reclaim_blocks_on_ascend=16,
        prefill_min_reclaim_blocks_on_ascend=32,
        defer_prefill_compression_on_ascend=False,
        log_decisions=True,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 2289
    runner._logger = logger
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 258048

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {},
    )

    assert signals == {}
    assert state_store.skipped is None
    assert any("reason=below_worker_threshold" in line for line in logger.lines)
    assert any("effective_kv=20544" in line for line in logger.lines)
    assert any("block_capacity=258048" in line for line in logger.lines)


def test_runner_keeps_existing_signal_at_worker_block_boundary():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=128)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=32639),
    }
    state_store = _StateStore()
    state_store.state = types.SimpleNamespace(
        prefill_len=32383,
        compression_count=1,
        current_cache_len=4353,
    )
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        kv_budget=4096,
        divide_length=128,
        min_reclaim_blocks_on_ascend=8,
        defer_prefill_compression_on_ascend=False,
        log_decisions=False,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 273
    runner._logger = logger
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 34 * 128

    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=32641,
        step=273,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=32383,
        scheduled_tokens=1,
    )

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {"req-1": signal},
    )

    assert signals["req-1"].req_id == signal.req_id
    assert signals["req-1"].force is True
    assert state_store.skipped is None


def test_runner_advances_compressed_decode_to_exact_block_boundary():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=128)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=32768),
    }
    state_store = _StateStore()
    state_store.state = types.SimpleNamespace(
        prefill_len=32768,
        compression_count=1,
        current_cache_len=8447,
        current_cache_len_step=300,
        current_cache_len_semantics="estimated_with_scheduled",
    )
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        kv_budget=8192,
        divide_length=128,
        min_reclaim_blocks_on_ascend=8,
        defer_prefill_compression_on_ascend=False,
        log_decisions=False,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 301
    runner._logger = logger
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 66 * 128

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {},
    )

    assert state_store.state.current_cache_len == 8448
    assert state_store.state.current_cache_len_step == 301
    assert signals["req-1"].estimated_cache_len == 8448
    assert signals["req-1"].force is True


def test_runner_does_not_double_advance_compressed_len_in_same_step():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=128)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=32768),
    }
    state_store = _StateStore()
    state_store.state = types.SimpleNamespace(
        prefill_len=32768,
        compression_count=1,
        current_cache_len=8448,
        current_cache_len_step=301,
        current_cache_len_semantics="estimated_with_scheduled",
    )
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        kv_budget=8192,
        divide_length=128,
        min_reclaim_blocks_on_ascend=8,
        defer_prefill_compression_on_ascend=False,
        log_decisions=False,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 301
    runner._logger = logger
    runner._log_execution_path = True
    runner._log_execution_path_core_only = False
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 66 * 128

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {},
    )

    assert state_store.state.current_cache_len == 8448
    assert signals["req-1"].estimated_cache_len == 8448
    assert signals["req-1"].force is True


class _PatchTable:
    def __init__(self, *, block_size=128, current_blocks=3, max_blocks=8):
        self.block_size = block_size
        self.max_num_blocks_per_req = max_blocks
        self.num_blocks_per_row = np.array([current_blocks], dtype=np.int32)


def _runner_for_scheduler_output_patch(
    *,
    tables,
    log_decisions=False,
    prefill_len=4096,
    cache_len_after=256,
    compression_scheduler_nct=4096,
    retained_cache_len=257,
):
    base_runner = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=128),
        input_batch=types.SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=types.SimpleNamespace(block_tables=tables),
        ),
    )
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=prefill_len, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=7,
        cache_len=cache_len_after,
        scheduled_tokens=1,
        scheduler_nct=compression_scheduler_nct,
    )
    runner = object.__new__(TriAttentionModelRunner)
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner.config = TriAttentionRuntimeConfig(log_decisions=log_decisions)
    runner._logger = _Logger()
    runner._pending_compression_events = [
        {
            "status": "applied",
            "req_id": "req-1",
            "cache_len_after": cache_len_after,
            "details": {"retained_cache_len": retained_cache_len},
        }
    ]
    return runner


def test_scheduler_output_patch_drops_stale_new_blocks_after_reclaim():
    runner = _runner_for_scheduler_output_patch(
        tables=[
            _PatchTable(current_blocks=3),
            _PatchTable(current_blocks=3),
        ],
    )
    scheduler_output = types.SimpleNamespace(
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=["req-1"],
            new_block_ids=[([99, 100], [199, 200])],
        )
    )

    runner._patch_scheduler_output_for_compressed_reqs(scheduler_output)

    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [([], [])]


def test_scheduler_output_patch_keeps_only_blocks_needed_for_retained_cache_len():
    runner = _runner_for_scheduler_output_patch(
        tables=[
            _PatchTable(current_blocks=2),
            _PatchTable(current_blocks=2),
        ],
    )
    scheduler_output = types.SimpleNamespace(
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=["req-1"],
            new_block_ids=[([99, 100], [199, 200])],
        )
    )

    runner._patch_scheduler_output_for_compressed_reqs(scheduler_output)

    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [([99], [199])]


def test_scheduler_output_patch_preserves_decode_growth_after_compression_anchor():
    runner = _runner_for_scheduler_output_patch(
        tables=[
            _PatchTable(current_blocks=32, max_blocks=64),
            _PatchTable(current_blocks=32, max_blocks=64),
        ],
        prefill_len=32768,
        cache_len_after=4096,
        compression_scheduler_nct=32768,
        retained_cache_len=4096,
    )
    scheduler_output = types.SimpleNamespace(
        num_scheduled_tokens={"req-1": 1},
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[33393],
            new_block_ids=[
                (
                    [90, 91, 92, 93, 94, 95],
                    [190, 191, 192, 193, 194, 195],
                )
            ],
        ),
    )

    runner._patch_scheduler_output_for_compressed_reqs(scheduler_output)

    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [
        ([90, 91, 92, 93, 94], [190, 191, 192, 193, 194])
    ]
