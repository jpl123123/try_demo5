from types import SimpleNamespace

from triattention.vllm.runtime.executor import CompressionExecutionResult
from triattention.vllm.runtime.runner_compression_actions import (
    execute_runner_compression_actions,
)
from triattention.vllm.runtime.signals import CompressionSignal
from triattention.vllm.runtime.state import RequestStateStore


class _Executor:
    def __init__(self, reason="zero_copy_recency_not_ready"):
        self.reason = reason
        self.calls = []

    def execute(self, *, req_id, signal, scheduler_output):
        self.calls.append(req_id)
        return CompressionExecutionResult(
            applied=False,
            reason=self.reason,
            cache_len_after=4096,
        )


class _AppliedExecutor:
    def execute(self, *, req_id, signal, scheduler_output):
        return CompressionExecutionResult(
            applied=True,
            reason="applied",
            cache_len_after=4096,
            details={"retained_cache_len": 4225},
        )


class _StateStore:
    def __init__(self):
        self.skipped_by_req = {}

    def mark_compression_skipped(self, **kwargs):
        self.skipped = kwargs
        self.skipped_by_req[kwargs["req_id"]] = kwargs


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class _CollectingLogger:
    def __init__(self):
        self.lines = []

    def debug(self, fmt, *args, **kwargs):
        self.lines.append(("debug", fmt % args if args else fmt))

    def info(self, fmt, *args, **kwargs):
        self.lines.append(("info", fmt % args if args else fmt))

    def exception(self, fmt, *args, **kwargs):
        self.lines.append(("exception", fmt % args if args else fmt))


def test_zero_copy_recency_not_ready_is_allowed_in_strict_mode():
    state_store = _StateStore()
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4096,
        step=4,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=1,
    )

    events = execute_runner_compression_actions(
        executor=_Executor(),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"zero_copy_recency_not_ready"},
        logger=_Logger(),
        log_decisions=False,
    )

    assert events == [
        {
            "req_id": "req-1",
            "step": 4,
            "status": "skipped",
            "reason": "zero_copy_recency_not_ready",
            "cache_len_after": 4096,
            "details": {},
            "scheduled_tokens": 1,
            "estimated_cache_len": 4096,
            "prefill_len": 10000,
        }
    ]
    assert state_store.skipped["reason"] == "zero_copy_recency_not_ready"


def test_prefill_compression_limit_is_allowed_in_strict_mode():
    state_store = _StateStore()
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4096,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=2048,
    )

    events = execute_runner_compression_actions(
        executor=_Executor(reason="prefill_compression_limit"),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"prefill_compression_limit"},
        logger=_Logger(),
        log_decisions=False,
    )

    assert events == [
        {
            "req_id": "req-1",
            "step": 6,
            "status": "skipped",
            "reason": "prefill_compression_limit",
            "cache_len_after": 4096,
            "details": {},
            "scheduled_tokens": 2048,
            "estimated_cache_len": 4096,
            "prefill_len": 10000,
        }
    ]
    assert state_store.skipped["reason"] == "prefill_compression_limit"


def test_initial_decode_grace_is_allowed_in_strict_mode():
    state_store = _StateStore()
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

    events = execute_runner_compression_actions(
        executor=_Executor(reason="initial_decode_grace"),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"initial_decode_grace"},
        logger=_Logger(),
        log_decisions=False,
    )

    assert events == [
        {
            "req_id": "req-1",
            "step": 6,
            "status": "skipped",
            "reason": "initial_decode_grace",
            "cache_len_after": 4096,
            "details": {},
            "scheduled_tokens": 1,
            "estimated_cache_len": 9864,
            "prefill_len": 9863,
        }
    ]
    assert state_store.skipped["reason"] == "initial_decode_grace"


def test_core_only_execution_path_suppresses_noisy_zero_copy_skip_logs():
    state_store = _StateStore()
    logger = _CollectingLogger()
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4096,
        step=8,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=1,
    )

    events = execute_runner_compression_actions(
        executor=_Executor(),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"zero_copy_recency_not_ready"},
        logger=logger,
        log_decisions=False,
        logging_enabled=True,
        log_execution_path=True,
        log_execution_path_core_only=True,
    )

    assert events[0]["reason"] == "zero_copy_recency_not_ready"
    assert logger.lines == []


def test_max_compressions_per_step_delays_excess_requests():
    state_store = _StateStore()
    executor = _Executor(reason="under_budget")
    signals = {}
    for idx in range(4):
        req_id = f"req-{idx}"
        signals[req_id] = CompressionSignal(
            req_id=req_id,
            should_compress=True,
            reason="length_threshold",
            estimated_cache_len=4096,
            step=9,
            kv_usage=None,
            protect_prefill=False,
            prefill_len=10000,
            scheduled_tokens=1,
        )

    events = execute_runner_compression_actions(
        executor=executor,
        state_store=state_store,
        scheduler_output=object(),
        signals=signals,
        strict_no_downgrade=False,
        allowed_strict_skip_reasons=set(),
        logger=_Logger(),
        log_decisions=False,
        max_compressions_per_step=2,
    )

    assert executor.calls == ["req-0", "req-1"]
    assert [event["reason"] for event in events].count("compression_step_limited") == 2
    assert state_store.skipped_by_req["req-2"]["reason"] == "compression_step_limited"
    assert state_store.skipped_by_req["req-3"]["reason"] == "compression_step_limited"


def test_max_compressions_per_step_does_not_delay_forced_boundary_requests():
    state_store = _StateStore()
    executor = _Executor(reason="under_budget")
    signals = {}
    for idx in range(4):
        req_id = f"req-{idx}"
        signals[req_id] = CompressionSignal(
            req_id=req_id,
            should_compress=True,
            reason="length_threshold",
            estimated_cache_len=6401,
            step=9,
            kv_usage=None,
            protect_prefill=False,
            prefill_len=10000,
            scheduled_tokens=1,
            force=True,
        )

    events = execute_runner_compression_actions(
        executor=executor,
        state_store=state_store,
        scheduler_output=object(),
        signals=signals,
        strict_no_downgrade=False,
        allowed_strict_skip_reasons=set(),
        logger=_Logger(),
        log_decisions=False,
        max_compressions_per_step=2,
    )

    assert executor.calls == ["req-0", "req-1", "req-2", "req-3"]
    assert "compression_step_limited" not in [event["reason"] for event in events]


def test_forced_boundary_request_bypasses_batch_queue_dedup():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=10000, protect_prefill=False)
    state_store.mark_compressed("req-1", step=8, cache_len=6400, scheduled_tokens=1)
    executor = _Executor(reason="under_budget")
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=6401,
        step=9,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=1,
        force=True,
    )

    events = execute_runner_compression_actions(
        executor=executor,
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=False,
        allowed_strict_skip_reasons=set(),
        logger=_Logger(),
        log_decisions=False,
        max_compressions_per_step=1,
    )

    assert executor.calls == ["req-1"]
    assert events[0]["reason"] == "under_budget"


def test_applied_compression_event_records_scheduler_nct_anchor():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=32383, protect_prefill=False)
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=32384,
        step=17,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=32383,
        scheduled_tokens=1,
    )
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[32383],
        ),
    )

    events = execute_runner_compression_actions(
        executor=_AppliedExecutor(),
        state_store=state_store,
        scheduler_output=scheduler_output,
        signals={"req-1": signal},
        strict_no_downgrade=False,
        allowed_strict_skip_reasons=set(),
        logger=_Logger(),
        log_decisions=False,
    )

    state = state_store.get("req-1")
    assert events[0]["cache_len_after"] == 4096
    assert events[0]["scheduler_nct"] == 32383
    assert state is not None
    assert state.cache_len_after_last_compression == 4096
    assert state.current_cache_len == 4097
