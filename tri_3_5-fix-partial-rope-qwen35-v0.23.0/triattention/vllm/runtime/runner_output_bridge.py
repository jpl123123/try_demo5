"""Runner output bridge helpers for TriAttention runtime.

Keeps `TriAttentionModelRunner` focused on orchestration while this module owns:
- base runner execute_model invocation under effective-input overrides
- side-channel compression event attachment to execute_model/sample_tokens outputs
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Iterator

from vllm.logger import logger

from .graph_mode_guard import force_ascend_eager_and_skip_compiled
from .input_adapter import active_effective_input_overrides, prepare_effective_input_overrides
from .input_patch_backend import assert_effective_overrides_consumed
from .logging_control import runtime_logging_enabled
from .phase_profile import (
    phase_elapsed_ms,
    phase_now,
    phase_profile_enabled,
    record_phase,
)
from .runner_struct_compat import debug_v1_override_path_enabled
from .thresholds import is_ascend_environment_available, is_ascend_runtime


class _TriattentionEventBag:
    """Picklable carrier for TriAttention compression events."""

    __slots__ = ("events",)

    def __init__(self, events):
        self.events = list(events)

    def __reduce__(self):
        return (_TriattentionEventBag, (list(self.events),))

    def __getstate__(self):
        return {"events": list(self.events)}

    def __setstate__(self, state):
        self.events = list(state.get("events", []))


def _scheduler_output_details(
    scheduler_output: Any,
    *,
    overrides: bool,
    sparse_overrides: bool,
) -> dict[str, Any]:
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    num_reqs = len(num_scheduled) if isinstance(num_scheduled, dict) else None
    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total_tokens is None and isinstance(num_scheduled, dict):
        try:
            total_tokens = sum(int(v) for v in num_scheduled.values())
        except Exception:
            total_tokens = None
    try:
        total_tokens_i = int(total_tokens)
    except Exception:
        total_tokens_i = None
    return {
        "num_reqs": num_reqs,
        "total_tokens": total_tokens_i,
        "overrides": int(overrides),
        "sparse_overrides": int(sparse_overrides),
    }


def _scheduled_request_count(scheduler_output: Any) -> int | None:
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    return len(num_scheduled) if isinstance(num_scheduled, dict) else None


def _has_sparse_effective_overrides(overrides: Any) -> bool:
    return (
        getattr(overrides, "seq_base_map", None) is not None
        or getattr(overrides, "pos_delta_map", None) is not None
        or getattr(overrides, "single_seq_base", None) is not None
        or int(getattr(overrides, "single_pos_delta", 0) or 0) != 0
    )


def _should_guard_ascend_multi_req_effective_overrides(
    *,
    base_runner: Any,
    scheduler_output: Any,
    overrides: Any,
    config: Any | None,
) -> bool:
    if not bool(
        getattr(
            config,
            "force_eager_multi_req_on_ascend_effective_overrides",
            True,
        )
    ):
        return False
    if not _has_sparse_effective_overrides(overrides):
        return False
    num_reqs = _scheduled_request_count(scheduler_output)
    if num_reqs is None or num_reqs <= 1:
        return False
    return is_ascend_runtime(base_runner) or is_ascend_environment_available()


@contextmanager
def _temporary_model_config_enforce_eager(
    base_runner: Any,
    *,
    enabled: bool,
) -> Iterator[bool]:
    model_config = getattr(base_runner, "model_config", None)
    if not enabled or model_config is None or not hasattr(model_config, "enforce_eager"):
        yield False
        return
    original = getattr(model_config, "enforce_eager")
    if bool(original):
        yield False
        return
    setattr(model_config, "enforce_eager", True)
    try:
        yield True
    finally:
        setattr(model_config, "enforce_eager", original)


def _execute_base_runner(
    *,
    base_runner: Any,
    scheduler_output: Any,
    intermediate_tensors: Any,
    overrides: bool,
    sparse_overrides: bool,
) -> Any:
    profile_enabled = phase_profile_enabled()
    t0 = phase_now() if profile_enabled else 0.0
    try:
        return base_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
    finally:
        if profile_enabled:
            record_phase(
                "base_runner_execute_model",
                phase_elapsed_ms(t0),
                _scheduler_output_details(
                    scheduler_output,
                    overrides=overrides,
                    sparse_overrides=sparse_overrides,
                ),
            )


def execute_base_model_with_effective_overrides(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    intermediate_tensors: Any = None,
    use_effective_overrides: bool = True,
    config: Any | None = None,
    perf_out: dict[str, float] | None = None,
) -> Any:
    """Execute base runner with current effective-length overrides applied."""
    perf_enabled = isinstance(perf_out, dict)
    if not use_effective_overrides:
        if perf_enabled:
            t0 = time.perf_counter()
        output = _execute_base_runner(
            base_runner=base_runner,
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            overrides=False,
            sparse_overrides=False,
        )
        if perf_enabled:
            t1 = time.perf_counter()
            perf_out["override_prep_ms"] = 0.0
            perf_out["base_exec_ms"] = (t1 - t0) * 1000.0
        return output

    if perf_enabled:
        t0 = time.perf_counter()
    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=config,
    )
    if perf_enabled:
        t1 = time.perf_counter()
    if (
        overrides.seq_base_map is None
        and overrides.pos_delta_map is None
        and overrides.single_seq_base is None
        and overrides.single_pos_delta == 0
    ):
        if perf_enabled:
            t2 = time.perf_counter()
        output = _execute_base_runner(
            base_runner=base_runner,
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
            overrides=True,
            sparse_overrides=False,
        )
        if perf_enabled:
            t3 = time.perf_counter()
            perf_out["override_prep_ms"] = (t1 - t0) * 1000.0
            perf_out["base_exec_ms"] = (t3 - t2) * 1000.0
        return output
    # Use sparse overrides in hot path to avoid per-step dense tensor copies.
    guard_ascend_graph_mode = _should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=overrides,
        config=config,
    )
    with active_effective_input_overrides(overrides):
        with force_ascend_eager_and_skip_compiled(guard_ascend_graph_mode):
            with _temporary_model_config_enforce_eager(
                base_runner,
                enabled=guard_ascend_graph_mode,
            ):
                if perf_enabled:
                    t2 = time.perf_counter()
                output = _execute_base_runner(
                    base_runner=base_runner,
                    scheduler_output=scheduler_output,
                    intermediate_tensors=intermediate_tensors,
                    overrides=True,
                    sparse_overrides=True,
                )
                if perf_enabled:
                    t3 = time.perf_counter()
        if (
            getattr(base_runner, "req_states", None) is not None
            or getattr(base_runner, "input_batch", None) is not None
            or debug_v1_override_path_enabled()
        ):
            assert_effective_overrides_consumed()
        if perf_enabled:
            perf_out["override_prep_ms"] = (t1 - t0) * 1000.0
            perf_out["base_exec_ms"] = (t3 - t2) * 1000.0
        return output


def attach_execute_model_compression_events(
    *,
    output: Any,
    pending_events: list[dict[str, Any]],
    scheduler_output: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to ModelRunnerOutput when possible.

    In vLLM V1's async path, ``execute_model`` returns ``None`` and the
    actual ``ModelRunnerOutput`` is produced later.  When that happens, attach
    events to ``scheduler_output`` as a same-process fallback, but keep them
    pending so the later ``sample_tokens`` output can carry them across
    executor process boundaries.

    Returns ``(output, remaining_pending_events)``.
    """
    applied_count = sum(1 for e in pending_events if e.get("status") == "applied")
    if output is None:
        if scheduler_output is not None and pending_events:
            setattr(
                scheduler_output,
                "triattention_compression_events",
                pending_events,
            )
            if runtime_logging_enabled():
                logger.debug(
                    "attach_events: output=None, attached %d events (%d applied) to scheduler_output (id=%d)",
                    len(pending_events), applied_count, id(scheduler_output),
                )
            return output, pending_events
        return output, pending_events
    _attach_triattention_events_via_kv_cache_events(output, pending_events)
    try:
        setattr(output, "triattention_compression_events", pending_events)
        if applied_count > 0 and runtime_logging_enabled():
            logger.debug(
                "attach_events: attached %d events (%d applied) to output type=%s",
                len(pending_events), applied_count, type(output).__name__,
            )
    except Exception:
        # Keep pending events for sample_tokens fallback path.
        return output, pending_events
    return output, []


def _attach_triattention_events_via_kv_cache_events(
    output: Any,
    pending_events: list[dict[str, Any]],
) -> bool:
    """Attach events through the vLLM declared cross-process output field."""
    if output is None or not pending_events:
        return False
    try:
        target = output
        if not hasattr(target, "kv_connector_output"):
            for attr_name in ("model_runner_output", "_model_runner_output"):
                candidate = getattr(target, attr_name, None)
                if candidate is not None and hasattr(candidate, "kv_connector_output"):
                    target = candidate
                    break
        kco = getattr(target, "kv_connector_output", None)
        if kco is None:
            from vllm.v1.outputs import KVConnectorOutput  # noqa: PLC0415

            kco = KVConnectorOutput()
            setattr(target, "kv_connector_output", kco)
        kco.kv_cache_events = _TriattentionEventBag(pending_events)
        return True
    except Exception:
        return False


def _read_triattention_events_from_kv_cache_events(
    model_runner_output: Any,
) -> list[dict[str, Any]] | None:
    """Read TriAttention events from the vLLM declared cross-process field."""
    kco = getattr(model_runner_output, "kv_connector_output", None)
    if kco is None:
        return None
    bag = getattr(kco, "kv_cache_events", None)
    if bag is None:
        return None
    events = getattr(bag, "events", None)
    if not isinstance(events, list):
        return None
    return events


def attach_sample_tokens_compression_events(
    *,
    output: Any,
    pending_events: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to sample_tokens output (fallback path)."""
    if output is None:
        return None, []
    _attach_triattention_events_via_kv_cache_events(output, pending_events)
    setattr(output, "triattention_compression_events", pending_events)
    return output, []
