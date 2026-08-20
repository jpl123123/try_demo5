"""Install kvpress-ascend monkeypatches into vLLM / vLLM-Ascend.

Patches (none of them modify vllm-ascend source files):

- ``vllm.v1.core.sched.scheduler.Scheduler``: ``__init__`` (attach config +
  effective-length tracker), ``schedule`` (effective-offset sync + signals),
  ``update_from_output`` (compression events + block reclaim);
- ``vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots``: allocation
  aligned with the effective (compressed) length;
- ``vllm_ascend.worker.worker.NPUWorker``: ``init_device`` / ``execute_model``
  to install the KVPressModelRunner proxy;
- ``vllm_ascend.worker.model_runner_v1.NPUModelRunner._prepare_inputs``:
  effective seq_lens/positions/slot-mapping overrides;
- relaxed KV-cache memory check (vLLM) so long ``--max-model-len`` starts when
  compression keeps real usage low.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..envs import KVPressRuntimeConfig
from ..logging_control import log_debug, log_info, log_warning
from .scheduler_hooks import (
    EffectiveLenTracker,
    apply_compression_events,
    build_signals,
    cleanup_finished_requests,
    sync_effective_kv_offsets_before_schedule,
)
from .worker_hooks import install_worker_patches

_PATCHED = False
_ORIG_SCHED_INIT: Optional[Callable[..., Any]] = None
_ORIG_SCHED_SCHEDULE: Optional[Callable[..., Any]] = None
_ORIG_SCHED_UPDATE_FROM_OUTPUT: Optional[Callable[..., Any]] = None
_ORIG_KVCACHE_ALLOCATE_SLOTS: Optional[Callable[..., Any]] = None


def _patched_scheduler_init(self, *args, **kwargs):
    assert _ORIG_SCHED_INIT is not None
    _ORIG_SCHED_INIT(self, *args, **kwargs)
    cfg = KVPressRuntimeConfig.from_env()
    self.kvpress_config = cfg
    self._kvpress_effective_len_tracker = EffectiveLenTracker()
    self.kvpress_prefill_lens = {}
    self._kvpress_step = 0
    if cfg.logging_enabled:
        log_info(
            "Scheduler initialized: press=%s ratio=%s budget=%s "
            "min_reclaim_blocks=%d defer_prefill=%s build=%s",
            cfg.press_name,
            cfg.compression_ratio,
            cfg.kv_budget or "auto",
            cfg.min_reclaim_blocks,
            cfg.defer_prefill_compression,
            cfg.build_id,
        )


def _patched_scheduler_schedule(self):
    assert _ORIG_SCHED_SCHEDULE is not None
    cfg = getattr(self, "kvpress_config", None)
    if cfg is None:
        return _ORIG_SCHED_SCHEDULE(self)

    tracker = getattr(self, "_kvpress_effective_len_tracker", None)
    if tracker is not None:
        sync_effective_kv_offsets_before_schedule(self, tracker)

    scheduler_output = _ORIG_SCHED_SCHEDULE(self)

    self._kvpress_step += 1
    block_size = int(getattr(self, "block_size", 0) or 0)
    if block_size <= 0:
        block_size = int(getattr(cfg, "block_size_hint", 0) or 0)
    if block_size <= 0:
        block_size = 128

    # Record prefill lengths from newly scheduled requests (best effort).
    new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    prefill_lens = getattr(self, "kvpress_prefill_lens", None)
    if isinstance(new_reqs, (list, tuple)) and isinstance(prefill_lens, dict):
        for item in new_reqs:
            request = item if not isinstance(item, tuple) else item[-1]
            req_id = getattr(request, "req_id", None)
            if req_id is None:
                continue
            num_prompt = getattr(request, "num_prompt_tokens", None)
            if num_prompt is not None:
                prefill_lens[str(req_id)] = int(num_prompt)

    # Scheduler-side candidate budget. kvpress ratio-based budgets depend on
    # the request length, which the scheduler does not know precisely; use a
    # reclaim-driven candidate threshold: with keep = (1-ratio)*T, the request
    # can reclaim min_reclaim_blocks only once
    # T >= min_reclaim_blocks*block_size/ratio. The worker validates the real
    # reclaim before compacting.
    ratio = max(0.01, min(0.99, float(cfg.compression_ratio)))
    if cfg.kv_budget > 0:
        candidate_budget = cfg.kv_budget
    else:
        candidate_budget = max(
            1,
            int(max(0, cfg.min_reclaim_blocks) * max(1, block_size) / ratio),
        )

    signals = build_signals(
        self,
        scheduler_output,
        tracker if tracker is not None else EffectiveLenTracker(),
        budget=candidate_budget,
        min_reclaim_blocks=cfg.min_reclaim_blocks,
        block_size=block_size,
        protect_prefill=cfg.protect_prefill,
        defer_prefill_compression=cfg.defer_prefill_compression,
        step=self._kvpress_step,
        log_decisions=cfg.log_decisions,
    )
    setattr(scheduler_output, "kvpress_signals", signals)
    setattr(scheduler_output, "kvpress_step", self._kvpress_step)
    return scheduler_output


def _patched_scheduler_update_from_output(self, scheduler_output, model_runner_output):
    assert _ORIG_SCHED_UPDATE_FROM_OUTPUT is not None
    outputs = _ORIG_SCHED_UPDATE_FROM_OUTPUT(self, scheduler_output, model_runner_output)

    cfg = getattr(self, "kvpress_config", None)
    tracker = getattr(self, "_kvpress_effective_len_tracker", None)
    if cfg is None or tracker is None:
        return outputs

    from .scheduler_hooks import _read_compression_events

    events = _read_compression_events(scheduler_output, model_runner_output)
    if events:
        applied = [e for e in events if e.get("status") == "applied"]
        log_debug(
            "update_from_output: received %d events (%d applied)",
            len(events), len(applied),
        )
        block_size = int(getattr(self, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = 128
        apply_compression_events(
            self,
            tracker,
            events,
            block_size=block_size,
            enable_block_reclaim=cfg.enable_experimental_block_reclaim,
            log_decisions=cfg.log_decisions,
        )

    # Advance effective lengths for requests that were compressed and are still
    # running (their tokens written after the compacted prefix this step).
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if isinstance(scheduled, dict):
        for req_id, num in scheduled.items():
            if tracker.get(str(req_id)) is None:
                continue
            try:
                tracker.advance(str(req_id), max(1, int(num)))
            except (TypeError, ValueError):
                continue

    cleanup_finished_requests(self, tracker)
    return outputs


def _patched_kv_cache_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    effective = getattr(request, "_kvpress_effective_kv_offset", None)
    logical = getattr(request, "num_computed_tokens", None)
    if not isinstance(effective, int) or not isinstance(logical, int):
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(self, request, num_new_tokens, *args, **kwargs)
    if effective >= logical:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(self, request, num_new_tokens, *args, **kwargs)
    kwargs = dict(kwargs)
    kwargs["delay_cache_blocks"] = True
    setattr(request, "num_computed_tokens", effective)
    try:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(self, request, num_new_tokens, *args, **kwargs)
    finally:
        setattr(request, "num_computed_tokens", logical)


def _relax_kv_cache_memory_check() -> None:
    try:
        import vllm.v1.core.kv_cache_utils as kv_utils

        legacy_check = getattr(kv_utils, "_check_enough_kv_cache_memory", None)
        if callable(legacy_check) and not getattr(legacy_check, "_kvpress_relaxed", False):

            def _relaxed_legacy(available_memory, get_needed_memory, max_model_len, estimate_max_model_len):
                if available_memory <= 0:
                    legacy_check(available_memory, get_needed_memory, max_model_len, estimate_max_model_len)
                    return
                needed = get_needed_memory()
                if needed > available_memory:
                    log_info(
                        "KV cache check relaxed: max_model_len=%d needs %.2f GiB but only "
                        "%.2f GiB available; kvpress compression keeps actual usage lower.",
                        max_model_len, needed / (1 << 30), available_memory / (1 << 30),
                    )

            setattr(_relaxed_legacy, "_kvpress_relaxed", True)
            kv_utils._check_enough_kv_cache_memory = _relaxed_legacy
    except Exception:
        log_warning("could not relax KV cache memory check", exc_info=True)


def _install_input_patch() -> bool:
    patched: list[str] = []
    try:
        import vllm_ascend.worker.model_runner_v1 as ascend_runner_v1

        original = getattr(ascend_runner_v1.NPUModelRunner, "_prepare_inputs", None)
        if callable(original) and not getattr(original, "_kvpress_patched", False):
            from .input_patch_v1 import make_patched_v1_prepare_inputs

            ascend_runner_v1.NPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(original)
            patched.append("vllm_ascend.worker.model_runner_v1.NPUModelRunner._prepare_inputs")
    except Exception:
        log_warning("could not install kvpress input patch (v1)", exc_info=True)
    if patched:
        log_info("Installed kvpress input patches: %s", ", ".join(patched))
    return bool(patched)


def install_kvpress_integration_monkeypatches(
    *,
    patch_scheduler: bool = True,
    patch_worker: bool = True,
) -> None:
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_KVCACHE_ALLOCATE_SLOTS
    if _PATCHED:
        return

    cfg = KVPressRuntimeConfig.from_env()
    if patch_scheduler:
        try:
            import vllm.v1.core.sched.scheduler as sched_mod
            import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod

            Scheduler = sched_mod.Scheduler
            KVCacheManager = kv_cache_manager_mod.KVCacheManager
            _ORIG_SCHED_INIT = Scheduler.__init__
            _ORIG_SCHED_SCHEDULE = Scheduler.schedule
            _ORIG_SCHED_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
            Scheduler.__init__ = _patched_scheduler_init
            Scheduler.schedule = _patched_scheduler_schedule
            Scheduler.update_from_output = _patched_scheduler_update_from_output
            _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
            KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots
        except Exception:
            log_warning("could not patch vLLM Scheduler/KVCacheManager", exc_info=True)
            raise

    if patch_worker:
        install_worker_patches()
        if cfg.preinstall_input_patch:
            _install_input_patch()

    _relax_kv_cache_memory_check()
    _PATCHED = True
    if cfg.logging_enabled:
        log_info(
            "Installed kvpress monkeypatch integration: scheduler=%s worker=%s "
            "press=%s build=%s",
            patch_scheduler,
            patch_worker,
            cfg.press_name,
            cfg.build_id,
        )
