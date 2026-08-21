"""KVPress model-runner proxy for kvpress-ascend.

Wraps the vLLM-Ascend ``NPUModelRunner`` and converts kvpress's per-step
compression loop (``BasePress`` hooks + ``press.compress`` on HF caches) into
the Ascend boundary flow:

1. pre-step: consume scheduler signals, validate on the worker side, run
   ``compress_request`` (gather -> press score -> per-head in-place block-cache
   compaction), shrink worker block rows, set effective input overrides and
   trim stale ``new_block_ids``;
2. forward: call the base runner (the patched ``_prepare_inputs`` applies the
   compressed view; attention hooks capture the fresh queries for the next
   boundary);
3. post-step: advance per-request effective length, attach compression events
   to the output, emit the per-inference probe log.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.press_bridge import build_press
from ..envs import KVPressRuntimeConfig
from ..logging_control import log_debug, log_info, log_warning, probe
from . import input_patch_state as _patch_state
from .attention_hooks import AttentionHooks
from .block_sync import (
    apply_worker_block_reclaim,
    truncate_request_state_block_ids,
)
from .compression_engine import (
    _request_block_capacity,
    _request_block_ids,
    _request_token_count,
    _table_block_size,
    compress_request,
)
from .thresholds import (
    compression_length_threshold,
    is_prefill_phase_for_limit,
    is_request_scheduled_as_prefill,
    resolve_request_prefill_len,
)
from .output_bridge import attach_events_to_output
from .request_key_compat import (
    iter_scheduled_new_requests,
    iter_scheduled_token_items,
)
from .signals import CompressionSignal, signals_from_scheduler_output
from .state import RequestStateStore


def _scheduled_items(scheduler_output: Any) -> list[tuple[str, int]]:
    """Normalized scheduled items (vLLM keys may be request objects)."""
    from .request_key_compat import iter_scheduled_token_items

    return list(iter_scheduled_token_items(scheduler_output))


def _resolve_prefill_len(request: Any) -> int:
    for attr in ("num_prompt_tokens", "prefill_len"):
        value = getattr(request, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    state = getattr(request, "state", None)
    if state is not None:
        value = getattr(state, "prefill_len", None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


class KVPressModelRunner:
    """Proxy around the vLLM-Ascend model runner."""

    def __init__(
        self,
        base_runner: Any,
        config: Optional[KVPressRuntimeConfig] = None,
    ):
        self._base_runner = base_runner
        self.config = config or KVPressRuntimeConfig.from_env()
        self.state_store = RequestStateStore()
        self.hooks = AttentionHooks()
        self.press = build_press(
            self.config.press_name,
            self.config.compression_ratio,
            window_size=self.config.window_size,
            n_sink=self.config.n_sink,
            seed=self.config.seed,
            target_size=self.config.target_size,
            compression_interval=self.config.compression_interval,
            prefer_installed=self.config.use_installed_kvpress,
        )
        self._last_step = 0
        self._pending_compression_events: list[dict[str, Any]] = []
        # 0 = score every layer (per-layer keep sets, best fidelity). A small
        # positive value samples scoring and reuses the nearest scored layer's
        # keep set for the rest (documented approximation).
        self._max_layers_to_score = int(self.config.max_layers_to_score)
        self._logged_new_request = False
        self._install_hooks()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_runner, name)

    # ------------------------------------------------------------------ hooks

    def _install_hooks(self) -> None:
        model = getattr(self._base_runner, "model", None)
        if model is None:
            log_warning("runner proxy: model not found; attention hooks disabled")
            return
        hooked = self.hooks.install(model)
        if hooked == 0:
            log_warning(
                "attention hooks found 0 layers; query-based presses will fall "
                "back to recency selection (keys-only presses still score "
                "directly from the block cache)"
            )
        if self.config.logging_enabled:
            log_info(
                "attention hooks installed: layers=%d press=%s ratio=%s "
                "budget=%s defer_prefill=%s build=%s",
                hooked,
                self.press.__class__.__name__,
                self.config.compression_ratio,
                self.config.kv_budget or "auto",
                self.config.defer_prefill_compression,
                self.config.build_id,
            )

    # ------------------------------------------------------------ pre-step

    def _register_new_requests(self, scheduler_output: Any) -> None:
        for req_id, request, num_prompt_tokens in iter_scheduled_new_requests(
            scheduler_output
        ):
            prefill_len = _resolve_prefill_len(request) or int(num_prompt_tokens)
            state = self.state_store.ensure(
                str(req_id),
                prefill_len=prefill_len,
                protect_prefill=self.config.protect_prefill,
            )
            if not self._logged_new_request and self.config.logging_enabled:
                log_info(
                    "registered request req=%s prefill_len=%d protect_prefill=%s",
                    req_id, prefill_len, self.config.protect_prefill,
                )
                self._logged_new_request = True

    def _cleanup_finished(self, scheduler_output: Any) -> None:
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if isinstance(finished, (list, tuple, set)):
            self.state_store.cleanup_finished([str(r) for r in finished])

    def _ensure_state_for_existing_request(self, req_id: str) -> Any:
        """Backfill request state lazily from the worker surfaces when the
        new-req registration did not run (mirrors TriAttention)."""
        state = self.state_store.get(req_id)
        if state is not None:
            return state
        prefill_len = 0
        requests_dict = getattr(self._base_runner, "requests", None)
        if isinstance(requests_dict, dict):
            req_state = requests_dict.get(req_id)
            if req_state is not None:
                prefill_len = resolve_request_prefill_len(req_state)
                if prefill_len <= 0:
                    prefill_len = int(getattr(req_state, "num_prompt_tokens", 0) or 0)
        if prefill_len <= 0:
            input_batch = getattr(self._base_runner, "input_batch", None)
            if input_batch is not None:
                req_id_to_index = getattr(input_batch, "req_id_to_index", None)
                num_prompt = getattr(input_batch, "num_prompt_tokens", None)
                if isinstance(req_id_to_index, dict) and num_prompt is not None:
                    req_index = req_id_to_index.get(req_id)
                    if isinstance(req_index, int):
                        try:
                            prefill_len = int(num_prompt[req_index])
                        except Exception:
                            pass
        state = self.state_store.ensure(
            str(req_id),
            prefill_len=prefill_len,
            protect_prefill=self.config.protect_prefill,
        )
        if self.config.log_decisions:
            log_debug(
                "backfilled runtime state for scheduled request: req=%s prefill_len=%d",
                req_id, prefill_len,
            )
        return state

    def _sync_worker_num_computed(self, scheduler_output: Any) -> None:
        """Mirror vLLM's num_computed_tokens into request state each step."""
        requests_dict = getattr(self._base_runner, "requests", None)
        input_batch = getattr(self._base_runner, "input_batch", None)
        req_id_to_index = (
            getattr(input_batch, "req_id_to_index", None) if input_batch else None
        )
        num_computed_cpu = getattr(input_batch, "num_computed_tokens_cpu", None) if input_batch else None
        for req_id, _scheduled in _scheduled_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None:
                continue
            nct = 0
            req_state = requests_dict.get(req_id) if isinstance(requests_dict, dict) else None
            if req_state is not None:
                nct = int(getattr(req_state, "num_computed_tokens", 0) or 0)
            if nct <= 0 and isinstance(req_id_to_index, dict) and num_computed_cpu is not None:
                req_index = req_id_to_index.get(req_id)
                if isinstance(req_index, int):
                    try:
                        nct = int(num_computed_cpu[req_index])
                    except Exception:
                        pass
            if nct > int(state.num_computed_tokens):
                state.num_computed_tokens = nct

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        return signals_from_scheduler_output(scheduler_output)

    def _compression_threshold(self, block_size: int) -> int:
        if self.config.kv_budget > 0:
            budget = self.config.kv_budget
        else:
            # Ratio-based press: the scheduler cannot know the per-request
            # budget before the length is known; use a reclaim-driven candidate
            # (keep ~ (1-ratio)*len, so reclaim >= min_reclaim_blocks once
            # len >= min_reclaim_blocks*block_size/ratio). The worker validates
            # the real budget in _execute_compression.
            ratio = max(0.01, min(0.99, float(self.config.compression_ratio)))
            budget = max(
                1,
                int(max(0, self.config.min_reclaim_blocks) * max(1, block_size) / ratio),
            )
        return compression_length_threshold(
            kv_budget=budget,
            min_reclaim_blocks=self.config.min_reclaim_blocks,
            block_size=block_size,
            protect_prefill=self.config.protect_prefill,
            include_prefill_in_budget=self.config.include_prefill_in_budget,
        )

    def _worker_self_triggers(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> dict[str, CompressionSignal]:
        """Supplement scheduler signals from worker-side state (async lag)."""
        block_size = _table_block_size(self._base_runner)
        threshold = self._compression_threshold(block_size)
        step = self._last_step
        block_size = _table_block_size(self._base_runner)
        for req_id, scheduled_tokens in _scheduled_items(scheduler_output):
            if req_id in signals:
                continue
            state = self._ensure_state_for_existing_request(req_id)
            if state.is_compressed:
                effective = int(state.current_cache_len)
            else:
                # First-compression self-trigger from the ACTUAL worker-side
                # length (block capacity / mirrored num_computed) — the
                # TriAttention philosophy: the worker derives the length
                # itself, so engine-core signal lag or an unpatched
                # engine-core scheduler cannot block compression.
                effective = int(state.num_computed_tokens)
                if effective <= 0:
                    actual = _request_block_capacity(self._base_runner, req_id)
                    if actual is not None:
                        effective = int(actual)
                if effective <= 0:
                    continue
            if effective + scheduled_tokens <= threshold:
                continue
            if self.config.defer_prefill_compression and is_prefill_phase_for_limit(
                scheduler_output=scheduler_output,
                req_id=req_id,
                scheduled_tokens=scheduled_tokens,
                prefill_len=int(state.prefill_len),
                num_computed_tokens=effective,
            ):
                continue
            signals[req_id] = CompressionSignal(
                req_id=req_id,
                should_compress=True,
                reason="worker_length_threshold",
                step=step,
                estimated_cache_len=effective,
                prefill_len=int(state.prefill_len),
                scheduled_tokens=scheduled_tokens,
                protect_prefill=self.config.protect_prefill,
            )
            log_debug(
                "worker self-trigger req=%s effective=%d scheduled=%d threshold=%d",
                req_id, effective, scheduled_tokens, threshold,
            )
        return signals

    def _is_prefill_step(self, state: Any, scheduled_tokens: int) -> bool:
        if state is None:
            return False
        return 0 < int(state.current_cache_len or 0) < int(state.prefill_len or 0)

    def _execute_compression(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        block_size = _table_block_size(self._base_runner)
        for req_id, signal in signals.items():
            if not signal.should_compress:
                continue
            state = self.state_store.ensure(
                req_id,
                prefill_len=int(signal.prefill_len or 0),
                protect_prefill=self.config.protect_prefill,
            )
            scheduled_tokens = max(1, int(signal.scheduled_tokens or 1))
            # Batch-queue dedup: consecutive decode steps with <=1 scheduled
            # token should not recompress immediately after a compression.
            if (
                state.compression_count > 0
                and state.last_compression_step >= 0
                and signal.step - state.last_compression_step <= 1
                and scheduled_tokens <= 1
                and not bool(signal.force)
            ):
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "skipped",
                        "reason": "batch_queue_dedup",
                        "cache_len_after": int(state.current_cache_len),
                    }
                )
                continue

            total_tokens = _request_token_count(self._base_runner, req_id, block_size)
            if total_tokens is None or total_tokens <= 0:
                block_ids = _request_block_ids(self._base_runner, req_id)
                if block_ids:
                    total_tokens = len(block_ids) * block_size
            if total_tokens is None or total_tokens <= 0:
                continue

            if self.config.defer_prefill_compression and is_prefill_phase_for_limit(
                scheduler_output=scheduler_output,
                req_id=req_id,
                scheduled_tokens=scheduled_tokens,
                prefill_len=int(state.prefill_len),
                num_computed_tokens=_request_block_capacity(self._base_runner, req_id),
            ):
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "skipped",
                        "reason": "prefill_incomplete",
                        "cache_len_after": int(total_tokens),
                    }
                )
                continue

            budget = self.config.resolved_budget(int(total_tokens))
            if (
                int(state.cache_len_after_last_compression) > 0
                and int(state.cache_len_after_last_compression) < total_tokens
            ):
                # Re-compression: keep the press budget; never downgrade below
                # the previous compressed length.
                budget = max(int(budget), int(state.cache_len_after_last_compression))
            keep_count = min(int(budget), int(total_tokens))

            event = compress_request(
                base_runner=self._base_runner,
                req_id=req_id,
                keep_count=keep_count,
                total_tokens=int(total_tokens),
                block_size=block_size,
                press=self.press,
                hooks=self.hooks,
                max_layers_to_score=self._max_layers_to_score,
                min_reclaim_blocks=self.config.min_reclaim_blocks,
                scheduled_tokens=scheduled_tokens,
            )
            if event.get("status") == "applied":
                retained = int(event.get("retained_cache_len", keep_count))
                required_blocks = (retained + block_size - 1) // block_size
                # Multi-group (MTP) models: reclaim each compressible group's
                # own row using the per-gid reclaim plan from the event.
                details = event.get("details")
                reclaim_groups = (
                    details.get("block_reclaim", {}).get("groups")
                    if isinstance(details, dict) and isinstance(details.get("block_reclaim"), dict)
                    else None
                )
                if isinstance(reclaim_groups, list) and reclaim_groups:
                    for group in reclaim_groups:
                        gid = int(group.get("gid", 0))
                        apply_worker_block_reclaim(
                            base_runner=self._base_runner,
                            req_id=req_id,
                            retained_cache_len=retained,
                            block_size=block_size,
                            gid=gid,
                        )
                        after = group.get("block_ids_after")
                        if isinstance(after, list):
                            truncate_request_state_block_ids(
                                base_runner=self._base_runner,
                                req_id=req_id,
                                required_blocks=len(after),
                                gid=gid,
                            )
                else:
                    apply_worker_block_reclaim(
                        base_runner=self._base_runner,
                        req_id=req_id,
                        retained_cache_len=retained,
                        block_size=block_size,
                    )
                    truncate_request_state_block_ids(
                        base_runner=self._base_runner,
                        req_id=req_id,
                        required_blocks=required_blocks,
                    )
                scheduler_nct = 0
                requests_dict = getattr(self._base_runner, "requests", None)
                if isinstance(requests_dict, dict):
                    req_state = requests_dict.get(req_id)
                    if req_state is not None:
                        scheduler_nct = int(
                            getattr(req_state, "num_computed_tokens", 0) or 0
                        )
                self.state_store.mark_compressed(
                    req_id,
                    step=signal.step,
                    cache_len=int(event.get("cache_len_after", keep_count)),
                    scheduled_tokens=scheduled_tokens,
                    scheduler_nct=scheduler_nct,
                )
            else:
                self.state_store.mark_compression_skipped(
                    req_id, event.get("reason", "unknown"), signal.step
                )
            events.append(event)
        return events

    def _trim_stale_new_block_ids(self, scheduler_output: Any) -> None:
        """Cap new_block_ids for compressed requests (async lookahead safety)."""
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return
        req_ids = getattr(cached_reqs, "req_ids", None)
        new_block_ids_list = getattr(cached_reqs, "new_block_ids", None)
        if not isinstance(req_ids, list) or not isinstance(new_block_ids_list, list):
            return
        if len(req_ids) != len(new_block_ids_list):
            return
        block_size = _table_block_size(self._base_runner)
        for i, req_id in enumerate(req_ids):
            state = self.state_store.get(str(req_id))
            if state is None or not state.is_compressed:
                continue
            group = new_block_ids_list[i]
            if not isinstance(group, (list, tuple)):
                continue
            required_blocks = (
                int(state.current_cache_len) + block_size - 1
            ) // block_size
            current_blocks = 0
            block_ids = _request_block_ids(self._base_runner, str(req_id))
            if block_ids:
                current_blocks = len(block_ids)
            limit = max(0, required_blocks - current_blocks)
            if len(group) > limit:
                trimmed = list(group[:limit])
                new_block_ids_list[i] = (
                    tuple(trimmed) if isinstance(group, tuple) else trimmed
                )
                log_debug(
                    "trimmed new_block_ids req=%s %d -> %d", req_id, len(group), limit,
                )

    def _prepare_effective_overrides(self, scheduler_output: Any) -> bool:
        """Set effective input overrides for compressed requests in the batch."""
        input_batch = getattr(self._base_runner, "input_batch", None)
        req_id_to_index = (
            getattr(input_batch, "req_id_to_index", None)
            if input_batch is not None
            else None
        )
        if not isinstance(req_id_to_index, dict):
            return False
        requests_dict = getattr(self._base_runner, "requests", None)
        base_by_req_idx: dict[int, int] = {}
        for req_id, _scheduled in _scheduled_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None or not state.is_compressed:
                continue
            req_idx = req_id_to_index.get(req_id)
            if not isinstance(req_idx, int):
                continue
            effective_base = int(state.current_cache_len)
            num_computed = 0
            if isinstance(requests_dict, dict):
                req_state = requests_dict.get(req_id)
                if req_state is not None:
                    num_computed = int(getattr(req_state, "num_computed_tokens", 0) or 0)
            if num_computed <= 0:
                num_computed = int(getattr(state, "num_computed_tokens", 0) or 0)
            if effective_base < num_computed:
                base_by_req_idx[int(req_idx)] = effective_base
        if not base_by_req_idx:
            return False
        _patch_state.set_effective_bases(base_by_req_idx)
        return True

    # ------------------------------------------------------------- main API

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        step = self._last_step + 1
        self._last_step = step
        self.hooks.reset_step(step)
        self._register_new_requests(scheduler_output)
        self._cleanup_finished(scheduler_output)
        self._sync_worker_num_computed(scheduler_output)

        signals = self._consume_signals(scheduler_output)
        signals = self._worker_self_triggers(scheduler_output, signals)
        events = self._execute_compression(scheduler_output, signals)
        self._pending_compression_events = events
        applied_count = sum(1 for e in events if e.get("status") == "applied")

        self._prepare_effective_overrides(scheduler_output)
        self._trim_stale_new_block_ids(scheduler_output)

        try:
            output = self._base_runner.execute_model(
                scheduler_output=scheduler_output,
                intermediate_tensors=intermediate_tensors,
            )
        finally:
            self._advance_compressed_lengths(scheduler_output)

        output, remaining = attach_events_to_output(output, events, scheduler_output)
        self._pending_compression_events = remaining

        if self.config.probe_enabled:
            events_by_req: dict[str, str] = {}
            for e in events:
                if isinstance(e, dict) and e.get("req_id") is not None:
                    events_by_req[str(e["req_id"])] = str(e.get("status", "none"))
            for req_id, scheduled_tokens in _scheduled_items(scheduler_output):
                state = self.state_store.get(req_id)
                seq_len = int(state.current_cache_len) if state is not None else 0
                budget = self.config.resolved_budget(max(1, seq_len + scheduled_tokens))
                probe(
                    "step=%d req=%s core_entered=1 hook_entered=%d "
                    "press=%s ratio=%.3f seq_len=%d budget=%d keep=%d "
                    "reclaimed_blocks=%d compress_events=%d last_event=%s",
                    step,
                    req_id,
                    int(self.hooks.captured_this_step),
                    self.press.__class__.__name__,
                    self.config.compression_ratio,
                    seq_len,
                    budget,
                    min(budget, seq_len),
                    sum(
                        1
                        for e in events
                        if e.get("req_id") == req_id
                        and (e.get("details") or {}).get("reclaimed_block_count", 0) > 0
                    ),
                    applied_count,
                    events_by_req.get(req_id, "none"),
                )
        return output

    def _advance_compressed_lengths(self, scheduler_output: Any) -> None:
        for req_id, scheduled_tokens in _scheduled_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None or not state.is_compressed:
                continue
            self.state_store.advance_cache_len(req_id, scheduled_tokens, self._last_step)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        """Delegate sampling and attach any pending compression events."""
        sample_fn = getattr(self._base_runner, "sample_tokens", None)
        if not callable(sample_fn):
            raise RuntimeError("KVPressModelRunner: base runner has no sample_tokens")
        output = sample_fn(grammar_output)
        if self._pending_compression_events:
            output, self._pending_compression_events = attach_events_to_output(
                output, self._pending_compression_events
            )
        return output
