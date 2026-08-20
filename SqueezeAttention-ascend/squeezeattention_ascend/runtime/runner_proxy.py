"""SqueezeAttention model-runner proxy for SqueezeAttention-ascend.

Converts SqueezeAttention's HF generation loop into the Ascend boundary flow:

1. pre-step: consume scheduler signals; validate worker-side; run the
   SqueezeAttention compression (per-layer recency keep sets on the block
   cache); shrink worker block rows; set effective input overrides; trim stale
   ``new_block_ids``;
2. during prefill: accumulate per-layer cosine-similarity (hidd_data) per
   request from the attention hooks; when the request's prefill completes,
   finalize the KMeans layer-wise budgets;
3. forward + post-step: advance effective lengths, attach events, probe log.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.budgets import LayerImportanceAccumulator
from ..envs import SqueezeRuntimeConfig
from ..logging_control import cluster_log, log_debug, log_info, log_warning, probe
from . import input_patch_state as _patch_state
from .attention_hooks import SqueezeAttentionHooks
from .block_sync import apply_worker_block_reclaim, truncate_request_state_block_ids
from .compression_engine import (
    _request_block_ids,
    _request_token_count,
    _table_block_size,
    compress_request,
    finalize_budgets,
)
from .output_bridge import attach_events_to_output
from .signals import CompressionSignal, signals_from_scheduler_output
from .state import RequestStateStore


def _scheduled_items(scheduler_output: Any) -> list[tuple[str, int]]:
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return []
    items = []
    for req_id, num in scheduled.items():
        try:
            items.append((str(req_id), max(1, int(num))))
        except (TypeError, ValueError):
            continue
    return items


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


class SqueezeAttentionModelRunner:
    """Proxy around the vLLM-Ascend model runner."""

    def __init__(
        self,
        base_runner: Any,
        config: Optional[SqueezeRuntimeConfig] = None,
    ):
        self._base_runner = base_runner
        self.config = config or SqueezeRuntimeConfig.from_env()
        self.state_store = RequestStateStore()
        self.hooks = SqueezeAttentionHooks()
        self.hooks.capture_queries = bool(
            self.config.mode == "class_weighted" and self.config.fake_key_padding
        )
        self._last_step = 0
        self._pending_compression_events: list[dict[str, Any]] = []
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
        if self.config.logging_enabled:
            log_info(
                "attention hooks installed: layers=%d mode=%s ini_size=%.3f "
                "class3=%.3f start_size=%d build=%s",
                hooked,
                self.config.mode,
                self.config.ini_size,
                self.config.class3_size,
                self.config.start_size,
                self.config.build_id,
            )

    # ------------------------------------------------------------ pre-step

    def _register_new_requests(self, scheduler_output: Any) -> None:
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        if not isinstance(new_reqs, (list, tuple)):
            return
        for item in new_reqs:
            request = item if not isinstance(item, tuple) else item[-1]
            req_id = getattr(request, "req_id", None)
            if req_id is None:
                continue
            prefill_len = _resolve_prefill_len(request)
            state = self.state_store.ensure(str(req_id), prefill_len=prefill_len)
            if "importance" not in state.extra:
                state.extra["importance"] = LayerImportanceAccumulator()
                state.extra["budgets_ready"] = False
                state.extra["sliding_windows"] = []
            if not self._logged_new_request and self.config.logging_enabled:
                log_info(
                    "registered request req=%s prefill_len=%d mode=%s",
                    req_id, prefill_len, self.config.mode,
                )
                self._logged_new_request = True

    def _cleanup_finished(self, scheduler_output: Any) -> None:
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if isinstance(finished, (list, tuple, set)):
            self.state_store.cleanup_finished([str(r) for r in finished])

    def _sync_worker_num_computed(self, scheduler_output: Any) -> None:
        """Mirror vLLM's num_computed_tokens into request state each step
        (kvpress's cache_position equivalent for prefill tracking)."""
        requests_dict = getattr(self._base_runner, "requests", None)
        if not isinstance(requests_dict, dict):
            return
        for req_id, _scheduled in _scheduled_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None:
                continue
            req_state = requests_dict.get(req_id)
            if req_state is None:
                continue
            nct = int(getattr(req_state, "num_computed_tokens", 0) or 0)
            if nct > int(state.num_computed_tokens):
                state.num_computed_tokens = nct

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        return signals_from_scheduler_output(scheduler_output)

    def _request_keep_count(self, state: Any, total_tokens: int, block_size: int) -> int:
        budgets = state.extra.get("sliding_windows") or []
        return self.config.resolved_k(budgets, block_size, total_tokens)

    def _compression_threshold(self, state: Any, block_size: int) -> int:
        keep_count = self._request_keep_count(state, 1 << 30, block_size)
        return max(1, keep_count + max(0, self.config.min_reclaim_blocks) * max(1, block_size))

    def _worker_self_triggers(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> dict[str, CompressionSignal]:
        block_size = _table_block_size(self._base_runner)
        for req_id, scheduled_tokens in _scheduled_items(scheduler_output):
            if req_id in signals:
                continue
            state = self.state_store.get(req_id)
            if state is None or not state.is_compressed:
                continue
            threshold = self._compression_threshold(state, block_size)
            effective = int(state.current_cache_len)
            if effective + scheduled_tokens <= threshold:
                continue
            signals[req_id] = CompressionSignal(
                req_id=req_id,
                should_compress=True,
                reason="worker_length_threshold",
                step=self._last_step,
                estimated_cache_len=effective,
                prefill_len=int(state.prefill_len),
                scheduled_tokens=scheduled_tokens,
            )
            log_debug(
                "worker self-trigger req=%s effective=%d scheduled=%d threshold=%d",
                req_id, effective, scheduled_tokens, threshold,
            )
        return signals

    def _capture_prefill_importance(self, scheduler_output: Any) -> None:
        """Compute per-token layer similarities AFTER the forward and slice
        them into per-request accumulators.

        The hooks only store tensor references (the forward must stay
        transparent to torch.compile); the actual cosine-similarity math runs
        here, outside any compiled region.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        input_batch = getattr(self._base_runner, "input_batch", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0)) if input_batch else 0
        if num_reqs <= 0:
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, dict):
            return
        arange_np = getattr(self._base_runner, "arange_np", np.arange(num_reqs))
        num_scheduled_tokens = np.zeros(num_reqs, dtype=np.int64)
        req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch else None
        if not isinstance(req_id_to_index, dict):
            return
        for req_id, num in scheduled.items():
            idx = req_id_to_index.get(req_id)
            if isinstance(idx, int) and 0 <= idx < num_reqs:
                try:
                    num_scheduled_tokens[idx] = max(1, int(num))
                except (TypeError, ValueError):
                    pass
        req_indices = np.repeat(arange_np[:num_reqs], num_scheduled_tokens)

        state_by_idx: dict[int, Any] = {}
        for req_id, idx in req_id_to_index.items():
            state = self.state_store.get(str(req_id))
            if state is not None and not state.extra.get("budgets_ready"):
                state_by_idx[int(idx)] = state

        num_layers = self.hooks.layer_count
        for layer_idx in range(num_layers):
            pair = self.hooks.get_capture_pair(layer_idx)
            if pair is None:
                continue
            try:
                layer_input, attn_out = pair
                # Paper's hidd_data: cosine similarity between the layer input
                # and the post-attention residual output.
                residual = layer_input.float() + attn_out.float()
                sims = F.cosine_similarity(layer_input.float(), residual, dim=-1)
                for idx, state in state_by_idx.items():
                    token_indices = np.nonzero(req_indices == idx)[0]
                    if token_indices.size == 0:
                        continue
                    start, end = int(token_indices[0]), int(token_indices[-1]) + 1
                    accumulator = state.extra.get("importance")
                    if accumulator is not None:
                        accumulator.add(layer_idx, sims, start, end)
            except Exception:  # pragma: no cover - capture must never break serving
                pass
            finally:
                self.hooks.clear_capture_pair(layer_idx)

    def _maybe_finalize_budgets(self, scheduler_output: Any) -> None:
        """After a request's prefill completes, run KMeans budget allocation."""
        for req_id, _scheduled in _scheduled_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None or state.extra.get("budgets_ready"):
                continue
            prefill_len = int(state.prefill_len or 0)
            if prefill_len <= 0 or int(state.num_computed_tokens) < prefill_len:
                continue
            accumulator = state.extra.get("importance")
            num_layers = self.hooks.layer_count
            if num_layers <= 0:
                num_layers = len(getattr(state.extra, "sliding_windows", [])) or 1
            importance = accumulator.means(num_layers) if accumulator is not None else [0.0] * num_layers
            budgets, diagnostics = finalize_budgets(
                base_runner=self._base_runner,
                req_id=req_id,
                layer_importance=importance,
                num_layers=num_layers,
                prompt_len=prefill_len,
                ini_size=self.config.ini_size,
                class3_size=self.config.class3_size,
                n_clusters=self.config.n_clusters,
                seed=self.config.kmeans_seed,
                log_budgets=self.config.log_budgets,
            )
            state.extra["sliding_windows"] = budgets
            state.extra["budgets_ready"] = True
            state.extra["budget_diagnostics"] = diagnostics
            if accumulator is not None:
                accumulator.clear()

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
            )
            scheduled_tokens = max(1, int(signal.scheduled_tokens or 1))
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

            if (
                self.config.defer_prefill_compression
                and 0 < int(state.num_computed_tokens or 0) < int(state.prefill_len or 0)
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

            if not state.extra.get("budgets_ready"):
                budgets = [0] * self.hooks.layer_count
                keep_count = self._request_keep_count(state, int(total_tokens), block_size)
            else:
                budgets = list(state.extra.get("sliding_windows") or [])
                keep_count = self._request_keep_count(state, int(total_tokens), block_size)
            if (
                int(state.cache_len_after_last_compression) > 0
                and int(state.cache_len_after_last_compression) < total_tokens
            ):
                keep_count = max(int(keep_count), int(state.cache_len_after_last_compression))
            keep_count = min(int(keep_count), int(total_tokens))

            event = compress_request(
                base_runner=self._base_runner,
                req_id=req_id,
                keep_count=keep_count,
                budgets=budgets,
                total_tokens=int(total_tokens),
                block_size=block_size,
                start_size=self.config.start_size,
                mode=self.config.mode,
                hooks=self.hooks,
                fake_key_padding=self.config.fake_key_padding,
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
                requests_dict = getattr(self._base_runner, "requests", None)
                scheduler_nct = 0
                if isinstance(requests_dict, dict):
                    req_state = requests_dict.get(req_id)
                    if req_state is not None:
                        scheduler_nct = int(getattr(req_state, "num_computed_tokens", 0) or 0)
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
            required_blocks = (int(state.current_cache_len) + block_size - 1) // block_size
            current_blocks = 0
            block_ids = _request_block_ids(self._base_runner, str(req_id))
            if block_ids:
                current_blocks = len(block_ids)
            limit = max(0, required_blocks - current_blocks)
            if len(group) > limit:
                trimmed = list(group[:limit])
                new_block_ids_list[i] = tuple(trimmed) if isinstance(group, tuple) else trimmed
                log_debug(
                    "trimmed new_block_ids req=%s %d -> %d", req_id, len(group), limit,
                )

    def _prepare_effective_overrides(self, scheduler_output: Any) -> bool:
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

        self._maybe_finalize_budgets(scheduler_output)

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

        # Layer-importance capture runs AFTER the forward (outside any
        # compiled region): the hooks stored references only.
        self._capture_prefill_importance(scheduler_output)

        output, remaining = attach_events_to_output(output, events, scheduler_output)
        self._pending_compression_events = remaining

        if self.config.probe_enabled:
            events_by_req: dict[str, str] = {}
            for e in events:
                if isinstance(e, dict) and e.get("req_id") is not None:
                    events_by_req[str(e["req_id"])] = str(e.get("status", "none"))
            for req_id, _scheduled in _scheduled_items(scheduler_output):
                state = self.state_store.get(req_id)
                seq_len = int(state.current_cache_len) if state is not None else 0
                budgets_ready = bool(state.extra.get("budgets_ready")) if state is not None else False
                block_size = _table_block_size(self._base_runner)
                k = self._request_keep_count(state, max(1, seq_len + 1), block_size) if state is not None else 0
                probe(
                    "step=%d req=%s core_entered=1 hook_entered=%d "
                    "mode=%s layers=%d budgets_ready=%d K=%d start=%d "
                    "ini=%.3f class3=%.3f seq_len=%d keep=%d "
                    "reclaimed_blocks=%d compress_events=%d last_event=%s",
                    step,
                    req_id,
                    int(self.hooks.captured_this_step),
                    self.config.mode,
                    self.hooks.layer_count,
                    int(budgets_ready),
                    k,
                    self.config.start_size,
                    self.config.ini_size,
                    self.config.class3_size,
                    seq_len,
                    min(k, seq_len),
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
        sample_fn = getattr(self._base_runner, "sample_tokens", None)
        if not callable(sample_fn):
            raise RuntimeError("SqueezeAttentionModelRunner: base runner has no sample_tokens")
        output = sample_fn(grammar_output)
        if self._pending_compression_events:
            output, self._pending_compression_events = attach_events_to_output(
                output, self._pending_compression_events
            )
        return output
