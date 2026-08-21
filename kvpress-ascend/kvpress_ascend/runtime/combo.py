"""Combo mode: SqueezeAttention layer-wise budgets x kvpress token-wise presses,
executed as ONE physical eviction per compression boundary.

The user's design question answered in code: the layer dimension
(SqueezeAttention KMeans budgets) decides how many tokens each layer may keep,
the token dimension (kvpress press scoring) decides WHICH tokens each layer
keeps, and a single in-place compaction + block-row shrink + scheduler reclaim
event performs the eviction once.

``KVPRESS_COMBO=1`` (with both plugins installed and enabled) installs a single
scheduler patch (kvpress signal/tracker/reclaim) and a single combo runner
proxy on the worker; the SqueezeAttention plugin detects combo mode and skips
its standalone install, so no double proxy / double eviction / double free can
occur.

Combo requires the ``squeezeattention_ascend`` package (pip installed
alongside). It is imported lazily at install time.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.kv_layout import (
    compact_request_kv_in_place_per_head,
    gather_request_kv_dense,
)
from ..core.press_bridge import build_press, select_keep_indices
from ..envs import KVPressRuntimeConfig
from ..logging_control import log_debug, log_info, log_warning, probe
from . import input_patch_state as _patch_state
from .block_sync import apply_worker_block_reclaim, truncate_request_state_block_ids
from .compression_engine import _request_block_ids, _request_token_count, _table_block_size
from .output_bridge import attach_events_to_output
from .request_key_compat import (
    iter_scheduled_new_requests,
    iter_scheduled_token_items,
)
from .signals import CompressionSignal, signals_from_scheduler_output
from .state import RequestStateStore

try:
    from squeezeattention_ascend.core.budgets import (
        LayerImportanceAccumulator,
        compute_layer_budgets,
    )
    from squeezeattention_ascend.core.selection import (
        pad_short_budget_layers_with_fake_keys,
    )
    from squeezeattention_ascend.envs import SqueezeRuntimeConfig
except Exception as exc:  # pragma: no cover - guarded at install time
    raise ImportError(
        "KVPRESS_COMBO=1 requires the SqueezeAttention-ascend package "
        "(pip install ./SqueezeAttention-ascend)"
    ) from exc


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return False
    is_compiling = getattr(compiler, "is_compiling", None)
    try:
        return bool(is_compiling and is_compiling())
    except Exception:
        return False


def _first_tensor_like(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item):
                return item
        return None
    return value if torch.is_tensor(value) else None


class ComboHooks:
    """Merged capture hooks: per-layer post-RoPE queries (press scoring) and
    per-layer (layer_input, attn_out) references (SqueezeAttention hidd_data).

    Fully transparent to torch.compile: reference-only storage inside the
    forward, compile-guarded, no tensor ops. The cosine similarity is computed
    by the runner after the forward.
    """

    def __init__(self) -> None:
        self._layer_queries: dict[int, torch.Tensor] = {}
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._layer_attn_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._layer_count = 0
        self._captured_any = False
        self._current_step = -1
        self._enabled = True

    def reset_step(self, step: int) -> None:
        self._current_step = int(step)
        self._captured_any = False

    def note_captured(self) -> None:
        self._captured_any = True

    @property
    def captured_this_step(self) -> bool:
        return self._captured_any

    @property
    def layer_count(self) -> int:
        return self._layer_count

    def get_query(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._layer_queries.get(int(layer_idx))

    def get_capture_pair(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        layer_input = self._layer_inputs.get(int(layer_idx))
        attn_out = self._layer_attn_outputs.get(int(layer_idx))
        if layer_input is None or attn_out is None:
            return None
        return layer_input, attn_out

    def clear_capture_pair(self, layer_idx: int) -> None:
        self._layer_inputs.pop(int(layer_idx), None)
        self._layer_attn_outputs.pop(int(layer_idx), None)

    def _wrap_layer_forward(self, layer: Any, layer_idx: int) -> None:
        original = getattr(layer, "forward", None)
        if not callable(original) or bool(getattr(original, "_combo_layer_hooked", False)):
            return
        hooks = self

        def _patched_forward(*args, **kwargs):
            if _is_compiling():
                return original(*args, **kwargs)
            try:
                if hooks._enabled:
                    hidden = kwargs.get("hidden_states")
                    if hidden is None and args:
                        hidden = args[0]
                    h_tensor = _first_tensor_like(hidden)
                    if h_tensor is not None:
                        hooks._layer_inputs[layer_idx] = h_tensor
            except Exception:
                pass
            return original(*args, **kwargs)

        setattr(_patched_forward, "_combo_layer_hooked", True)
        setattr(layer, "forward", _patched_forward)
        self._hooks.append(layer)

    def _wrap_attention_forward(self, attn_module: Any, layer_idx: int) -> None:
        original = getattr(attn_module, "forward", None)
        if not callable(original) or bool(getattr(original, "_combo_attn_hooked", False)):
            return
        hooks = self

        def _patched_forward(*args, **kwargs):
            if _is_compiling():
                return original(*args, **kwargs)
            try:
                if hooks._enabled:
                    query = kwargs.get("query")
                    if query is None and len(args) >= 2:
                        query = args[1]
                    q_tensor = _first_tensor_like(query)
                    if q_tensor is not None:
                        hooks._layer_queries[layer_idx] = q_tensor
                        hooks.note_captured()
                    layer_input = hooks._layer_inputs.get(layer_idx)
                    if layer_input is not None:
                        result = original(*args, **kwargs)
                        attn_out = _first_tensor_like(result)
                        if attn_out is not None and attn_out.shape == layer_input.shape:
                            hooks._layer_attn_outputs[layer_idx] = attn_out
                            hooks.note_captured()
                        return result
                return original(*args, **kwargs)
            except Exception:
                return original(*args, **kwargs)

        setattr(_patched_forward, "_combo_attn_hooked", True)
        setattr(attn_module, "forward", _patched_forward)
        self._hooks.append(attn_module)

    def install(self, model: Any) -> int:
        from .attention_hooks import _resolve_model_layers

        layers = _resolve_model_layers(model)
        hooked = 0
        for local_idx, layer in enumerate(layers):
            layer_idx = int(
                getattr(layer, "layer_idx", None)
                or getattr(layer, "layer_id", None)
                or local_idx
            )
            self._layer_count = max(self._layer_count, layer_idx + 1)
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                self_attn = getattr(layer, "attention", None)
            if self_attn is not None:
                backend_attn = getattr(self_attn, "attention", None)
                target = backend_attn if backend_attn is not None else self_attn
                self._wrap_attention_forward(target, layer_idx)
                self._wrap_layer_forward(layer, layer_idx)
                hooked += 1
        return hooked


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


def combo_compress_request(
    *,
    base_runner: Any,
    req_id: str,
    keep_count: int,
    budgets: list[int],
    total_tokens: int,
    block_size: int,
    press: Any,
    hooks: Any,
    mode: str,
    fake_key_padding: bool = False,
    min_reclaim_blocks: int = 1,
    scheduled_tokens: int = 1,
) -> dict[str, Any]:
    """One physical eviction: per-layer press keep sets (token dimension) with
    per-layer budgets (layer dimension), single compaction + reclaim plan."""
    from .group_resolver import resolve_group_tensors

    if keep_count <= 0 or keep_count >= total_tokens:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "under_budget" if keep_count >= total_tokens else "empty_keep",
            "cache_len_after": total_tokens,
        }
    group_tensors = resolve_group_tensors(base_runner)
    if not group_tensors:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "no_kv_tensors",
            "cache_len_after": total_tokens,
        }

    retained_cache_len = keep_count + max(1, int(scheduled_tokens))
    required_blocks = (retained_cache_len + block_size - 1) // block_size

    total_reclaimed = 0
    compacted_layers = 0
    padded_slots = 0
    applied_any = False
    reclaim_groups: list[dict[str, Any]] = []

    for gid, layer_tensors in group_tensors.items():
        block_ids = _request_block_ids(base_runner, req_id, gid)
        if not block_ids:
            continue
        current_blocks = len(block_ids)
        reclaimable = current_blocks - required_blocks
        if reclaimable <= 0:
            continue
        group_reclaim: Optional[dict[str, Any]] = {
            "gid": gid,
            "block_ids_before": list(block_ids),
            "block_ids_after": list(block_ids[:required_blocks]),
            "required_blocks": required_blocks,
            "reclaimable_blocks": max(0, reclaimable),
            "scored_layers": 0,
        }
        for layer_idx, kv_cache in layer_tensors:
            try:
                keys, _values = gather_request_kv_dense(
                    kv_cache, block_ids, block_size, total_tokens
                )
                layer_budget = budgets[layer_idx] if layer_idx < len(budgets) else keep_count
                if mode == "class_weighted":
                    layer_keep = min(int(layer_budget), int(total_tokens))
                    layer_keep = max(1, min(layer_keep, keep_count))
                else:
                    layer_keep = keep_count
                queries = hooks.get_query(layer_idx) if hooks is not None else None
                keep_tensor = select_keep_indices(
                    press, keys, int(layer_keep), queries=queries
                )
                compact_request_kv_in_place_per_head(
                    kv_cache,
                    block_ids,
                    block_size,
                    keep_tensor,
                    total_tokens,
                    preserve_dropped_tokens=False,
                    prefix_only=True,
                )
                compacted_layers += 1
                applied_any = True
                if group_reclaim is not None:
                    group_reclaim["scored_layers"] = int(group_reclaim["scored_layers"]) + 1
                if mode == "class_weighted" and fake_key_padding and hooks is not None:
                    query = hooks.get_query(layer_idx)
                    if query is not None:
                        key_cache = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
                        try:
                            padded_slots += pad_short_budget_layers_with_fake_keys(
                                key_cache=key_cache,
                                block_ids=block_ids,
                                block_size=block_size,
                                query=query,
                                keep_count=int(layer_keep),
                                total_tokens=total_tokens,
                                max_keep_count=keep_count,
                                num_kv_heads=int(key_cache.shape[2]),
                            )
                        except Exception as exc:
                            log_warning(
                                "combo fake-key padding failed req=%s layer=%d: %s",
                                req_id, layer_idx, exc,
                            )
            except Exception as exc:  # pragma: no cover - per-layer safety
                log_warning(
                    "combo compression layer failed req=%s gid=%d layer=%d: %s: %s",
                    req_id, gid, layer_idx, type(exc).__name__, exc,
                )
        if group_reclaim is not None and int(group_reclaim["scored_layers"]) > 0:
            reclaim_groups.append(group_reclaim)
        total_reclaimed += max(0, reclaimable)

    if not applied_any:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "no_layers_scored",
            "cache_len_after": total_tokens,
        }
    if min_reclaim_blocks > 0 and total_reclaimed < min_reclaim_blocks:
        return {
            "req_id": req_id,
            "status": "skipped",
            "reason": "below_min_reclaim",
            "cache_len_after": total_tokens,
        }

    probe(
        "COMPRESS req=%s mode=combo press=%s before=%d after=%d retained=%d "
        "reclaimed_blocks=%d layers_compacted=%d fake_key_slots=%d",
        req_id,
        press.__class__.__name__,
        total_tokens,
        keep_count,
        retained_cache_len,
        total_reclaimed,
        compacted_layers,
        padded_slots,
    )
    return {
        "req_id": req_id,
        "status": "applied",
        "reason": "combo_compaction",
        "cache_len_after": keep_count,
        "effective_cache_len_after": keep_count,
        "retained_cache_len": retained_cache_len,
        "details": {
            "mode": "combo",
            "press": press.__class__.__name__,
            "effective_tokens_before": total_tokens,
            "keep_count": keep_count,
            "retained_cache_len": retained_cache_len,
            "reclaimed_block_count": total_reclaimed,
            "fake_key_slots": padded_slots,
            "block_reclaim": {"mode": "truncate_tail", "groups": reclaim_groups},
        },
    }


class KVPressSqueezeComboRunner:
    """Single proxy: one eviction per boundary, composed of the layer
    dimension (SqueezeAttention budgets) and the token dimension (kvpress
    press)."""

    def __init__(self, base_runner: Any, config: Optional[KVPressRuntimeConfig] = None):
        self._base_runner = base_runner
        self.config = config or KVPressRuntimeConfig.from_env()
        self.squeeze_config = SqueezeRuntimeConfig.from_env()
        self.state_store = RequestStateStore()
        self.hooks = ComboHooks()
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
        self._logged_new_request = False
        self._install_hooks()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_runner, name)

    def _install_hooks(self) -> None:
        model = getattr(self._base_runner, "model", None)
        if model is None:
            log_warning("combo runner: model not found; hooks disabled")
            return
        hooked = self.hooks.install(model)
        log_info(
            "combo attention hooks installed: layers=%d press=%s "
            "kv_budget=%s squeeze_ini=%.3f squeeze_class3=%.3f build=%s",
            hooked,
            self.press.__class__.__name__,
            self.config.kv_budget or "auto",
            self.squeeze_config.ini_size,
            self.squeeze_config.class3_size,
            self.config.build_id,
        )

    # ------------------------------------------------------------ pre-step

    def _register_new_requests(self, scheduler_output: Any) -> None:
        for req_id, request, num_prompt_tokens in iter_scheduled_new_requests(
            scheduler_output
        ):
            prefill_len = _resolve_prefill_len(request) or int(num_prompt_tokens)
            state = self.state_store.ensure(str(req_id), prefill_len=prefill_len)
            if "importance" not in state.extra:
                state.extra["importance"] = LayerImportanceAccumulator()
                state.extra["budgets_ready"] = False
                state.extra["sliding_windows"] = []
            if not self._logged_new_request and self.config.logging_enabled:
                log_info(
                    "combo registered request req=%s prefill_len=%d",
                    req_id, prefill_len,
                )
                self._logged_new_request = True

    def _cleanup_finished(self, scheduler_output: Any) -> None:
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if isinstance(finished, (list, tuple, set)):
            self.state_store.cleanup_finished([str(r) for r in finished])

    def _sync_worker_num_computed(self, scheduler_output: Any) -> None:
        requests_dict = getattr(self._base_runner, "requests", None)
        if not isinstance(requests_dict, dict):
            return
        for req_id, _scheduled in iter_scheduled_token_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None:
                continue
            req_state = requests_dict.get(req_id)
            if req_state is None:
                continue
            nct = int(getattr(req_state, "num_computed_tokens", 0) or 0)
            if nct > int(state.num_computed_tokens):
                state.num_computed_tokens = nct

    def _capture_prefill_importance(self, scheduler_output: Any) -> None:
        """Post-step: cosine similarity (hidd_data) sliced per request."""
        input_batch = getattr(self._base_runner, "input_batch", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0)) if input_batch else 0
        if num_reqs <= 0:
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, dict):
            return
        req_id_to_index = (
            getattr(input_batch, "req_id_to_index", None) if input_batch else None
        )
        if not isinstance(req_id_to_index, dict):
            return
        arange_np = getattr(self._base_runner, "arange_np", np.arange(num_reqs))
        num_scheduled_tokens = np.zeros(num_reqs, dtype=np.int64)
        for req_id, num in scheduled.items():
            idx = req_id_to_index.get(req_id)
            if isinstance(idx, int) and 0 <= idx < num_reqs:
                try:
                    num_scheduled_tokens[idx] = max(1, int(num))
                except (TypeError, ValueError):
                    pass
        req_indices = np.repeat(arange_np[:num_reqs], num_scheduled_tokens)
        state_by_idx = {
            int(idx): state
            for req_id, idx in req_id_to_index.items()
            if (state := self.state_store.get(str(req_id))) is not None
            and not state.extra.get("budgets_ready")
        }
        for layer_idx in range(self.hooks.layer_count):
            pair = self.hooks.get_capture_pair(layer_idx)
            if pair is None:
                continue
            try:
                layer_input, attn_out = pair
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
            except Exception:
                pass
            finally:
                self.hooks.clear_capture_pair(layer_idx)

    def _maybe_finalize_budgets(self, scheduler_output: Any) -> None:
        for req_id, _scheduled in iter_scheduled_token_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None or state.extra.get("budgets_ready"):
                continue
            prefill_len = int(state.prefill_len or 0)
            if prefill_len <= 0 or int(state.num_computed_tokens) < prefill_len:
                continue
            accumulator = state.extra.get("importance")
            num_layers = self.hooks.layer_count or 1
            importance = (
                accumulator.means(num_layers)
                if accumulator is not None
                else [0.0] * num_layers
            )
            budgets, diagnostics = compute_layer_budgets(
                layer_importance=importance,
                num_layers=num_layers,
                ini_size=self.squeeze_config.ini_size,
                class3_size=self.squeeze_config.class3_size,
                prompt_len=prefill_len,
                n_clusters=self.squeeze_config.n_clusters,
                seed=self.squeeze_config.kmeans_seed,
            )
            state.extra["sliding_windows"] = budgets
            state.extra["budgets_ready"] = True
            state.extra["budget_diagnostics"] = diagnostics
            if self.squeeze_config.log_budgets:
                from ..logging_control import log_info as _li

                _li(
                    "[CLUSTER] combo budgets req=%s layers=%d prompt_len=%d "
                    "class_sizes=%s budgets=%s",
                    req_id, num_layers, prefill_len,
                    diagnostics["class_sizes"], budgets,
                )
            if accumulator is not None:
                accumulator.clear()

    def _request_keep_count(self, state: Any, total_tokens: int, block_size: int) -> int:
        if self.config.kv_budget > 0:
            k = self.config.kv_budget
        elif self.squeeze_config.kv_budget > 0:
            k = self.squeeze_config.kv_budget
        else:
            budgets = state.extra.get("sliding_windows") or []
            if budgets:
                k = max(1, max(int(b) for b in budgets))
            else:
                k = max(1, int(total_tokens * self.squeeze_config.ini_size))
        return min(int(total_tokens), k)

    def _compression_threshold(self, state: Any, block_size: int) -> int:
        keep_count = self._request_keep_count(state, 1 << 30, block_size)
        return max(1, keep_count + max(0, self.config.min_reclaim_blocks) * max(1, block_size))

    def _worker_self_triggers(
        self,
        scheduler_output: Any,
        signals: dict[str, CompressionSignal],
    ) -> dict[str, CompressionSignal]:
        block_size = _table_block_size(self._base_runner)
        for req_id, scheduled_tokens in iter_scheduled_token_items(scheduler_output):
            if req_id in signals:
                continue
            state = self.state_store.get(req_id)
            if state is None:
                continue
            threshold = self._compression_threshold(state, block_size)
            if state.is_compressed:
                effective = int(state.current_cache_len)
            else:
                if int(state.num_computed_tokens) <= 0:
                    continue
                effective = int(state.num_computed_tokens)
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
        return signals

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
                req_id, prefill_len=int(signal.prefill_len or 0)
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
            budgets = list(state.extra.get("sliding_windows") or [])
            keep_count = self._request_keep_count(state, int(total_tokens), block_size)
            if (
                int(state.cache_len_after_last_compression) > 0
                and int(state.cache_len_after_last_compression) < total_tokens
            ):
                keep_count = max(int(keep_count), int(state.cache_len_after_last_compression))
            keep_count = min(int(keep_count), int(total_tokens))

            event = combo_compress_request(
                base_runner=self._base_runner,
                req_id=req_id,
                keep_count=keep_count,
                budgets=budgets,
                total_tokens=int(total_tokens),
                block_size=block_size,
                press=self.press,
                hooks=self.hooks,
                mode=self.squeeze_config.mode,
                fake_key_padding=self.squeeze_config.fake_key_padding,
                min_reclaim_blocks=self.config.min_reclaim_blocks,
                scheduled_tokens=scheduled_tokens,
            )
            if event.get("status") == "applied":
                retained = int(event.get("retained_cache_len", keep_count))
                required_blocks = (retained + block_size - 1) // block_size
                details = event.get("details")
                reclaim_groups = (
                    details.get("block_reclaim", {}).get("groups")
                    if isinstance(details, dict)
                    and isinstance(details.get("block_reclaim"), dict)
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
        for req_id, _scheduled in iter_scheduled_token_items(scheduler_output):
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
            if effective_base < num_computed:
                base_by_req_idx[int(req_idx)] = effective_base
        if not base_by_req_idx:
            return False
        _patch_state.set_effective_bases(base_by_req_idx)
        return True

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

        self._capture_prefill_importance(scheduler_output)

        output, remaining = attach_events_to_output(output, events, scheduler_output)
        self._pending_compression_events = remaining

        if self.config.probe_enabled:
            events_by_req: dict[str, str] = {}
            for e in events:
                if isinstance(e, dict) and e.get("req_id") is not None:
                    events_by_req[str(e["req_id"])] = str(e.get("status", "none"))
            for req_id, _scheduled in iter_scheduled_token_items(scheduler_output):
                state = self.state_store.get(req_id)
                seq_len = int(state.current_cache_len) if state is not None else 0
                budgets_ready = (
                    bool(state.extra.get("budgets_ready")) if state is not None else False
                )
                block_size = _table_block_size(self._base_runner)
                k = (
                    self._request_keep_count(state, max(1, seq_len + 1), block_size)
                    if state is not None
                    else 0
                )
                probe(
                    "step=%d req=%s core_entered=1 hook_entered=%d "
                    "mode=combo layers=%d budgets_ready=%d K=%d start=%d "
                    "press=%s ini=%.3f class3=%.3f seq_len=%d keep=%d "
                    "reclaimed_blocks=%d compress_events=%d last_event=%s",
                    step,
                    req_id,
                    int(self.hooks.captured_this_step),
                    self.hooks.layer_count,
                    int(budgets_ready),
                    k,
                    self.squeeze_config.start_size,
                    self.press.__class__.__name__,
                    self.squeeze_config.ini_size,
                    self.squeeze_config.class3_size,
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
        for req_id, scheduled_tokens in iter_scheduled_token_items(scheduler_output):
            state = self.state_store.get(req_id)
            if state is None or not state.is_compressed:
                continue
            self.state_store.advance_cache_len(req_id, scheduled_tokens, self._last_step)

    def _consume_signals(self, scheduler_output: Any) -> dict[str, CompressionSignal]:
        return signals_from_scheduler_output(scheduler_output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        sample_fn = getattr(self._base_runner, "sample_tokens", None)
        if not callable(sample_fn):
            raise RuntimeError("combo runner: base runner has no sample_tokens")
        output = sample_fn(grammar_output)
        if self._pending_compression_events:
            output, self._pending_compression_events = attach_events_to_output(
                output, self._pending_compression_events
            )
        return output


# ---------------------------------------------------------------------------
# Installation (single scheduler patch + single combo worker proxy)
# ---------------------------------------------------------------------------

_ORIG_ASCEND_WORKER_METHODS: dict[type, dict[str, Any]] = {}


def _install_combo_runner_proxy(worker: Any) -> None:
    if getattr(worker, "_combo_runner_proxy_installed", False):
        return
    base_runner = getattr(worker, "model_runner", None)
    if base_runner is None:
        log_warning("combo worker: model_runner not available yet")
        return
    if isinstance(base_runner, KVPressSqueezeComboRunner):
        worker._combo_runner_proxy_installed = True
        return
    worker.model_runner = KVPressSqueezeComboRunner(base_runner=base_runner)
    worker._combo_runner_proxy_installed = True
    log_info(
        "Worker injected combo runner proxy (layer budgets x token press, "
        "single eviction per boundary): press=%s kv_budget=%s ini=%.3f class3=%.3f",
        worker.model_runner.press.__class__.__name__,
        worker.model_runner.config.kv_budget or "auto",
        worker.model_runner.squeeze_config.ini_size,
        worker.model_runner.squeeze_config.class3_size,
    )


def _patched_combo_worker_init_device(self):
    _resolve_original_worker_method(self, "init_device")(self)
    _install_combo_runner_proxy(self)


def _patched_combo_worker_execute_model(self, scheduler_output):
    if not getattr(self, "_combo_runner_proxy_installed", False):
        signals = getattr(scheduler_output, "kvpress_signals", None)
        if signals:
            _install_combo_runner_proxy(self)
    return _resolve_original_worker_method(self, "execute_model")(self, scheduler_output)


def _resolve_original_worker_method(worker: Any, method_name: str) -> Any:
    for cls in type(worker).__mro__:
        methods = _ORIG_ASCEND_WORKER_METHODS.get(cls)
        if methods is not None and method_name in methods:
            return methods[method_name]
    raise RuntimeError(f"missing_original_ascend_worker_method:{method_name}")


def _install_combo_worker_patches() -> None:
    try:
        import vllm_ascend.worker.worker as ascend_worker_mod

        worker_cls = ascend_worker_mod.NPUWorker
        if getattr(worker_cls, "init_device", None) is _patched_combo_worker_init_device:
            return
        methods = {
            "init_device": worker_cls.init_device,
            "execute_model": worker_cls.execute_model,
        }
        worker_cls.init_device = _patched_combo_worker_init_device
        worker_cls.execute_model = _patched_combo_worker_execute_model
        _ORIG_ASCEND_WORKER_METHODS[worker_cls] = methods
        log_info(
            "Installed combo worker patches for Ascend: vllm_ascend.worker.worker.NPUWorker"
        )
    except Exception:
        log_warning("could not install combo worker patches", exc_info=True)


def install_combo_monkeypatches() -> None:
    """Install the single-patch combo pipeline (scheduler + worker + input)."""
    from .monkeypatch import install_kvpress_integration_monkeypatches

    # Scheduler + KV cache manager + input patch + relaxed memory check via the
    # (single) kvpress scheduler pipeline; signals/tracker/reclaim all reuse it.
    install_kvpress_integration_monkeypatches(
        patch_scheduler=True,
        patch_worker=False,
    )
    _install_combo_worker_patches()
    os.environ.setdefault("KVPRESS_COMBO_ACTIVE", "1")
    log_info(
        "combo mode installed: layer dimension (SqueezeAttention budgets) x "
        "token dimension (kvpress press), ONE eviction per boundary; "
        "SqueezeAttention standalone plugin will skip.",
    )
