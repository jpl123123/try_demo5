"""Attention-layer hooks for kvpress-ascend.

Converts kvpress's HF mechanism (``BasePress.__call__`` registering forward
hooks on every ``self_attn`` layer to observe queries and hidden states) into
hooks on vLLM-Ascend decoder-layer attention modules.

Two capture surfaces (cheap; references only unless the press needs a copy):

- per-layer post-RoPE **queries** (and keys/values when present) captured from
  the vLLM attention layer forward (``layer.self_attn.attention.forward`` or
  ``layer.self_attn.forward``);
- per-layer **hidden states** (decoder layer input) for window presses.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch

from ..logging_control import attention_hook_log, log_warning


def _first_tensor_like(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item):
                return item
        return None
    if torch.is_tensor(value):
        return value
    return None


def _infer_layer_idx(layer_name: str, layer_obj: Any, fallback: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", str(layer_name))
    if matches:
        return int(matches[-1])
    return int(fallback)


class AttentionHooks:
    """Registry of per-layer capture hooks installed on a loaded model."""

    def __init__(self) -> None:
        self._layer_queries: dict[int, torch.Tensor] = {}
        self._layer_hidden_states: dict[int, torch.Tensor] = {}
        self._layer_step: dict[int, int] = {}
        self._hooks: list[Any] = []
        self._layer_count = 0
        self._captured_any = False
        self._current_step = -1
        self._enabled = True

    # -- capture state -----------------------------------------------------

    def reset_step(self, step: int) -> None:
        self._current_step = int(step)
        self._captured_any = False

    def mark_step(self, step: int) -> None:
        self._current_step = int(step)

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

    def get_hidden_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._layer_hidden_states.get(int(layer_idx))

    def all_query_layers(self) -> list[int]:
        return sorted(self._layer_queries.keys())

    def sample_query_layer_indices(self, limit: int = 0) -> list[int]:
        """Layers whose captured queries are available (optionally capped)."""
        indices = self.all_query_layers()
        if limit <= 0 or len(indices) <= limit:
            return indices
        return [indices[int(round(i * (len(indices) - 1) / (limit - 1)))] for i in range(limit)]

    # -- install -----------------------------------------------------------

    def _wrap_attention_forward(self, attn_module: Any, layer_idx: int, label: str) -> None:
        original = getattr(attn_module, "forward", None)
        if not callable(original) or bool(getattr(original, "_kvpress_hooked", False)):
            return
        hooks = self

        def _patched_forward(*args, **kwargs):
            try:
                if hooks._enabled:
                    # vLLM attention layer signature:
                    # forward(positions, query, key, value, kv_cache, attn_metadata, ...)
                    query = kwargs.get("query")
                    if query is None and len(args) >= 2:
                        query = args[1]
                    q_tensor = _first_tensor_like(query)
                    if q_tensor is not None and q_tensor.dim() >= 2:
                        hooks._layer_queries[layer_idx] = q_tensor
                        hooks.note_captured()
                        attention_hook_log(
                            "attention capture layer=%d q_shape=%s step=%d",
                            layer_idx, tuple(q_tensor.shape), hooks._current_step,
                        )
            except Exception:  # pragma: no cover - capture must never break forward
                pass
            return original(*args, **kwargs)

        setattr(_patched_forward, "_kvpress_hooked", True)
        setattr(attn_module, "forward", _patched_forward)
        self._hooks.append(attn_module)

    def _wrap_layer_forward(self, layer: Any, layer_idx: int) -> None:
        original = getattr(layer, "forward", None)
        if not callable(original) or bool(getattr(original, "_kvpress_layer_hooked", False)):
            return
        hooks = self

        def _patched_forward(*args, **kwargs):
            try:
                if hooks._enabled:
                    hidden = kwargs.get("hidden_states")
                    if hidden is None and args:
                        hidden = args[0]
                    h_tensor = _first_tensor_like(hidden)
                    if h_tensor is not None:
                        hooks._layer_hidden_states[layer_idx] = h_tensor
            except Exception:  # pragma: no cover
                pass
            return original(*args, **kwargs)

        setattr(_patched_forward, "_kvpress_layer_hooked", True)
        setattr(layer, "forward", _patched_forward)
        self._hooks.append(layer)

    def install(self, model: Any) -> int:
        """Install hooks on all decoder layers of the loaded vLLM model.

        Returns the number of layers hooked. Locates layers via the common
        vLLM container attributes (``model.model.layers`` / ``model.layers`` /
        ``model.decoder.layers`` / ``model.transformer.layers``).
        """
        roots = [
            model,
            getattr(model, "model", None),
            getattr(model, "decoder", None),
            getattr(model, "transformer", None),
            getattr(model, "language_model", None),
        ]
        layers: list[Any] = []
        for root in roots:
            if root is None:
                continue
            for attr_name in ("layers", "h", "blocks"):
                candidate = getattr(root, attr_name, None)
                if isinstance(candidate, (list, tuple)):
                    layers = list(candidate)
                    break
                try:
                    candidate_list = [candidate[idx] for idx in range(len(candidate))]
                    if candidate_list:
                        layers = candidate_list
                        break
                except Exception:
                    continue
            if layers:
                break

        hooked = 0
        for local_idx, layer in enumerate(layers):
            layer_idx = _infer_layer_idx(local_idx, layer, local_idx)
            self._layer_count = max(self._layer_count, layer_idx + 1)
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                self_attn = getattr(layer, "attention", None)
            if self_attn is not None:
                # vLLM decoder layers wrap the backend Attention layer either
                # directly (self_attn is the Attention module) or as
                # self_attn.attention.
                backend_attn = getattr(self_attn, "attention", None)
                target = backend_attn if backend_attn is not None else self_attn
                self._wrap_attention_forward(target, layer_idx, f"layer[{layer_idx}]")
                self._wrap_layer_forward(layer, layer_idx)
                hooked += 1
        return hooked

    def uninstall(self) -> None:
        self._enabled = False
        self._hooks.clear()
        self._layer_queries.clear()
        self._layer_hidden_states.clear()

    def log_summary(self) -> str:
        return (
            f"layers={self._layer_count} captured_layers={len(self._layer_queries)}"
            f" captured_this_step={self.captured_this_step}"
        )
