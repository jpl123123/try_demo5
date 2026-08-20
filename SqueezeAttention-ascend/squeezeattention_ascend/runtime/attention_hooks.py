"""Attention-layer hooks for SqueezeAttention-ascend.

Converts SqueezeAttention's HF mechanism (replacing decoder layers with
``LlamaAttention_squeeze`` / ``LlamaDecoderLayer_squeeze`` that observe layer
input/output hidden states) into hooks on vLLM-Ascend decoder layers:

- per-layer **per-token cosine similarity** between the layer input hidden
  states and the post-self-attention residual output (the paper's
  ``hidd_data``, computed during prefill);
- per-layer post-RoPE **queries** (needed only by the experimental
  ``class_weighted`` fake-key padding mode).
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..logging_control import log_debug, log_warning


def _first_tensor_like(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item):
                return item
        return None
    if torch.is_tensor(value):
        return value
    return None


def _infer_layer_idx(layer_name: Any, layer_obj: Any, fallback: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", str(layer_name))
    if matches:
        return int(matches[-1])
    return int(fallback)


def _is_compiling() -> bool:
    """True while torch.compile / dynamo is tracing this code.

    The hooks must be fully transparent to the tracer: tensor ops (or values
    consumed by Python) inside a traced region become extra graph outputs and
    desync the npugraph_ex / AOT artifacts (``ValueError: too many values to
    unpack``). While tracing, the hooks short-circuit to the original forward.
    """
    compiler = getattr(torch, "compiler", None)
    if compiler is None:  # torch < 2.0 (test environments)
        return False
    is_compiling = getattr(compiler, "is_compiling", None)
    try:
        return bool(is_compiling and is_compiling())
    except Exception:  # pragma: no cover - never break the forward
        return False


class SqueezeAttentionHooks:
    """Per-layer similarity + query capture hooks on a loaded model."""

    def __init__(self) -> None:
        self._layer_similarities: dict[int, torch.Tensor] = {}
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._layer_attn_outputs: dict[int, torch.Tensor] = {}
        self._layer_queries: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._layer_count = 0
        self._captured_any = False
        self._current_step = -1
        self._enabled = True
        self.capture_queries = False

    # -- capture state -----------------------------------------------------

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

    def get_similarities(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._layer_similarities.get(int(layer_idx))

    def get_capture_pair(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(layer_input, attn_output) captured this step (references)."""
        layer_input = self._layer_inputs.get(int(layer_idx))
        attn_out = self._layer_attn_outputs.get(int(layer_idx))
        if layer_input is None or attn_out is None:
            return None
        return layer_input, attn_out

    def clear_capture_pair(self, layer_idx: int) -> None:
        self._layer_inputs.pop(int(layer_idx), None)
        self._layer_attn_outputs.pop(int(layer_idx), None)

    def get_query(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._layer_queries.get(int(layer_idx))

    # -- install -----------------------------------------------------------

    def _wrap_layer_forward(self, layer: Any, layer_idx: int) -> None:
        original = getattr(layer, "forward", None)
        if not callable(original) or bool(getattr(original, "_squeeze_layer_hooked", False)):
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
                        hooks.note_captured()
            except Exception:  # pragma: no cover
                pass
            return original(*args, **kwargs)

        setattr(_patched_forward, "_squeeze_layer_hooked", True)
        setattr(layer, "forward", _patched_forward)
        self._hooks.append(layer)

    def _wrap_attention_forward(self, attn_module: Any, layer_idx: int) -> None:
        original = getattr(attn_module, "forward", None)
        if not callable(original) or bool(getattr(original, "_squeeze_attn_hooked", False)):
            return
        hooks = self

        def _patched_forward(*args, **kwargs):
            if _is_compiling():
                return original(*args, **kwargs)
            try:
                if hooks._enabled:
                    if hooks.capture_queries:
                        query = kwargs.get("query")
                        if query is None and len(args) >= 2:
                            query = args[1]
                        q_tensor = _first_tensor_like(query)
                        if q_tensor is not None:
                            hooks._layer_queries[layer_idx] = q_tensor
                    layer_input = hooks._layer_inputs.get(layer_idx)
                    if layer_input is not None:
                        result = original(*args, **kwargs)
                        attn_out = _first_tensor_like(result)
                        if attn_out is not None and attn_out.shape == layer_input.shape:
                            # Reference-only capture: the paper's hidd_data
                            # cosine similarity is computed by the runner proxy
                            # AFTER the forward (outside any compiled region).
                            # No tensor ops may run here: they would be traced
                            # into the compiled graph and change its outputs.
                            hooks._layer_attn_outputs[layer_idx] = attn_out
                            hooks.note_captured()
                        return result
                return original(*args, **kwargs)
            except Exception:  # pragma: no cover - capture must never break forward
                return original(*args, **kwargs)

        setattr(_patched_forward, "_squeeze_attn_hooked", True)
        setattr(attn_module, "forward", _patched_forward)
        self._hooks.append(attn_module)

    def install(self, model: Any) -> int:
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
                backend_attn = getattr(self_attn, "attention", None)
                target = backend_attn if backend_attn is not None else self_attn
                self._wrap_attention_forward(target, layer_idx)
                self._wrap_layer_forward(layer, layer_idx)
                hooked += 1
        return hooked

    def uninstall(self) -> None:
        self._enabled = False
        self._hooks.clear()
        self._layer_similarities.clear()
        self._layer_inputs.clear()
        self._layer_attn_outputs.clear()
        self._layer_queries.clear()

    def log_summary(self) -> str:
        return (
            f"layers={self._layer_count} sim_layers={len(self._layer_similarities)}"
            f" captured_this_step={self.captured_this_step}"
        )
