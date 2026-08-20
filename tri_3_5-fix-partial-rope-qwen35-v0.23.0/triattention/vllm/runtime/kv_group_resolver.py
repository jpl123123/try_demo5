"""Resolve vLLM KV cache tensors grouped by kv-cache group/layer for TriAttention runtime."""

from __future__ import annotations

import re
from typing import Any

import torch

from .kv_compaction import register_kv_layout_axis_hint

_NON_COMPRESSIBLE_LAYER_NAME_MARKERS = (
    "linear_attn",
    "mamba",
    "gated_delta",
    "gdn",
)

_NON_COMPRESSIBLE_SPEC_MARKERS = (
    "mamba",
    "encoderonlyattention",
    "crossattention",
)

_COMPRESSIBLE_SPEC_MARKERS = (
    "fullattention",
    "mlaattention",
)


def infer_layer_idx(layer_name: str, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", layer_name)
    if matches:
        return int(matches[-1])
    return fallback_idx


def _infer_kv_axis_from_group_backend(base_runner: Any, gid: int) -> int | None:
    attn_groups = getattr(base_runner, "attn_groups", None)
    if not isinstance(attn_groups, (list, tuple)):
        return None
    if gid < 0 or gid >= len(attn_groups):
        return None
    group = attn_groups[gid]
    backend = getattr(group, "backend", None)
    if backend is None:
        return None

    backend_cls = backend if isinstance(backend, type) else backend.__class__
    get_kv_cache_shape = getattr(backend_cls, "get_kv_cache_shape", None)
    if callable(get_kv_cache_shape):
        try:
            # Probe with num_blocks=3 to avoid (2, 2, ...) ambiguity.
            shape = tuple(
                int(x)
                for x in get_kv_cache_shape(
                    3,   # num_blocks
                    16,  # block_size (vLLM backends require multiple of 16)
                    1,   # num_kv_heads
                    1,   # head_size
                )
            )
            if len(shape) >= 2:
                dim0_is_kv = shape[0] == 2
                dim1_is_kv = shape[1] == 2
                if dim0_is_kv ^ dim1_is_kv:
                    return 0 if dim0_is_kv else 1
        except Exception:
            pass

    # Conservative fallback for fake backends in tests or unknown vLLM variants.
    module_name = str(getattr(backend_cls, "__module__", ""))
    cls_name = str(getattr(backend_cls, "__name__", ""))
    ident = f"{module_name}.{cls_name}".lower()
    if "flash_attn" in ident:
        return 0
    if "triton_attn" in ident:
        return 1
    return None


def _normalize_kv_cache_ref(raw: Any) -> Any | None:
    if isinstance(raw, torch.Tensor):
        return raw
    if (
        isinstance(raw, (list, tuple))
        and len(raw) >= 2
        and isinstance(raw[0], torch.Tensor)
        and isinstance(raw[1], torch.Tensor)
    ):
        return tuple(raw)
    return None


def _kv_cache_ref_key(cache_ref: Any) -> tuple[int, ...]:
    if isinstance(cache_ref, torch.Tensor):
        return (int(cache_ref.data_ptr()),)
    return tuple(
        int(tensor.data_ptr())
        for tensor in cache_ref
        if isinstance(tensor, torch.Tensor)
    )


def _spec_ident(spec: Any) -> str:
    spec_cls = spec if isinstance(spec, type) else spec.__class__
    module_name = str(getattr(spec_cls, "__module__", ""))
    cls_name = str(getattr(spec_cls, "__name__", ""))
    return f"{module_name}.{cls_name}".lower()


def _is_noncompressible_layer_name(layer_name: Any) -> bool:
    ident = str(layer_name).lower()
    return any(marker in ident for marker in _NON_COMPRESSIBLE_LAYER_NAME_MARKERS)


def _looks_like_attention_layer_name(layer_name: Any) -> bool:
    ident = str(layer_name).lower()
    return "attn" in ident or "attention" in ident


def _is_compressible_kv_spec(spec: Any) -> bool | None:
    if spec is None:
        return None

    nested_specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested_specs, dict):
        decisions = [
            _is_compressible_kv_spec(nested_spec)
            for nested_spec in nested_specs.values()
        ]
        known = [decision for decision in decisions if decision is not None]
        if not known:
            return None
        return bool(known) and all(known)

    ident = _spec_ident(spec)
    if any(marker in ident for marker in _NON_COMPRESSIBLE_SPEC_MARKERS):
        return False
    if any(marker in ident for marker in _COMPRESSIBLE_SPEC_MARKERS):
        return True
    if ident.endswith("attentionspec"):
        return True
    return None


def _group_spec_decision(group: Any, layer_names: list[Any]) -> bool | None:
    group_spec = getattr(group, "kv_cache_spec", None)
    nested_specs = getattr(group_spec, "kv_cache_specs", None)
    if isinstance(nested_specs, dict):
        decisions: list[bool | None] = []
        for layer_name in layer_names:
            layer_spec = nested_specs.get(layer_name)
            decisions.append(_is_compressible_kv_spec(layer_spec))
        known = [decision for decision in decisions if decision is not None]
        if not known:
            return None
        return bool(known) and all(known)
    return _is_compressible_kv_spec(group_spec)


def is_compressible_kv_group(group: Any) -> bool | None:
    """Return whether TriAttention can compact this KV-cache group.

    Qwen3.5-style hybrid models expose both full-attention KV groups and
    linear/mamba state groups. TriAttention should only move/reclaim the
    full-attention KV groups; state-space groups are not token KV histories.
    """
    layer_names_raw = getattr(group, "layer_names", None)
    if not isinstance(layer_names_raw, (list, tuple)):
        return None
    layer_names = list(layer_names_raw)
    if not layer_names:
        return None
    if any(_is_noncompressible_layer_name(layer_name) for layer_name in layer_names):
        return False

    spec_decision = _group_spec_decision(group, layer_names)
    if spec_decision is not None:
        return spec_decision
    if any(_looks_like_attention_layer_name(layer_name) for layer_name in layer_names):
        return True
    return None


def resolve_compressible_group_ids(base_runner: Any) -> set[int] | None:
    """Resolve KV-cache group ids that are safe for TriAttention compaction.

    ``None`` means the runtime could not inspect group metadata and should keep
    the legacy all-groups behavior.
    """
    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return None

    group_ids: set[int] = set()
    saw_known_decision = False
    for gid, group in enumerate(kv_cache_groups):
        decision = is_compressible_kv_group(group)
        if decision is None:
            continue
        saw_known_decision = True
        if decision:
            group_ids.add(int(gid))

    if not saw_known_decision:
        return None
    return group_ids


def resolve_group_tensors(base_runner: Any) -> dict[int, list[tuple[int, Any]]]:
    """Resolve kv cache tensors for each kv cache group.

    Returns:
        gid -> list of (layer_idx, kv_cache_ref)
    """
    group_tensors: dict[int, list[tuple[int, Any]]] = {}

    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    compilation_config = getattr(base_runner, "compilation_config", None)
    static_forward_context = (
        getattr(compilation_config, "static_forward_context", None)
        if compilation_config is not None
        else None
    )

    if kv_cache_config is None or not isinstance(static_forward_context, dict):
        fallback = getattr(base_runner, "kv_caches", None)
        if isinstance(fallback, list):
            tensors = [
                (idx, cache_ref)
                for idx, raw in enumerate(fallback)
                if (cache_ref := _normalize_kv_cache_ref(raw)) is not None
            ]
            if tensors:
                group_tensors[0] = tensors
        return group_tensors

    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return group_tensors

    compressible_group_ids = resolve_compressible_group_ids(base_runner)

    for gid, group in enumerate(kv_cache_groups):
        if compressible_group_ids is not None and gid not in compressible_group_ids:
            continue
        layer_names = getattr(group, "layer_names", None)
        if not isinstance(layer_names, (list, tuple)):
            continue
        tensors: list[tuple[int, Any]] = []
        seen_ptrs: set[tuple[int, ...]] = set()
        for local_idx, layer_name in enumerate(layer_names):
            layer = static_forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache_list = getattr(layer, "kv_cache", None)
            if isinstance(kv_cache_list, list) and kv_cache_list:
                cache_ref = _normalize_kv_cache_ref(kv_cache_list[0])
            else:
                cache_ref = _normalize_kv_cache_ref(kv_cache_list)
            if cache_ref is None:
                continue
            ptr = _kv_cache_ref_key(cache_ref)
            if not ptr:
                continue
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors.append(
                (
                    infer_layer_idx(
                        layer_name=layer_name,
                        layer_obj=layer,
                        fallback_idx=local_idx,
                    ),
                    cache_ref,
                )
            )
        if tensors:
            kv_axis_hint = _infer_kv_axis_from_group_backend(base_runner=base_runner, gid=gid)
            if kv_axis_hint is not None:
                for _layer_idx, cache_ref in tensors:
                    if not isinstance(cache_ref, torch.Tensor):
                        continue
                    try:
                        register_kv_layout_axis_hint(cache_ref, kv_axis_hint)
                    except ValueError:
                        # Best effort registration only; compaction path will fail-fast if
                        # an ambiguous layout cannot be safely disambiguated.
                        pass
            group_tensors[gid] = tensors
    return group_tensors
