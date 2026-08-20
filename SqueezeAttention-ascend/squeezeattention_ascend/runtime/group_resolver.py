"""KV-cache group / layer tensor resolution for SqueezeAttention-ascend.

Converts SqueezeAttention's HF assumption (per-layer dense KV in a ``DynamicCache``)
into the vLLM-Ascend reality: per-layer block-cache tensors organized in
KV-cache groups (hybrid/MTP models expose multiple groups, e.g. the main
full-attention group plus a draft group). Only compressible (full-attention)
groups are compacted. Adapted from tri_3_5's kv_group_resolver.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from ..core.kv_layout import register_kv_layout_axis_hint

_NON_COMPRESSIBLE_LAYER_NAME_MARKERS = ("linear_attn", "mamba", "gated_delta", "gdn")
_NON_COMPRESSIBLE_SPEC_MARKERS = ("mamba", "encoderonlyattention", "crossattention")
_COMPRESSIBLE_SPEC_MARKERS = ("fullattention", "mlaattention")


def infer_layer_idx(layer_name: Any, layer_obj: Any, fallback_idx: int) -> int:
    for attr in ("layer_idx", "layer_id", "idx"):
        value = getattr(layer_obj, attr, None)
        if isinstance(value, int):
            return value
    matches = re.findall(r"\d+", str(layer_name))
    if matches:
        return int(matches[-1])
    return int(fallback_idx)


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
    return tuple(int(t.data_ptr()) for t in cache_ref if isinstance(t, torch.Tensor))


def _spec_ident(spec: Any) -> str:
    spec_cls = spec if isinstance(spec, type) else spec.__class__
    return f"{getattr(spec_cls, '__module__', '')}.{getattr(spec_cls, '__name__', '')}".lower()


def _is_compressible_kv_spec(spec: Any) -> bool | None:
    if spec is None:
        return None
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict):
        decisions = [_is_compressible_kv_spec(s) for s in nested.values()]
        known = [d for d in decisions if d is not None]
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


def is_compressible_kv_group(group: Any) -> bool | None:
    layer_names = getattr(group, "layer_names", None)
    if not isinstance(layer_names, (list, tuple)) or not layer_names:
        return None
    if any(
        any(marker in str(name).lower() for marker in _NON_COMPRESSIBLE_LAYER_NAME_MARKERS)
        for name in layer_names
    ):
        return False
    group_spec = getattr(group, "kv_cache_spec", None)
    nested = getattr(group_spec, "kv_cache_specs", None)
    if isinstance(nested, dict):
        decisions = [
            _is_compressible_kv_spec(nested.get(name))
            for name in layer_names
        ]
        known = [d for d in decisions if d is not None]
        if known:
            return bool(known) and all(known)
    decision = _is_compressible_kv_spec(group_spec)
    if decision is not None:
        return decision
    if any("attn" in str(name).lower() or "attention" in str(name).lower() for name in layer_names):
        return True
    return None


def resolve_compressible_group_ids(base_runner: Any) -> set[int] | None:
    kv_cache_config = getattr(base_runner, "kv_cache_config", None)
    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not isinstance(kv_cache_groups, (list, tuple)):
        return None
    group_ids: set[int] = set()
    saw_known = False
    for gid, group in enumerate(kv_cache_groups):
        decision = is_compressible_kv_group(group)
        if decision is None:
            continue
        saw_known = True
        if decision:
            group_ids.add(int(gid))
    if not saw_known:
        return None
    return group_ids


def resolve_group_tensors(base_runner: Any) -> dict[int, list[tuple[int, Any]]]:
    """Resolve ``gid -> [(layer_idx, kv_cache_ref), ...]`` for compressible groups.

    Falls back to ``base_runner.kv_caches`` (a flat per-layer list) when group
    metadata is unavailable.
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
                (idx, ref)
                for idx, raw in enumerate(fallback)
                if (ref := _normalize_kv_cache_ref(raw)) is not None
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
            if not ptr or ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors.append(
                (
                    infer_layer_idx(layer_name, layer, local_idx),
                    cache_ref,
                )
            )
        if tensors:
            group_tensors[int(gid)] = tensors
    return group_tensors
