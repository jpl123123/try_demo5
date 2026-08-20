"""HF-aligned TriAttention selector implementation for TriAttention runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .config import TriAttentionRuntimeConfig
from .constants import TRITON_SCORING_REQUIRED_MARKER
from .fast_recency_guard import has_available_sparse_stats
from .kv_compaction import build_keep_token_indices, gather_request_k_dense_range

try:
    from vllm.logger import logger as _runtime_logger
except Exception:  # pragma: no cover - fallback for lightweight tests
    _runtime_logger = logging.getLogger(__name__)


class _StatsShapeMismatch(RuntimeError):
    """Raised when sparse stats cannot score the current runtime KV shape."""


def build_triattention_selector(
    config: TriAttentionRuntimeConfig,
    base_runner: Any | None = None,
) -> tuple[
    Callable[..., dict[str, Any] | None] | None,
    Callable[..., dict[str, Any] | None] | None,
    str,
]:
    """Build TriAttention selector callable.

    The returned selector emits either:
    - {"mode": "shared", "indices": Tensor|list[int]}
    - {"mode": "per_head", "indices": Tensor|list[list[int]]}
    """
    requested_pruning_mode = config.pruning_mode
    if requested_pruning_mode == "per_layer" and not bool(
        getattr(config, "allow_per_layer_mode", False)
    ):
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:per_layer_mode_disabled:"
            "set allow_per_layer_mode=True for explicit opt-in"
        )

    strict_triton_required = bool(
        config.enable_experimental_kv_compaction and config.require_triton_scoring
    )
    recency_accuracy_guarded = (
        bool(getattr(config, "fast_recency_accuracy_guard", True))
        and has_available_sparse_stats(config)
    )
    if bool(getattr(config, "fast_recency_only", False)) and not recency_accuracy_guarded:

        def _select_keep_indices_recency(
            *,
            total_tokens: int,
            prefill_len: int,
            protect_prefill: bool,
            budget_total: int,
            **_: Any,
        ) -> dict[str, Any] | None:
            keep_indices = build_keep_token_indices(
                total_tokens=total_tokens,
                kv_budget=budget_total,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                include_prefill_in_budget=True,
            )
            if keep_indices is None:
                return None
            return {"mode": "shared", "indices": keep_indices}

        def _select_keep_indices_for_group_recency(
            *,
            total_tokens: int,
            prefill_len: int,
            protect_prefill: bool,
            budget_total: int,
            **_: Any,
        ) -> dict[str, Any] | None:
            return _select_keep_indices_recency(
                total_tokens=total_tokens,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                budget_total=budget_total,
            )

        setattr(_select_keep_indices_recency, "_supports_paged", True)
        setattr(_select_keep_indices_for_group_recency, "_supports_paged_group", True)
        return (
            _select_keep_indices_recency,
            _select_keep_indices_for_group_recency,
            "enabled:recency_only",
        )

    if config.sparse_stats_path is None:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_set"
            )
        return None, None, "stats_path_not_set"

    stats_path = Path(config.sparse_stats_path).expanduser()
    if not stats_path.exists():
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_found"
            )
        return None, None, "stats_path_not_found"

    try:
        from triattention.vllm.core.config import TriAttentionConfig
        from triattention.vllm.core.compressor import TriAttentionCompressor
        from triattention.vllm.core.scoring import (
            compute_scores_pytorch,
            compute_scores_triton,
        )
        from triattention.vllm.core.utils import normalize_scores
    except Exception as exc:  # pragma: no cover - import safety
        raise RuntimeError(
            f"{TRITON_SCORING_REQUIRED_MARKER}:import_failed:{type(exc).__name__}"
        ) from exc

    if requested_pruning_mode not in {"per_layer", "per_head", "per_layer_per_head"}:
        if strict_triton_required:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:unsupported_pruning_mode:{requested_pruning_mode}"
            )
        return None, None, f"unsupported_pruning_mode:{requested_pruning_mode}"
    # Keep per-head score tensor and decide aggregation in selector;
    # this matches HF path better than forcing mean aggregation inside scoring.
    pruning_mode = "per_head"
    per_head_semantics = config.per_head_selection_semantics

    def _resolve_effective_model_path() -> Path | None:
        if getattr(config, "model_path", None) is not None:
            return Path(config.model_path)
        if base_runner is None:
            return None
        candidates: list[Any] = []
        candidates.append(getattr(getattr(base_runner, "model_config", None), "model", None))
        candidates.append(
            getattr(
                getattr(getattr(base_runner, "vllm_config", None), "model_config", None),
                "model",
                None,
            )
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return Path(candidate)
            if isinstance(candidate, Path):
                return candidate
        return None

    effective_model_path = _resolve_effective_model_path()

    def _resolve_effective_device() -> torch.device | None:
        if base_runner is None:
            return None
        candidates: list[Any] = [
            getattr(base_runner, "device", None),
            getattr(getattr(base_runner, "device_config", None), "device", None),
            getattr(
                getattr(getattr(base_runner, "vllm_config", None), "device_config", None),
                "device",
                None,
            ),
        ]
        for candidate in candidates:
            if isinstance(candidate, torch.device):
                return candidate
            if isinstance(candidate, str) and candidate.strip():
                try:
                    return torch.device(candidate.strip())
                except Exception:
                    continue
        return None

    effective_device = _resolve_effective_device()
    device_type = effective_device.type if effective_device is not None else None
    def _is_ascend_runner() -> bool:
        if base_runner is None:
            return False
        module_name = type(base_runner).__module__
        if isinstance(module_name, str) and module_name.startswith("vllm_ascend."):
            return True
        runner_repr = repr(type(base_runner)).lower()
        if "vllm_ascend" in runner_repr:
            return True
        device_config = getattr(base_runner, "device_config", None)
        vllm_device_config = getattr(
            getattr(base_runner, "vllm_config", None),
            "device_config",
            None,
        )
        for candidate in (device_config, vllm_device_config):
            for attr_name in ("device", "device_type"):
                raw = getattr(candidate, attr_name, None)
                if raw is None:
                    continue
                value = str(raw).lower()
                if "npu" in value or "ascend" in value:
                    return True
        return False

    is_ascend_runner = _is_ascend_runner()
    scoring_backend = getattr(config, "scoring_backend", "auto").strip().lower()
    use_torch_scoring = scoring_backend in {"torch", "pytorch"} or (
        scoring_backend == "auto"
        and (is_ascend_runner or device_type not in {None, "cuda"})
    )
    score_backend_name = "torch" if use_torch_scoring else "triton"
    log_execution_path = bool(
        getattr(config, "logging_enabled", True)
        and getattr(config, "log_execution_path", True)
    )
    log_core_trace = bool(
        log_execution_path
        and getattr(config, "log_core_trace", False)
    )
    log_selector_debug = bool(
        log_execution_path
        and getattr(config, "log_selector_debug", False)
    )

    tri_cfg_kwargs: dict[str, Any] = {}
    if effective_device is not None:
        tri_cfg_kwargs["device"] = effective_device

    def _resolve_tensor_parallel_info() -> tuple[int, int]:
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_size = int(get_tensor_model_parallel_world_size())
            tp_rank = int(get_tensor_model_parallel_rank())
            if tp_size > 0:
                return max(0, tp_rank), max(1, tp_size)
        except Exception:
            pass

        if base_runner is None:
            return 0, 1
        parallel_configs = [
            getattr(base_runner, "parallel_config", None),
            getattr(getattr(base_runner, "vllm_config", None), "parallel_config", None),
        ]
        for parallel_config in parallel_configs:
            if parallel_config is None:
                continue
            tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
            tp_rank = int(
                getattr(
                    parallel_config,
                    "tensor_parallel_rank",
                    getattr(base_runner, "tp_rank", 0),
                )
                or 0
            )
            return max(0, tp_rank), max(1, tp_size)
        return 0, 1

    tp_rank, tp_size = _resolve_tensor_parallel_info()
    score_max_layers = max(0, int(getattr(config, "score_max_layers", 0) or 0))
    score_layer_stride = max(1, int(getattr(config, "score_layer_stride", 1) or 1))

    def _pick_uniform_entries(entries: list[Any], limit: int) -> list[Any]:
        if limit <= 0 or len(entries) <= limit:
            return entries
        if limit == 1:
            return [entries[-1]]
        last = len(entries) - 1
        positions: list[int] = []
        seen: set[int] = set()
        for i in range(limit):
            pos = int(round((i * last) / float(limit - 1)))
            if pos not in seen:
                positions.append(pos)
                seen.add(pos)
        # Rounding can duplicate positions for very small lists; fill from the
        # tail because later layers tend to be more task-specific.
        pos = last
        while len(positions) < limit and pos >= 0:
            if pos not in seen:
                positions.append(pos)
                seen.add(pos)
            pos -= 1
        return [entries[pos] for pos in sorted(positions[:limit])]

    def _filter_layer_entries_for_scoring(entries: list[Any]) -> list[Any]:
        if not entries:
            return entries
        filtered = [
            entry
            for pos, entry in enumerate(entries)
            if pos % score_layer_stride == 0
        ]
        if not filtered:
            filtered = [entries[-1]]
        return _pick_uniform_entries(filtered, score_max_layers)

    tri_cfg = TriAttentionConfig(
        stats_path=stats_path,
        model_path=effective_model_path,
        kv_budget=config.kv_budget,
        divide_length=config.divide_length,
        pruning_mode=pruning_mode,
        score_aggregation=config.sparse_score_aggregation,
        sparse_normalize_scores=config.sparse_normalize_scores,
        window_size=min(config.window_size, max(config.kv_budget - 1, 0)),
        include_prefill_in_budget=config.include_prefill_in_budget,
        protect_prefill=config.protect_prefill,
        disable_mlr=config.disable_mlr,
        disable_trig=config.disable_trig,
        disable_top_n_high_freq=config.disable_top_n_high_freq,
        use_triton_scoring=not use_torch_scoring,
        use_trig_cache=not use_torch_scoring,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        **tri_cfg_kwargs,
    )
    compressor = TriAttentionCompressor(tri_cfg)
    available_layers_sorted: tuple[int, ...] | None = None
    available_layers_set: set[int] | None = None
    def _resolve_effective_recent_count(total_tokens: int) -> int:
        if total_tokens <= 0 or config.window_size <= 0:
            return 0
        # The runtime selector must preserve the same trailing protection window
        # regardless of request lifecycle details. Tying this to transient
        # "recent_unabsorbed" bookkeeping lets live serve requests under-protect
        # the tail (often collapsing to zero) even though fresh/offline
        # selection correctly preserves `window_size` tokens. That divergence
        # changes the keep set and cascades into output corruption.
        return min(config.window_size, total_tokens)

    def _resolve_layer_idx_for_stats(layer_idx: int) -> int:
        nonlocal available_layers_sorted
        nonlocal available_layers_set
        compressor._lazy_init()
        if available_layers_sorted is None or available_layers_set is None:
            available_layers_sorted = tuple(sorted(compressor.head_stats.keys()))
            available_layers_set = set(available_layers_sorted)
        if not available_layers_sorted:
            raise RuntimeError("empty_head_stats")
        if layer_idx in available_layers_set:
            return layer_idx
        return available_layers_sorted[layer_idx % len(available_layers_sorted)]

    reduced_head_stats_cache: dict[tuple[Any, ...], tuple[dict[str, torch.Tensor], torch.Tensor]] = {}
    tp_sliced_head_stats_cache: dict[tuple[int, int, int, int], tuple[dict[str, torch.Tensor], torch.Tensor]] = {}

    def _kv_cache_key_tensor(kv_cache: Any) -> torch.Tensor:
        if isinstance(kv_cache, torch.Tensor):
            return kv_cache
        if (
            isinstance(kv_cache, (list, tuple))
            and kv_cache
            and isinstance(kv_cache[0], torch.Tensor)
        ):
            return kv_cache[0]
        raise RuntimeError(f"unsupported_kv_cache_ref:{type(kv_cache).__name__}")

    def _kv_cache_num_heads(kv_cache: Any) -> int:
        key_tensor = _kv_cache_key_tensor(kv_cache)
        if key_tensor.ndim == 4:
            return int(key_tensor.shape[2])
        if key_tensor.ndim == 5:
            return int(key_tensor.shape[3])
        raise RuntimeError(f"unsupported_kv_cache_rank:{key_tensor.ndim}")

    def _kv_cache_device(kv_cache: Any) -> torch.device:
        return _kv_cache_key_tensor(kv_cache).device

    def _kv_cache_head_dim(kv_cache: Any) -> int:
        key_tensor = _kv_cache_key_tensor(kv_cache)
        if key_tensor.ndim in {4, 5}:
            return int(key_tensor.shape[-1])
        raise RuntimeError(f"unsupported_kv_cache_rank:{key_tensor.ndim}")

    def _runtime_freq_count_from_head_dim(head_dim: int) -> int:
        # For partial rotary models (e.g. Qwen3.5), freq_count != head_dim // 2.
        # Use the resolved partial_rotary_factor from tri_cfg (which comes from
        # stats metadata) to derive the expected runtime freq_count from the
        # runtime KV cache's head_dim. This correctly detects genuine stats-vs-
        # runtime mismatches (different head_dim or different partial_rotary)
        # while not falsely flagging partial-rotary models as mismatched.
        partial_factor = float(getattr(tri_cfg, "partial_rotary_factor", 1.0) or 1.0)
        rotary_dim = int(int(head_dim) * partial_factor)
        return max(0, rotary_dim // 2)

    def _runtime_freq_count_from_kv_cache(kv_cache: Any) -> int:
        return _runtime_freq_count_from_head_dim(_kv_cache_head_dim(kv_cache))

    def _runtime_freq_count_from_keys(keys_dense: torch.Tensor) -> int:
        return _runtime_freq_count_from_head_dim(int(keys_dense.shape[-1]))

    def _stats_freq_count(
        score_head_stats: dict[str, torch.Tensor],
        score_freq_scale_sq: torch.Tensor,
    ) -> tuple[int | None, str | None]:
        freq_dims: dict[str, int] = {}
        q_mean_complex = score_head_stats.get("q_mean_complex")
        if isinstance(q_mean_complex, torch.Tensor) and q_mean_complex.ndim >= 2:
            if int(q_mean_complex.shape[-1]) == 2 and q_mean_complex.ndim >= 3:
                freq_dims["q_mean_complex"] = int(q_mean_complex.shape[-2])
            else:
                freq_dims["q_mean_complex"] = int(q_mean_complex.shape[-1])
        q_abs_mean = score_head_stats.get("q_abs_mean")
        if isinstance(q_abs_mean, torch.Tensor) and q_abs_mean.ndim >= 1:
            freq_dims["q_abs_mean"] = int(q_abs_mean.shape[-1])
        if isinstance(score_freq_scale_sq, torch.Tensor) and score_freq_scale_sq.ndim >= 1:
            freq_dims["freq_scale_sq"] = int(score_freq_scale_sq.shape[-1])
        omega = getattr(compressor, "omega", None)
        if isinstance(omega, torch.Tensor) and omega.ndim >= 1:
            freq_dims["omega"] = int(omega.shape[-1])
        known_dims = {value for value in freq_dims.values() if value > 0}
        if not known_dims:
            return None, None
        if len(known_dims) > 1:
            details = ",".join(f"{name}={value}" for name, value in sorted(freq_dims.items()))
            return None, f"stats_freq_internal_mismatch:{details}"
        return known_dims.pop(), None

    def _stats_shape_mismatch_reason(
        *,
        layer_idx: int,
        runtime_freq_count: int,
        score_head_stats: dict[str, torch.Tensor],
        score_freq_scale_sq: torch.Tensor,
    ) -> str | None:
        stats_freq_count, internal_reason = _stats_freq_count(
            score_head_stats,
            score_freq_scale_sq,
        )
        if internal_reason is not None:
            return f"{internal_reason}:layer={int(layer_idx)}"
        if stats_freq_count is None or runtime_freq_count <= 0:
            return None
        if int(stats_freq_count) != int(runtime_freq_count):
            return (
                "stats_shape_mismatch:"
                f"layer={int(layer_idx)}:"
                f"stats_freq={int(stats_freq_count)}:"
                f"runtime_freq={int(runtime_freq_count)}"
            )
        return None

    stats_shape_mismatch_warnings: set[str] = set()

    def _warn_stats_shape_mismatch_once(
        *,
        req_id: str | None,
        gid: int | None,
        layer_idx: int | None,
        mode: str,
        reason: str,
    ) -> None:
        key = f"{mode}:{layer_idx}:{reason}"
        if key in stats_shape_mismatch_warnings:
            return
        stats_shape_mismatch_warnings.add(key)
        _runtime_logger.warning(
            "TriAttention sparse stats incompatible with runtime KV; "
            "using recency fallback req=%s gid=%s layer=%s mode=%s "
            "reason=%s stats_path=%s",
            req_id,
            gid,
            layer_idx,
            mode,
            reason,
            str(stats_path),
        )

    def _slice_tensor_parallel_layer_stats(
        *,
        resolved_layer_idx: int,
        layer_stats: dict[str, torch.Tensor],
        layer_freq_scale_sq: torch.Tensor,
        runtime_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        stats_heads = int(layer_freq_scale_sq.shape[0])
        if tp_size <= 1 or runtime_heads <= 0 or stats_heads == runtime_heads:
            return layer_stats, layer_freq_scale_sq
        if stats_heads % tp_size != 0:
            return layer_stats, layer_freq_scale_sq
        local_stats_heads = stats_heads // tp_size
        if local_stats_heads < runtime_heads or local_stats_heads % runtime_heads != 0:
            return layer_stats, layer_freq_scale_sq
        local_rank = min(max(0, int(tp_rank)), tp_size - 1)
        cache_key = (resolved_layer_idx, stats_heads, local_rank, tp_size)
        cached = tp_sliced_head_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        start = local_rank * local_stats_heads
        end = start + local_stats_heads
        sliced_stats: dict[str, torch.Tensor] = {}
        for name, value in layer_stats.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == stats_heads
            ):
                sliced_stats[name] = value[start:end].contiguous()
            else:
                sliced_stats[name] = value
        sliced_freq_scale_sq = layer_freq_scale_sq[start:end].contiguous()
        sliced = (sliced_stats, sliced_freq_scale_sq)
        tp_sliced_head_stats_cache[cache_key] = sliced
        return sliced

    def _reduce_head_stats_to_target(
        *,
        layer_stats: dict[str, torch.Tensor],
        layer_freq_scale_sq: torch.Tensor,
        target_heads: int,
        cache_key: tuple[Any, ...],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        cached = reduced_head_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        source_heads = int(layer_freq_scale_sq.shape[0])
        if source_heads == target_heads:
            reduced = (layer_stats, layer_freq_scale_sq)
            reduced_head_stats_cache[cache_key] = reduced
            return reduced
        if target_heads <= 0 or source_heads % target_heads != 0:
            raise RuntimeError(
                f"incompatible_head_mapping:source={source_heads},target={target_heads}"
            )
        group_size = source_heads // target_heads

        reduced_stats: dict[str, torch.Tensor] = {}
        q_abs_mean = layer_stats.get("q_abs_mean")
        if isinstance(q_abs_mean, torch.Tensor):
            reduced_stats["q_abs_mean"] = (
                q_abs_mean.reshape(target_heads, group_size, q_abs_mean.shape[1])
                .mean(dim=1)
                .contiguous()
            )

        q_mean_complex = layer_stats.get("q_mean_complex")
        if isinstance(q_mean_complex, torch.Tensor):
            reduced_stats["q_mean_complex"] = (
                q_mean_complex.reshape(
                    target_heads,
                    group_size,
                    q_mean_complex.shape[1],
                    q_mean_complex.shape[2],
                )
                .mean(dim=1)
                .contiguous()
            )

        reduced_freq_scale_sq = (
            layer_freq_scale_sq.reshape(
                target_heads,
                group_size,
                layer_freq_scale_sq.shape[1],
            )
            .mean(dim=1)
            .contiguous()
        )
        reduced = (reduced_stats, reduced_freq_scale_sq)
        reduced_head_stats_cache[cache_key] = reduced
        return reduced

    def _compute_layer_scores(
        keys_dense: torch.Tensor,
        *,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = int(keys_dense.shape[1])
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        mismatch_reason = _stats_shape_mismatch_reason(
            layer_idx=layer_idx,
            runtime_freq_count=_runtime_freq_count_from_keys(keys_dense),
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
        )
        if mismatch_reason is not None:
            raise _StatsShapeMismatch(mismatch_reason)

        scores = _compute_layer_scores_raw(
            keys_dense=keys_dense,
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            round_start=round_start,
        )

        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    def _resolve_layer_score_inputs(
        *,
        layer_idx: int,
        runtime_heads: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, int]:
        resolved_layer_idx = _resolve_layer_idx_for_stats(layer_idx)
        layer_head_stats = compressor.head_stats[resolved_layer_idx]
        layer_freq_scale_sq = layer_head_stats.get("freq_scale_sq")
        if not isinstance(layer_freq_scale_sq, torch.Tensor):
            freq_scale_sq = getattr(compressor, "freq_scale_sq", None)
            if (
                isinstance(freq_scale_sq, torch.Tensor)
                and 0 <= int(resolved_layer_idx) < int(freq_scale_sq.shape[0])
            ):
                layer_freq_scale_sq = freq_scale_sq[resolved_layer_idx]
            else:
                raise _StatsShapeMismatch(
                    f"stats_freq_missing:layer={int(resolved_layer_idx)}"
                )
        layer_head_stats, layer_freq_scale_sq = _slice_tensor_parallel_layer_stats(
            resolved_layer_idx=resolved_layer_idx,
            layer_stats=layer_head_stats,
            layer_freq_scale_sq=layer_freq_scale_sq,
            runtime_heads=runtime_heads,
        )
        stats_heads = int(layer_freq_scale_sq.shape[0])
        use_hf_group_max = (
            stats_heads != runtime_heads
            and (
                (
                    requested_pruning_mode == "per_head"
                    and per_head_semantics == "hf_aligned_global_per_head"
                )
                or requested_pruning_mode == "per_layer_per_head"
            )
        )
        score_head_stats = layer_head_stats
        score_freq_scale_sq = layer_freq_scale_sq
        group_size = 1
        if use_hf_group_max:
            if runtime_heads <= 0 or stats_heads % runtime_heads != 0:
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:incompatible_head_mapping:source={stats_heads},target={runtime_heads}"
                )
            group_size = stats_heads // runtime_heads
        elif stats_heads != runtime_heads:
            score_head_stats, score_freq_scale_sq = _reduce_head_stats_to_target(
                layer_stats=layer_head_stats,
                layer_freq_scale_sq=layer_freq_scale_sq,
                target_heads=runtime_heads,
                cache_key=(
                    "tp_local",
                    resolved_layer_idx,
                    runtime_heads,
                    tp_rank,
                    tp_size,
                    stats_heads,
                ),
            )
        return score_head_stats, score_freq_scale_sq, use_hf_group_max, group_size

    def _reduce_grouped_head_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        group_size: int,
        aggregate_mode: str,
    ) -> torch.Tensor:
        grouped = scores.view(
            scores.shape[0],
            runtime_heads,
            group_size,
            scores.shape[-1],
        )
        if aggregate_mode == "mean":
            return grouped.mean(dim=2)
        return grouped.max(dim=2).values

    def _layer_group_aggregation_mode() -> str:
        if requested_pruning_mode == "per_layer_per_head":
            return config.layer_perhead_aggregation
        return "max"

    def _compute_layer_scores_raw(
        *,
        keys_dense: torch.Tensor,
        score_head_stats: dict[str, torch.Tensor],
        score_freq_scale_sq: torch.Tensor,
        use_hf_group_max: bool,
        group_size: int,
        round_start: int,
    ) -> torch.Tensor:
        score_inputs = (
            keys_dense.repeat_interleave(group_size, dim=1).contiguous()
            if use_hf_group_max and group_size > 1
            else keys_dense
        )
        mismatch_reason = _stats_shape_mismatch_reason(
            layer_idx=-1,
            runtime_freq_count=_runtime_freq_count_from_keys(score_inputs),
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
        )
        if mismatch_reason is not None:
            raise _StatsShapeMismatch(mismatch_reason)
        try:
            if use_torch_scoring:
                return compute_scores_pytorch(
                    key_states=score_inputs,
                    cache_positions=None,
                    head_stats=score_head_stats,
                    omega=compressor.omega,
                    offsets=compressor.offsets,
                    freq_scale_sq=score_freq_scale_sq,
                    config=tri_cfg,
                    round_start=round_start,
                )
            return compute_scores_triton(
                key_states=score_inputs,
                cache_positions=None,
                head_stats=score_head_stats,
                omega=compressor.omega,
                offsets=compressor.offsets,
                freq_scale_sq=score_freq_scale_sq,
                config=tri_cfg,
                round_start=round_start,
                trig_cache=getattr(compressor, "trig_cache", None),
            )
        except _StatsShapeMismatch:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:score_failed:{score_backend_name}:{type(exc).__name__}"
            ) from exc

    def _finalize_layer_scores(
        *,
        scores: torch.Tensor,
        runtime_heads: int,
        use_hf_group_max: bool,
        group_size: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:

        if config.sparse_normalize_scores:
            scores = normalize_scores(scores)
        mutate_scores = (
            config.window_size > 0
            or (protect_prefill and prefill_len > 0)
        )
        if mutate_scores:
            scores = scores.clone()
        if config.window_size > 0:
            total_tokens = int(scores.shape[-1])
            recent_count = _resolve_effective_recent_count(total_tokens)
            if recent_count > 0:
                scores[..., total_tokens - recent_count :] = float("inf")
        if protect_prefill and prefill_len > 0:
            scores[..., :prefill_len] = float("inf")
        if use_hf_group_max:
            scores = _reduce_grouped_head_scores(
                scores=scores,
                runtime_heads=runtime_heads,
                group_size=group_size,
                aggregate_mode=_layer_group_aggregation_mode(),
            )
        return scores

    def _compute_layer_scores_paged(
        *,
        kv_cache: Any,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        layer_idx: int,
        round_start: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        runtime_heads = _kv_cache_num_heads(kv_cache)
        (
            score_head_stats,
            score_freq_scale_sq,
            use_hf_group_max,
            group_size,
        ) = _resolve_layer_score_inputs(
            layer_idx=layer_idx,
            runtime_heads=runtime_heads,
        )
        mismatch_reason = _stats_shape_mismatch_reason(
            layer_idx=layer_idx,
            runtime_freq_count=_runtime_freq_count_from_kv_cache(kv_cache),
            score_head_stats=score_head_stats,
            score_freq_scale_sq=score_freq_scale_sq,
        )
        if mismatch_reason is not None:
            raise _StatsShapeMismatch(mismatch_reason)
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
        chunks: list[torch.Tensor] = []
        start = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            keys_chunk = gather_request_k_dense_range(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                start_token=start,
                num_tokens=curr_tokens,
            )
            chunk_scores = _compute_layer_scores_raw(
                keys_dense=keys_chunk,
                score_head_stats=score_head_stats,
                score_freq_scale_sq=score_freq_scale_sq,
                use_hf_group_max=use_hf_group_max,
                group_size=group_size,
                round_start=round_start,
            )
            chunks.append(chunk_scores)
            start += curr_tokens
        scores = torch.cat(chunks, dim=-1)
        return _finalize_layer_scores(
            scores=scores,
            runtime_heads=runtime_heads,
            use_hf_group_max=use_hf_group_max,
            group_size=group_size,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )

    def _build_token_guard_mask(
        *,
        start_token: int,
        num_tokens: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        if config.window_size <= 0 and not (protect_prefill and prefill_len > 0):
            return None
        token_positions = torch.arange(
            start_token,
            start_token + num_tokens,
            device=device,
            dtype=torch.long,
        )
        guard_mask = torch.zeros_like(token_positions, dtype=torch.bool)
        if config.window_size > 0:
            recent_count = _resolve_effective_recent_count(total_tokens)
            window_start = max(0, total_tokens - recent_count)
            guard_mask |= token_positions >= window_start
        if protect_prefill and prefill_len > 0:
            guard_mask |= token_positions < prefill_len
        return guard_mask

    def _apply_token_guards(
        *,
        scores: torch.Tensor,
        start_token: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
    ) -> torch.Tensor:
        guard_mask = _build_token_guard_mask(
            start_token=start_token,
            num_tokens=int(scores.shape[-1]),
            total_tokens=total_tokens,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
            device=scores.device,
        )
        if guard_mask is None:
            return scores
        # Avoid host sync on guard_mask.any().item() in hot path.
        # masked_fill is a no-op when guard_mask has no true elements.
        return scores.masked_fill(guard_mask.view(1, 1, -1), float("inf"))

    def _score_chunk_tokens(block_size: int, total_tokens: int) -> int:
        upper = max(block_size, int(config.score_chunk_max_tokens))
        # Small/medium effective lengths do not need chunking; avoiding chunk splits
        # reduces Python loop overhead and kernel launches in the hot scoring path.
        if total_tokens <= upper:
            return max(block_size, total_tokens)
        return upper

    def _log_selector_scoring_enter(
        *,
        req_id: str | None,
        gid: int | None,
        layer_idx: int | None,
        mode: str,
        total_tokens: int,
        budget_total: int,
        round_start: int,
        chunk_tokens: int,
        layer_indices: list[int] | None = None,
    ) -> None:
        if not log_execution_path:
            return
        layer_count = len(layer_indices) if layer_indices is not None else None
        _runtime_logger.info(
            "TRIATTN_EXEC_PATH selector_scoring_enter req=%s gid=%s layer=%s "
            "mode=%s backend=%s total_tokens=%d budget_total=%d round_start=%d "
            "chunk_tokens=%d trig_enabled=%s normalize=%s layer_count=%s",
            req_id,
            gid,
            layer_idx,
            mode,
            score_backend_name,
            int(total_tokens),
            int(budget_total),
            int(round_start),
            int(chunk_tokens),
            not bool(getattr(config, "disable_trig", False)),
            bool(getattr(config, "sparse_normalize_scores", False)),
            layer_count,
        )

    def _keep_count_for_log(mode: str, indices: Any) -> int:
        try:
            if mode == "per_head":
                if hasattr(indices, "ndim") and hasattr(indices, "shape"):
                    return int(indices.shape[1]) if int(indices.ndim) == 2 else -1
                if isinstance(indices, list):
                    if not indices:
                        return 0
                    first_row = indices[0]
                    return len(first_row) if isinstance(first_row, list) else -1
                return -1
            if hasattr(indices, "numel"):
                return int(indices.numel())
            if isinstance(indices, list):
                return len(indices)
        except Exception:
            return -1
        return -1

    def _log_selector_scoring_exit(
        *,
        req_id: str | None,
        gid: int | None,
        layer_idx: int | None,
        mode: str,
        result_mode: str | None,
        keep_count: int | None,
        reason: str | None = None,
        layer_indices: list[int] | None = None,
    ) -> None:
        if not log_core_trace:
            return
        layer_count = len(layer_indices) if layer_indices is not None else None
        _runtime_logger.info(
            "TRIATTN_CORE_TRACE exit selector_scoring req=%s gid=%s layer=%s "
            "mode=%s backend=%s result_mode=%s keep_count=%s reason=%s "
            "layer_count=%s",
            req_id,
            gid,
            layer_idx,
            mode,
            score_backend_name,
            result_mode,
            keep_count,
            reason,
            layer_count,
        )

    def _selector_debug(
        *,
        execution_path: str,
        total_tokens: int,
        chunk_tokens: int,
        layer_indices: list[int] | None = None,
        total_layer_count: int | None = None,
    ) -> dict[str, Any]:
        if not log_selector_debug:
            return {}
        debug = {
            "debug_execution_path": execution_path,
            "debug_score_backend": score_backend_name,
            "debug_recent_count": _resolve_effective_recent_count(total_tokens),
            "debug_chunk_tokens": int(chunk_tokens),
            "debug_trig_enabled": not bool(getattr(config, "disable_trig", False)),
            "debug_normalize_scores": bool(
                getattr(config, "sparse_normalize_scores", False)
            ),
        }
        if layer_indices is not None:
            debug["debug_score_layer_indices"] = list(layer_indices)
        if total_layer_count is not None:
            debug["debug_score_layer_count_total"] = int(total_layer_count)
        return debug

    def _recency_fallback_result(
        *,
        req_id: str | None,
        gid: int | None,
        layer_idx: int | None,
        mode: str,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        budget_total: int,
        chunk_tokens: int,
        reason: str,
        layer_indices: list[int] | None = None,
        total_layer_count: int | None = None,
    ) -> dict[str, Any] | None:
        _warn_stats_shape_mismatch_once(
            req_id=req_id,
            gid=gid,
            layer_idx=layer_idx,
            mode=mode,
            reason=reason,
        )
        keep_indices = build_keep_token_indices(
            total_tokens=total_tokens,
            kv_budget=budget_total,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
            include_prefill_in_budget=True,
        )
        if keep_indices is None:
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode=mode,
                result_mode=None,
                keep_count=None,
                reason=f"{reason}:recency_fallback_unavailable",
                layer_indices=layer_indices,
            )
            return None
        result = {
            "mode": "shared",
            "indices": keep_indices,
            "semantic": "recency_fallback_stats_incompatible",
            **_selector_debug(
                execution_path=f"selector_hf>{mode}>recency_fallback",
                total_tokens=total_tokens,
                chunk_tokens=chunk_tokens,
                layer_indices=layer_indices,
                total_layer_count=total_layer_count,
            ),
        }
        _log_selector_scoring_exit(
            req_id=req_id,
            gid=gid,
            layer_idx=layer_idx,
            mode=mode,
            result_mode="shared:recency_fallback_stats_incompatible",
            keep_count=_keep_count_for_log("shared", keep_indices),
            reason=reason,
            layer_indices=layer_indices,
        )
        return result

    def _select_keep_indices_paged_streaming(
        *,
        kv_cache: Any,
        block_ids: list[int] | torch.Tensor,
        block_size: int,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
        req_id: str | None = None,
        gid: int | None = None,
    ) -> dict[str, Any]:
        chunk_tokens = _score_chunk_tokens(block_size, total_tokens)
        _log_selector_scoring_enter(
            req_id=req_id,
            gid=gid,
            layer_idx=layer_idx,
            mode="paged_layer",
            total_tokens=total_tokens,
            budget_total=budget_total,
            round_start=round_start,
            chunk_tokens=chunk_tokens,
            layer_indices=[int(layer_idx)],
        )
        try:
            runtime_heads = _kv_cache_num_heads(kv_cache)
            (
                score_head_stats,
                score_freq_scale_sq,
                use_hf_group_max,
                group_size,
            ) = _resolve_layer_score_inputs(
                layer_idx=layer_idx,
                runtime_heads=runtime_heads,
            )
            mismatch_reason = _stats_shape_mismatch_reason(
                layer_idx=layer_idx,
                runtime_freq_count=_runtime_freq_count_from_kv_cache(kv_cache),
                score_head_stats=score_head_stats,
                score_freq_scale_sq=score_freq_scale_sq,
            )
            if mismatch_reason is not None:
                raise _StatsShapeMismatch(mismatch_reason)
        except _StatsShapeMismatch as exc:
            return _recency_fallback_result(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="paged_layer",
                total_tokens=total_tokens,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                budget_total=budget_total,
                chunk_tokens=chunk_tokens,
                reason=str(exc),
                layer_indices=[int(layer_idx)],
                total_layer_count=1,
            )
        k = min(budget_total, total_tokens)
        if k <= 0:
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="paged_layer",
                result_mode="shared",
                keep_count=0,
                reason="non_positive_budget",
                layer_indices=[int(layer_idx)],
            )
            return {"mode": "shared", "indices": []}

        norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        raw_chunk_scores_cache: list[torch.Tensor] | None = None
        if config.sparse_normalize_scores:
            eps = 1e-8
            sum_vec: torch.Tensor | None = None
            sumsq_vec: torch.Tensor | None = None
            count = 0
            raw_chunk_scores_cache = []
            start = 0
            while start < total_tokens:
                curr_tokens = min(chunk_tokens, total_tokens - start)
                keys_chunk = gather_request_k_dense_range(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    block_size=block_size,
                    start_token=start,
                    num_tokens=curr_tokens,
                )
                raw_scores = _compute_layer_scores_raw(
                    keys_dense=keys_chunk,
                    score_head_stats=score_head_stats,
                    score_freq_scale_sq=score_freq_scale_sq,
                    use_hf_group_max=use_hf_group_max,
                    group_size=group_size,
                    round_start=round_start,
                )[0]
                raw_chunk_scores_cache.append(raw_scores)
                raw_fp32 = raw_scores.to(dtype=torch.float32)
                chunk_sum = raw_fp32.sum(dim=-1)
                chunk_sumsq = (raw_fp32 * raw_fp32).sum(dim=-1)
                if sum_vec is None:
                    sum_vec = chunk_sum
                    sumsq_vec = chunk_sumsq
                else:
                    sum_vec = sum_vec + chunk_sum
                    sumsq_vec = sumsq_vec + chunk_sumsq
                count += curr_tokens
                start += curr_tokens
            if sum_vec is None or sumsq_vec is None or count <= 0:
                _log_selector_scoring_exit(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=layer_idx,
                    mode="paged_layer",
                    result_mode=None,
                    keep_count=None,
                    reason="empty_normalization_stats",
                    layer_indices=[int(layer_idx)],
                )
                return None
            mean = sum_vec / float(count)
            if count > 1:
                var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
            else:
                var = torch.zeros_like(mean)
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var)
            std_safe = torch.where(std < eps, torch.ones_like(std), std)
            norm_stats = (mean, std_safe)

        # normalize_scores is z-score along token axis (affine monotonic per head/layer),
        # but for paths that aggregate across heads (e.g. max), normalization must be
        # preserved for HF alignment semantics. We use a two-pass chunked statistics
        # accumulation above instead of materializing full sequence scores.
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head:
            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
        else:
            best_scores = None
            best_indices = None

        start = 0
        chunk_idx = 0
        while start < total_tokens:
            curr_tokens = min(chunk_tokens, total_tokens - start)
            if raw_chunk_scores_cache is not None and chunk_idx < len(raw_chunk_scores_cache):
                chunk_scores = raw_chunk_scores_cache[chunk_idx].unsqueeze(0)
            else:
                keys_chunk = gather_request_k_dense_range(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    block_size=block_size,
                    start_token=start,
                    num_tokens=curr_tokens,
                )
                chunk_scores = _compute_layer_scores_raw(
                    keys_dense=keys_chunk,
                    score_head_stats=score_head_stats,
                    score_freq_scale_sq=score_freq_scale_sq,
                    use_hf_group_max=use_hf_group_max,
                    group_size=group_size,
                    round_start=round_start,
                )
            if norm_stats is not None:
                mean, std_safe = norm_stats
                chunk_scores = (
                    chunk_scores - mean.view(1, -1, 1)
                ) / std_safe.view(1, -1, 1)
            if use_hf_group_max:
                chunk_scores = _reduce_grouped_head_scores(
                    scores=chunk_scores,
                    runtime_heads=runtime_heads,
                    group_size=group_size,
                    aggregate_mode=_layer_group_aggregation_mode(),
                )
            chunk_scores = _apply_token_guards(
                scores=chunk_scores,
                start_token=start,
                total_tokens=total_tokens,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
            )

            if wants_per_head and chunk_scores.ndim == 3:
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            else:
                if chunk_scores.ndim == 3:
                    chunk_scores = chunk_scores.max(dim=1).values
                cand_k = min(k, int(chunk_scores.shape[-1]))
                cand = torch.topk(
                    chunk_scores[0],
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
            start += curr_tokens
            chunk_idx += 1

        if best_indices is None:
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="paged_layer",
                result_mode="shared",
                keep_count=0,
                reason="no_best_indices",
                layer_indices=[int(layer_idx)],
            )
            return {"mode": "shared", "indices": []}
        if wants_per_head and best_indices.ndim == 2:
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            result = {
                "mode": "per_head",
                "indices": keep_per_head,
                **_selector_debug(
                    execution_path="selector_hf>paged_layer>triattention_scoring",
                    total_tokens=total_tokens,
                    chunk_tokens=chunk_tokens,
                    layer_indices=[int(layer_idx)],
                    total_layer_count=1,
                ),
            }
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="paged_layer",
                result_mode="per_head",
                keep_count=_keep_count_for_log("per_head", keep_per_head),
                layer_indices=[int(layer_idx)],
            )
            return result
        keep = torch.sort(best_indices, dim=-1).values.contiguous()
        result = {
            "mode": "shared",
            "indices": keep,
            **_selector_debug(
                execution_path="selector_hf>paged_layer>triattention_scoring",
                total_tokens=total_tokens,
                chunk_tokens=chunk_tokens,
                layer_indices=[int(layer_idx)],
                total_layer_count=1,
            ),
        }
        _log_selector_scoring_exit(
            req_id=req_id,
            gid=gid,
            layer_idx=layer_idx,
            mode="paged_layer",
            result_mode="shared",
            keep_count=_keep_count_for_log("shared", keep),
            layer_indices=[int(layer_idx)],
        )
        return result

    def _select_keep_indices(
        *,
        keys_dense: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        block_ids: list[int] | torch.Tensor | None = None,
        block_size: int | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
        req_id: str | None = None,
        gid: int | None = None,
    ) -> dict[str, Any] | None:
        if total_tokens <= budget_total:
            return {"mode": "shared", "indices": list(range(total_tokens))}
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None

        if keys_dense is not None:
            _log_selector_scoring_enter(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="dense_layer",
                total_tokens=total_tokens,
                budget_total=budget_total,
                round_start=round_start,
                chunk_tokens=total_tokens,
                layer_indices=[int(layer_idx)],
            )
            try:
                scores = _compute_layer_scores(
                    keys_dense=keys_dense,
                    layer_idx=layer_idx,
                    round_start=round_start,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                )
            except _StatsShapeMismatch as exc:
                return _recency_fallback_result(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=layer_idx,
                    mode="dense_layer",
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    budget_total=budget_total,
                    chunk_tokens=total_tokens,
                    reason=str(exc),
                    layer_indices=[int(layer_idx)],
                    total_layer_count=1,
                )
        elif kv_cache is not None and block_ids is not None and block_size is not None:
            paged_result = _select_keep_indices_paged_streaming(
                kv_cache=kv_cache,
                block_ids=block_ids,
                block_size=block_size,
                total_tokens=total_tokens,
                layer_idx=layer_idx,
                round_start=round_start,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                budget_total=budget_total,
                req_id=req_id,
                gid=gid,
            )
            return paged_result
        else:
            raise RuntimeError("missing scoring inputs for selector")

        k = min(int(budget_total), int(scores.shape[-1]))
        if k <= 0:
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="dense_layer",
                result_mode="shared",
                keep_count=0,
                reason="non_positive_budget",
                layer_indices=[int(layer_idx)],
            )
            return {"mode": "shared", "indices": []}
        wants_per_head = requested_pruning_mode in {"per_head", "per_layer_per_head"}
        if wants_per_head and scores.ndim == 3:
            topk = torch.topk(
                scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices[0]
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            result = {
                "mode": "per_head",
                "indices": keep_per_head,
                **_selector_debug(
                    execution_path="selector_hf>dense_layer>triattention_scoring",
                    total_tokens=total_tokens,
                    chunk_tokens=total_tokens,
                    layer_indices=[int(layer_idx)],
                    total_layer_count=1,
                ),
            }
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=layer_idx,
                mode="dense_layer",
                result_mode="per_head",
                keep_count=_keep_count_for_log("per_head", keep_per_head),
                layer_indices=[int(layer_idx)],
            )
            return result

        scores_agg = scores
        if scores_agg.ndim == 3:
            scores_agg = scores_agg.max(dim=1).values
        selected = torch.topk(
            scores_agg,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices[0]
        keep = torch.sort(selected).values.contiguous()
        result = {
            "mode": "shared",
            "indices": keep,
            **_selector_debug(
                execution_path="selector_hf>dense_layer>triattention_scoring",
                total_tokens=total_tokens,
                chunk_tokens=total_tokens,
                layer_indices=[int(layer_idx)],
                total_layer_count=1,
            ),
        }
        _log_selector_scoring_exit(
            req_id=req_id,
            gid=gid,
            layer_idx=layer_idx,
            mode="dense_layer",
            result_mode="shared",
            keep_count=_keep_count_for_log("shared", keep),
            layer_indices=[int(layer_idx)],
        )
        return result

    def _select_keep_indices_for_group_per_head(
        *,
        layer_inputs: list[tuple[int, torch.Tensor]] | None = None,
        layer_input_iter: Callable[[], Iterable[tuple[int, torch.Tensor]]] | None = None,
        layer_kv_iter: Callable[
            [],
            Iterable[tuple[int, Any, list[int] | torch.Tensor, int]],
        ]
        | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        round_start: int,
        budget_total: int,
        req_id: str | None = None,
        gid: int | None = None,
    ) -> dict[str, Any] | None:
        if requested_pruning_mode != "per_head":
            return None
        if per_head_semantics != "hf_aligned_global_per_head":
            return None
        if total_tokens <= budget_total:
            head_count = 0
            first_item: Any | None = None
            indices_device = torch.device("cpu")
            if layer_inputs:
                head_count = int(layer_inputs[0][1].shape[1])
                indices_device = layer_inputs[0][1].device
            elif layer_input_iter is not None:
                first_item = next(iter(layer_input_iter()), None)
                if first_item is not None:
                    head_count = int(first_item[1].shape[1])
                    indices_device = first_item[1].device
            elif layer_kv_iter is not None:
                first_item = next(iter(layer_kv_iter()), None)
                if first_item is not None:
                    head_count = _kv_cache_num_heads(first_item[1])
                    indices_device = _kv_cache_device(first_item[1])
            if head_count <= 0:
                return {"mode": "per_head", "indices": []}
            all_indices = torch.arange(
                total_tokens,
                dtype=torch.long,
                device=indices_device,
            )
            return {
                "mode": "per_head",
                "indices": all_indices.unsqueeze(0).expand(head_count, -1).contiguous(),
            }
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            return None
        if layer_kv_iter is not None:
            iter_inputs = layer_kv_iter()
            iter_mode = "paged"
        elif layer_input_iter is not None:
            iter_inputs = layer_input_iter()
            iter_mode = "dense_iter"
        else:
            iter_inputs = layer_inputs or []
            iter_mode = "dense_list"
        if not iter_inputs:
            return None

        if iter_mode == "paged":
            group_agg_mode = os.environ.get(
                "TRIATTN_RUNTIME_DEBUG_GROUP_PERHEAD_AGG_MODE",
                "mean",
            ).strip().lower()
            if group_agg_mode not in {"mean", "max"}:
                group_agg_mode = "mean"
            layer_entries = list(iter_inputs)
            if not layer_entries:
                return None
            total_layer_entries = len(layer_entries)
            layer_entries = _filter_layer_entries_for_scoring(layer_entries)
            k = min(budget_total, total_tokens)
            if k <= 0:
                _log_selector_scoring_exit(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode="paged_global_per_head",
                    result_mode="per_head:hf_aligned_global_per_head",
                    keep_count=0,
                    reason="non_positive_budget",
                    layer_indices=[int(entry[0]) for entry in layer_entries],
                )
                return {"mode": "per_head", "indices": []}
            prepared_layers: list[dict[str, Any]] = []
            try:
                for layer_idx, kv_cache, block_ids, layer_block_size in layer_entries:
                    runtime_heads = _kv_cache_num_heads(kv_cache)
                    (
                        score_head_stats,
                        score_freq_scale_sq,
                        use_hf_group_max,
                        group_size,
                    ) = _resolve_layer_score_inputs(
                        layer_idx=layer_idx,
                        runtime_heads=runtime_heads,
                    )
                    mismatch_reason = _stats_shape_mismatch_reason(
                        layer_idx=layer_idx,
                        runtime_freq_count=_runtime_freq_count_from_kv_cache(kv_cache),
                        score_head_stats=score_head_stats,
                        score_freq_scale_sq=score_freq_scale_sq,
                    )
                    if mismatch_reason is not None:
                        raise _StatsShapeMismatch(mismatch_reason)
                    prepared_layers.append(
                        {
                            "layer_idx": layer_idx,
                            "kv_cache": kv_cache,
                            "block_ids": block_ids,
                            "block_size": layer_block_size,
                            "runtime_heads": runtime_heads,
                            "score_head_stats": score_head_stats,
                            "score_freq_scale_sq": score_freq_scale_sq,
                            "use_hf_group_max": use_hf_group_max,
                            "group_size": group_size,
                        }
                    )
            except _StatsShapeMismatch as exc:
                fallback_block_size = min(int(entry[3]) for entry in layer_entries)
                return _recency_fallback_result(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode="paged_global_per_head",
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    budget_total=budget_total,
                    chunk_tokens=_score_chunk_tokens(fallback_block_size, total_tokens),
                    reason=str(exc),
                    layer_indices=[int(entry[0]) for entry in layer_entries],
                    total_layer_count=total_layer_entries,
                )

            prepared_layer_indices = [int(entry["layer_idx"]) for entry in prepared_layers]

            min_block_size = min(entry["block_size"] for entry in prepared_layers)
            chunk_tokens = _score_chunk_tokens(min_block_size, total_tokens)
            _log_selector_scoring_enter(
                req_id=req_id,
                gid=gid,
                layer_idx=None,
                mode="paged_global_per_head",
                total_tokens=total_tokens,
                budget_total=budget_total,
                round_start=round_start,
                chunk_tokens=chunk_tokens,
                layer_indices=prepared_layer_indices,
            )
            norm_stats: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(prepared_layers)
            raw_scores_cache_by_layer: list[list[torch.Tensor] | None] = [None] * len(prepared_layers)
            if config.sparse_normalize_scores:
                eps = 1e-8
                for layer_pos, entry in enumerate(prepared_layers):
                    sum_vec: torch.Tensor | None = None
                    sumsq_vec: torch.Tensor | None = None
                    count = 0
                    layer_raw_scores: list[torch.Tensor] = []
                    start = 0
                    while start < total_tokens:
                        curr_tokens = min(chunk_tokens, total_tokens - start)
                        keys_chunk = gather_request_k_dense_range(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            start_token=start,
                            num_tokens=curr_tokens,
                        )
                        raw_scores = _compute_layer_scores_raw(
                            keys_dense=keys_chunk,
                            score_head_stats=entry["score_head_stats"],
                            score_freq_scale_sq=entry["score_freq_scale_sq"],
                            use_hf_group_max=entry["use_hf_group_max"],
                            group_size=entry["group_size"],
                            round_start=round_start,
                        )[0]
                        layer_raw_scores.append(raw_scores)
                        raw_fp32 = raw_scores.to(dtype=torch.float32)
                        chunk_sum = raw_fp32.sum(dim=-1)
                        chunk_sumsq = (raw_fp32 * raw_fp32).sum(dim=-1)
                        if sum_vec is None:
                            sum_vec = chunk_sum
                            sumsq_vec = chunk_sumsq
                        else:
                            sum_vec = sum_vec + chunk_sum
                            sumsq_vec = sumsq_vec + chunk_sumsq
                        count += curr_tokens
                        start += curr_tokens

                    if (
                        sum_vec is None
                        or sumsq_vec is None
                        or count <= 0
                    ):
                        _log_selector_scoring_exit(
                            req_id=req_id,
                            gid=gid,
                            layer_idx=None,
                            mode="paged_global_per_head",
                            result_mode=None,
                            keep_count=None,
                            reason="empty_normalization_stats",
                            layer_indices=prepared_layer_indices,
                        )
                        return None
                    mean = sum_vec / float(count)
                    if count > 1:
                        var = (sumsq_vec - float(count) * (mean * mean)) / float(count - 1)
                    else:
                        var = torch.zeros_like(mean)
                    var = torch.clamp(var, min=0.0)
                    std = torch.sqrt(var)
                    std_safe = torch.where(std < eps, torch.ones_like(std), std)
                    norm_stats[layer_pos] = (
                        mean,
                        std_safe,
                    )
                    raw_scores_cache_by_layer[layer_pos] = layer_raw_scores

            best_scores: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None
            start = 0
            chunk_idx = 0
            while start < total_tokens:
                curr_tokens = min(chunk_tokens, total_tokens - start)
                chunk_guard_mask = _build_token_guard_mask(
                    start_token=start,
                    num_tokens=curr_tokens,
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    device=_kv_cache_device(prepared_layers[0]["kv_cache"]),
                )
                chunk_agg: torch.Tensor | None = None
                layer_count = 0
                for layer_pos, entry in enumerate(prepared_layers):
                    layer_raw_cache = raw_scores_cache_by_layer[layer_pos]
                    if layer_raw_cache is not None and chunk_idx < len(layer_raw_cache):
                        chunk_scores = layer_raw_cache[chunk_idx].unsqueeze(0)
                    else:
                        keys_chunk = gather_request_k_dense_range(
                            kv_cache=entry["kv_cache"],
                            block_ids=entry["block_ids"],
                            block_size=entry["block_size"],
                            start_token=start,
                            num_tokens=curr_tokens,
                        )
                        chunk_scores = _compute_layer_scores_raw(
                            keys_dense=keys_chunk,
                            score_head_stats=entry["score_head_stats"],
                            score_freq_scale_sq=entry["score_freq_scale_sq"],
                            use_hf_group_max=entry["use_hf_group_max"],
                            group_size=entry["group_size"],
                            round_start=round_start,
                        )
                    if config.sparse_normalize_scores:
                        mean, std_safe = norm_stats[layer_pos] or (None, None)
                        if mean is None or std_safe is None:
                            _log_selector_scoring_exit(
                                req_id=req_id,
                                gid=gid,
                                layer_idx=None,
                                mode="paged_global_per_head",
                                result_mode=None,
                                keep_count=None,
                                reason="missing_normalization_stats",
                                layer_indices=prepared_layer_indices,
                            )
                            return None
                        chunk_scores = (chunk_scores - mean.view(1, -1, 1)) / std_safe.view(1, -1, 1)
                    if chunk_guard_mask is not None:
                        chunk_scores = chunk_scores.masked_fill(
                            chunk_guard_mask.view(1, 1, -1),
                            float("inf"),
                        )
                    if entry["use_hf_group_max"]:
                        chunk_scores = _reduce_grouped_head_scores(
                            scores=chunk_scores,
                            runtime_heads=entry["runtime_heads"],
                            group_size=entry["group_size"],
                            aggregate_mode="max",
                        )
                    if chunk_scores.ndim != 3:
                        raise RuntimeError(
                            f"unexpected_score_rank_for_per_head:{chunk_scores.ndim}"
                        )
                    layer_scores = chunk_scores[0]
                    if chunk_agg is None:
                        chunk_agg = layer_scores.clone()
                    else:
                        if group_agg_mode == "max":
                            chunk_agg = torch.maximum(chunk_agg, layer_scores)
                        else:
                            chunk_agg.add_(layer_scores)
                    layer_count += 1

                if chunk_agg is None or layer_count <= 0:
                    _log_selector_scoring_exit(
                        req_id=req_id,
                        gid=gid,
                        layer_idx=None,
                        mode="paged_global_per_head",
                        result_mode=None,
                        keep_count=None,
                        reason="empty_chunk_aggregate",
                        layer_indices=prepared_layer_indices,
                    )
                    return None
                chunk_final = (
                    chunk_agg
                    if group_agg_mode == "max"
                    else chunk_agg.div(float(layer_count))
                )
                cand_k = min(k, int(chunk_final.shape[-1]))
                cand = torch.topk(
                    chunk_final,
                    k=cand_k,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                cand_scores = cand.values
                cand_indices = cand.indices + start
                if best_scores is None or best_indices is None:
                    best_scores = cand_scores
                    best_indices = cand_indices
                else:
                    merged_scores = torch.cat([best_scores, cand_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, cand_indices], dim=-1)
                    merge_k = min(k, int(merged_scores.shape[-1]))
                    picked = torch.topk(
                        merged_scores,
                        k=merge_k,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )
                    best_scores = picked.values
                    best_indices = torch.gather(
                        merged_indices,
                        dim=-1,
                        index=picked.indices,
                    )
                start += curr_tokens
                chunk_idx += 1

            if best_indices is None:
                _log_selector_scoring_exit(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode="paged_global_per_head",
                    result_mode=None,
                    keep_count=None,
                    reason="no_best_indices",
                    layer_indices=prepared_layer_indices,
                )
                return None
            keep_per_head = torch.sort(best_indices, dim=-1).values.contiguous()
            result = {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": group_agg_mode,
                **_selector_debug(
                    execution_path="selector_hf>paged_global_per_head>triattention_scoring",
                    total_tokens=total_tokens,
                    chunk_tokens=chunk_tokens,
                    layer_indices=prepared_layer_indices,
                    total_layer_count=total_layer_entries,
                ),
            }
            if log_selector_debug:
                result.update(
                    {
                        "debug_group_layer_indices": prepared_layer_indices,
                        "debug_group_layer_count_total": total_layer_entries,
                    }
                )
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=None,
                mode="paged_global_per_head",
                result_mode="per_head:hf_aligned_global_per_head",
                keep_count=_keep_count_for_log("per_head", keep_per_head),
                layer_indices=prepared_layer_indices,
            )
            return result
        else:
            if iter_mode in {"dense_iter", "dense_list"}:
                iter_inputs = _filter_layer_entries_for_scoring(list(iter_inputs))
            dense_layer_indices = [int(layer_idx) for layer_idx, _keys_dense in iter_inputs]
            _log_selector_scoring_enter(
                req_id=req_id,
                gid=gid,
                layer_idx=None,
                mode=iter_mode,
                total_tokens=total_tokens,
                budget_total=budget_total,
                round_start=round_start,
                chunk_tokens=total_tokens,
                layer_indices=dense_layer_indices,
            )
            aggregated_scores: torch.Tensor | None = None
            layer_count = 0
            try:
                for layer_idx, keys_dense in iter_inputs:
                    scores = _compute_layer_scores(
                        keys_dense=keys_dense,
                        layer_idx=layer_idx,
                        round_start=round_start,
                        prefill_len=prefill_len,
                        protect_prefill=protect_prefill,
                    )
                    if scores.ndim != 3:
                        raise RuntimeError(
                            f"unexpected_score_rank_for_per_head:{scores.ndim}"
                        )
                    layer_scores = scores[0]
                    if aggregated_scores is None:
                        aggregated_scores = layer_scores.clone()
                    else:
                        aggregated_scores.add_(layer_scores)
                    layer_count += 1
            except _StatsShapeMismatch as exc:
                return _recency_fallback_result(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode=iter_mode,
                    total_tokens=total_tokens,
                    prefill_len=prefill_len,
                    protect_prefill=protect_prefill,
                    budget_total=budget_total,
                    chunk_tokens=total_tokens,
                    reason=str(exc),
                    layer_indices=dense_layer_indices,
                    total_layer_count=len(dense_layer_indices),
                )
            if aggregated_scores is None or layer_count <= 0:
                _log_selector_scoring_exit(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode=iter_mode,
                    result_mode=None,
                    keep_count=None,
                    reason="empty_aggregate",
                    layer_indices=dense_layer_indices,
                )
                return None
            aggregated_scores.div_(layer_count)
            k = min(budget_total, aggregated_scores.shape[-1])
            if k <= 0:
                _log_selector_scoring_exit(
                    req_id=req_id,
                    gid=gid,
                    layer_idx=None,
                    mode=iter_mode,
                    result_mode="per_head:hf_aligned_global_per_head",
                    keep_count=0,
                    reason="non_positive_budget",
                    layer_indices=dense_layer_indices,
                )
                return {"mode": "per_head", "indices": []}

            topk = torch.topk(
                aggregated_scores,
                k=k,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
            result = {
                "mode": "per_head",
                "indices": keep_per_head,
                "semantic": "hf_aligned_global_per_head",
                "group_agg_mode": "mean",
                **_selector_debug(
                    execution_path="selector_hf>dense_global_per_head>triattention_scoring",
                    total_tokens=total_tokens,
                    chunk_tokens=total_tokens,
                    layer_indices=dense_layer_indices,
                    total_layer_count=len(dense_layer_indices),
                ),
            }
            if log_selector_debug:
                result.update(
                    {
                        "debug_group_layer_indices": dense_layer_indices,
                        "debug_group_layer_count_total": len(dense_layer_indices),
                    }
                )
            _log_selector_scoring_exit(
                req_id=req_id,
                gid=gid,
                layer_idx=None,
                mode=iter_mode,
                result_mode="per_head:hf_aligned_global_per_head",
                keep_count=_keep_count_for_log("per_head", keep_per_head),
                layer_indices=dense_layer_indices,
            )
            return result

    setattr(_select_keep_indices, "_supports_paged", True)
    setattr(_select_keep_indices_for_group_per_head, "_supports_paged_group", True)
    layer_filter_status = (
        f":score_layers=max{score_max_layers or 'all'},stride{score_layer_stride}"
    )
    return (
        _select_keep_indices,
        _select_keep_indices_for_group_per_head,
        f"enabled:{score_backend_name}:tp={tp_rank}/{tp_size}{layer_filter_status}",
    )
