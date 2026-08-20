"""Configuration for TriAttention runtime integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_QWEN3_STATS = "qwen3_32b_int4_stats.pt"
_DEFAULT_GPT_OSS_STATS = "gpt_oss_120b_stats.pt"


def _packaged_stats_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "stats" / filename


def _resolve_packaged_sparse_stats_path(model_path_raw: str | None) -> Path | None:
    model_hint = (model_path_raw or "").lower().replace("_", "-")
    candidates: list[str] = []
    if "gpt" in model_hint and "oss" in model_hint:
        candidates.append(_DEFAULT_GPT_OSS_STATS)
    if "qwen" in model_hint:
        candidates.append(_DEFAULT_QWEN3_STATS)
    candidates.extend([_DEFAULT_QWEN3_STATS, _DEFAULT_GPT_OSS_STATS])

    seen: set[str] = set()
    for filename in candidates:
        if filename in seen:
            continue
        seen.add(filename)
        path = _packaged_stats_path(filename)
        if path.exists():
            return path
    return None


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


@dataclass
class TriAttentionRuntimeConfig:
    """Runtime config loaded by scheduler and worker.

    Phase 1B provides lifecycle + trigger signaling, with an optional
    experimental KV compaction hook for gather/score/select/scatter.
    """

    kv_budget: int = 2048
    divide_length: int = 128
    protect_prefill: bool = False
    disable_compression: bool = False

    enable_kv_usage_trigger: bool = False
    kv_usage_trigger: float = 0.98
    kv_usage_release: float = 0.90

    enable_experimental_kv_compaction: bool = True
    enable_experimental_block_reclaim: bool = True
    require_triton_scoring: bool = True
    require_physical_reclaim: bool = True
    logging_enabled: bool = True
    log_decisions: bool = True
    log_execution_path: bool = False
    log_execution_path_core_only: bool = False
    log_core_trace: bool = False
    log_selector_debug: bool = False
    fail_on_effective_len_regression: bool = True
    effective_len_regression_ratio: float = 0.9
    effective_len_guard_divide_multiples: int = 2
    defer_prefill_compression: bool = False
    defer_prefill_compression_on_ascend: bool = True
    min_decode_tokens_before_compress: int = 0
    min_decode_tokens_before_compress_on_ascend: int = 0
    score_chunk_max_tokens: int = 4096
    score_max_layers: int = 0
    score_max_layers_on_ascend: int = 0
    score_layer_stride: int = 1
    fast_recency_only: bool = False
    fast_recency_accuracy_guard: bool = True
    fast_recency_long_context_guard: bool = False
    fast_recency_long_context_guard_tokens: int = 16384
    min_reclaim_blocks: int = 1
    min_reclaim_blocks_on_ascend: int = 8
    prefill_min_reclaim_blocks_on_ascend: int = 32
    prefill_max_compressions_on_ascend: int = 1
    log_all_worker_events: bool = False
    enable_async_compression_boundary: bool = False
    enable_zero_copy_recency: bool = True
    zero_copy_recency_only_on_ascend: bool = True
    enable_packed_pos_delta_on_ascend: bool = False
    auto_fast_recency_on_ascend: bool = True
    early_install_proxy_on_ascend: bool = True
    preinstall_input_patch: bool = True
    force_eager_multi_req_on_ascend_effective_overrides: bool = False
    max_compressions_per_step_on_ascend: int = 4

    # Optional TriAttention-style scoring path (used by runtime hook when enabled).
    sparse_stats_path: Path | None = None
    model_path: Path | None = None
    pruning_mode: str = "per_head"
    sparse_score_aggregation: str = "mean"
    sparse_normalize_scores: bool = True
    window_size: int = 128
    include_prefill_in_budget: bool = True
    per_head_selection_semantics: str = "hf_aligned_global_per_head"
    layer_perhead_aggregation: str = "max"
    per_layer_aggregation: str = "max"
    allow_per_layer_mode: bool = False
    scoring_backend: str = "auto"
    disable_mlr: bool = False
    disable_trig: bool = False
    disable_top_n_high_freq: int = 0

    @classmethod
    def from_env(cls, prefix: str = "TRIATTN_RUNTIME_") -> "TriAttentionRuntimeConfig":
        env = os.environ

        def _get_raw(name: str) -> str | None:
            return env.get(prefix + name)

        def maybe_int(name: str, default: int) -> int:
            raw = _get_raw(name)
            return default if raw is None else int(raw)

        def maybe_float(name: str, default: float) -> float:
            raw = _get_raw(name)
            return default if raw is None else float(raw)

        def maybe_bool(name: str, default: bool) -> bool:
            raw = _get_raw(name)
            return default if raw is None else _parse_bool(raw)

        def maybe_str(name: str, default: str | None) -> str | None:
            raw = _get_raw(name)
            if raw is None:
                return default
            raw = raw.strip()
            return raw if raw else default

        sparse_stats_path_raw = maybe_str("SPARSE_STATS_PATH", None)
        model_path_raw = maybe_str("MODEL_PATH", None)
        packaged_sparse_stats_path = (
            _resolve_packaged_sparse_stats_path(model_path_raw)
            if sparse_stats_path_raw is None
            else None
        )
        sparse_stats_path_effective = (
            sparse_stats_path_raw
            or (
                str(packaged_sparse_stats_path)
                if packaged_sparse_stats_path is not None
                else None
            )
        )
        sparse_stats_path_candidate = (
            Path(sparse_stats_path_effective)
            if sparse_stats_path_effective
            else None
        )
        if sparse_stats_path_candidate is not None:
            try:
                stats_path_exists = sparse_stats_path_candidate.expanduser().exists()
            except (OSError, RuntimeError, ValueError):
                stats_path_exists = False
            if not stats_path_exists:
                fallback_sparse_stats_path = _resolve_packaged_sparse_stats_path(
                    model_path_raw
                )
                if fallback_sparse_stats_path is not None:
                    sparse_stats_path_candidate = fallback_sparse_stats_path
        logging_enabled = maybe_bool("LOGGING", cls.logging_enabled)
        log_decisions = maybe_bool("LOG_DECISIONS", cls.log_decisions)
        log_execution_path = maybe_bool(
            "LOG_EXECUTION_PATH",
            cls.log_execution_path,
        )
        log_execution_path_core_only = maybe_bool(
            "LOG_EXECUTION_PATH_CORE_ONLY",
            cls.log_execution_path_core_only,
        )
        log_core_trace = maybe_bool("LOG_CORE_TRACE", cls.log_core_trace)
        log_selector_debug = maybe_bool(
            "LOG_SELECTOR_DEBUG",
            cls.log_selector_debug,
        )
        log_all_worker_events = maybe_bool(
            "LOG_ALL_WORKER_EVENTS",
            cls.log_all_worker_events,
        )
        if not logging_enabled:
            log_decisions = False
            log_execution_path = False
            log_execution_path_core_only = False
            log_core_trace = False
            log_selector_debug = False
            log_all_worker_events = False
        if not log_execution_path:
            log_execution_path_core_only = False
            log_core_trace = False
            log_selector_debug = False
        fast_recency_only = maybe_bool("FAST_RECENCY_ONLY", cls.fast_recency_only)
        fast_recency_accuracy_guard_default = cls.fast_recency_accuracy_guard
        if (
            fast_recency_only
            and _get_raw("FAST_RECENCY_ONLY") is not None
            and _get_raw("FAST_RECENCY_ACCURACY_GUARD") is None
            and sparse_stats_path_effective is None
        ):
            fast_recency_accuracy_guard_default = False

        config = cls(
            kv_budget=maybe_int("KV_BUDGET", cls.kv_budget),
            divide_length=maybe_int("DIVIDE_LENGTH", cls.divide_length),
            protect_prefill=maybe_bool("PROTECT_PREFILL", cls.protect_prefill),
            disable_compression=maybe_bool(
                "DISABLE_COMPRESSION", cls.disable_compression
            ),
            enable_kv_usage_trigger=maybe_bool(
                "ENABLE_KV_USAGE_TRIGGER", cls.enable_kv_usage_trigger
            ),
            kv_usage_trigger=maybe_float("KV_USAGE_TRIGGER", cls.kv_usage_trigger),
            kv_usage_release=maybe_float("KV_USAGE_RELEASE", cls.kv_usage_release),
            enable_experimental_kv_compaction=maybe_bool(
                "ENABLE_EXPERIMENTAL_KV_COMPACTION",
                cls.enable_experimental_kv_compaction,
            ),
            enable_experimental_block_reclaim=maybe_bool(
                "ENABLE_EXPERIMENTAL_BLOCK_RECLAIM",
                cls.enable_experimental_block_reclaim,
            ),
            require_triton_scoring=maybe_bool(
                "REQUIRE_TRITON_SCORING",
                cls.require_triton_scoring,
            ),
            require_physical_reclaim=maybe_bool(
                "REQUIRE_PHYSICAL_RECLAIM",
                cls.require_physical_reclaim,
            ),
            logging_enabled=logging_enabled,
            log_decisions=log_decisions,
            log_execution_path=log_execution_path,
            log_execution_path_core_only=log_execution_path_core_only,
            log_core_trace=log_core_trace,
            log_selector_debug=log_selector_debug,
            fail_on_effective_len_regression=maybe_bool(
                "FAIL_ON_EFFECTIVE_LEN_REGRESSION",
                cls.fail_on_effective_len_regression,
            ),
            effective_len_regression_ratio=maybe_float(
                "EFFECTIVE_LEN_REGRESSION_RATIO",
                cls.effective_len_regression_ratio,
            ),
            effective_len_guard_divide_multiples=maybe_int(
                "EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES",
                cls.effective_len_guard_divide_multiples,
            ),
            defer_prefill_compression=maybe_bool(
                "DEFER_PREFILL_COMPRESSION",
                cls.defer_prefill_compression,
            ),
            defer_prefill_compression_on_ascend=maybe_bool(
                "DEFER_PREFILL_COMPRESSION_ON_ASCEND",
                cls.defer_prefill_compression_on_ascend,
            ),
            min_decode_tokens_before_compress=maybe_int(
                "MIN_DECODE_TOKENS_BEFORE_COMPRESS",
                cls.min_decode_tokens_before_compress,
            ),
            min_decode_tokens_before_compress_on_ascend=maybe_int(
                "MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND",
                cls.min_decode_tokens_before_compress_on_ascend,
            ),
            score_chunk_max_tokens=maybe_int(
                "SCORE_CHUNK_MAX_TOKENS",
                cls.score_chunk_max_tokens,
            ),
            score_max_layers=maybe_int(
                "SCORE_MAX_LAYERS",
                cls.score_max_layers,
            ),
            score_max_layers_on_ascend=maybe_int(
                "SCORE_MAX_LAYERS_ON_ASCEND",
                cls.score_max_layers_on_ascend,
            ),
            score_layer_stride=maybe_int(
                "SCORE_LAYER_STRIDE",
                cls.score_layer_stride,
            ),
            fast_recency_only=fast_recency_only,
            fast_recency_accuracy_guard=maybe_bool(
                "FAST_RECENCY_ACCURACY_GUARD",
                fast_recency_accuracy_guard_default,
            ),
            fast_recency_long_context_guard=maybe_bool(
                "FAST_RECENCY_LONG_CONTEXT_GUARD",
                cls.fast_recency_long_context_guard,
            ),
            fast_recency_long_context_guard_tokens=maybe_int(
                "FAST_RECENCY_LONG_CONTEXT_GUARD_TOKENS",
                cls.fast_recency_long_context_guard_tokens,
            ),
            min_reclaim_blocks=maybe_int(
                "MIN_RECLAIM_BLOCKS",
                cls.min_reclaim_blocks,
            ),
            min_reclaim_blocks_on_ascend=maybe_int(
                "MIN_RECLAIM_BLOCKS_ON_ASCEND",
                cls.min_reclaim_blocks_on_ascend,
            ),
            prefill_min_reclaim_blocks_on_ascend=maybe_int(
                "PREFILL_MIN_RECLAIM_BLOCKS_ON_ASCEND",
                cls.prefill_min_reclaim_blocks_on_ascend,
            ),
            prefill_max_compressions_on_ascend=maybe_int(
                "PREFILL_MAX_COMPRESSIONS_ON_ASCEND",
                cls.prefill_max_compressions_on_ascend,
            ),
            log_all_worker_events=log_all_worker_events,
            enable_async_compression_boundary=maybe_bool(
                "ENABLE_ASYNC_COMPRESSION_BOUNDARY",
                cls.enable_async_compression_boundary,
            ),
            enable_zero_copy_recency=maybe_bool(
                "ENABLE_ZERO_COPY_RECENCY",
                cls.enable_zero_copy_recency,
            ),
            zero_copy_recency_only_on_ascend=maybe_bool(
                "ZERO_COPY_RECENCY_ONLY_ON_ASCEND",
                cls.zero_copy_recency_only_on_ascend,
            ),
            enable_packed_pos_delta_on_ascend=maybe_bool(
                "ENABLE_PACKED_POS_DELTA_ON_ASCEND",
                cls.enable_packed_pos_delta_on_ascend,
            ),
            auto_fast_recency_on_ascend=maybe_bool(
                "AUTO_FAST_RECENCY_ON_ASCEND",
                cls.auto_fast_recency_on_ascend,
            ),
            early_install_proxy_on_ascend=maybe_bool(
                "EARLY_INSTALL_PROXY_ON_ASCEND",
                cls.early_install_proxy_on_ascend,
            ),
            preinstall_input_patch=maybe_bool(
                "PREINSTALL_INPUT_PATCH",
                cls.preinstall_input_patch,
            ),
            force_eager_multi_req_on_ascend_effective_overrides=maybe_bool(
                "FORCE_EAGER_MULTI_REQ_ON_ASCEND_EFFECTIVE_OVERRIDES",
                cls.force_eager_multi_req_on_ascend_effective_overrides,
            ),
            max_compressions_per_step_on_ascend=maybe_int(
                "MAX_COMPRESSIONS_PER_STEP_ON_ASCEND",
                cls.max_compressions_per_step_on_ascend,
            ),
            sparse_stats_path=sparse_stats_path_candidate,
            model_path=Path(model_path_raw) if model_path_raw else None,
            pruning_mode=maybe_str("PRUNING_MODE", cls.pruning_mode) or cls.pruning_mode,
            sparse_score_aggregation=(
                maybe_str("SPARSE_SCORE_AGGREGATION", cls.sparse_score_aggregation)
                or cls.sparse_score_aggregation
            ),
            sparse_normalize_scores=maybe_bool(
                "SPARSE_NORMALIZE_SCORES", cls.sparse_normalize_scores
            ),
            window_size=maybe_int("WINDOW_SIZE", cls.window_size),
            include_prefill_in_budget=maybe_bool(
                "INCLUDE_PREFILL_IN_BUDGET", cls.include_prefill_in_budget
            ),
            per_head_selection_semantics=(
                maybe_str(
                    "PER_HEAD_SELECTION_SEMANTICS",
                    cls.per_head_selection_semantics,
                )
                or cls.per_head_selection_semantics
            ),
            layer_perhead_aggregation=(
                maybe_str(
                    "LAYER_PERHEAD_AGGREGATION",
                    cls.layer_perhead_aggregation,
                )
                or cls.layer_perhead_aggregation
            ),
            per_layer_aggregation=(
                maybe_str(
                    "PER_LAYER_AGGREGATION",
                    cls.per_layer_aggregation,
                )
                or cls.per_layer_aggregation
            ),
            allow_per_layer_mode=maybe_bool(
                "ALLOW_PER_LAYER_MODE", cls.allow_per_layer_mode
            ),
            scoring_backend=(
                maybe_str("SCORING_BACKEND", cls.scoring_backend)
                or cls.scoring_backend
            ).lower(),
            disable_mlr=maybe_bool("DISABLE_MLR", cls.disable_mlr),
            disable_trig=maybe_bool("DISABLE_TRIG", cls.disable_trig),
            disable_top_n_high_freq=maybe_int(
                "DISABLE_TOP_N_HIGH_FREQ", cls.disable_top_n_high_freq
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be > 0, got {self.kv_budget}")
        if self.divide_length <= 0:
            raise ValueError(
                f"divide_length must be > 0, got {self.divide_length}"
            )
        if not (0.0 < self.kv_usage_trigger <= 1.0):
            raise ValueError(
                "kv_usage_trigger must be in (0, 1], "
                f"got {self.kv_usage_trigger}"
            )
        if not (0.0 <= self.kv_usage_release <= 1.0):
            raise ValueError(
                "kv_usage_release must be in [0, 1], "
                f"got {self.kv_usage_release}"
            )
        if self.kv_usage_release > self.kv_usage_trigger:
            raise ValueError(
                "kv_usage_release should be <= kv_usage_trigger to avoid "
                "hysteresis inversion"
            )
        if self.pruning_mode not in {"per_layer", "per_head", "per_layer_per_head"}:
            raise ValueError(
                "pruning_mode must be one of {'per_layer','per_head','per_layer_per_head'}, "
                f"got {self.pruning_mode!r}"
            )
        if self.pruning_mode == "per_layer" and not self.allow_per_layer_mode:
            raise ValueError(
                "pruning_mode='per_layer' is disabled by default in the runtime to prevent "
                "accidental use. Set allow_per_layer_mode=True "
                "(env TRIATTN_RUNTIME_ALLOW_PER_LAYER_MODE=1) for explicit opt-in."
            )
        if self.sparse_score_aggregation not in {"mean", "max"}:
            raise ValueError(
                "sparse_score_aggregation must be 'mean' or 'max', "
                f"got {self.sparse_score_aggregation!r}"
            )
        if self.per_head_selection_semantics not in {
            "legacy_layer_local",
            "hf_aligned_global_per_head",
        }:
            raise ValueError(
                "per_head_selection_semantics must be one of "
                "{'legacy_layer_local','hf_aligned_global_per_head'}, "
                f"got {self.per_head_selection_semantics!r}"
            )
        if self.layer_perhead_aggregation not in {"max", "mean"}:
            raise ValueError(
                "layer_perhead_aggregation must be 'max' or 'mean', "
                f"got {self.layer_perhead_aggregation!r}"
            )
        if self.per_layer_aggregation not in {"max", "mean", "pure_mean"}:
            raise ValueError(
                "per_layer_aggregation must be one of {'max','mean','pure_mean'}, "
                f"got {self.per_layer_aggregation!r}"
            )
        if self.scoring_backend not in {"auto", "triton", "torch", "pytorch"}:
            raise ValueError(
                "scoring_backend must be one of {'auto','triton','torch','pytorch'}, "
                f"got {self.scoring_backend!r}"
            )
        if self.window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {self.window_size}")
        if self.disable_top_n_high_freq < 0:
            raise ValueError(
                "disable_top_n_high_freq must be >= 0, "
                f"got {self.disable_top_n_high_freq}"
            )
        if self.fast_recency_long_context_guard_tokens < 0:
            raise ValueError(
                "fast_recency_long_context_guard_tokens must be >= 0, "
                f"got {self.fast_recency_long_context_guard_tokens}"
            )
        if not (0.0 < self.effective_len_regression_ratio <= 1.0):
            raise ValueError(
                "effective_len_regression_ratio must be in (0, 1], "
                f"got {self.effective_len_regression_ratio}"
            )
        if self.effective_len_guard_divide_multiples < 1:
            raise ValueError(
                "effective_len_guard_divide_multiples must be >= 1, "
                f"got {self.effective_len_guard_divide_multiples}"
            )
        if self.score_max_layers < 0:
            raise ValueError(
                "score_max_layers must be >= 0, "
                f"got {self.score_max_layers}"
            )
        if self.score_max_layers_on_ascend < 0:
            raise ValueError(
                "score_max_layers_on_ascend must be >= 0, "
                f"got {self.score_max_layers_on_ascend}"
            )
        if self.score_layer_stride < 1:
            raise ValueError(
                "score_layer_stride must be >= 1, "
                f"got {self.score_layer_stride}"
            )
        if self.min_reclaim_blocks < 0:
            raise ValueError(
                "min_reclaim_blocks must be >= 0, "
                f"got {self.min_reclaim_blocks}"
            )
        if self.min_reclaim_blocks_on_ascend < 0:
            raise ValueError(
                "min_reclaim_blocks_on_ascend must be >= 0, "
                f"got {self.min_reclaim_blocks_on_ascend}"
            )
        if self.prefill_min_reclaim_blocks_on_ascend < 0:
            raise ValueError(
                "prefill_min_reclaim_blocks_on_ascend must be >= 0, "
                f"got {self.prefill_min_reclaim_blocks_on_ascend}"
            )
        if self.prefill_max_compressions_on_ascend < 0:
            raise ValueError(
                "prefill_max_compressions_on_ascend must be >= 0, "
                f"got {self.prefill_max_compressions_on_ascend}"
            )
        if self.max_compressions_per_step_on_ascend < 0:
            raise ValueError(
                "max_compressions_per_step_on_ascend must be >= 0, "
                f"got {self.max_compressions_per_step_on_ascend}"
            )
        if self.score_chunk_max_tokens < 1:
            raise ValueError(
                "score_chunk_max_tokens must be >= 1, "
                f"got {self.score_chunk_max_tokens}"
            )
        # The previous constraint requiring require_triton_scoring=True
        # alongside enable_experimental_kv_compaction has been relaxed.
        # The PyTorch scoring path is mathematically equivalent to the
        # Triton kernel and supports compaction, so the "fallback
        # downgrade" concern is not applicable.
        if (
            self.enable_experimental_kv_compaction
            and self.require_physical_reclaim
            and not self.enable_experimental_block_reclaim
        ):
            raise ValueError(
                "enable_experimental_kv_compaction requires "
                "enable_experimental_block_reclaim=True when "
                "require_physical_reclaim=True"
            )
