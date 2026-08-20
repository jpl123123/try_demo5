"""kvpress-ascend runtime configuration from environment variables.

All knobs are prefixed with ``KVPRESS_`` (the tool this package adapts).
``KVPRESS_ENABLE`` (alias ``KVPRESS``) is the master activation switch that the
vLLM plugin entry point checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


# ---------------------------------------------------------------------------
# Supported presses (Ascend-native scoring bridge). Mask-based presses that
# require per-head *different counts* cannot be physically expressed in the
# Ascend block cache / attention kernels and are rejected at startup.
# ---------------------------------------------------------------------------
SUPPORTED_PRESSES = frozenset(
    {
        "KnormPress",
        "StreamingLLMPress",
        "RandomPress",
        "SnapKVPress",
        "TOVAPress",
        "ObservedAttentionPress",
        "DecodingPress",
    }
)

MASK_BASED_PRESSES = frozenset(
    {
        "AdaKVPress",
        "CriticalAdaKVPress",
        "CriticalKVPress",
        "PyramidKVPress",
        "DuoAttentionPress",
        "CURPress",
        "CapPress",
    }
)


@dataclass
class KVPressRuntimeConfig:
    """Runtime configuration for the kvpress-ascend adapter."""

    enabled: bool = False
    press_name: str = "KnormPress"
    compression_ratio: float = 0.5
    target_size: int = 0
    window_size: int = 64
    n_sink: int = 4
    compression_interval: int = 512
    seed: Optional[int] = None
    kv_budget: int = 0

    # Press source
    use_installed_kvpress: bool = True

    # Behavior controls
    defer_prefill_compression: bool = True
    min_reclaim_blocks: int = 1
    max_compressions_per_step: int = 1
    protect_prefill: bool = True
    include_prefill_in_budget: bool = True
    enable_experimental_block_reclaim: bool = True
    enable_async_boundary_sync: bool = False
    early_install_proxy: bool = True
    preinstall_input_patch: bool = True
    block_size_hint: int = 0
    max_layers_to_score: int = 0  # 0 = score all layers

    # Logging controls
    logging_enabled: bool = True
    probe_enabled: bool = True
    log_attention_hook: bool = False
    log_decisions: bool = False
    log_selector_debug: bool = False

    # Diagnostics / build info
    build_id: str = "kvpress-ascend-v1-20260820"
    runtime_step: int = field(default=0, repr=False)

    @staticmethod
    def from_env() -> "KVPressRuntimeConfig":
        enabled = env_bool("KVPRESS_ENABLE", False) or env_bool("KVPRESS", False)
        press_name = env_str("KVPRESS_PRESS", "KnormPress")
        if press_name in MASK_BASED_PRESSES:
            raise ValueError(
                f"KVPRESS_PRESS={press_name} is a mask-based press (head-wise "
                "different counts). Ascend attention kernels cannot express per-head "
                "masking; choose one of: " + ", ".join(sorted(SUPPORTED_PRESSES))
            )
        if press_name not in SUPPORTED_PRESSES:
            raise ValueError(
                f"KVPRESS_PRESS={press_name} unsupported; choose one of: "
                + ", ".join(sorted(SUPPORTED_PRESSES))
            )
        logging_enabled = env_bool("KVPRESS_RUNTIME_LOGGING", True)
        return KVPressRuntimeConfig(
            enabled=enabled,
            press_name=press_name,
            compression_ratio=env_float("KVPRESS_COMPRESSION_RATIO", 0.5),
            target_size=env_int("KVPRESS_TARGET_SIZE", 0),
            window_size=env_int("KVPRESS_WINDOW_SIZE", 64),
            n_sink=env_int("KVPRESS_SINK_TOKENS", 4),
            compression_interval=env_int("KVPRESS_COMPRESSION_INTERVAL", 512),
            seed=env_int("KVPRESS_SEED", 0) or None,
            kv_budget=env_int("KVPRESS_KV_BUDGET", 0),
            use_installed_kvpress=env_bool("KVPRESS_USE_INSTALLED", True),
            defer_prefill_compression=env_bool("KVPRESS_DEFER_PREFILL_COMPRESSION", True),
            min_reclaim_blocks=env_int("KVPRESS_MIN_RECLAIM_BLOCKS", 1),
            max_compressions_per_step=env_int("KVPRESS_MAX_COMPRESSIONS_PER_STEP", 1),
            protect_prefill=env_bool("KVPRESS_PROTECT_PREFILL", True),
            include_prefill_in_budget=env_bool("KVPRESS_INCLUDE_PREFILL_IN_BUDGET", True),
            enable_experimental_block_reclaim=env_bool(
                "KVPRESS_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM", True
            ),
            enable_async_boundary_sync=env_bool("KVPRESS_ENABLE_ASYNC_BOUNDARY_SYNC", False),
            early_install_proxy=env_bool("KVPRESS_EARLY_INSTALL_PROXY", True),
            preinstall_input_patch=env_bool("KVPRESS_PREINSTALL_INPUT_PATCH", True),
            block_size_hint=env_int("KVPRESS_BLOCK_SIZE_HINT", 0),
            max_layers_to_score=env_int("KVPRESS_MAX_LAYERS_TO_SCORE", 0),
            logging_enabled=logging_enabled,
            probe_enabled=env_bool("KVPRESS_PROBE", True) and logging_enabled,
            log_attention_hook=env_bool("KVPRESS_LOG_ATTENTION_HOOK", False),
            log_decisions=env_bool("KVPRESS_LOG_DECISIONS", False),
            log_selector_debug=env_bool("KVPRESS_LOG_SELECTOR_DEBUG", False),
        )

    def resolved_budget(self, seq_len: int) -> int:
        """Per-request KV budget in tokens.

        ``KVPRESS_KV_BUDGET`` (absolute) wins when set; otherwise the budget is
        derived from the press compression ratio (DecodingPress target_size wins
        over ratio when both are set).
        """
        if self.kv_budget > 0:
            return self.kv_budget
        if self.press_name == "DecodingPress" and self.target_size > 0:
            return self.target_size
        n_kept = int(seq_len * (1.0 - self.compression_ratio))
        if self.target_size > 0:
            n_kept = min(n_kept, self.target_size)
        return max(1, n_kept)
