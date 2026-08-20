"""SqueezeAttention-ascend runtime configuration from environment variables.

All knobs are prefixed with ``SQUEEZE_`` (the tool this package adapts).
``SQUEEZE_ENABLE`` (alias ``SQUEEZE``) is the master activation switch checked
by the vLLM plugin entry point.
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


@dataclass
class SqueezeRuntimeConfig:
    """Runtime configuration for the SqueezeAttention-ascend adapter."""

    enabled: bool = False
    ini_size: float = 0.21
    class3_size: float = 0.08
    start_size: int = 4
    mode: str = "uniform"  # uniform | class_weighted
    n_clusters: int = 3
    kmeans_seed: Optional[int] = None
    kv_budget: int = 0

    # Behavior controls
    defer_prefill_compression: bool = True
    min_reclaim_blocks: int = 1
    max_compressions_per_step: int = 1
    enable_experimental_block_reclaim: bool = True
    fake_key_padding: bool = False
    early_install_proxy: bool = True
    preinstall_input_patch: bool = True
    block_size_hint: int = 0
    max_layers_to_score: int = 0  # 0 = all layers (recency selection is cheap)

    # Logging controls
    logging_enabled: bool = True
    probe_enabled: bool = True
    log_budgets: bool = True
    log_decisions: bool = False

    build_id: str = "squeezeattention-ascend-v1-20260820"
    runtime_step: int = field(default=0, repr=False)

    @staticmethod
    def from_env() -> "SqueezeRuntimeConfig":
        enabled = env_bool("SQUEEZE_ENABLE", False) or env_bool("SQUEEZE", False)
        mode = env_str("SQUEEZE_MODE", "uniform").strip().lower()
        if mode not in {"uniform", "class_weighted"}:
            raise ValueError(
                f"SQUEEZE_MODE={mode} unsupported; choose 'uniform' or 'class_weighted'"
            )
        logging_enabled = env_bool("SQUEEZE_RUNTIME_LOGGING", True)
        return SqueezeRuntimeConfig(
            enabled=enabled,
            ini_size=env_float("SQUEEZE_INI_SIZE", 0.21),
            class3_size=env_float("SQUEEZE_CLASS3_SIZE", 0.08),
            start_size=env_int("SQUEEZE_START_SIZE", 4),
            mode=mode,
            n_clusters=max(2, env_int("SQUEEZE_KMEANS_CLUSTERS", 3)),
            kmeans_seed=env_int("SQUEEZE_KMEANS_SEED", 0) or None,
            kv_budget=env_int("SQUEEZE_KV_BUDGET", 0),
            defer_prefill_compression=env_bool("SQUEEZE_DEFER_PREFILL_COMPRESSION", True),
            min_reclaim_blocks=env_int("SQUEEZE_MIN_RECLAIM_BLOCKS", 1),
            max_compressions_per_step=env_int("SQUEEZE_MAX_COMPRESSIONS_PER_STEP", 1),
            enable_experimental_block_reclaim=env_bool(
                "SQUEEZE_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM", True
            ),
            fake_key_padding=env_bool("SQUEEZE_FAKE_KEY_PADDING", False),
            early_install_proxy=env_bool("SQUEEZE_EARLY_INSTALL_PROXY", True),
            preinstall_input_patch=env_bool("SQUEEZE_PREINSTALL_INPUT_PATCH", True),
            block_size_hint=env_int("SQUEEZE_BLOCK_SIZE_HINT", 0),
            max_layers_to_score=env_int("SQUEEZE_MAX_LAYERS_TO_SCORE", 0),
            logging_enabled=logging_enabled,
            probe_enabled=env_bool("SQUEEZE_PROBE", True) and logging_enabled,
            log_budgets=env_bool("SQUEEZE_LOG_BUDGETS", True),
            log_decisions=env_bool("SQUEEZE_LOG_DECISIONS", False),
        )

    def total_budget_fraction(self, num_layers: int) -> float:
        """Total KV budget fraction over all layers (the SqueezeAttention
        conservation invariant: ``num_layers * ini_size``)."""
        return max(0.0, float(num_layers) * self.ini_size)

    def resolved_k(self, budgets: list[int], block_size: int, total_tokens: int) -> int:
        """Uniform keep count K for the shared block row (``uniform`` mode).

        Uses the maximum per-layer budget, block-aligned, capped by the current
        token count. This is the largest budget any layer requires, which is
        the only count physically expressible with a shared block table and a
        uniform per-request seq_len.
        """
        if self.kv_budget > 0:
            k = self.kv_budget
        elif budgets:
            k = max(1, max(int(b) for b in budgets))
        else:
            k = max(1, int(total_tokens * self.ini_size))
        return min(int(total_tokens), k)
