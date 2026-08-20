"""Ascend-specific runtime defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import TriAttentionRuntimeConfig

_AUTO_FAST_RECENCY_MIN_RECLAIM_BLOCKS_ON_ASCEND = 8
_AUTO_ASCEND_SPARSE_MIN_RECLAIM_BLOCKS = 16
_AUTO_ASCEND_SCORE_MAX_LAYERS = 8


def apply_ascend_fast_recency_defaults(
    config: TriAttentionRuntimeConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    if not bool(getattr(config, "auto_fast_recency_on_ascend", True)):
        return
    env_map = os.environ if env is None else env
    if (
        env_map.get("TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND") is None
        and not bool(getattr(config, "fast_recency_only", False))
        and int(getattr(config, "min_reclaim_blocks_on_ascend", 0) or 0)
        < _AUTO_ASCEND_SPARSE_MIN_RECLAIM_BLOCKS
    ):
        config.min_reclaim_blocks_on_ascend = _AUTO_ASCEND_SPARSE_MIN_RECLAIM_BLOCKS

    if (
        env_map.get("TRIATTN_RUNTIME_SCORE_MAX_LAYERS") is None
        and env_map.get("TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND") is None
        and int(getattr(config, "score_max_layers_on_ascend", 0) or 0) <= 0
    ):
        config.score_max_layers_on_ascend = _AUTO_ASCEND_SCORE_MAX_LAYERS
    if env_map.get("TRIATTN_RUNTIME_SCORE_MAX_LAYERS") is None:
        ascend_score_limit = max(
            0,
            int(getattr(config, "score_max_layers_on_ascend", 0) or 0),
        )
        if (
            ascend_score_limit > 0
            and int(getattr(config, "score_max_layers", 0) or 0) <= 0
        ):
            config.score_max_layers = ascend_score_limit

    if bool(getattr(config, "fast_recency_only", False)):
        # Zero-copy recency compaction is cheap on Ascend, but very small
        # decode-time reclaim windows add scheduler/worker churn without moving
        # the model-forward bottleneck. Keep the documented Ascend default even
        # when a stale reclaim env is present; disable auto-fast-recency to keep
        # a custom interval.
        config.min_reclaim_blocks_on_ascend = (
            _AUTO_FAST_RECENCY_MIN_RECLAIM_BLOCKS_ON_ASCEND
        )
        if (
            env_map.get("TRIATTN_RUNTIME_SCORE_MAX_LAYERS") is None
            and env_map.get("TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND") is None
        ):
            if int(getattr(config, "score_max_layers_on_ascend", 0) or 0) <= 0:
                config.score_max_layers_on_ascend = _AUTO_ASCEND_SCORE_MAX_LAYERS
            if int(getattr(config, "score_max_layers", 0) or 0) <= 0:
                config.score_max_layers = max(
                    1,
                    int(config.score_max_layers_on_ascend),
                )
