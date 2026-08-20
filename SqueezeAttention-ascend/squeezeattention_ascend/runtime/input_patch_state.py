"""Effective-input override state shared between the runner proxy and the
patched ``_prepare_inputs``.

This is the Ascend equivalent of SqueezeAttention's HF ``cache_position`` re-mapping:
after compression the model must see the *compressed* history length (effective
base) instead of vLLM's stale ``num_computed_tokens``, and new tokens must be
slotted right after the kept prefix.
"""

from __future__ import annotations

from typing import Optional

# Per-batch effective overrides set by the runner proxy before the base
# execute_model call and consumed (cleared) by the patched _prepare_inputs.
ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX: dict[int, int] = {}
ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = False
ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE: Optional[int] = None
ACTIVE_EFFECTIVE_MAX_SEQ_LEN: Optional[int] = None
ACTIVE_EXPECTED_REQ_IDS: Optional[list[str]] = None
ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU: Optional[object] = None
ACTIVE_EXPECTED_QUERY_LENS_CPU: Optional[object] = None


def reset() -> None:
    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_OVERRIDES_ENABLED
    global ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE, ACTIVE_EFFECTIVE_MAX_SEQ_LEN
    global ACTIVE_EXPECTED_REQ_IDS, ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    global ACTIVE_EXPECTED_QUERY_LENS_CPU
    ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = {}
    ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = False
    ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = None
    ACTIVE_EFFECTIVE_MAX_SEQ_LEN = None
    ACTIVE_EXPECTED_REQ_IDS = None
    ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU = None
    ACTIVE_EXPECTED_QUERY_LENS_CPU = None


def set_effective_bases(base_by_req_idx: dict[int, int]) -> None:
    """Activate overrides for one step: req_idx -> effective base (compressed
    history length before this step's tokens)."""
    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_OVERRIDES_ENABLED
    ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = dict(base_by_req_idx)
    ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = bool(base_by_req_idx)


def set_single_effective_seq_base(base: Optional[int]) -> None:
    global ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE, ACTIVE_EFFECTIVE_OVERRIDES_ENABLED
    ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = base
    ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = ACTIVE_EFFECTIVE_OVERRIDES_ENABLED or base is not None


def set_active_effective_max_seq_len(value: Optional[int]) -> None:
    global ACTIVE_EFFECTIVE_MAX_SEQ_LEN
    ACTIVE_EFFECTIVE_MAX_SEQ_LEN = value


def mark_consumed() -> None:
    """Called by the patched _prepare_inputs after applying the overrides."""
    global ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX, ACTIVE_EFFECTIVE_OVERRIDES_ENABLED
    global ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE
    ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX = {}
    ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE = None
    ACTIVE_EFFECTIVE_OVERRIDES_ENABLED = False
