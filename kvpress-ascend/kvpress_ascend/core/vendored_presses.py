"""Vendored fallback press definitions for kvpress-ascend.

These mirror the parameter semantics of the supported kvpress presses
(https://github.com/NVIDIA/kvpress) but are torch-only: they do not depend on
``transformers`` or ``kvpress`` itself, so the adapter works standalone.

Scoring itself is implemented natively in :mod:`kvpress_ascend.core.press_bridge`
against Ascend data paths (post-RoPE captured queries + dense-gathered cache K);
these classes only carry the press parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScorerPressBase:
    """Base for score-based presses (kvpress ScorerPress equivalent)."""

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"


@dataclass
class KnormPress(ScorerPressBase):
    """Key-norm based compression (kvpress KnormPress)."""


@dataclass
class RandomPress(ScorerPressBase):
    """Random compression baseline (kvpress RandomPress)."""

    seed: Optional[int] = None


@dataclass
class StreamingLLMPress(ScorerPressBase):
    """StreamingLLM sink+recent window (kvpress StreamingLLMPress)."""

    n_sink: int = 4


@dataclass
class SnapKVPress(ScorerPressBase):
    """SnapKV window-attention scoring (kvpress SnapKVPress)."""

    window_size: int = 64
    kernel_size: int = 5


@dataclass
class TOVAPress(ScorerPressBase):
    """TOVA last-token attention scoring (kvpress TOVAPress)."""


@dataclass
class ObservedAttentionPress(ScorerPressBase):
    """Observed attention scoring (kvpress ObservedAttentionPress)."""


@dataclass
class DecodingPress:
    """Decoding-interval wrapper (kvpress DecodingPress)."""

    base_press: ScorerPressBase = field(default_factory=KnormPress)
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256

    def __post_init__(self):
        assert self.compression_interval > 0, "compression_interval must be greater than 0"
        assert self.target_size > 0, "target_size must be greater than 0"
        assert isinstance(self.base_press, ScorerPressBase), "DecodingPress requires a ScorerPress"

    @property
    def compression_ratio(self) -> float:
        return self.base_press.compression_ratio


VENDORED_PRESS_REGISTRY = {
    "KnormPress": KnormPress,
    "RandomPress": RandomPress,
    "StreamingLLMPress": StreamingLLMPress,
    "SnapKVPress": SnapKVPress,
    "TOVAPress": TOVAPress,
    "ObservedAttentionPress": ObservedAttentionPress,
    "DecodingPress": DecodingPress,
}
