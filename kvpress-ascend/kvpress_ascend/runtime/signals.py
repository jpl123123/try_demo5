"""Compression signal passed from the patched Scheduler to the worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompressionSignal:
    """Scheduler-side request that a KV compression boundary was reached.

    kvpress's HF mechanism compresses at prefill/decode boundaries tracked by
    ``cache_position``; this signal is its vLLM-Ascend equivalent, computed by
    the patched ``Scheduler.schedule`` and consumed by the runner proxy.
    """

    req_id: str
    should_compress: bool = False
    reason: str = "length_threshold"
    step: int = 0
    estimated_cache_len: int = 0
    prefill_len: int = 0
    scheduled_tokens: int = 1
    kv_usage: Optional[float] = None
    protect_prefill: bool = True
    force: bool = False
    is_prefill_step: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "should_compress": self.should_compress,
            "reason": self.reason,
            "step": self.step,
            "estimated_cache_len": self.estimated_cache_len,
            "prefill_len": self.prefill_len,
            "scheduled_tokens": self.scheduled_tokens,
            "kv_usage": self.kv_usage,
            "protect_prefill": self.protect_prefill,
            "force": self.force,
            "is_prefill_step": self.is_prefill_step,
        }


def signals_from_scheduler_output(scheduler_output: Any) -> dict[str, CompressionSignal]:
    raw = getattr(scheduler_output, "kvpress_signals", None)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, CompressionSignal] = {}
    for req_id, sig in raw.items():
        if isinstance(sig, CompressionSignal):
            out[str(req_id)] = sig
        elif isinstance(sig, dict):
            out[str(req_id)] = CompressionSignal(**sig)
    return out
