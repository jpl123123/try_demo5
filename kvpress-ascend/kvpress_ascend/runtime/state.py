"""Per-request runtime state for kvpress-ascend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RequestState:
    """Mirror of kvpress's per-cache state (compressed length, phase, press
    step counters) adapted to vLLM-Ascend request lifetimes."""

    req_id: str
    prefill_len: int = 0
    num_computed_tokens: int = 0
    current_cache_len: int = 0
    current_cache_len_step: int = -1
    compression_count: int = 0
    last_compression_step: int = -1
    cache_len_after_last_compression: int = 0
    nct_at_last_compression: int = 0
    is_compressed: bool = False
    press_step_counts: int = 0  # DecodingPress interval counter (per request)
    last_skipped_reason: Optional[str] = None
    protected: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class RequestStateStore:
    """Keyed by request id; mirrors tri_3_5's state store."""

    def __init__(self) -> None:
        self._states: dict[str, RequestState] = {}

    def ensure(self, req_id: str, prefill_len: int = 0, protect_prefill: bool = True) -> RequestState:
        state = self._states.get(str(req_id))
        if state is None:
            state = RequestState(
                req_id=str(req_id),
                prefill_len=int(prefill_len),
                protected=bool(protect_prefill),
            )
            self._states[str(req_id)] = state
        elif int(prefill_len) > int(state.prefill_len):
            state.prefill_len = int(prefill_len)
        return state

    def get(self, req_id: str) -> Optional[RequestState]:
        return self._states.get(str(req_id))

    def remove(self, req_id: str) -> None:
        self._states.pop(str(req_id), None)

    def has_compressed_request_in(self, req_ids: list[str]) -> bool:
        return any(
            (state := self._states.get(str(req_id))) is not None and state.is_compressed
            for req_id in req_ids
        )

    def has_active_compressed_requests(self) -> bool:
        return any(state.is_compressed for state in self._states.values())

    def mark_compressed(
        self,
        req_id: str,
        *,
        step: int,
        cache_len: int,
        scheduled_tokens: int,
        scheduler_nct: Optional[int] = None,
    ) -> None:
        state = self.ensure(req_id)
        state.compression_count += 1
        state.last_compression_step = int(step)
        state.cache_len_after_last_compression = int(cache_len)
        if scheduler_nct is not None:
            state.nct_at_last_compression = int(scheduler_nct)
        state.is_compressed = True
        state.current_cache_len = int(cache_len)
        state.current_cache_len_step = int(step)
        state.press_step_counts = 0

    def mark_compression_skipped(self, req_id: str, reason: str, step: int) -> None:
        state = self.ensure(req_id)
        state.last_skipped_reason = str(reason)

    def advance_cache_len(self, req_id: str, scheduled_tokens: int, step: int) -> int:
        state = self.ensure(req_id)
        if state.current_cache_len_step == int(step):
            return int(state.current_cache_len)
        next_len = int(state.current_cache_len) + max(1, int(scheduled_tokens))
        state.current_cache_len = next_len
        state.current_cache_len_step = int(step)
        return next_len

    def set_cache_len(self, req_id: str, cache_len: int, step: int) -> None:
        state = self.ensure(req_id)
        state.current_cache_len = int(cache_len)
        state.current_cache_len_step = int(step)

    def cleanup_finished(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            self.remove(req_id)
