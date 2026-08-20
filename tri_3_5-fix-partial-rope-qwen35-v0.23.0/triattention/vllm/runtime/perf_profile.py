"""Lightweight aggregated perf profiling for TriAttention runtime (env gated)."""

from __future__ import annotations

import os
from pathlib import Path
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .logging_control import runtime_profile_logging_enabled


def _env_enabled(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


@dataclass
class _E2EPhaseStats:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.calls)


@dataclass
class TriAttentionPerfProfile:
    """Aggregates per-step timing counters with sparse logging."""

    logger: Any
    enabled: bool = False
    log_every_steps: int = 200
    e2e_enabled: bool = False
    e2e_log_every_steps: int = 200
    total_steps: int = 0
    steps_with_overrides: int = 0
    steps_with_trigger: int = 0
    compress_calls: int = 0
    compress_applied: int = 0
    compress_skipped: int = 0
    compress_errors: int = 0
    reclaimed_blocks: int = 0
    t_state_ms: float = 0.0
    t_compress_ms: float = 0.0
    t_reclaim_ms: float = 0.0
    t_override_prep_ms: float = 0.0
    t_base_exec_ms: float = 0.0
    t_total_exec_ms: float = 0.0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    apply_reasons: Counter[str] = field(default_factory=Counter)
    selector_statuses: Counter[str] = field(default_factory=Counter)
    reclaim_modes: Counter[str] = field(default_factory=Counter)
    cudagraph_modes: Counter[str] = field(default_factory=Counter)
    e2e_steps: int = 0
    e2e_samples: int = 0
    e2e_phase_stats: dict[str, _E2EPhaseStats] = field(default_factory=dict)
    e2e_last_step: dict[str, float] = field(default_factory=dict)
    e2e_last_sample: dict[str, float] = field(default_factory=dict)
    sink_dir: str | None = None

    @classmethod
    def from_env(cls, logger: Any) -> "TriAttentionPerfProfile":
        sink_dir = os.environ.get("TRIATTN_RUNTIME_PERF_SINK_DIR")
        return cls(
            logger=logger,
            enabled=runtime_profile_logging_enabled(
                "TRIATTN_RUNTIME_PERF_PROFILE",
                "0",
            ),
            log_every_steps=max(1, _env_int("TRIATTN_RUNTIME_PERF_LOG_EVERY", 200)),
            e2e_enabled=runtime_profile_logging_enabled(
                "TRIATTN_RUNTIME_E2E_PROFILE",
                "0",
            ),
            e2e_log_every_steps=max(
                1,
                _env_int(
                    "TRIATTN_RUNTIME_E2E_LOG_EVERY",
                    max(1, _env_int("TRIATTN_RUNTIME_PERF_LOG_EVERY", 200)),
                ),
            ),
            sink_dir=sink_dir,
        )

    def timer(self) -> "_Timer":
        return _Timer()

    def record_step(
        self,
        *,
        has_trigger: bool,
        uses_overrides: bool,
        t_state_ms: float,
        t_compress_ms: float,
        t_reclaim_ms: float,
        t_override_prep_ms: float,
        t_base_exec_ms: float,
        t_total_exec_ms: float,
    ) -> None:
        if not self.enabled:
            return
        self.total_steps += 1
        if has_trigger:
            self.steps_with_trigger += 1
        if uses_overrides:
            self.steps_with_overrides += 1
        self.t_state_ms += t_state_ms
        self.t_compress_ms += t_compress_ms
        self.t_reclaim_ms += t_reclaim_ms
        self.t_override_prep_ms += t_override_prep_ms
        self.t_base_exec_ms += t_base_exec_ms
        self.t_total_exec_ms += t_total_exec_ms
        if self.total_steps % self.log_every_steps == 0:
            self._log_summary()

    def record_compression_events(self, events: list[dict[str, Any]] | None) -> None:
        if not self.enabled or not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            self.compress_calls += 1
            status = str(event.get("status", ""))
            details = event.get("details")
            if not isinstance(details, dict):
                details = {}
            if status == "applied":
                self.compress_applied += 1
                reason = event.get("reason")
                if isinstance(reason, str):
                    self.apply_reasons[reason] += 1
                selector_status = details.get("selector_status")
                if isinstance(selector_status, str):
                    self.selector_statuses[selector_status] += 1
                block_reclaim = event.get("block_reclaim")
                if not isinstance(block_reclaim, dict):
                    block_reclaim = details.get("block_reclaim")
                reclaim_mode = (
                    block_reclaim.get("mode")
                    if isinstance(block_reclaim, dict)
                    else None
                )
                if isinstance(reclaim_mode, str):
                    self.reclaim_modes[reclaim_mode] += 1
                reclaimed = details.get("reclaimed_block_count")
                if isinstance(reclaimed, int):
                    self.reclaimed_blocks += max(0, reclaimed)
            elif status == "skipped":
                self.compress_skipped += 1
                reason = event.get("reason")
                if isinstance(reason, str):
                    self.skip_reasons[reason] += 1
            elif status == "error":
                self.compress_errors += 1

    def _log_summary(self) -> None:
        steps = max(1, self.total_steps)
        top_skips = ",".join(
            f"{reason}:{count}" for reason, count in self.skip_reasons.most_common(5)
        )
        top_applied = ",".join(
            f"{reason}:{count}" for reason, count in self.apply_reasons.most_common(5)
        )
        top_selectors = ",".join(
            f"{reason}:{count}"
            for reason, count in self.selector_statuses.most_common(5)
        )
        line = (
            "TRIATTN_PERF "
            f"steps={self.total_steps} trig_steps={self.steps_with_trigger} "
            f"override_steps={self.steps_with_overrides} "
            f"compress_calls={self.compress_calls} applied={self.compress_applied} "
            f"skipped={self.compress_skipped} errors={self.compress_errors} "
            f"reclaimed_blocks={self.reclaimed_blocks} "
            f"avg_ms(total={self.t_total_exec_ms / steps:.2f} "
            f"state={self.t_state_ms / steps:.2f} "
            f"compress={self.t_compress_ms / steps:.2f} "
            f"reclaim={self.t_reclaim_ms / steps:.2f} "
            f"override_prep={self.t_override_prep_ms / steps:.2f} "
            f"base_exec={self.t_base_exec_ms / steps:.2f}) "
            f"top_apply_reasons={top_applied or 'none'} "
            f"top_skip_reasons={top_skips or 'none'} "
            f"selector_statuses={top_selectors or 'none'} "
            f"reclaim_modes={dict(self.reclaim_modes)} "
            f"cudagraph_modes={dict(self.cudagraph_modes)}"
        )
        self.logger.info("%s", line)
        if self.sink_dir:
            try:
                sink_dir = Path(self.sink_dir)
                sink_dir.mkdir(parents=True, exist_ok=True)
                sink_path = sink_dir / f"triattn_perf_{os.getpid()}.log"
                with sink_path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
            except Exception:
                pass

    def record_model_output(self, output: Any) -> None:
        if not self.enabled or output is None:
            return
        try:
            stats = getattr(output, "cudagraph_stats", None)
        except Exception:
            return
        if stats is None:
            return
        runtime_mode = None
        try:
            runtime_mode = getattr(stats, "runtime_mode", None)
        except Exception:
            runtime_mode = None
        if runtime_mode is not None:
            self.cudagraph_modes[str(runtime_mode)] += 1

    def record_e2e_step(
        self,
        phases_ms: dict[str, float] | None,
        *,
        num_reqs: int | None = None,
        total_tokens: int | None = None,
        has_trigger: bool = False,
        uses_overrides: bool = False,
        pending_events: int = 0,
    ) -> None:
        if not self.e2e_enabled or not isinstance(phases_ms, dict):
            return
        self.e2e_steps += 1
        self.e2e_last_step = {
            name: float(elapsed)
            for name, elapsed in phases_ms.items()
            if isinstance(elapsed, (int, float))
        }
        for name, elapsed in self.e2e_last_step.items():
            self.e2e_phase_stats.setdefault(name, _E2EPhaseStats()).record(elapsed)
        if self.e2e_steps % self.e2e_log_every_steps == 0:
            self._log_e2e_summary(
                num_reqs=num_reqs,
                total_tokens=total_tokens,
                has_trigger=has_trigger,
                uses_overrides=uses_overrides,
                pending_events=pending_events,
            )

    def record_e2e_sample(
        self,
        phases_ms: dict[str, float] | None,
        *,
        pending_events: int = 0,
    ) -> None:
        if not self.e2e_enabled or not isinstance(phases_ms, dict):
            return
        self.e2e_samples += 1
        self.e2e_last_sample = {
            name: float(elapsed)
            for name, elapsed in phases_ms.items()
            if isinstance(elapsed, (int, float))
        }
        for name, elapsed in self.e2e_last_sample.items():
            self.e2e_phase_stats.setdefault(name, _E2EPhaseStats()).record(elapsed)
        if self.e2e_steps > 0 and self.e2e_steps % self.e2e_log_every_steps == 0:
            self._log_e2e_summary(pending_events=pending_events)

    def _format_e2e_phase(self, item: tuple[str, _E2EPhaseStats]) -> str:
        name, stats = item
        return (
            f"{name}:calls={stats.calls},avg={stats.avg_ms:.2f},"
            f"max={stats.max_ms:.2f},total={stats.total_ms:.2f}"
        )

    def _format_last_e2e(self, phases: dict[str, float]) -> str:
        if not phases:
            return "none"
        top = sorted(phases.items(), key=lambda item: item[1], reverse=True)[:8]
        return "|".join(f"{name}={elapsed:.2f}" for name, elapsed in top)

    def _log_e2e_summary(
        self,
        *,
        num_reqs: int | None = None,
        total_tokens: int | None = None,
        has_trigger: bool | None = None,
        uses_overrides: bool | None = None,
        pending_events: int | None = None,
    ) -> None:
        if not self.e2e_phase_stats:
            return
        top_total = sorted(
            self.e2e_phase_stats.items(),
            key=lambda item: item[1].total_ms,
            reverse=True,
        )[:8]
        top_avg = sorted(
            self.e2e_phase_stats.items(),
            key=lambda item: item[1].avg_ms,
            reverse=True,
        )[:8]
        line = (
            "TRIATTN_E2E_PERF "
            f"steps={self.e2e_steps} samples={self.e2e_samples} "
            f"num_reqs={num_reqs if num_reqs is not None else 'na'} "
            f"total_tokens={total_tokens if total_tokens is not None else 'na'} "
            f"has_trigger={int(bool(has_trigger)) if has_trigger is not None else 'na'} "
            f"uses_overrides={int(bool(uses_overrides)) if uses_overrides is not None else 'na'} "
            f"pending_events={pending_events if pending_events is not None else 'na'} "
            f"top_total={'|'.join(self._format_e2e_phase(item) for item in top_total)} "
            f"top_avg={'|'.join(self._format_e2e_phase(item) for item in top_avg)} "
            f"last_step={self._format_last_e2e(self.e2e_last_step)} "
            f"last_sample={self._format_last_e2e(self.e2e_last_sample)}"
        )
        self.logger.info("%s", line)
        if self.sink_dir:
            try:
                sink_dir = Path(self.sink_dir)
                sink_dir.mkdir(parents=True, exist_ok=True)
                sink_path = sink_dir / f"triattn_e2e_perf_{os.getpid()}.log"
                with sink_path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
            except Exception:
                pass


class _Timer:
    __slots__ = ("_t0",)

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
