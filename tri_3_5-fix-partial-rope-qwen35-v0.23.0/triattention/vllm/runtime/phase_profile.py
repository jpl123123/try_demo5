"""Host-side phase timing for TriAttention runtime integration.

This profiler intentionally uses host wall-clock timing only. It does not
insert device synchronizations, so it can expose Python-side blocking points
without materially changing Ascend execution behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
import os
from pathlib import Path
import time
from typing import Any, Callable

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


def _runtime_logger() -> Any:
    try:
        from vllm.logger import logger

        return logger
    except Exception:
        import logging

        return logging.getLogger(__name__)


def _fmt_details(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    parts: list[str] = []
    for key in sorted(details):
        value = details[key]
        if value is None:
            continue
        if isinstance(value, float):
            value_s = f"{value:.2f}"
        else:
            value_s = str(value)
        value_s = value_s.replace(" ", "_").replace(",", ";")
        parts.append(f"{key}={value_s}")
    return ",".join(parts)


@dataclass
class _PhaseStats:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    last_details: str = ""
    detail_values: Counter[str] = field(default_factory=Counter)

    def record(self, elapsed_ms: float, details: dict[str, Any] | None) -> None:
        self.calls += 1
        elapsed = max(0.0, float(elapsed_ms))
        self.total_ms += elapsed
        self.max_ms = max(self.max_ms, elapsed)
        self.last_ms = elapsed
        detail_s = _fmt_details(details)
        if detail_s:
            self.last_details = detail_s
            for item in detail_s.split(","):
                if item:
                    self.detail_values[item] += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.calls)


@dataclass
class TriAttentionPhaseProfile:
    logger: Any
    enabled: bool = False
    log_every_calls: int = 200
    top_n: int = 8
    sink_dir: str | None = None
    total_calls: int = 0
    phases: dict[str, _PhaseStats] = field(default_factory=dict)
    model_probe_labels: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "TriAttentionPhaseProfile":
        perf_log_every = _env_int("TRIATTN_RUNTIME_PERF_LOG_EVERY", 200)
        default_phase_every = max(1, perf_log_every * 10)
        return cls(
            logger=_runtime_logger(),
            enabled=runtime_profile_logging_enabled(
                "TRIATTN_RUNTIME_PHASE_PROFILE",
                "0",
            ),
            log_every_calls=max(
                1,
                _env_int("TRIATTN_RUNTIME_PHASE_LOG_EVERY", default_phase_every),
            ),
            top_n=max(1, _env_int("TRIATTN_RUNTIME_PHASE_TOP_N", 12)),
            sink_dir=os.environ.get("TRIATTN_RUNTIME_PERF_SINK_DIR"),
        )

    def record_phase(
        self,
        phase: str,
        elapsed_ms: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.total_calls += 1
        stats = self.phases.setdefault(phase, _PhaseStats())
        stats.record(elapsed_ms, details)
        if self.total_calls % self.log_every_calls == 0:
            self._log_summary()

    def register_model_probes(self, labels: list[str] | tuple[str, ...]) -> None:
        if not self.enabled:
            return
        self.model_probe_labels = tuple(labels)

    def _format_phase(self, item: tuple[str, _PhaseStats]) -> str:
        name, stats = item
        return (
            f"{name}:calls={stats.calls},avg={stats.avg_ms:.2f},"
            f"max={stats.max_ms:.2f},total={stats.total_ms:.1f}"
        )

    @staticmethod
    def _is_model_submodule_phase(name: str) -> bool:
        return name.startswith(
            (
                "model_layer_",
                "model_self_attn_",
                "model_mlp_",
            )
        )

    def _top_phases(
        self,
        phases: list[tuple[str, _PhaseStats]],
        metric: str,
    ) -> list[tuple[str, _PhaseStats]]:
        if metric == "avg":
            key = lambda item: item[1].avg_ms
        else:
            key = lambda item: item[1].total_ms
        return sorted(phases, key=key, reverse=True)[: self.top_n]

    def _log_summary(self) -> None:
        if not self.phases:
            return
        all_phases = list(self.phases.items())
        top_total = self._top_phases(all_phases, "total")
        top_avg = self._top_phases(all_phases, "avg")
        model_phases = [
            item for item in all_phases if self._is_model_submodule_phase(item[0])
        ]
        model_top_total = self._top_phases(model_phases, "total")
        model_top_avg = self._top_phases(model_phases, "avg")
        detail_items = [
            (
                name,
                stats.last_ms,
                stats.last_details,
            )
            for name, stats in self.phases.items()
            if stats.last_details
        ]
        detail_items.sort(key=lambda item: item[1], reverse=True)
        details = ";".join(
            f"{name}:last={last_ms:.2f},{detail_s}"
            for name, last_ms, detail_s in detail_items[:8]
        )
        line = (
            "TRIATTN_PHASE_PERF "
            f"calls={self.total_calls} "
            f"top_total={'|'.join(self._format_phase(item) for item in top_total)} "
            f"top_avg={'|'.join(self._format_phase(item) for item in top_avg)}"
        )
        if self.model_probe_labels:
            status = "active" if model_phases else "installed_no_records"
            line = (
                f"{line} model_probe_status={status},"
                f"installed={len(self.model_probe_labels)},"
                f"recorded={len(model_phases)}"
            )
        if model_top_total:
            line = (
                f"{line} model_top_total="
                f"{'|'.join(self._format_phase(item) for item in model_top_total)}"
            )
        if model_top_avg:
            line = (
                f"{line} model_top_avg="
                f"{'|'.join(self._format_phase(item) for item in model_top_avg)}"
            )
        if details:
            line = f"{line} details={details}"
        self.logger.info("%s", line)
        if self.sink_dir:
            try:
                sink_dir = Path(self.sink_dir)
                sink_dir.mkdir(parents=True, exist_ok=True)
                sink_path = sink_dir / f"triattn_phase_perf_{os.getpid()}.log"
                with sink_path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
            except Exception:
                pass


_PROFILE: TriAttentionPhaseProfile | None = None


def get_phase_profile() -> TriAttentionPhaseProfile:
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = TriAttentionPhaseProfile.from_env()
    return _PROFILE


def reset_phase_profile_for_tests() -> None:
    global _PROFILE
    _PROFILE = None


def phase_profile_enabled() -> bool:
    return bool(get_phase_profile().enabled)


def phase_now() -> float:
    return time.perf_counter()


def phase_elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def record_phase(
    phase: str,
    elapsed_ms: float,
    details: dict[str, Any] | None = None,
) -> None:
    get_phase_profile().record_phase(phase, elapsed_ms, details)


def register_model_probes(labels: list[str] | tuple[str, ...]) -> None:
    get_phase_profile().register_model_probes(labels)


def _is_phase_timed(func: Any) -> bool:
    return bool(getattr(func, "_triattention_phase_timed", False))


def make_timed_wrapper(
    phase: str,
    original: Callable[..., Any],
    details_fn: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any] | None]
    | None = None,
) -> Callable[..., Any]:
    """Wrap a callable with env-gated host-side phase timing."""

    if _is_phase_timed(original):
        return original

    @wraps(original)
    def _wrapped(*args, **kwargs):
        if not phase_profile_enabled():
            return original(*args, **kwargs)
        t0 = phase_now()
        result: Any = None
        failed = False
        try:
            result = original(*args, **kwargs)
            return result
        except Exception:
            failed = True
            raise
        finally:
            details: dict[str, Any] | None = None
            if details_fn is not None:
                try:
                    details = details_fn(args, kwargs, result)
                except Exception:
                    details = None
            if failed:
                details = dict(details or {})
                details["status"] = "error"
            record_phase(phase, phase_elapsed_ms(t0), details)

    setattr(_wrapped, "_triattention_phase_timed", True)
    setattr(_wrapped, "_triattention_phase_original", original)
    return _wrapped
