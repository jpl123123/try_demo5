"""TriAttention vLLM runtime — scheduler, worker, and hook extensions for vLLM."""

from .config import TriAttentionRuntimeConfig
from .effective_len_tracker import EffectiveCacheLenTracker
from .executor import CompressionExecutionResult, CompressionExecutor
from .plan_models import KeepPlan, PlacementPlan, ReclaimEvent, ReclaimGroup
from .planner import CompressionPlanner
from .signals import CompressionSignal
from .state import RequestCompressionState, RequestStateStore

# Public alias for the current/default runtime config name.
TriAttentionConfig = TriAttentionRuntimeConfig

__all__ = [
    "TriAttentionRuntimeConfig",
    "TriAttentionConfig",
    "EffectiveCacheLenTracker",
    "CompressionExecutionResult",
    "CompressionExecutor",
    "CompressionPlanner",
    "CompressionSignal",
    "KeepPlan",
    "PlacementPlan",
    "ReclaimEvent",
    "ReclaimGroup",
    "RequestCompressionState",
    "RequestStateStore",
    "install_runner_compression_hook",
]


def __getattr__(name: str):
    if name == "install_runner_compression_hook":
        from .hook_impl import install_runner_compression_hook

        return install_runner_compression_hook
    raise AttributeError(name)
