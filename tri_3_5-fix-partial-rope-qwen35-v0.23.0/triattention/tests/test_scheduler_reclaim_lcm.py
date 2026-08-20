"""Tests for TriAttention scheduler reclaim with multi-group LCM block sizes.

On vLLM versions where the scheduler's ``self.block_size`` is the LCM of all
KV-cache group block sizes (multi-group hybrid/MTP models such as
Qwen3.5-MTP on vLLM-Ascend v0.23.0), the gid-agnostic ``required_blocks``
computed from ``self.block_size`` can be far smaller than the real per-group
physical block count. The reclaim accounting must use each group's own block
size (from ``kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size``)
so the scheduler matches what the worker actually freed.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np


class _Logger:
    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def _ensure_vllm_stubs():
    """Stub out the vllm modules that scheduler.py imports at module load."""
    if "vllm" not in sys.modules:
        sys.modules["vllm"] = types.SimpleNamespace()
    vllm = sys.modules["vllm"]
    # vllm.config.VllmConfig
    if not hasattr(vllm, "config"):
        sys.modules["vllm.config"] = types.SimpleNamespace(VllmConfig=object)
    vllm.config = sys.modules.get("vllm.config") or types.SimpleNamespace(VllmConfig=object)
    if "vllm.logger" not in sys.modules:
        sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())
    # vllm.multimodal
    if "vllm.multimodal" not in sys.modules:
        sys.modules["vllm.multimodal"] = types.SimpleNamespace(
            MULTIMODAL_REGISTRY=object,
            MultiModalRegistry=object,
        )
    # vllm.v1.core.sched.output
    if "vllm.v1.core.sched.output" not in sys.modules:
        sys.modules["vllm.v1.core.sched.output"] = types.SimpleNamespace(
            SchedulerOutput=object
        )
    # vllm.v1.core.sched.scheduler
    if "vllm.v1.core.sched.scheduler" not in sys.modules:
        sys.modules["vllm.v1.core.sched.scheduler"] = types.SimpleNamespace(
            Scheduler=object
        )
    # vllm.v1.kv_cache_interface
    if "vllm.v1.kv_cache_interface" not in sys.modules:
        sys.modules["vllm.v1.kv_cache_interface"] = types.SimpleNamespace(
            KVCacheConfig=object
        )
    # vllm.v1.outputs
    if "vllm.v1.outputs" not in sys.modules:
        sys.modules["vllm.v1.outputs"] = types.SimpleNamespace(
            ModelRunnerOutput=object
        )
    # vllm.v1.structured_output
    if "vllm.v1.structured_output" not in sys.modules:
        sys.modules["vllm.v1.structured_output"] = types.SimpleNamespace(
            StructuredOutputManager=object
        )


_ensure_vllm_stubs()

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig  # noqa: E402
from triattention.vllm.runtime.scheduler import TriAttentionScheduler  # noqa: E402


class _Block:
    def __init__(self, block_id):
        self.block_id = block_id


class _Manager:
    def __init__(self, blocks):
        self.req_to_blocks = blocks
        self.num_cached_block = {}


class _Spec:
    def __init__(self, block_size):
        self.block_size = block_size


class _Group:
    def __init__(self, block_size):
        self.kv_cache_spec = _Spec(block_size)


class _KVCacheConfig:
    def __init__(self, group_block_sizes):
        self.kv_cache_groups = [_Group(bs) for bs in group_block_sizes]


class _Coordinator:
    def __init__(self, managers):
        self.single_type_managers = managers


class _KVCacheManager:
    def __init__(self, managers):
        self.coordinator = _Coordinator(managers)


class _Request:
    def __init__(self):
        self.num_computed_tokens = 0


def _make_scheduler(*, scheduler_block_size, group_block_sizes, managers):
    """Build a minimal object exposing the fields _apply_compression_events
    reads, without instantiating the real upstream Scheduler."""
    sched = TriAttentionScheduler.__new__(TriAttentionScheduler)
    sched.triattention_config = TriAttentionRuntimeConfig.from_env()
    sched.triattention_config.enable_experimental_block_reclaim = True
    sched.triattention_config.require_physical_reclaim = True
    sched.triattention_config.logging_enabled = False
    sched.triattention_config.log_decisions = False
    sched.block_size = scheduler_block_size
    sched.kv_cache_config = _KVCacheConfig(group_block_sizes)
    sched.kv_cache_manager = _KVCacheManager(managers)
    sched.requests = {}
    sched._prefill_lens = {}
    sched._prefill_compression_counts = {}
    sched._length_threshold_cache = {}
    sched._last_signal_log_steps = {}
    sched._long_context_guard_logged = set()
    sched._effective_len_tracker = type(
        "_E", (), {"apply_compression": lambda *a, **k: None}
    )()
    sched._triattention_step = 0
    return sched


def test_reclaim_passes_when_scheduler_block_size_is_lcm_of_groups():
    """Multi-group model: scheduler block_size is LCM, but per-group physical
    block size matches the worker's view, so kept_len == required_blocks(gid)."""
    # Two groups with block sizes 128 and 2048 -> LCM = 2048.
    # retained_cache_len=14464 -> ceil(14464/128)=113 blocks for gid=0,
    # ceil(14464/2048)=8 blocks for gid=1.
    group_block_sizes = [128, 2048]
    scheduler_block_size = 2048  # LCM
    # Worker produced 113 block ids for gid=0 (ceil(14464/128)=113).
    kept_ids = list(range(113))
    # Original request had 310 blocks (ceil(39642/128)) for gid=0.
    original_blocks = [_Block(i) for i in range(310)]
    managers = [
        _Manager({"req-1": list(original_blocks)}),
        _Manager({"req-1": list(original_blocks)}),
    ]
    sched = _make_scheduler(
        scheduler_block_size=scheduler_block_size,
        group_block_sizes=group_block_sizes,
        managers=managers,
    )
    sched.requests["req-1"] = _Request()

    event = {
        "status": "applied",
        "req_id": "req-1",
        "step": 1,
        "cache_len_after": 14336,
        "scheduled_tokens": 1,
        "details": {
            "retained_cache_len": 14464,
            "block_reclaim": {
                "mode": "truncate_tail",
                "groups": [
                    {
                        "gid": 0,
                        "block_ids_before": list(range(310)),
                        "block_ids_after": kept_ids,
                        "block_ids_removed": list(range(113, 310)),
                    }
                ],
            },
        },
    }

    # With the old (gid-agnostic) code, required_blocks = ceil(14464/2048) = 8,
    # kept_len=113 -> assertion "kept_len != required_blocks" would fire.
    # With the fix, required_blocks_by_gid[0] = ceil(14464/128) = 113 = kept_len.
    sched._apply_compression_events([event])

    # gid=0 was truncated to 113 blocks; gid=1 was not in the explicit groups
    # but is in expected_shrink_gids so it gets synthesized to its own
    # required_blocks (ceil(14464/2048)=8).
    assert len(managers[0].req_to_blocks["req-1"]) == 113
    # gid=1 synthesized truncation uses group 1's block_size=2048 -> 8 blocks.
    assert len(managers[1].req_to_blocks["req-1"]) == 8


def test_resolve_group_block_sizes_falls_back_to_self_block_size():
    """When kv_cache_config lacks group specs, fall back to self.block_size."""
    group_block_sizes = [128]
    managers = [_Manager({"req-1": [_Block(0)]})]
    sched = _make_scheduler(
        scheduler_block_size=128,
        group_block_sizes=group_block_sizes,
        managers=managers,
    )
    sizes = sched._resolve_group_block_sizes(managers)
    assert sizes == [128]

    # Missing kv_cache_config -> all fall back.
    sched.kv_cache_config = None
    sizes = sched._resolve_group_block_sizes(managers)
    assert sizes == [128]


def test_reclaim_single_group_unchanged_behavior():
    """Single-group model: scheduler block_size == group block_size, so the
    per-gid path yields the same result as the legacy gid-agnostic path."""
    group_block_sizes = [128]
    scheduler_block_size = 128
    kept_ids = list(range(100))
    original_blocks = [_Block(i) for i in range(200)]
    managers = [_Manager({"req-1": list(original_blocks)})]
    sched = _make_scheduler(
        scheduler_block_size=scheduler_block_size,
        group_block_sizes=group_block_sizes,
        managers=managers,
    )
    sched.requests["req-1"] = _Request()

    event = {
        "status": "applied",
        "req_id": "req-1",
        "step": 1,
        "cache_len_after": 12736,
        "scheduled_tokens": 1,
        "details": {
            "retained_cache_len": 12865,
            "block_reclaim": {
                "mode": "truncate_tail",
                "groups": [
                    {
                        "gid": 0,
                        "block_ids_before": list(range(200)),
                        "block_ids_after": kept_ids,
                        "block_ids_removed": list(range(100, 200)),
                    }
                ],
            },
        },
    }
    # ceil(12865/128) = 101, but kept_len=100 -> assertion fires regardless.
    # Adjust kept_ids to match: ceil(12865/128)=101.
    kept_ids_correct = list(range(101))
    event["details"]["block_reclaim"]["groups"][0]["block_ids_after"] = kept_ids_correct
    event["details"]["block_reclaim"]["groups"][0]["block_ids_removed"] = list(range(101, 200))

    sched._apply_compression_events([event])
    assert len(managers[0].req_to_blocks["req-1"]) == 101
