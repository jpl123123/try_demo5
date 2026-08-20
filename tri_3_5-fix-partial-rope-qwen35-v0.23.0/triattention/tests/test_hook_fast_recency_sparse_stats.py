from pathlib import Path
import sys
import types
from types import SimpleNamespace

if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        Tensor=object,
        is_tensor=lambda value: False,
    )

from triattention.vllm.runtime import hook_impl
from triattention.vllm.runtime.constants import TRITON_SCORING_REQUIRED_MARKER
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.selector_hf import build_triattention_selector
from triattention.vllm.runtime.signals import CompressionSignal


def _runner(*, block_size: int = 128):
    req_state = SimpleNamespace(
        num_computed_tokens=4096,
        block_ids=[list(range(50))],
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        device_config=SimpleNamespace(device="npu"),
        kv_caches=[object()],
        requests={"req-1": req_state},
    )


def _signal():
    return CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4097,
        step=3,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=0,
        scheduled_tokens=1,
    )


def _patch_pipeline(monkeypatch, reason: str = "pipeline_reached"):
    monkeypatch.setattr(
        hook_impl,
        "_build_triattention_selector",
        lambda config, base_runner=None: (None, None, "enabled:test"),
    )
    monkeypatch.setattr(
        hook_impl,
        "_resolve_group_tensors",
        lambda base_runner: {0: [(0, object())]},
    )

    calls = []

    def _pipeline(**kwargs):
        calls.append(kwargs)
        return {
            "applied": False,
            "reason": reason,
            "cache_len_after": kwargs["effective_tokens"],
        }

    monkeypatch.setattr(hook_impl, "run_group_compaction_pipeline", _pipeline)
    return calls


def test_sparse_stats_accuracy_guard_bypasses_ascend_zero_copy_wait(monkeypatch, tmp_path):
    calls = _patch_pipeline(monkeypatch)
    stats_path = tmp_path / "triattention-stats.pt"
    stats_path.write_bytes(b"stats")
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        sparse_stats_path=stats_path,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        require_triton_scoring=False,
        require_physical_reclaim=False,
        enable_zero_copy_recency=True,
        zero_copy_recency_only_on_ascend=True,
        defer_prefill_compression_on_ascend=False,
    )

    hook = hook_impl.make_runner_compression_hook(
        base_runner=_runner(),
        config=config,
    )
    result = hook("req-1", _signal(), SimpleNamespace())

    assert result["reason"] == "pipeline_reached"
    assert len(calls) == 1


def test_pure_fast_recency_still_waits_for_ascend_zero_copy(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        enable_experimental_kv_compaction=True,
        enable_experimental_block_reclaim=True,
        require_triton_scoring=False,
        require_physical_reclaim=False,
        enable_zero_copy_recency=True,
        zero_copy_recency_only_on_ascend=True,
        defer_prefill_compression_on_ascend=False,
    )

    hook = hook_impl.make_runner_compression_hook(
        base_runner=_runner(block_size=96),
        config=config,
    )
    result = hook("req-1", _signal(), SimpleNamespace())

    assert result["reason"] == "zero_copy_recency_not_ready"
    assert calls == []


def test_fast_recency_missing_stats_path_stays_recency_selector(tmp_path):
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        sparse_stats_path=tmp_path / "missing-stats.pt",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
    )

    selector, group_selector, selector_status = build_triattention_selector(config)

    assert selector is not None
    assert group_selector is not None
    assert selector_status == "enabled:recency_only"


def test_strict_sparse_scoring_still_fails_when_stats_missing(tmp_path):
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        sparse_stats_path=tmp_path / "missing-stats.pt",
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
    )

    try:
        build_triattention_selector(config)
    except RuntimeError as exc:
        assert (
            f"{TRITON_SCORING_REQUIRED_MARKER}:stats_path_not_found"
            in str(exc)
        )
    else:
        raise AssertionError("missing sparse stats should still fail strict scoring")
