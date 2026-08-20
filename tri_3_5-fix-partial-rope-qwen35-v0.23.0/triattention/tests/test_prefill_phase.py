from types import SimpleNamespace

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.hook_runtime_context import build_hook_runtime_context
from triattention.vllm.runtime.prefill_phase import (
    is_prefill_phase_for_limit,
    is_request_scheduled_as_prefill,
    is_request_scheduled_as_spec_decode,
)
from triattention.vllm.runtime.signals import CompressionSignal


class _AscendRunner:
    pass


_AscendRunner.__module__ = "vllm_ascend.test"


def _signal():
    return CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=2048,
        step=10,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=1,
    )


def test_first_ascend_decode_hook_enters_core_by_default():
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=9864,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=9863,
        scheduled_tokens=1,
    )

    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(num_computed_tokens=9863, block_ids=[[1] * 80]),
        req_runtime_state=SimpleNamespace(
            compression_count=0,
            current_cache_len=0,
            last_absorbed_cache_len=9863,
        ),
        signal=signal,
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once=set(),
        original_block_ids_by_group=[[1] * 80],
        block_size_hint=128,
    )

    assert not context.should_defer_recompress
    assert context.defer_reason is None


def test_first_ascend_decode_hook_can_opt_into_initial_grace():
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=9864,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=9863,
        scheduled_tokens=1,
    )

    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
            min_decode_tokens_before_compress_on_ascend=2048,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(num_computed_tokens=9863, block_ids=[[1] * 80]),
        req_runtime_state=SimpleNamespace(
            compression_count=0,
            current_cache_len=0,
            last_absorbed_cache_len=9863,
        ),
        signal=signal,
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once=set(),
        original_block_ids_by_group=[[1] * 80],
        block_size_hint=128,
    )

    assert context.should_defer_recompress
    assert context.defer_reason == "initial_decode_grace"


def test_later_ascend_decode_hook_exits_initial_grace():
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=9863 + 2048,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=9863,
        scheduled_tokens=1,
    )

    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(
            num_computed_tokens=9863 + 2047,
            block_ids=[[1] * 96],
        ),
        req_runtime_state=SimpleNamespace(
            compression_count=0,
            current_cache_len=0,
            last_absorbed_cache_len=9863,
        ),
        signal=signal,
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once=set(),
        original_block_ids_by_group=[[1] * 96],
        block_size_hint=128,
    )

    assert not context.should_defer_recompress
    assert context.defer_reason is None


def test_prefill_phase_uses_scheduler_new_reqs():
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req-1")],
    )

    assert is_request_scheduled_as_prefill(scheduler_output, "req-1")
    assert is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id="req-1",
        scheduled_tokens=1,
        prefill_len=10000,
        num_computed_tokens=10000,
    )


def test_spec_decode_multi_token_step_is_not_prefill_after_prompt():
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={"req-1": [11, 12, 13]},
    )

    assert is_request_scheduled_as_spec_decode(scheduler_output, "req-1")
    assert not is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id="req-1",
        scheduled_tokens=4,
        prefill_len=10000,
        num_computed_tokens=10000,
    )


def test_spec_decode_multi_token_step_still_prefill_when_scheduled_as_new_req():
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req-1")],
        scheduled_spec_decode_tokens={"req-1": [11, 12, 13]},
    )

    assert is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id="req-1",
        scheduled_tokens=4,
        prefill_len=10000,
        num_computed_tokens=9000,
    )


def test_multi_token_without_spec_decode_remains_prefill():
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )

    assert is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id="req-1",
        scheduled_tokens=4,
        prefill_len=10000,
        num_computed_tokens=None,
    )


def test_non_spec_multi_token_step_keeps_chunked_prefill_semantics():
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )

    assert is_prefill_phase_for_limit(
        scheduler_output=scheduler_output,
        req_id="req-1",
        scheduled_tokens=4,
        prefill_len=10000,
        num_computed_tokens=10000,
    )


def test_decode_after_compression_is_not_prefill_limit():
    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            prefill_max_compressions_on_ascend=1,
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(num_computed_tokens=10000, block_ids=[[1] * 16]),
        req_runtime_state=SimpleNamespace(
            compression_count=1,
            current_cache_len=2048,
            last_absorbed_cache_len=2048,
        ),
        signal=_signal(),
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once={"req-1"},
        original_block_ids_by_group=[[1] * 16],
        block_size_hint=128,
    )

    assert context.defer_reason != "prefill_compression_limit"


def test_decode_recompress_not_deferred_at_physical_capacity_boundary():
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=32641,
        step=273,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=32383,
        scheduled_tokens=1,
        force=True,
    )

    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
            kv_budget=4096,
            divide_length=128,
            min_reclaim_blocks_on_ascend=8,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(
            num_computed_tokens=32639,
            block_ids=[[1] * 34],
        ),
        req_runtime_state=SimpleNamespace(
            compression_count=1,
            current_cache_len=4353,
            last_absorbed_cache_len=4096,
        ),
        signal=signal,
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once={"req-1"},
        original_block_ids_by_group=[[1] * 34],
        block_size_hint=128,
    )

    assert context.effective_tokens == 4352
    assert not context.should_defer_recompress
    assert context.defer_reason is None


def test_hybrid_short_linear_group_does_not_cap_full_attention_length():
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=32091,
        step=1,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=32089,
        scheduled_tokens=1,
    )

    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
            kv_budget=6144,
            min_reclaim_blocks_on_ascend=1,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(
            num_computed_tokens=32090,
            block_ids=[
                [1] * 22,
                [2] * 2,
            ],
        ),
        req_runtime_state=SimpleNamespace(
            compression_count=0,
            current_cache_len=0,
            last_absorbed_cache_len=0,
        ),
        signal=signal,
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once=set(),
        original_block_ids_by_group=[
            [1] * 22,
            [2] * 2,
        ],
        block_size_hint=1536,
        compressible_group_ids={0},
    )

    assert context.effective_tokens == 32090
    assert context.budget_total == 6144
    assert not context.should_defer_recompress
    assert context.defer_reason is None


def test_prefill_after_compression_still_hits_prefill_limit():
    context = build_hook_runtime_context(
        base_runner=_AscendRunner(),
        config=TriAttentionRuntimeConfig(
            prefill_max_compressions_on_ascend=1,
            enable_experimental_kv_compaction=True,
            defer_prefill_compression_on_ascend=False,
        ),
        req_id="req-1",
        req_state=SimpleNamespace(num_computed_tokens=4096, block_ids=[[1] * 16]),
        req_runtime_state=SimpleNamespace(
            compression_count=1,
            current_cache_len=2048,
            last_absorbed_cache_len=2048,
        ),
        signal=_signal(),
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[],
            num_scheduled_tokens={"req-1": 1},
        ),
        compressed_once={"req-1"},
        original_block_ids_by_group=[[1] * 16],
        block_size_hint=128,
    )

    assert context.defer_reason == "prefill_compression_limit"
