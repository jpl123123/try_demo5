import sys
import types

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig

try:
    import torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(Tensor=object)

from triattention.vllm.runtime.hook_group_pipeline import (  # noqa: E402
    try_build_recency_tail_block_remap,
)
from triattention.vllm.runtime.layout_engine import truncate_tail_reclaim_group


def _config():
    return TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        enable_zero_copy_recency=True,
        enable_experimental_block_reclaim=True,
    )


def test_zero_copy_tail_remap_preserves_decode_trailing_block():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[list(range(80))],
        effective_tokens=10000,
        budget_total=2048,
        block_size=128,
        retained_token_padding=129,
    )

    assert outcome is not None
    assert outcome.cache_len_after == 1936
    assert outcome.mutable_block_ids_by_group == [list(range(63, 80))]
    assert outcome.block_reclaim_groups[0].block_ids_removed == list(range(63))


def test_zero_copy_tail_remap_still_handles_exact_block_table():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[list(range(79))],
        effective_tokens=10000,
        budget_total=2048,
        block_size=128,
    )

    assert outcome is not None
    assert outcome.cache_len_after == 1936
    assert outcome.mutable_block_ids_by_group == [list(range(63, 79))]
    assert outcome.block_reclaim_groups[0].block_ids_removed == list(range(63))


def test_zero_copy_tail_remap_borrows_slack_blocks_on_aligned_tail():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[list(range(100))],
        effective_tokens=12800,
        budget_total=6400,
        block_size=128,
        retained_token_padding=129,
    )

    assert outcome is not None
    assert outcome.cache_len_after == 6400
    assert outcome.mutable_block_ids_by_group == [list(range(50, 100)) + [48, 49]]
    assert outcome.block_reclaim_groups[0].block_ids_removed == list(range(48))


def test_zero_copy_tail_remap_skips_noncompressible_hybrid_group():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[
            list(range(80)),
            [200, 201],
        ],
        effective_tokens=10000,
        budget_total=2048,
        block_size=128,
        compressible_group_ids={0},
    )

    assert outcome is not None
    assert outcome.mutable_block_ids_by_group == [
        list(range(63, 79)),
        [200, 201],
    ]
    assert [group.gid for group in outcome.block_reclaim_groups] == [0]


def test_truncate_tail_reclaim_preserves_current_decode_write_block():
    kept, removed, group = truncate_tail_reclaim_group(
        gid=0,
        normalized_block_ids=[10, 11, 12, 13],
        cache_len_after=256,
        block_size=128,
        retained_token_padding=1,
    )

    assert kept == [10, 11, 12]
    assert removed == [13]
    assert group is not None
    assert group.block_ids_after == [10, 11, 12]
    assert group.block_ids_removed == [13]


def test_truncate_tail_reclaim_preserves_next_decode_slack_block():
    kept, removed, group = truncate_tail_reclaim_group(
        gid=0,
        normalized_block_ids=list(range(40)),
        cache_len_after=4096,
        block_size=128,
        retained_token_padding=129,
    )

    assert kept == list(range(34))
    assert removed == list(range(34, 40))
    assert group is not None
    assert group.block_ids_after == list(range(34))
