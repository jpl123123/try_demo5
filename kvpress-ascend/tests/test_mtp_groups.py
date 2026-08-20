"""Multi-KV-group (Qwen3.5-MTP style) handling tests.

The user's launch uses ``qwen3_5_mtp`` with 3 speculative tokens: the model has
multiple KV-cache groups (main full-attention group + draft group). Only the
compressible (full-attention) groups may be compacted; the draft group's row
must remain untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

from kvpress_ascend.runtime.group_resolver import resolve_compressible_group_ids, resolve_group_tensors  # noqa: E402


def _make_mtp_runner():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner(num_layers=4, block_size=8, num_groups=2)
    # Group 0 = main full attention (compressible); group 1 = draft
    # (also full attention here; a linear/mamba group would be excluded by the
    # spec markers).
    runner.kv_cache_config.kv_cache_groups[1].name = "draft"
    return runner


def test_compressible_group_resolution():
    runner = _make_mtp_runner()
    ids = resolve_compressible_group_ids(runner)
    assert ids == {0, 1}  # both full attention in this stub
    tensors = resolve_group_tensors(runner)
    assert set(tensors.keys()) == {0, 1}
    for gid, layers in tensors.items():
        assert len(layers) == 4


def test_non_compressible_spec_excluded():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner(num_layers=2, block_size=8, num_groups=2)
    # Simulate a mamba/linear-attn draft group: layer names carry the marker.
    runner.kv_cache_config.kv_cache_groups[1].layer_names = [
        "model.layers.0.linear_attn",
        "model.layers.1.linear_attn",
    ]
    ids = resolve_compressible_group_ids(runner)
    assert ids == {0}
    tensors = resolve_group_tensors(runner)
    assert set(tensors.keys()) == {0}


def test_mtp_loop_compacts_only_compressible_group():
    """End-to-end with two groups: the compressible group compacts; the
    non-compressible (linear-attn draft) group row stays untouched."""
    from conftest import install_kvpress_plugin
    from sim_loop import SimLoop

    sim_env.install_stubs()
    install_kvpress_plugin(
        {
            "KVPRESS_ENABLE": "1",
            "KVPRESS_PRESS": "StreamingLLMPress",
            "KVPRESS_COMPRESSION_RATIO": "0.5",
            "KVPRESS_SINK_TOKENS": "4",
            "KVPRESS_MIN_RECLAIM_BLOCKS": "1",
        }
    )
    sim = SimLoop(block_size=8, num_blocks=1024)
    # Give the worker a two-group block table; group 1 is a non-compressible
    # linear-attn draft group (its layers carry the marker and have no static
    # forward context entry, so the resolver excludes it from compaction).
    from vllm_ascend.worker.block_table import BlockTable, MultiGroupBlockTable

    base_runner = sim._base_runner()
    tables = [
        BlockTable(8, 16, 64),
        BlockTable(8, 16, 64),
    ]
    base_runner.block_table = MultiGroupBlockTable(tables)
    base_runner.num_groups = 2
    # Add a second, non-compressible KV-cache group (linear-attn draft).
    from vllm_ascend.worker.model_runner_v1 import StubKVGroup

    base_runner.kv_cache_config.kv_cache_groups = list(
        base_runner.kv_cache_config.kv_cache_groups
    )
    base_runner.kv_cache_config.kv_cache_groups.append(
        StubKVGroup(1, "draft", [
            "model.layers.0.linear_attn",
            "model.layers.1.linear_attn",
            "model.layers.2.linear_attn",
            "model.layers.3.linear_attn",
        ])
    )

    sim.add_request("req-1", num_prompt_tokens=64)
    sim.prefill("req-1", 64, chunk=64)
    before = sim.row_blocks("req-1")
    # The draft group maintains its own row (its own block allocation).
    tables[1].add_row([100, 101, 102, 103, 104, 105, 106, 107], 0)
    sim.decode("req-1", steps=2)

    batch = base_runner.input_batch
    g0_blocks = int(batch.block_table.block_tables[0].num_blocks_per_row[0])
    g1_blocks = int(batch.block_table.block_tables[1].num_blocks_per_row[0])
    assert g0_blocks < before
    # Draft group row keeps its full prefill row (never compacted).
    assert g1_blocks == 8
    # Its block ids are untouched.
    assert batch.block_table.block_tables[1].block_table.np[0, :8].tolist() == [
        100, 101, 102, 103, 104, 105, 106, 107,
    ]
