"""Input-patch unit tests: effective seq_lens / positions / slot mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sim_env  # noqa: E402

sim_env.install_stubs()

from kvpress_ascend.runtime import input_patch_state as _ps  # noqa: E402
from kvpress_ascend.runtime.input_patch_v1 import (  # noqa: E402
    make_patched_v1_prepare_inputs,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner  # noqa: E402


def _make_runner_with_row():
    runner = NPUModelRunner(num_layers=2, block_size=8, max_num_reqs=4)
    req = type("R", (), {"req_id": "req-1", "num_computed_tokens": 100, "num_prompt_tokens": 200, "block_ids": [1, 3, 5, 7, 9]})()
    runner.requests["req-1"] = req
    # Worker block row: 5 blocks -> 40 token capacity.
    runner.block_table.add_row([1, 3, 5, 7, 9], 0)
    runner.input_batch = type(
        "B",
        (),
        {
            "req_id_to_index": {"req-1": 0},
            "num_computed_tokens_cpu": np.array([100, 0, 0, 0], dtype=np.int64),
            "num_prompt_tokens": np.array([200, 0, 0, 0], dtype=np.int64),
            "num_reqs": 1,
            "block_table": runner.block_table,
        },
    )()
    return runner


def _make_scheduler_output():
    so = type(
        "SO",
        (),
        {
            "num_scheduled_tokens": {"req-1": 1},
            "total_num_scheduled_tokens": 1,
        },
    )()
    return so


def test_patch_rewrites_seq_lens_positions_and_slots():
    runner = _make_runner_with_row()
    original = NPUModelRunner._prepare_inputs
    patched = make_patched_v1_prepare_inputs(original)

    # Activate overrides: effective base 34 (compressed history), vLLM thinks 100.
    _ps.reset()
    _ps.set_effective_bases({0: 34})

    num_scheduled_tokens = np.array([1], dtype=np.int64)
    patched(runner, _make_scheduler_output(), num_scheduled_tokens)

    assert int(runner.seq_lens.np[0]) == 35  # 34 + 1 scheduled
    assert int(runner.optimistic_seq_lens_cpu[0]) == 35
    # Position of the decoded token lands right after the kept prefix.
    assert int(runner._positions_np_buf[0]) == 34
    # Slot mapping points into the request row: position 34 -> block 4 of the
    # row (block id 9), offset 2.
    slot = int(runner.block_table.slot_mapping.np[0])
    assert slot // 8 in [1, 3, 5, 7, 9]
    assert slot % 8 == 2
    # Overrides consumed after the step.
    assert not _ps.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED


def test_patch_noop_without_overrides():
    runner = _make_runner_with_row()
    original = NPUModelRunner._prepare_inputs
    patched = make_patched_v1_prepare_inputs(original)
    _ps.reset()
    num_scheduled_tokens = np.array([1], dtype=np.int64)
    patched(runner, _make_scheduler_output(), num_scheduled_tokens)
    # vLLM's own view (100 + 1) untouched.
    assert int(runner.seq_lens.np[0]) == 101


def test_single_request_fast_path_base():
    runner = _make_runner_with_row()
    original = NPUModelRunner._prepare_inputs
    patched = make_patched_v1_prepare_inputs(original)
    _ps.reset()
    _ps.set_single_effective_seq_base(34)
    num_scheduled_tokens = np.array([1], dtype=np.int64)
    patched(runner, _make_scheduler_output(), num_scheduled_tokens)
    assert int(runner.seq_lens.np[0]) == 35
    assert int(runner._positions_np_buf[0]) == 34
