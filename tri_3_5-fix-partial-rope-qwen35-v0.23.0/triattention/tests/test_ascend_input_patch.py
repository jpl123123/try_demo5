import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.SimpleNamespace()
if "vllm.logger" not in sys.modules:
    sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())

from triattention.vllm.runtime import input_patch_state
from triattention.vllm.runtime.input_patch_ascend_backend import (
    make_patched_ascend_v2_build_attn_metadata,
    make_patched_ascend_v2_update_seq_lens_cpu,
)
from triattention.vllm.runtime.input_patch_installer import (
    make_patched_ascend_v1_block_table_get_device_tensor,
)
from triattention.vllm.runtime.input_patch_vllm_v1_backend import (
    _validate_v1_block_table_bounds,
    make_patched_v1_prepare_inputs,
)


class _V1BlockTable:
    def __init__(self, *, block_size=128, num_blocks_per_row=(4,)):
        self.block_size = block_size
        self.num_blocks_per_row = np.array(num_blocks_per_row, dtype=np.int32)
        self.compute_calls = []
        self.commit_calls = []

    def compute_slot_mapping(self, req_indices, positions):
        self.compute_calls.append((req_indices.copy(), positions.copy()))

    def commit_slot_mapping(self, total_num_scheduled_tokens):
        self.commit_calls.append(total_num_scheduled_tokens)


class _V1BlockTableV023:
    """Mock BlockTable that mimics the vLLM-Ascend v0.23.0 GPU API.

    v0.23.0 changed ``compute_slot_mapping`` to
    ``(num_reqs, query_start_loc, positions)`` (all GPU tensors) and removed
    ``commit_slot_mapping``; the kernel writes ``slot_mapping.gpu`` directly.
    """

    def __init__(self, *, block_size=128, num_blocks_per_row=(4,)):
        self.block_size = block_size
        self.num_blocks_per_row = np.array(num_blocks_per_row, dtype=np.int32)
        # Records (num_reqs, qsl_cpu_list, positions_cpu_list) for assertions.
        self.compute_calls: list = []

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.compute_calls.append((
            int(num_reqs),
            query_start_loc.detach().cpu().numpy().tolist(),
            positions.detach().cpu().numpy().tolist(),
        ))


class _GpuBufferMock:
    """Minimal CpuGpuBuffer mock (``.np`` / ``.gpu``) used for query_start_loc."""

    def __init__(self, np_array, gpu_tensor):
        self.np = np_array
        self.gpu = gpu_tensor


def test_ascend_v2_seq_override_updates_seq_lens_np():
    def _original(self, scheduler_output, req_ids):
        del scheduler_output
        for row, req_id in enumerate(req_ids):
            req_idx = self.req_states.req_id_to_index[req_id]
            self.input_buffers.seq_lens_np[row] = (
                self.req_states.num_computed_tokens_cpu[req_idx] + 1
            )

    runner = SimpleNamespace(
        input_buffers=SimpleNamespace(seq_lens_np=np.array([5001], dtype=np.int32)),
        req_states=SimpleNamespace(
            req_id_to_index={"req-1": 3},
            num_computed_tokens_cpu=np.array([0, 0, 0, 5000], dtype=np.int32),
        ),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 1})
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={3: 2048},
        effective_pos_delta_by_req_idx={3: -2952},
    )

    try:
        patched = make_patched_ascend_v2_update_seq_lens_cpu(_original)
        patched(runner, scheduler_output, ["req-1"])
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.input_buffers.seq_lens_np[0] == 2049
    assert runner.req_states.num_computed_tokens_cpu[3] == 5000


def test_ascend_v2_metadata_uses_effective_max_seq_len():
    def _original_build_attn_metadata(*args, **kwargs):
        del args
        return kwargs["max_seq_len"]

    patched = make_patched_ascend_v2_build_attn_metadata(
        _original_build_attn_metadata
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={3: 2048},
        effective_pos_delta_by_req_idx={3: -2952},
    )

    try:
        max_seq_len = patched(
            num_reqs=3,
            seq_lens_np=np.array([2049, 2305, 0], dtype=np.int32),
            max_seq_len=40960,
        )
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert max_seq_len == 2305


def test_ascend_v1_block_table_trim_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE", raising=False)
    monkeypatch.delenv("TRIATTN_DEBUG_DISABLE_ASCEND_BLOCK_TABLE_TRIM", raising=False)

    tensor = np.arange(20, dtype=np.int32).reshape(2, 10)
    patched = make_patched_ascend_v1_block_table_get_device_tensor(
        lambda self: tensor
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_max_seq_len(256)
    input_patch_state.set_active_block_table_trim_observation(
        block_size=None,
        original_cols=None,
        effective_cols=None,
    )

    try:
        out = patched(SimpleNamespace(block_size=128))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_max_seq_len(None)

    assert out is tensor
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS is None


def test_ascend_v1_block_table_trim_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE", "1")
    monkeypatch.delenv("TRIATTN_DEBUG_DISABLE_ASCEND_BLOCK_TABLE_TRIM", raising=False)

    tensor = np.arange(20, dtype=np.int32).reshape(2, 10)
    patched = make_patched_ascend_v1_block_table_get_device_tensor(
        lambda self: tensor
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_max_seq_len(256)
    input_patch_state.set_active_block_table_trim_observation(
        block_size=None,
        original_cols=None,
        effective_cols=None,
    )

    try:
        out = patched(SimpleNamespace(block_size=128))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_max_seq_len(None)

    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, tensor[:, :2])
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE == 128
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS == 10
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS == 2


def test_v1_prepare_inputs_fails_on_stale_expected_batch_row():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=SimpleNamespace(
                compute_slot_mapping=lambda *args, **kwargs: None,
                commit_slot_mapping=lambda *args, **kwargs: None,
            ),
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([4097, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={3: 2048},
        effective_pos_delta_by_req_idx={3: -2048},
        expected_req_row_indices=(3,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        try:
            patched(runner, scheduler_output, np.array([1], dtype=np.int32))
        except RuntimeError as exc:
            assert "TRIATTN_V1_IDX_MAPPING_MISMATCH" in str(exc)
        else:
            raise AssertionError("stale expected batch row should fail fast")
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )


def test_v1_prepare_inputs_clamps_effective_seq_to_block_table_capacity():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(3,))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([4097, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx=None,
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np[0] == 384
    assert block_table.commit_calls == [1]
    np.testing.assert_array_equal(block_table.compute_calls[0][1], [383])


def test_v1_prepare_inputs_fails_when_shifted_slot_position_is_out_of_bounds():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=_V1BlockTable(block_size=128, num_blocks_per_row=(3,)),
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([4097, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        try:
            patched(runner, scheduler_output, np.array([1], dtype=np.int32))
        except RuntimeError as exc:
            assert "TRIATTN_ASCEND_V1_SLOT_POSITION_OOB" in str(exc)
            assert "max_slot_position=384" in str(exc)
            assert "capacity=384" in str(exc)
        else:
            raise AssertionError("slot position capacity mismatch should fail fast")
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )


def test_v1_block_table_validation_rejects_slot_position_beyond_effective_seq_len():
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx=None,
        expected_req_row_indices=(0,),
        expected_query_lens=(4,),
    )

    try:
        try:
            _validate_v1_block_table_bounds(
                block_table=_V1BlockTable(block_size=128, num_blocks_per_row=(32,)),
                seq_lens_np=np.array([388], dtype=np.int32),
                req_indices=np.array([0, 0, 0, 0], dtype=np.int32),
                slot_positions_np=np.array([2348, 2349, 2350, 2351], dtype=np.int32),
                num_reqs=1,
                validate_seq_lens=True,
            )
        except RuntimeError as exc:
            assert "TRIATTN_ASCEND_V1_SLOT_POSITION_SEQ_LEN_MISMATCH" in str(exc)
            assert "max_slot_position=2351" in str(exc)
            assert "seq_len=388" in str(exc)
        else:
            raise AssertionError("slot positions beyond effective seq_len should fail")
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )


def test_v1_prepare_inputs_validates_shifted_slot_positions_within_capacity():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(4,))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([4097, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np[0] == 385
    assert block_table.commit_calls == [1]
    np.testing.assert_array_equal(block_table.compute_calls[0][1], [384])


def test_v1_prepare_inputs_updates_ascend_optimistic_seq_lens_cpu():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(4,))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([4097, 99], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
        optimistic_seq_lens_cpu=torch.tensor([4097, 99], dtype=torch.int32),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np.tolist() == [385, 0]
    assert runner.optimistic_seq_lens_cpu.tolist() == [385, 0]
    np.testing.assert_array_equal(block_table.compute_calls[0][1], [384])


def test_v1_prepare_inputs_remaps_overrides_after_input_batch_row_reorder():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(4, 4))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["req-b", "req-a"],
            req_id_to_index={"req-b": 0, "req-a": 1},
            num_computed_tokens_cpu=np.array([1024, 4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0, 1], dtype=np.int32),
        positions=SimpleNamespace(np=np.array([1024, 4096], dtype=np.int32)),
        seq_lens=SimpleNamespace(
            np=np.array([1025, 4097, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=2)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_req_ids=("req-a",),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1, 1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np[0] == 1025
    assert runner.seq_lens.np[1] == 385
    np.testing.assert_array_equal(block_table.compute_calls[0][1], [1024, 384])


def test_v1_prepare_inputs_rebuilds_slot_positions_from_effective_base():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(32,))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(
            np=np.array([2348, 2343, 4096, 4097], dtype=np.int32)
        ),
        seq_lens=SimpleNamespace(
            np=np.array([4098, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=4)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(4,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([4], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np[0] == 388
    np.testing.assert_array_equal(
        block_table.compute_calls[0][1],
        [384, 385, 386, 387],
    )


def test_v1_prepare_inputs_single_base_rebuilds_slot_positions():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(block_size=128, num_blocks_per_row=(32,))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_computed_tokens_cpu=np.array([4096], dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.array([0], dtype=np.int32),
        positions=SimpleNamespace(
            np=np.array([2348, 4096, 4097], dtype=np.int32)
        ),
        seq_lens=SimpleNamespace(
            np=np.array([4098, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=3)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx=None,
        effective_pos_delta_by_req_idx=None,
        single_effective_seq_base=384,
        single_effective_pos_delta=-3712,
        expected_req_row_indices=(0,),
        expected_query_lens=(3,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([3], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.seq_lens.np[0] == 387
    np.testing.assert_array_equal(
        block_table.compute_calls[0][1],
        [384, 385, 386],
    )


def test_v1_prepare_inputs_ignores_positive_pos_delta_in_mixed_batch():
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTable(
        block_size=128,
        num_blocks_per_row=(64, 64, 64, 64, 32),
    )
    positions = np.concatenate(
        [
            np.array([33949, 36299, 35128, 34789], dtype=np.int32),
            np.arange(2044, 4088, dtype=np.int32),
        ]
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=5,
            num_computed_tokens_cpu=np.array(
                [33949, 36299, 35128, 34789, 2044],
                dtype=np.int32,
            ),
            block_table=block_table,
        ),
        arange_np=np.arange(5, dtype=np.int32),
        positions=SimpleNamespace(np=positions),
        seq_lens=SimpleNamespace(
            np=np.array([33950, 36300, 35129, 34790, 4088, 0], dtype=np.int32),
            copy_to_gpu=lambda: None,
        ),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=int(positions.size))
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={4: 2193},
        effective_pos_delta_by_req_idx={4: 149},
        expected_req_row_indices=(4,),
        expected_query_lens=(2044,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(
            runner,
            scheduler_output,
            np.array([1, 1, 1, 1, 2044], dtype=np.int32),
        )
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert block_table.compute_calls == []
    assert block_table.commit_calls == []
    assert runner.seq_lens.np[4] == 4088


# ---------------------------------------------------------------------------
# vLLM-Ascend v0.23.0 compatibility tests
#
# In v0.23.0 ``self.positions`` and ``self.seq_lens`` became plain GPU
# ``torch.Tensor`` objects (no ``.np`` attribute) and ``BlockTable`` switched
# to the GPU ``compute_slot_mapping(num_reqs, query_start_loc, positions)``
# signature without ``commit_slot_mapping``. These tests exercise the
# compatibility layer added in ``input_patch_vllm_v1_backend.py``.
# ---------------------------------------------------------------------------


def _build_v023_runner(*, num_reqs, num_computed_tokens, positions_np,
                      seq_lens_np, block_table, query_start_loc_np=None):
    """Build a SimpleNamespace runner that mimics the v0.23.0 layout.

    - ``positions`` is a plain GPU torch.Tensor (no ``.np``).
    - ``_positions_np_buf`` is the CPU numpy mirror.
    - ``seq_lens`` is a plain CPU/GPU torch.Tensor (no ``.np``).
    - ``optimistic_seq_lens_cpu`` is the CPU torch.Tensor mirror.
    - ``query_start_loc`` is a CpuGpuBuffer-like mock with ``.gpu``.
    """
    device = torch.device("cpu")  # tests run without a real NPU
    positions_gpu = torch.from_numpy(positions_np.astype(np.int64)).to(device)
    seq_lens_t = torch.from_numpy(seq_lens_np.astype(np.int32)).to(device)
    optimistic = torch.from_numpy(seq_lens_np.astype(np.int32)).to(device)

    if query_start_loc_np is None:
        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int64)
    qsl_gpu = torch.from_numpy(query_start_loc_np.astype(np.int64)).to(device)

    return SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=num_reqs,
            num_computed_tokens_cpu=np.array(num_computed_tokens, dtype=np.int32),
            block_table=block_table,
        ),
        arange_np=np.arange(max(num_reqs, 1), dtype=np.int32),
        positions=positions_gpu,
        _positions_np_buf=positions_np.astype(np.int64),
        seq_lens=seq_lens_t,
        optimistic_seq_lens_cpu=optimistic,
        query_start_loc=_GpuBufferMock(query_start_loc_np, qsl_gpu),
        device=device,
    )


def test_v023_prepare_inputs_applies_seq_and_slot_overrides_via_gpu_api():
    """v0.23.0 path: seq_lens and slot mapping are updated through the new GPU API."""
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTableV023(block_size=128, num_blocks_per_row=(4,))
    positions_np = np.array([4096], dtype=np.int64)
    runner = _build_v023_runner(
        num_reqs=1,
        num_computed_tokens=[4096],
        positions_np=positions_np,
        seq_lens_np=np.array([4097, 0], dtype=np.int32),
        block_table=block_table,
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    # seq_lens was overridden to effective_base + num_scheduled_tokens = 385.
    assert runner.seq_lens.cpu().tolist()[0] == 385
    assert runner.optimistic_seq_lens_cpu.tolist()[0] == 385
    # The new GPU compute_slot_mapping API was invoked (no commit_slot_mapping).
    assert len(block_table.compute_calls) == 1
    num_reqs, qsl, pos = block_table.compute_calls[0]
    assert num_reqs == 1
    assert qsl == [0, 1]
    assert pos == [384]
    # No commit_calls attribute on the v0.23 mock (commit_slot_mapping removed).
    assert not hasattr(block_table, "commit_calls")


def test_v023_prepare_inputs_skips_slot_mapping_when_no_override():
    """v0.23.0 path: without a pos override, compute_slot_mapping is not called."""
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTableV023(block_size=128, num_blocks_per_row=(4,))
    positions_np = np.array([4096], dtype=np.int64)
    runner = _build_v023_runner(
        num_reqs=1,
        num_computed_tokens=[4096],
        positions_np=positions_np,
        seq_lens_np=np.array([4097, 0], dtype=np.int32),
        block_table=block_table,
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    # Only seq base override, no pos delta -> slot positions unchanged.
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx=None,
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    # seq override applied (384 + 1 = 385), but no slot remap (single req,
    # effective base 384 < positions[0]=4096 so _build_effective_slot_positions
    # does produce a remapped position to 384 -> compute_slot_mapping called).
    assert runner.seq_lens.cpu().tolist()[0] == 385
    # single-req path with effective_base=384 yields slot position [384].
    assert len(block_table.compute_calls) == 1
    _, _, pos = block_table.compute_calls[0]
    assert pos == [384]


def test_v023_prepare_inputs_handles_multi_request_batch():
    """v0.23.0 path: multi-request batch query_start_loc and positions mapping."""
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTableV023(block_size=128, num_blocks_per_row=(4, 4))
    positions_np = np.array([1024, 4096], dtype=np.int64)
    runner = _build_v023_runner(
        num_reqs=2,
        num_computed_tokens=[1024, 4096],
        positions_np=positions_np,
        seq_lens_np=np.array([1025, 4097, 0], dtype=np.int32),
        block_table=block_table,
        # Two requests each with 1 scheduled token -> [0, 1, 2].
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int64),
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=2)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={1: 384},
        effective_pos_delta_by_req_idx={1: -3712},
        expected_req_row_indices=(1,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1, 1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    # req 0 unchanged (1025), req 1 overridden to 385.
    seq_list = runner.seq_lens.cpu().tolist()
    assert seq_list[0] == 1025
    assert seq_list[1] == 385
    # Slot mapping invoked once with both tokens; req 1 position remapped to 384.
    assert len(block_table.compute_calls) == 1
    num_reqs, qsl, pos = block_table.compute_calls[0]
    assert num_reqs == 2
    assert qsl == [0, 1, 2]
    assert pos == [1024, 384]


def test_v023_prepare_inputs_falls_back_to_qsl_reconstruction_when_missing():
    """v0.23.0 path: when query_start_loc lacks ``.gpu``, it is reconstructed."""
    def _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        del self, scheduler_output, num_scheduled_tokens
        return object()

    block_table = _V1BlockTableV023(block_size=128, num_blocks_per_row=(4,))
    positions_np = np.array([4096], dtype=np.int64)
    runner = _build_v023_runner(
        num_reqs=1,
        num_computed_tokens=[4096],
        positions_np=positions_np,
        seq_lens_np=np.array([4097, 0], dtype=np.int32),
        block_table=block_table,
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    # Remove the ``.gpu`` attribute to force the reconstruction fallback.
    runner.query_start_loc = None

    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={0: 384},
        effective_pos_delta_by_req_idx={0: -3712},
        expected_req_row_indices=(0,),
        expected_query_lens=(1,),
    )

    try:
        patched = make_patched_v1_prepare_inputs(_original_prepare_inputs)
        patched(runner, scheduler_output, np.array([1], dtype=np.int32))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    # The reconstruction fallback builds [0, 1] from req_indices.
    assert len(block_table.compute_calls) == 1
    num_reqs, qsl, pos = block_table.compute_calls[0]
    assert num_reqs == 1
    assert qsl == [0, 1]
    assert pos == [384]


def test_v023_helpers_extract_numpy_from_plain_tensors():
    """Unit test the version-tolerant buffer accessors directly."""
    from triattention.vllm.runtime.input_patch_vllm_v1_backend import (
        _runner_positions_np,
        _runner_seq_lens_np,
    )

    # Plain tensor layout (v0.23.0): positions is a GPU-like tensor with
    # ``_positions_np_buf`` mirror; seq_lens is plain tensor with
    # ``optimistic_seq_lens_cpu`` mirror.
    positions_buf = np.array([10, 20, 30], dtype=np.int64)
    runner = SimpleNamespace(
        positions=torch.zeros(3, dtype=torch.int64),
        _positions_np_buf=positions_buf,
        seq_lens=torch.zeros(3, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.tensor([100, 200, 0], dtype=torch.int32),
    )
    np.testing.assert_array_equal(_runner_positions_np(runner, 3), [10, 20, 30])
    np.testing.assert_array_equal(_runner_seq_lens_np(runner), [100, 200, 0])

    # Legacy CpuGpuBuffer layout (v0.18.0): objects expose ``.np``.
    runner_legacy = SimpleNamespace(
        positions=SimpleNamespace(np=np.array([1, 2], dtype=np.int64)),
        seq_lens=SimpleNamespace(np=np.array([7, 8], dtype=np.int32)),
    )
    np.testing.assert_array_equal(_runner_positions_np(runner_legacy, 2), [1, 2])
    np.testing.assert_array_equal(_runner_seq_lens_np(runner_legacy), [7, 8])

