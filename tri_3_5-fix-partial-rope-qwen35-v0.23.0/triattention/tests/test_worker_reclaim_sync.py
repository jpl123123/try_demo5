import sys
import types
from types import SimpleNamespace

import numpy as np


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

from triattention.vllm.runtime.worker_reclaim_sync import (  # noqa: E402
    apply_worker_block_reclaim_events,
)


class _Table:
    def __init__(self, rows):
        self.block_table = SimpleNamespace(np=np.array(rows, dtype=np.int32))
        self.num_blocks_per_row = np.array(
            [len([v for v in row if int(v) != 0]) for row in rows],
            dtype=np.int32,
        )

    def add_row(self, block_ids, row_idx):
        self.num_blocks_per_row[row_idx] = 0
        if block_ids:
            self.block_table.np[row_idx, :len(block_ids)] = block_ids
            self.num_blocks_per_row[row_idx] = len(block_ids)


def test_worker_reclaim_truncate_clears_stale_block_table_tail():
    table = _Table([[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=table,
        ),
        requests={"req-1": SimpleNamespace(block_ids=[[11, 12, 13, 14, 15, 16]])},
    )

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 300,
            }
        ],
    )

    assert table.num_blocks_per_row[0] == 3
    np.testing.assert_array_equal(table.block_table.np[0], [11, 12, 13, 0, 0, 0])
    np.testing.assert_array_equal(table.block_table.np[1], [21, 22, 23, 24, 25, 26])
    assert base_runner.requests["req-1"].block_ids == [[11, 12, 13]]


def test_worker_reclaim_uses_retained_cache_len_for_current_decode_block():
    table = _Table([[11, 12, 13, 14, 15, 16]])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=table,
        ),
        requests={"req-1": SimpleNamespace(block_ids=[[11, 12, 13, 14, 15, 16]])},
    )

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 256,
                "details": {"retained_cache_len": 257},
            }
        ],
    )

    assert table.num_blocks_per_row[0] == 3
    np.testing.assert_array_equal(table.block_table.np[0], [11, 12, 13, 0, 0, 0])
    assert base_runner.requests["req-1"].block_ids == [[11, 12, 13]]


def test_worker_reclaim_preserves_decode_slack_from_retained_cache_len():
    table = _Table([list(range(1, 41))])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=table,
        ),
        requests={"req-1": SimpleNamespace(block_ids=[list(range(1, 41))])},
    )

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 4096,
                "details": {"retained_cache_len": 4225},
            }
        ],
    )

    assert table.num_blocks_per_row[0] == 34
    np.testing.assert_array_equal(
        table.block_table.np[0],
        list(range(1, 35)) + [0] * 6,
    )
    assert base_runner.requests["req-1"].block_ids == [list(range(1, 35))]


def test_worker_reclaim_with_explicit_group_preserves_other_hybrid_groups():
    table0 = _Table([[101, 102, 103, 104, 105, 106]])
    table1 = _Table([[201, 202]])
    block_table = SimpleNamespace(block_tables=[table0, table1])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=block_table,
        ),
        requests={
            "req-1": SimpleNamespace(
                block_ids=[
                    [101, 102, 103, 104, 105, 106],
                    [201, 202],
                ]
            )
        },
    )

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 384,
                "block_reclaim": {
                    "mode": "truncate_tail",
                    "groups": [
                        {
                            "gid": 0,
                            "block_ids_before": [101, 102, 103, 104, 105, 106],
                            "block_ids_after": [101, 102, 103],
                            "block_ids_removed": [104, 105, 106],
                        }
                    ],
                },
            }
        ],
    )

    assert table0.num_blocks_per_row[0] == 3
    assert table1.num_blocks_per_row[0] == 2
    np.testing.assert_array_equal(table0.block_table.np[0], [101, 102, 103, 0, 0, 0])
    np.testing.assert_array_equal(table1.block_table.np[0], [201, 202])
    assert base_runner.requests["req-1"].block_ids == [
        [101, 102, 103],
        [201, 202],
    ]


def test_worker_reclaim_remap_rewrites_row_and_clears_stale_tail():
    table0 = _Table([[101, 102, 103, 104, 105, 106]])
    table1 = _Table([[201, 202, 203, 204, 205, 206]])
    block_table = SimpleNamespace(block_tables=[table0, table1])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=block_table,
        ),
        requests={
            "req-1": SimpleNamespace(
                block_ids=([101, 102, 103, 104, 105, 106], [201, 202, 203, 204, 205, 206])
            )
        },
    )

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 384,
                "block_reclaim": {
                    "mode": "remap_tail",
                    "groups": [
                        {"gid": 0, "block_ids_after": [104, 105, 106]},
                        {"gid": 1, "block_ids_after": [204, 205, 206]},
                    ],
                },
            }
        ],
    )

    assert table0.num_blocks_per_row[0] == 3
    assert table1.num_blocks_per_row[0] == 3
    np.testing.assert_array_equal(table0.block_table.np[0], [104, 105, 106, 0, 0, 0])
    np.testing.assert_array_equal(table1.block_table.np[0], [204, 205, 206, 0, 0, 0])
    assert base_runner.requests["req-1"].block_ids == (
        [104, 105, 106],
        [204, 205, 206],
    )


def test_worker_reclaim_remap_preserves_borrowed_slack_blocks():
    table = _Table([list(range(100))])
    base_runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            block_table=table,
        ),
        requests={"req-1": SimpleNamespace(block_ids=[list(range(100))])},
    )
    block_ids_after = list(range(50, 100)) + [48, 49]

    apply_worker_block_reclaim_events(
        base_runner=base_runner,
        events=[
            {
                "status": "applied",
                "req_id": "req-1",
                "cache_len_after": 6400,
                "details": {"retained_cache_len": 6529},
                "block_reclaim": {
                    "mode": "remap_tail",
                    "groups": [{"gid": 0, "block_ids_after": block_ids_after}],
                },
            }
        ],
    )

    assert table.num_blocks_per_row[0] == 52
    np.testing.assert_array_equal(
        table.block_table.np[0],
        block_ids_after + [0] * 48,
    )
    assert base_runner.requests["req-1"].block_ids == [block_ids_after]
