import sys
import types
from types import SimpleNamespace
import pickle


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
try:
    import torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(
            Tensor=object,
            is_tensor=lambda value: False,
        )

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.input_adapter import (
    EffectiveInputOverrides,
    prepare_effective_input_overrides,
)
from triattention.vllm.runtime import runner_output_bridge as bridge
from triattention.vllm.runtime.state import RequestStateStore


class _AscendRunner:
    pass


_AscendRunner.__module__ = "vllm_ascend.test"


def _overrides():
    return EffectiveInputOverrides(
        seq_base_map={0: 2048},
        pos_delta_map={0: -30000},
        single_seq_base=2048,
        single_pos_delta=-30000,
    )


def test_ascend_multi_req_effective_overrides_keep_graph_by_default():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})

    assert not bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(),
    )


def test_ascend_multi_req_effective_overrides_can_force_eager_guard():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})

    assert bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(
            force_eager_multi_req_on_ascend_effective_overrides=True,
        ),
    )


def test_single_req_effective_overrides_keep_graph_mode_eligible():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1})

    assert not bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(),
    )


def test_graph_guard_respects_config_opt_out():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})

    assert not bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(
            force_eager_multi_req_on_ascend_effective_overrides=False,
        ),
    )


def test_temporary_enforce_eager_restores_original_value():
    base_runner = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))

    with bridge._temporary_model_config_enforce_eager(base_runner, enabled=True) as active:
        assert active
        assert base_runner.model_config.enforce_eager

    assert not base_runner.model_config.enforce_eager


def test_effective_overrides_prefer_active_input_batch_rows():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=4096, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=7,
        cache_len=2048,
        scheduled_tokens=1,
        scheduler_nct=4096,
    )
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"req-1": 5}),
        input_batch=SimpleNamespace(req_id_to_index={"req-1": 0}),
        requests={"req-1": SimpleNamespace(num_computed_tokens=4096)},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-1": 1},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[4096],
        ),
    )

    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=TriAttentionRuntimeConfig(),
    )

    assert overrides.seq_base_map == {0: 2048}
    assert overrides.pos_delta_map == {0: -2048}
    assert overrides.expected_req_row_indices == (0,)


def test_effective_overrides_ignore_compression_anchor_after_scheduler_nct_rollback():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=32768, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=7,
        cache_len=384,
        scheduled_tokens=1,
        scheduler_nct=32768,
    )
    state_store.update_cache_len("req-1", 384 + 8192, step=8)
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"req-1": 0}),
        input_batch=SimpleNamespace(req_id_to_index={"req-1": 0}),
        requests={"req-1": SimpleNamespace(num_computed_tokens=7392)},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-1": 8192},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[7392],
        ),
        triattention_step=8,
    )

    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=TriAttentionRuntimeConfig(),
    )

    assert overrides.seq_base_map == {0: 384}
    assert overrides.pos_delta_map == {0: -7008}
    assert overrides.single_seq_base == 384


def test_effective_overrides_keep_full_budget_after_decode_compression():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=32383, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=17,
        cache_len=4096,
        scheduled_tokens=1,
        scheduler_nct=32383,
    )
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"req-1": 0}),
        input_batch=SimpleNamespace(req_id_to_index={"req-1": 0}),
        requests={"req-1": SimpleNamespace(num_computed_tokens=32639)},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-1": 1},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[32639],
        ),
        triattention_step=273,
    )

    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=TriAttentionRuntimeConfig(),
    )

    assert overrides.seq_base_map == {0: 4352}
    assert overrides.pos_delta_map == {0: -28287}
    assert overrides.single_seq_base == 4352


def test_effective_overrides_skip_positive_delta_from_stale_compression_anchor():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=4096, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=17,
        cache_len=4237,
        scheduled_tokens=1,
        scheduler_nct=4088,
    )
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"req-1": 0}),
        input_batch=SimpleNamespace(req_id_to_index={"req-1": 0}),
        requests={"req-1": SimpleNamespace(num_computed_tokens=4088)},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-1": 1},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[4088],
        ),
        triattention_step=18,
    )

    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=TriAttentionRuntimeConfig(),
    )

    assert overrides.seq_base_map is None
    assert overrides.pos_delta_map is None
    assert overrides.single_seq_base is None
    assert overrides.single_pos_delta == 0


def test_execute_model_none_keeps_events_for_sample_tokens_output():
    event = {
        "status": "applied",
        "req_id": "req-1",
        "cache_len_after": 4096,
    }
    scheduler_output = SimpleNamespace()

    output, pending = bridge.attach_execute_model_compression_events(
        output=None,
        pending_events=[event],
        scheduler_output=scheduler_output,
    )

    assert output is None
    assert pending == [event]
    assert scheduler_output.triattention_compression_events == [event]

    inner_output = SimpleNamespace(kv_connector_output=SimpleNamespace())
    sample_output = SimpleNamespace(_model_runner_output=inner_output)
    sample_output, pending = bridge.attach_sample_tokens_compression_events(
        output=sample_output,
        pending_events=pending,
    )

    assert sample_output.triattention_compression_events == [event]
    assert bridge._read_triattention_events_from_kv_cache_events(inner_output) == [event]
    assert pending == []


def test_triattention_event_bag_survives_pickle_round_trip():
    event = {
        "status": "applied",
        "req_id": "req-1",
        "cache_len_after": 4096,
    }

    restored = pickle.loads(pickle.dumps(bridge._TriattentionEventBag([event])))

    assert restored.events == [event]
