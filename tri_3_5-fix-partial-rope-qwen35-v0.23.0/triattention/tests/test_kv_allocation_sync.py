from types import SimpleNamespace

from triattention.vllm.runtime.kv_allocation_sync import (
    resolve_current_effective_cache_len,
    update_request_effective_kv_offset,
    prepare_request_effective_num_computed,
)


def test_scheduler_event_consumption_advances_effective_cache_len_from_anchor():
    current = resolve_current_effective_cache_len(
        cache_len_after=4096,
        scheduler_nct=32383,
        num_computed_tokens=32384,
        scheduled_tokens=1,
    )

    assert current == 4097


def test_allocation_offset_tracks_effective_boundary_before_native_boundary():
    request = SimpleNamespace(num_computed_tokens=32384)
    current = resolve_current_effective_cache_len(
        cache_len_after=4096,
        scheduler_nct=32383,
        num_computed_tokens=request.num_computed_tokens,
        scheduled_tokens=1,
    )
    update_request_effective_kv_offset(request=request, cache_len_after=current)

    request.num_computed_tokens = 32639
    effective = prepare_request_effective_num_computed(request)

    assert effective == 4352
