"""Stub: vllm.v1.core.kv_cache_utils (memory check symbols)."""


def _check_enough_kv_cache_memory(available_memory, get_needed_memory, max_model_len, estimate_max_model_len):
    needed = get_needed_memory()
    if needed > available_memory:
        raise ValueError("stub: not enough KV cache memory")


def check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory):
    needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
    if needed > available_memory:
        raise ValueError("stub: not enough KV cache memory")


def max_memory_usage_bytes(vllm_config, kv_cache_specs):
    return 1 << 34


def estimate_max_model_len(vllm_config, kv_cache_spec, available_memory):
    return 8192
