import sys
import types
from types import SimpleNamespace

try:
    import torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(
            Tensor=object,
            is_tensor=lambda value: False,
        )

from triattention.vllm.runtime.kv_group_resolver import (
    is_compressible_kv_group,
    resolve_compressible_group_ids,
)


class FullAttentionSpec:
    pass


class MambaSpec:
    pass


class UniformTypeKVCacheSpecs:
    def __init__(self, kv_cache_specs):
        self.kv_cache_specs = kv_cache_specs


def _group(layer_names, spec):
    return SimpleNamespace(layer_names=layer_names, kv_cache_spec=spec)


def test_qwen35_full_attention_group_is_compressible():
    group = _group(
        ["model.layers.3.self_attn"],
        FullAttentionSpec(),
    )

    assert is_compressible_kv_group(group) is True


def test_qwen35_linear_attention_group_is_not_compressible():
    group = _group(
        ["model.layers.4.linear_attn"],
        MambaSpec(),
    )

    assert is_compressible_kv_group(group) is False


def test_resolve_compressible_group_ids_skips_linear_attention_group():
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                _group(["model.layers.3.self_attn"], FullAttentionSpec()),
                _group(["model.layers.4.linear_attn"], MambaSpec()),
            ],
        ),
    )

    assert resolve_compressible_group_ids(runner) == {0}


def test_uniform_group_with_mixed_specs_is_not_compressible():
    group = _group(
        ["model.layers.3.self_attn", "model.layers.4.linear_attn"],
        UniformTypeKVCacheSpecs(
            {
                "model.layers.3.self_attn": FullAttentionSpec(),
                "model.layers.4.linear_attn": MambaSpec(),
            }
        ),
    )

    assert is_compressible_kv_group(group) is False
