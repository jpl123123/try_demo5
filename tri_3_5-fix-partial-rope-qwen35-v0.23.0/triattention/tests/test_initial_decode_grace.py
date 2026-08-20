from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.thresholds import (
    initial_decode_compression_grace_tokens,
    should_defer_initial_decode_compression,
)


def test_initial_decode_grace_defaults_to_opt_in():
    config = TriAttentionRuntimeConfig()

    assert initial_decode_compression_grace_tokens(config, is_ascend=False) == 0
    assert initial_decode_compression_grace_tokens(config, is_ascend=True) == 0


def test_initial_decode_grace_defers_first_ascend_decode_when_configured():
    config = TriAttentionRuntimeConfig(
        min_decode_tokens_before_compress_on_ascend=2048,
    )

    assert should_defer_initial_decode_compression(
        config=config,
        effective_tokens=9865,
        prefill_len=9863,
        is_ascend=True,
        is_prefill_step=False,
        compressed_once=False,
    )
    assert not should_defer_initial_decode_compression(
        config=config,
        effective_tokens=9863 + 2048,
        prefill_len=9863,
        is_ascend=True,
        is_prefill_step=False,
        compressed_once=False,
    )


def test_initial_decode_grace_does_not_affect_prefill_or_compressed_requests():
    config = TriAttentionRuntimeConfig()

    assert not should_defer_initial_decode_compression(
        config=config,
        effective_tokens=9865,
        prefill_len=9863,
        is_ascend=True,
        is_prefill_step=True,
        compressed_once=False,
    )
    assert not should_defer_initial_decode_compression(
        config=config,
        effective_tokens=9865,
        prefill_len=9863,
        is_ascend=True,
        is_prefill_step=False,
        compressed_once=True,
    )


def test_initial_decode_grace_env_override(monkeypatch):
    monkeypatch.setenv(
        "TRIATTN_RUNTIME_MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND",
        "1024",
    )

    config = TriAttentionRuntimeConfig.from_env()

    assert initial_decode_compression_grace_tokens(config, is_ascend=True) == 1024
