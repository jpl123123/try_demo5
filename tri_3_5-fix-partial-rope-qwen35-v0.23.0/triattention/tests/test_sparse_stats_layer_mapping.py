from types import SimpleNamespace

import torch

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.selector_hf import build_triattention_selector
from triattention.vllm.core.utils import load_frequency_stats


def test_rkv_sparse_sampled_layers_preserve_max_layer_index(tmp_path):
    stats_path = tmp_path / "qwen35_sparse_layers.pt"
    freq_count = 4
    stats = {}
    for layer_idx in (0, 7, 15, 19):
        for head_idx in range(2):
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.ones(freq_count),
                "q_mean_imag": torch.zeros(freq_count),
                "q_abs_mean": torch.ones(freq_count),
            }
    torch.save(
        {
            "metadata": {
                "head_dim": freq_count * 2,
                "rope_style": "half",
                "sampled_heads": [[layer_idx, 0] for layer_idx in (0, 7, 15, 19)],
            },
            "stats": stats,
        },
        stats_path,
    )

    metadata, head_stats = load_frequency_stats(
        stats_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=2,
    )

    assert metadata["num_layers"] == 20
    assert sorted(head_stats) == [0, 7, 15, 19]
    assert head_stats[19]["freq_scale_sq"].shape == (2, freq_count)


def test_selector_falls_back_when_stats_freq_mismatches_runtime_kv(tmp_path):
    stats_path = tmp_path / "freq4_stats.pt"
    freq_count = 4
    layer_stats = {
        0: {
            "q_mean_complex": torch.zeros(2, freq_count, 2),
            "q_abs_mean": torch.ones(2, freq_count),
            "freq_scale_sq": torch.ones(2, freq_count),
        },
    }
    torch.save(
        {
            "metadata": {
                "num_attention_heads": 2,
                "num_kv_heads": 2,
                "head_dim": freq_count * 2,
                "num_layers": 1,
                "rope_style": "half",
            },
            "layer_stats": layer_stats,
        },
        stats_path,
    )
    config = TriAttentionRuntimeConfig(
        kv_budget=4,
        window_size=1,
        sparse_stats_path=stats_path,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
        scoring_backend="torch",
        sparse_normalize_scores=False,
        log_execution_path=False,
    )

    selector, _group_selector, status = build_triattention_selector(
        config,
        base_runner=SimpleNamespace(device=torch.device("cpu")),
    )

    assert selector is not None
    assert status.startswith("enabled:torch")
    # Runtime head_dim=16 means runtime freq_count=8, while stats only have 4.
    result = selector(
        keys_dense=torch.zeros(1, 2, 8, 16),
        total_tokens=8,
        prefill_len=0,
        protect_prefill=False,
        layer_idx=0,
        round_start=8,
        budget_total=4,
        req_id="req",
        gid=0,
    )

    assert result["mode"] == "shared"
    assert result["semantic"] == "recency_fallback_stats_incompatible"
    assert result["indices"] == [4, 5, 6, 7]


def test_partial_rope_stats_score_without_crash(tmp_path):
    """Qwen3.5-style partial-RoPE stats must score, not crash in _lazy_init.

    Reproduces the vLLM-Ascend 0.23.0rc1 crash where a stats file with
    ``head_dim=256``/``rotary_dim=64`` (``partial_rotary_factor=0.25``) carried
    an ``inv_freq`` of only 32 elements.  The old ``_init_rope`` required
    ``head_dim // 2 = 128`` elements and raised ``ValueError``.
    """
    freq_count = 32
    head_dim = 256
    rotary_dim = 64
    num_kv_heads = 4
    num_layers = 2

    stats = {}
    for layer_idx in range(num_layers):
        for head_idx in range(24):
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.randn(freq_count),
                "q_mean_imag": torch.randn(freq_count),
                "q_abs_mean": torch.rand(freq_count) + 0.1,
            }

    stats_path = tmp_path / "qwen35_partial_rope.pt"
    torch.save(
        {
            "metadata": {
                "head_dim": head_dim,
                "rotary_dim": rotary_dim,
                "freq_count": freq_count,
                "partial_rotary_factor": 0.25,
                "inv_freq": [0.1 * (i + 1) for i in range(freq_count)],
                "num_attention_heads": 24,
                "num_kv_heads": num_kv_heads,
                "rope_style": "half",
                "rope_type": "default",
                "full_attention_layers": list(range(num_layers)),
                "sampled_heads": [
                    [l, h]
                    for l in range(num_layers)
                    for h in range(24)
                ],
            },
            "stats": stats,
        },
        stats_path,
    )

    config = TriAttentionRuntimeConfig(
        kv_budget=16,
        window_size=4,
        sparse_stats_path=stats_path,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
        scoring_backend="torch",
        sparse_normalize_scores=False,
        log_execution_path=False,
    )

    selector, _group_selector, status = build_triattention_selector(
        config,
        base_runner=SimpleNamespace(device=torch.device("cpu")),
    )
    assert selector is not None
    assert status.startswith("enabled:torch")

    # Runtime K tensor carries the full head_dim=256; scoring must only use the
    # rotated 64 channels and not crash.
    keys_dense = torch.randn(1, num_kv_heads, 64, head_dim)
    result = selector(
        keys_dense=keys_dense,
        total_tokens=64,
        prefill_len=0,
        protect_prefill=False,
        layer_idx=0,
        round_start=64,
        budget_total=16,
        req_id="req",
        gid=0,
    )
    # Sparse scoring must run (NOT recency fallback).
    assert result["mode"] == "per_head"
    assert result.get("semantic", "sparse_scoring") != "recency_fallback_stats_incompatible"


def test_partial_rope_rotary_dim_inferred_from_inv_freq(tmp_path):
    """Partial-RoPE geometry must come from explicit metadata fields.

    The runtime does NOT infer rotary_dim from inv_freq length alone (that
    would be ambiguous). Stats files for partial-RoPE models must carry
    ``partial_rotary_factor`` / ``rotary_dim`` / ``freq_count`` explicitly,
    matching the vLLM-Ascend 0.18.0 production contract. Without those fields,
    a short inv_freq is treated as a genuine mismatch and ``_init_rope`` raises.
    """
    freq_count = 32
    head_dim = 256
    num_kv_heads = 4
    num_layers = 1

    layer_stats = {
        0: {
            "q_mean_complex": torch.randn(num_kv_heads, freq_count, 2),
            "q_abs_mean": torch.rand(num_kv_heads, freq_count) + 0.1,
            "freq_scale_sq": torch.rand(num_kv_heads, freq_count) + 0.1,
        },
    }
    stats_path = tmp_path / "partial_rope_inferred.pt"
    torch.save(
        {
            "metadata": {
                "num_attention_heads": 24,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_layers": num_layers,
                "rope_style": "half",
                "inv_freq": torch.randn(freq_count),  # NO rotary_dim/freq_count/partial_rotary_factor
            },
            "layer_stats": layer_stats,
        },
        stats_path,
    )

    from triattention.vllm.core.config import TriAttentionConfig
    from triattention.vllm.core.compressor import TriAttentionCompressor

    cfg = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=16,
        divide_length=8,
        window_size=4,
        pruning_mode="per_head",
        use_triton_scoring=False,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    comp = TriAttentionCompressor(cfg)
    # Without explicit partial-RoPE metadata, a 32-element inv_freq on a
    # head_dim=256 model is treated as a mismatch and _init_rope raises.
    raised = False
    try:
        comp._lazy_init()
    except ValueError as exc:
        raised = "fewer elements" in str(exc)
    assert raised, "expected ValueError about inv_freq length without explicit partial-RoPE metadata"


def test_partial_rope_explicit_freq_count_field(tmp_path):
    """Stats with explicit rotary_dim/freq_count/partial_rotary_factor score."""
    freq_count = 32
    head_dim = 256
    rotary_dim = 64
    num_kv_heads = 4
    num_layers = 1

    layer_stats = {
        0: {
            "q_mean_complex": torch.randn(num_kv_heads, freq_count, 2),
            "q_abs_mean": torch.rand(num_kv_heads, freq_count) + 0.1,
            "freq_scale_sq": torch.rand(num_kv_heads, freq_count) + 0.1,
        },
    }
    stats_path = tmp_path / "partial_rope_explicit.pt"
    torch.save(
        {
            "metadata": {
                "num_attention_heads": 24,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_layers": num_layers,
                "rope_style": "half",
                "rotary_dim": rotary_dim,
                "freq_count": freq_count,
                "partial_rotary_factor": 0.25,
                "inv_freq": [0.1 * (i + 1) for i in range(freq_count)],
            },
            "layer_stats": layer_stats,
        },
        stats_path,
    )

    from triattention.vllm.core.config import TriAttentionConfig
    from triattention.vllm.core.compressor import TriAttentionCompressor

    cfg = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=16,
        divide_length=8,
        window_size=4,
        pruning_mode="per_head",
        use_triton_scoring=False,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    comp = TriAttentionCompressor(cfg)
    comp._lazy_init()
    assert cfg.rotary_dim == 64
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-6
    assert cfg.freq_count == 32
    assert tuple(comp.inv_freq.shape) == (32,)
    assert tuple(comp.freq_scale_sq.shape) == (num_layers, num_kv_heads, freq_count)
