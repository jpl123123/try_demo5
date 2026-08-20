from triattention.vllm.runtime.ascend_defaults import (
    apply_ascend_fast_recency_defaults,
)
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig


def test_auto_fast_recency_keeps_sparse_scoring_default():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert not config.fast_recency_only
    assert config.fast_recency_accuracy_guard
    assert config.score_max_layers == 8
    assert config.score_max_layers_on_ascend == 8
    assert config.min_reclaim_blocks_on_ascend == 16


def test_auto_fast_recency_respects_explicit_user_mode():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ONLY": "0"},
    )

    assert not config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_auto_fast_recency_respects_accuracy_guard():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD": "1"},
    )

    assert config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_explicit_fast_recency_from_env_uses_packaged_stats(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_FAST_RECENCY_ONLY", "1")
    monkeypatch.delenv("TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_SPARSE_STATS_PATH", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_MODEL_PATH", raising=False)

    config = TriAttentionRuntimeConfig.from_env()

    assert config.fast_recency_only
    assert config.fast_recency_accuracy_guard
    assert config.sparse_stats_path is not None
    assert config.sparse_stats_path.name == "qwen3_32b_int4_stats.pt"
    assert config.sparse_stats_path.exists()


def test_packaged_stats_match_gpt_oss_model_hint(monkeypatch):
    monkeypatch.delenv("TRIATTN_RUNTIME_SPARSE_STATS_PATH", raising=False)
    monkeypatch.setenv("TRIATTN_RUNTIME_MODEL_PATH", "/models/gpt-oss-120b")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.sparse_stats_path is not None
    assert config.sparse_stats_path.name == "gpt_oss_120b_stats.pt"
    assert config.sparse_stats_path.exists()


def test_missing_env_stats_path_falls_back_to_packaged_stats(monkeypatch):
    monkeypatch.setenv(
        "TRIATTN_RUNTIME_SPARSE_STATS_PATH",
        "/tmp/triattention-missing-stats.pt",
    )
    monkeypatch.delenv("TRIATTN_RUNTIME_MODEL_PATH", raising=False)

    config = TriAttentionRuntimeConfig.from_env()

    assert config.sparse_stats_path is not None
    assert config.sparse_stats_path.name == "qwen3_32b_int4_stats.pt"
    assert config.sparse_stats_path.exists()


def test_explicit_fast_recency_with_stats_keeps_accuracy_guard(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_FAST_RECENCY_ONLY", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_SPARSE_STATS_PATH", "/tmp/triattention-stats.pt")
    monkeypatch.delenv("TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD", raising=False)

    config = TriAttentionRuntimeConfig.from_env()

    assert config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_auto_fast_recency_keeps_documented_decode_reclaim_interval():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert config.min_reclaim_blocks_on_ascend == 8


def test_auto_fast_recency_bounds_ascend_score_layers():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert config.score_max_layers == 8
    assert config.score_max_layers_on_ascend == 8


def test_auto_fast_recency_respects_explicit_score_layer_env():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
        score_max_layers=4,
        score_max_layers_on_ascend=4,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_SCORE_MAX_LAYERS": "4"},
    )

    assert config.score_max_layers == 4
    assert config.score_max_layers_on_ascend == 4


def test_sparse_ascend_defaults_respect_explicit_score_layer_env():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        auto_fast_recency_on_ascend=True,
        score_max_layers=0,
        score_max_layers_on_ascend=4,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND": "4"},
    )

    assert config.score_max_layers == 4
    assert config.score_max_layers_on_ascend == 4


def test_sparse_ascend_defaults_respect_explicit_reclaim_env():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        auto_fast_recency_on_ascend=True,
        min_reclaim_blocks_on_ascend=8,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND": "8"},
    )

    assert config.min_reclaim_blocks_on_ascend == 8


def test_sparse_ascend_defaults_can_be_disabled():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        auto_fast_recency_on_ascend=False,
        score_max_layers=0,
        score_max_layers_on_ascend=0,
        min_reclaim_blocks_on_ascend=8,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert config.score_max_layers == 0
    assert config.score_max_layers_on_ascend == 0
    assert config.min_reclaim_blocks_on_ascend == 8


def test_auto_fast_recency_overrides_stale_reclaim_interval():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
        min_reclaim_blocks_on_ascend=8,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND": "8"},
    )

    assert config.min_reclaim_blocks_on_ascend == 8


def test_auto_fast_recency_can_be_disabled_to_keep_accuracy_guard():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=False,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD": "1"},
    )

    assert config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_early_install_proxy_on_ascend_defaults_to_eager():
    assert TriAttentionRuntimeConfig().early_install_proxy_on_ascend


def test_fast_recency_long_context_guard_defaults_to_core_entry():
    assert not TriAttentionRuntimeConfig().fast_recency_long_context_guard


def test_multi_req_effective_overrides_keep_graph_by_default():
    assert not TriAttentionRuntimeConfig().force_eager_multi_req_on_ascend_effective_overrides


def test_max_compressions_per_step_on_ascend_default_limits_bursts():
    assert TriAttentionRuntimeConfig().max_compressions_per_step_on_ascend == 4


def test_max_compressions_per_step_on_ascend_from_env(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_MAX_COMPRESSIONS_PER_STEP_ON_ASCEND", "8")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.max_compressions_per_step_on_ascend == 8


def test_multi_req_effective_override_guard_from_env(monkeypatch):
    monkeypatch.setenv(
        "TRIATTN_RUNTIME_FORCE_EAGER_MULTI_REQ_ON_ASCEND_EFFECTIVE_OVERRIDES",
        "1",
    )

    config = TriAttentionRuntimeConfig.from_env()

    assert config.force_eager_multi_req_on_ascend_effective_overrides
