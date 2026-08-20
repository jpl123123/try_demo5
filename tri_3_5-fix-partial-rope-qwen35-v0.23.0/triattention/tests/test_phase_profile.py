from triattention.vllm.runtime.phase_profile import (
    TriAttentionPhaseProfile,
    make_timed_wrapper,
    phase_profile_enabled,
    reset_phase_profile_for_tests,
)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


def test_phase_profile_logs_top_phases_and_details():
    logger = _Logger()
    profile = TriAttentionPhaseProfile(
        logger=logger,
        enabled=True,
        log_every_calls=2,
    )

    profile.record_phase(
        "base_runner_execute_model",
        10.0,
        {"num_reqs": 16, "total_tokens": 16, "overrides": 1},
    )
    profile.record_phase(
        "ascend_v2_build_attn_metadata",
        2.0,
        {"max_seq_in": 40960, "max_seq_out": 2048},
    )

    line = logger.lines[-1]
    assert "TRIATTN_PHASE_PERF calls=2" in line
    assert "base_runner_execute_model:calls=1,avg=10.00" in line
    assert "ascend_v2_build_attn_metadata:calls=1,avg=2.00" in line
    assert "max_seq_in=40960" in line
    assert "max_seq_out=2048" in line


def test_phase_profile_logs_model_submodule_top_phases():
    logger = _Logger()
    profile = TriAttentionPhaseProfile(
        logger=logger,
        enabled=True,
        log_every_calls=4,
        top_n=2,
    )

    profile.record_phase("ascend_v1_model_forward", 30.0)
    profile.record_phase(
        "model_layer_forward[layer=0]",
        11.0,
        {"layer": 0, "kind": "layer"},
    )
    profile.record_phase(
        "model_self_attn_forward[layer=0]",
        7.0,
        {"layer": 0, "kind": "self_attn"},
    )
    profile.record_phase("model_mlp_forward[layer=0]", 3.0, {"layer": 0})

    line = logger.lines[-1]
    assert "model_top_total=model_layer_forward[layer=0]:calls=1,avg=11.00" in line
    assert "model_self_attn_forward[layer=0]:calls=1,avg=7.00" in line
    assert "model_mlp_forward[layer=0]:calls=1,avg=3.00" not in line
    assert "model_top_avg=model_layer_forward[layer=0]:calls=1,avg=11.00" in line


def test_phase_profile_reports_installed_model_probes_without_records():
    logger = _Logger()
    profile = TriAttentionPhaseProfile(
        logger=logger,
        enabled=True,
        log_every_calls=1,
    )

    profile.register_model_probes(["layer[0].forward", "layer[0].mlp.forward"])
    profile.record_phase("ascend_v1_model_forward", 30.0)

    line = logger.lines[-1]
    assert "model_probe_status=installed_no_records,installed=2,recorded=0" in line
    assert "model_top_total=" not in line


def test_phase_profile_reports_active_model_probes():
    logger = _Logger()
    profile = TriAttentionPhaseProfile(
        logger=logger,
        enabled=True,
        log_every_calls=2,
    )

    profile.register_model_probes(["layer[0].forward"])
    profile.record_phase("ascend_v1_model_forward", 30.0)
    profile.record_phase("model_layer_forward[layer=0]", 11.0)

    line = logger.lines[-1]
    assert "model_probe_status=active,installed=1,recorded=1" in line
    assert "model_top_total=model_layer_forward[layer=0]:calls=1,avg=11.00" in line


def test_phase_profile_top_n_env(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_PROFILE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_TOP_N", "17")
    reset_phase_profile_for_tests()

    assert TriAttentionPhaseProfile.from_env().top_n == 17
    reset_phase_profile_for_tests()


def test_phase_profile_has_dedicated_env_gate(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_PERF_PROFILE", "1")
    monkeypatch.delenv("TRIATTN_RUNTIME_PHASE_PROFILE", raising=False)
    reset_phase_profile_for_tests()

    assert not phase_profile_enabled()

    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_PROFILE", "1")
    reset_phase_profile_for_tests()

    assert phase_profile_enabled()
    reset_phase_profile_for_tests()


def test_timed_wrapper_preserves_fast_path_when_phase_profile_disabled(monkeypatch):
    calls = []

    def original(value):
        calls.append(value)
        return value + 1

    monkeypatch.delenv("TRIATTN_RUNTIME_PHASE_PROFILE", raising=False)
    reset_phase_profile_for_tests()

    wrapped = make_timed_wrapper("sample_phase", original)

    assert wrapped(41) == 42
    assert calls == [41]
    reset_phase_profile_for_tests()
