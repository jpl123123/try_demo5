from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.perf_profile import TriAttentionPerfProfile
from triattention.vllm.runtime.phase_profile import (
    phase_profile_enabled,
    reset_phase_profile_for_tests,
)


class _Logger:
    def info(self, fmt, *args):
        raise AssertionError("profile logging should be disabled")


def test_runtime_logging_defaults_keep_decision_logs_but_quiet_exec_path(monkeypatch):
    monkeypatch.delenv("TRIATTN_RUNTIME_LOGGING", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_DECISIONS", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_CORE_TRACE", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS", raising=False)

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert config.log_decisions
    assert not config.log_execution_path
    assert not config.log_execution_path_core_only
    assert not config.log_core_trace
    assert not config.log_selector_debug
    assert not config.log_all_worker_events


def test_runtime_logging_master_overrides_verbose_subswitches(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_DECISIONS", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_CORE_TRACE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS", "1")

    config = TriAttentionRuntimeConfig.from_env()

    assert not config.logging_enabled
    assert not config.log_decisions
    assert not config.log_execution_path
    assert not config.log_execution_path_core_only
    assert not config.log_core_trace
    assert not config.log_selector_debug
    assert not config.log_all_worker_events


def test_runtime_execution_path_log_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_CORE_TRACE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG", "1")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert not config.log_execution_path
    assert not config.log_execution_path_core_only
    assert not config.log_core_trace
    assert not config.log_selector_debug


def test_runtime_execution_path_core_only_can_be_enabled(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY", "1")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert config.log_execution_path
    assert config.log_execution_path_core_only


def test_runtime_verbose_trace_streams_require_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_CORE_TRACE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG", "1")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert config.log_execution_path
    assert config.log_core_trace
    assert config.log_selector_debug


def test_runtime_logging_master_disables_perf_profiles(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_PERF_PROFILE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_E2E_PROFILE", "1")

    profile = TriAttentionPerfProfile.from_env(_Logger())

    assert not profile.enabled
    assert not profile.e2e_enabled


def test_runtime_logging_master_disables_phase_profile(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_PROFILE", "1")
    reset_phase_profile_for_tests()

    assert not phase_profile_enabled()

    reset_phase_profile_for_tests()
