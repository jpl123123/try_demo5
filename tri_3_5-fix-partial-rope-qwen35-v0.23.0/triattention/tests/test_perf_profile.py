from triattention.vllm.runtime.perf_profile import TriAttentionPerfProfile


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


def test_perf_profile_counts_applied_and_skipped_events():
    logger = _Logger()
    profile = TriAttentionPerfProfile(
        logger=logger,
        enabled=True,
        log_every_steps=1,
    )

    profile.record_compression_events(
        [
            {
                "status": "applied",
                "reason": "kv_compacted:zero_copy_tail",
                "details": {
                    "selector_status": "enabled:recency_only",
                    "reclaimed_block_count": 66,
                    "block_reclaim": {"mode": "remap_tail"},
                },
                "block_reclaim": {"mode": "remap_tail"},
            },
            {
                "status": "skipped",
                "reason": "fast_recency_long_context_guard",
                "details": {},
            },
        ]
    )
    profile.record_step(
        has_trigger=True,
        uses_overrides=True,
        t_state_ms=1.0,
        t_compress_ms=2.0,
        t_reclaim_ms=3.0,
        t_override_prep_ms=4.0,
        t_base_exec_ms=5.0,
        t_total_exec_ms=6.0,
    )

    assert profile.compress_calls == 2
    assert profile.compress_applied == 1
    assert profile.compress_skipped == 1
    assert profile.reclaimed_blocks == 66
    line = logger.lines[-1]
    assert "top_apply_reasons=kv_compacted:zero_copy_tail:1" in line
    assert "top_skip_reasons=fast_recency_long_context_guard:1" in line
    assert "reclaim_modes={'remap_tail': 1}" in line


def test_e2e_profile_logs_top_phases_and_last_step():
    logger = _Logger()
    profile = TriAttentionPerfProfile(
        logger=logger,
        e2e_enabled=True,
        e2e_log_every_steps=2,
    )

    profile.record_e2e_step(
        {
            "register_new_requests": 1.0,
            "base_execute_model": 10.0,
            "execute_model_total": 12.0,
        },
        num_reqs=1,
        total_tokens=16,
        has_trigger=True,
        uses_overrides=True,
        pending_events=2,
    )
    assert logger.lines == []

    profile.record_e2e_sample(
        {
            "base_sample_tokens": 3.0,
            "attach_sample_tokens_events": 0.5,
        },
        pending_events=0,
    )
    profile.record_e2e_step(
        {
            "register_new_requests": 2.0,
            "base_execute_model": 20.0,
            "execute_model_total": 24.0,
        },
        num_reqs=2,
        total_tokens=32,
        has_trigger=False,
        uses_overrides=True,
        pending_events=0,
    )

    line = logger.lines[-1]
    assert line.startswith("TRIATTN_E2E_PERF steps=2 samples=1")
    assert "num_reqs=2" in line
    assert "total_tokens=32" in line
    assert "top_total=execute_model_total:calls=2,avg=18.00,max=24.00,total=36.00" in line
    assert "base_execute_model:calls=2,avg=15.00,max=20.00,total=30.00" in line
    assert "last_step=execute_model_total=24.00|base_execute_model=20.00" in line
    assert "last_sample=base_sample_tokens=3.00|attach_sample_tokens_events=0.50" in line


def test_e2e_profile_has_dedicated_env_gate(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_PERF_PROFILE", "1")
    monkeypatch.delenv("TRIATTN_RUNTIME_E2E_PROFILE", raising=False)
    profile = TriAttentionPerfProfile.from_env(_Logger())
    assert profile.enabled
    assert not profile.e2e_enabled

    monkeypatch.setenv("TRIATTN_RUNTIME_E2E_PROFILE", "1")
    profile = TriAttentionPerfProfile.from_env(_Logger())
    assert profile.e2e_enabled
