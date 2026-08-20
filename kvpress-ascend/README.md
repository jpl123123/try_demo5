# kvpress-ascend: KVPress monkeypatch adapter for vLLM-Ascend v0.23.0

Monkeypatch integration that adapts the kvpress KV-cache compression mechanism to
vLLM-Ascend (`vllm-ascend-releases-v0.23.0`) **without modifying any vLLM-Ascend
source**. See the root `PLAN.md` / `TODO.md` for the full design and
mechanism-conversion table, and `tri_3_5-fix-partial-rope-qwen35-v0.23.0` for the
reference integration this port is modeled on.

## Install

```bash
pip install ./kvpress-ascend
```

## Activation

```bash
export KVPRESS_ENABLE=1        # or: export KVPRESS=1
vllm serve <model> ...         # your normal vLLM-Ascend launch, unchanged
```

> Note: `export kvpress` is not valid shell syntax; use `export KVPRESS_ENABLE=1`
> (or the `KVPRESS=1` alias) before launching. With the switch off the plugin is
> a no-op, so it can stay installed alongside SqueezeAttention-ascend.

## Per-inference verification switch (is the patch really live?)

```bash
export KVPRESS_PROBE=1         # default on
export KVPRESS_RUNTIME_LOGGING=1
```

Every model step prints one probe line per scheduled request:

```
[KVPRESS-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  press=KnormPress ratio=0.500 seq_len=2048 budget=1024 keep=1024
  reclaimed_blocks=8 compress_events=1 last_event=applied
```

- `core_entered=1` — the runner-proxy `execute_model` interception ran (the
  kvpress compression boundary flow is active).
- `hook_entered=1` — the attention hooks captured at least one layer's query
  this step (the kvpress scoring input path is live).
- Both `0` ⇒ the patch is not active: check `KVPRESS_ENABLE`, the install, and
  the `[KVPRESS-ASCEND]` startup logs.

Compression events additionally log `[KVPRESS-ASCEND][PROBE] COMPRESS ...` with
`before/after/retained/reclaimed_blocks/groups/layers_compacted`.

## Mechanism conversion summary

| kvpress (HF transformers) | kvpress-ascend (vLLM-Ascend) |
|---|---|
| `BasePress.__call__` hooks on every `self_attn` | `AttentionHooks` on vLLM decoder-layer attention modules (captures post-RoPE queries) |
| `extract_keys_and_values` from `DynamicCache` (dense `[bsz,H,T,D]`) | `kv_layout.gather_request_kv_dense` from the Ascend paged block cache |
| `ScorerPress.compress` topk gather into a new dense tensor | per-head keep indices (sorted) + in-place per-head compaction of the request's existing blocks |
| compressed `cache_layer.keys` write-back (physical shrink) | block-row shrink + scheduler-side block reclaim |
| `cache_position` re-mapping | effective-length tracker + `_prepare_inputs` overrides (seq_lens / positions / slot mapping) |
| `is_prefilling(cache_position, q_len)` | scheduler/worker prefill-phase gates (`KVPRESS_DEFER_PREFILL_COMPRESSION`) |
| `DecodingPress` interval + target size | same semantics via `KVPRESS_TARGET_SIZE` / `KVPRESS_COMPRESSION_INTERVAL` |

## Environment variables

| Env | Default | Meaning |
|---|---|---|
| `KVPRESS_ENABLE` / `KVPRESS` | off | plugin activation |
| `KVPRESS_PRESS` | `KnormPress` | `KnormPress` \| `StreamingLLMPress` \| `RandomPress` \| `SnapKVPress` \| `TOVAPress` \| `ObservedAttentionPress` \| `DecodingPress` |
| `KVPRESS_COMPRESSION_RATIO` | 0.5 | press compression ratio (kept = `(1-ratio)*len`) |
| `KVPRESS_TARGET_SIZE` | 0 | decode target KV tokens (DecodingPress); 0 = ratio-driven |
| `KVPRESS_WINDOW_SIZE` | 64 | SnapKV window |
| `KVPRESS_SINK_TOKENS` | 4 | StreamingLLM sink tokens |
| `KVPRESS_COMPRESSION_INTERVAL` | 512 | DecodingPress interval |
| `KVPRESS_KV_BUDGET` | 0 | absolute per-request budget (overrides ratio when > 0) |
| `KVPRESS_MIN_RECLAIM_BLOCKS` | 1 | minimum freed blocks to run a compaction |
| `KVPRESS_MAX_COMPRESSIONS_PER_STEP` | 1 | guard |
| `KVPRESS_MAX_LAYERS_TO_SCORE` | 0 | 0 = score all layers; N = sampled scoring (nearest-layer keep reuse) |
| `KVPRESS_DEFER_PREFILL_COMPRESSION` | 1 | compress only after full prefill (stable Ascend mode) |
| `KVPRESS_USE_INSTALLED` | 1 | prefer installed `kvpress` classes, else vendored fallback |
| `KVPRESS_RUNTIME_LOGGING` | 1 | master logging |
| `KVPRESS_PROBE` | 1 | per-step core-entry probe logs |
| `KVPRESS_LOG_ATTENTION_HOOK` | 0 | per-layer hook capture debug logs |
| `KVPRESS_LOG_DECISIONS` | 0 | signal/decision debug logs |

Mask-based (head-wise different counts) presses such as `AdaKVPress`,
`PyramidKVPress`, `CriticalKVPress` are rejected at startup with an explicit log:
Ascend attention kernels cannot express per-head masking, so those presses are
not physically realizable here.

## Simulated-debug tests (no NPU needed)

```bash
pip install pytest
cd kvpress-ascend/tests && python -m pytest ../
```

The suite ships stub modules mirroring the vllm-ascend v0.23.0 surfaces
(`NPUWorker`, `NPUModelRunner`, `BlockTable`/`MultiGroupBlockTable`, V1
scheduler, KV cache manager) and runs the real patch code on CPU: plugin
activation, scheduler signals, in-place compaction content checks, block
reclaim, `_prepare_inputs` overrides, probe logging switches, MTP multi-group
handling, and both-plugins coexistence.
