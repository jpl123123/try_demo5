# vLLM-Ascend Integration

TriAttention can run through vLLM-Ascend by using the same runtime scheduler and
KV compaction path as the CUDA vLLM backend, with three Ascend-specific changes:

- `vllm_ascend.worker.worker.NPUWorker` is patched so the TriAttention model
  runner proxy is installed after the NPU model runner is created.
- vLLM-Ascend input preparation is patched so compressed KV length is reflected
  in NPU `seq_lens`, CPU `seq_lens_np`, and slot mappings. Without this,
  attention metadata can keep reading the original long context after KV has
  already been compacted, which commonly shows up as repeated tokens.
- NPU execution defaults to PyTorch/torch_npu scoring instead of the CUDA
  Triton scoring kernel.

The dense Ascend KV layout is supported in both forms:

- vLLM CUDA-style combined cache: `[2, num_blocks, block_size, num_kv_heads, head_dim]`
- vLLM-Ascend split cache: `(k_cache, v_cache)` where each tensor is
  `[num_blocks, block_size, num_kv_heads, head_dim]`

## Installation

Install vLLM and vLLM-Ascend first, then install this package in the same Python
environment:

```bash
pip install -e .
```

The vLLM plugin entry point activates automatically. You can disable it with:

```bash
export ENABLE_TRIATTENTION=0
```

## Server Example

```bash
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/model_stats.pt
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_DIVIDE_LENGTH=128
export TRIATTN_RUNTIME_WINDOW_SIZE=128
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=1
export TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY=0
export TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND=1
export TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH=1
export TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1
export TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND=0
export TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE=0

# auto = Triton on CUDA, PyTorch/torch_npu on NPU.
export TRIATTN_RUNTIME_SCORING_BACKEND=auto

vllm serve /path/to/model \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 1024
```

Recommended first-run settings:

- Use `--enforce-eager` while validating correctness and memory behavior.
- Disable prefix caching because compressed KV entries no longer match vLLM's
  original prefix-cache block hashes.
- Keep `--max-num-batched-tokens` modest so prefill chunks do not overshoot the
  KV budget before the compression boundary is reached.
- On Ascend, `TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=1` is the
  default. Compression is first applied after the full prompt prefill has
  completed, which is the most stable mode for long prompts on NPU attention
  backends. Set it to `0` only after validating streaming prefill compression on
  your vLLM-Ascend version.

## Scoring Backend

`TRIATTN_RUNTIME_SCORING_BACKEND` accepts:

| Value | Behavior |
|-------|----------|
| `auto` | Uses CUDA Triton on CUDA devices and PyTorch/torch_npu on NPU devices |
| `torch` / `pytorch` | Forces the PyTorch scoring path |
| `triton` | Forces the CUDA Triton scoring path |

On vLLM-Ascend, leave this as `auto` or set it to `torch`.

For correctness, the PyTorch/torch_npu scoring path explicitly promotes KV keys,
Q statistics, RoPE frequencies, and frequency scales to `float32` before scoring.
The KV cache itself remains in the model dtype; only the transient scoring chunks
are promoted.

## Runtime Logging Controls

Use one master switch for quiet performance runs:

```bash
export TRIATTN_RUNTIME_LOGGING=0
```

This suppresses TriAttention startup, scheduler/worker decision, compression
event, execution-path trace, and `TRIATTN_PERF` / `TRIATTN_E2E_PERF` /
`TRIATTN_PHASE_PERF` profile logs. It overrides the detailed log/profile
switches below. Error and safety warning logs are still emitted.

For debugging, enable the master switch and only the streams needed for the run:

```bash
export TRIATTN_RUNTIME_LOGGING=1
export TRIATTN_RUNTIME_LOG_DECISIONS=1
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH=1
export TRIATTN_RUNTIME_LOG_CORE_TRACE=0
export TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG=0
export TRIATTN_RUNTIME_PERF_PROFILE=1
export TRIATTN_RUNTIME_E2E_PROFILE=1
export TRIATTN_RUNTIME_PHASE_PROFILE=1
```

To show only the core execution path and avoid repetitive expected-skip logs,
use:

```bash
export TRIATTN_RUNTIME_LOGGING=1
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH=1
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY=1
export TRIATTN_RUNTIME_LOG_CORE_TRACE=0
export TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG=0
export TRIATTN_RUNTIME_LOG_DECISIONS=0
export TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS=0
export TRIATTN_RUNTIME_PERF_PROFILE=0
export TRIATTN_RUNTIME_E2E_PROFILE=0
export TRIATTN_RUNTIME_PHASE_PROFILE=0
```

When a single reproduction needs the full selector/compaction trace, opt into
the expensive streams explicitly:

```bash
export TRIATTN_RUNTIME_LOG_CORE_TRACE=1
export TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG=1
```

Leave those off for performance runs. They can serialize per-layer lists and
nested selector payloads once per worker/rank.

## Expected Logs

With `TRIATTN_RUNTIME_LOGGING=1`, look for these startup log lines:

```text
[TriAttention] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True
Installed TriAttention runtime worker patches for Ascend: vllm_ascend.worker.worker.NPUWorker
Installed TriAttention runtime input patches: ... vllm_ascend.worker.model_runner_v1.NPUModelRunner ...
```

Recent builds also include `build=ascend-core-stats-default-v21-20260603` in the
plugin, scheduler, and worker logs. If that build id is missing, the running
container is still loading an older installed package or stale source path.
If the `vllm_ascend.worker.worker.NPUWorker` patch line is absent, check for a
nearby warning about importing or patching `NPUWorker`; recent builds report
that failure instead of silently relying on the upstream GPU worker patch.

Compression events should report a status like `selector_status=enabled:torch:tp=1/2`
when the first compression boundary is reached on NPU. The `tp=rank/size`
suffix confirms that runtime scoring is using this worker's tensor-parallel
head shard. On vLLM-Ascend with `TRIATTN_RUNTIME_SCORING_BACKEND=auto`, the
status should say `enabled:torch`, not `enabled:triton`.

With `TRIATTN_RUNTIME_LOG_EXECUTION_PATH=1`, each real compression boundary
also emits `TRIATTN_EXEC_PATH` markers. Look for the sequence
`runner_execute_model_compression_boundary`, `worker_hook_enter`,
`group_pipeline_enter`, and `selector_scoring_enter`; the selector marker
prints `backend=torch` on Ascend and `trig_enabled=True` when the sparse
TriAttention trigonometric scoring path is active. It prints layer counts
instead of full layer lists unless selector debug is explicitly enabled.

If no real compression boundary is reached because a safety guard suppresses
the trigger, the runner emits `TRIATTN_EXEC_PATH runner_trigger_guard` with the
reason and key lengths. These are pre-core guard logs: `core_entered=False`
means no `worker_hook_enter`, `group_pipeline_enter`, or
`selector_scoring_enter` marker is expected for that skipped boundary. For
example, pure recency diagnostics with
`TRIATTN_RUNTIME_FAST_RECENCY_ONLY=1`, no sparse stats accuracy guard, and the
explicitly enabled long-context guard report
`reason=fast_recency_long_context_guard` above
`TRIATTN_RUNTIME_FAST_RECENCY_LONG_CONTEXT_GUARD_TOKENS` instead of entering
`worker_hook_enter`; the log hint points to either making sparse stats visible
to the worker process or disabling the long-context guard for a pure recency
diagnostic run.

With `TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY=1`, high-frequency
`runner_execute_model_compression_boundary`, `runner_executor_enter`,
`worker_hook_runtime_context`, and expected skip result logs are suppressed.
The important markers remain: `hook_installed`, `runner_trigger_guard`,
`worker_hook_enter`, `zero_copy_tail_enter`, `group_pipeline_enter`,
`selector_scoring_enter`, and applied or unexpected result logs.
Verbose `TRIATTN_CORE_TRACE` enter/exit logs remain off unless
`TRIATTN_RUNTIME_LOG_CORE_TRACE=1` is set.

For sparse-stat accuracy runs on Ascend, this is the recommended minimum
configuration:

```bash
export TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1
export TRIATTN_RUNTIME_LOGGING=1
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH=1
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY=1
export TRIATTN_RUNTIME_LOG_CORE_TRACE=0
export TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG=0
export TRIATTN_RUNTIME_LOG_DECISIONS=0
export TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS=0
export TRIATTN_RUNTIME_PERF_PROFILE=0
export TRIATTN_RUNTIME_E2E_PROFILE=0
export TRIATTN_RUNTIME_PHASE_PROFILE=0
```

With that configuration, the runtime prefers packaged sparse stats when no
explicit `TRIATTN_RUNTIME_SPARSE_STATS_PATH` is set. `FAST_RECENCY_ONLY=1` is
no longer required for Ascend validation, and when it is set the accuracy guard
still pulls the run back to sparse TriAttention selection if packaged or
explicit stats are available. After `worker_hook_enter`, the next core markers
should be `group_pipeline_enter` and `selector_scoring_enter`. If a run still
reports `zero_copy_recency_not_ready`, it is on the pure-recency diagnostic
path; confirm that `TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1` is visible to
the worker process and that no explicit missing stats path overrides the
packaged stats.

For a long prompt on Ascend, it is normal to see skipped compression events with
`reason=prefill_incomplete` during chunked prefill. The first real compression
should happen once the prompt has finished prefill and decode starts.

If proxy injection is visible but no compression line appears, the runtime now
backfills request state from the NPU runner and allows the first decode step to
trigger compression even when vLLM-Ascend's scheduler counters still lag behind
the full prompt length.

The default `TRIATTN_RUNTIME_MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND=0`
keeps that first eligible decode boundary active, so a real sparse run should
reach `worker_hook_enter`, `group_pipeline_enter`, and
`selector_scoring_enter` as soon as the prompt has finished prefill and the
length/reclaim threshold is satisfied. If a deployment needs to delay the first
post-prefill compaction for TPOT experiments, set this variable explicitly.

## Performance Tuning

On Ascend, the default path keeps sparse-stat per-head selection and caps
scoring to 8 sampled layers by default so sparse selection does not dominate
decode latency. The runtime still avoids the expensive full `[kept + dropped]`
KV rewrite when tail blocks are physically reclaimed. For latency-sensitive
serving after validating output quality, try:

```bash
export TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND=8
export TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND=16
export TRIATTN_RUNTIME_PREFILL_MIN_RECLAIM_BLOCKS_ON_ASCEND=32
export TRIATTN_RUNTIME_PREFILL_MAX_COMPRESSIONS_ON_ASCEND=1
export TRIATTN_RUNTIME_SPARSE_NORMALIZE_SCORES=0
export TRIATTN_RUNTIME_PERF_PROFILE=1
export TRIATTN_RUNTIME_PERF_LOG_EVERY=50
```

`TRIATTN_RUNTIME_PERF_PROFILE=1` keeps the low-overhead aggregated
`TRIATTN_PERF` counters enabled when `TRIATTN_RUNTIME_LOGGING=1`. To localize end-to-end blocking time without
installing method wrappers, enable the E2E profiler:

```bash
export TRIATTN_RUNTIME_E2E_PROFILE=1
export TRIATTN_RUNTIME_E2E_LOG_EVERY=50
```

`TRIATTN_E2E_PERF` reports aggregated `top_total` and `top_avg` timing across
the runner boundary, including state updates, compression actions,
effective-length override preparation, base `execute_model`, execute-output
event attachment, and `sample_tokens`. If `base_execute_model` dominates, the
remaining bottleneck is inside the backend model runner rather than the
TriAttention orchestration path.

Deeper per-phase runner and model probes are diagnostic-only and add Python
wrapper overhead on the decode hot path, so they are controlled separately.
Enable them only after `TRIATTN_E2E_PERF` shows the backend runner itself is the
blocking phase:

```bash
export TRIATTN_RUNTIME_PHASE_PROFILE=1
export TRIATTN_RUNTIME_PHASE_LOG_EVERY=500
export TRIATTN_RUNTIME_PHASE_TOP_N=16
export TRIATTN_RUNTIME_MODEL_PHASE_PROBES=1
```

Phase logs include `model_top_total` and `model_top_avg` when model submodule
probes are installed. Use those fields to distinguish a layer, attention, or
MLP hotspot after `ascend_v1_model_forward` dominates the backend runner.

`TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND` now has an effective Ascend default
of `8`. It is used only when `TRIATTN_RUNTIME_SCORE_MAX_LAYERS=0`. Set
`SCORE_MAX_LAYERS` explicitly to force a value for every backend. Set the
Ascend value to `0` explicitly to score all layers, or try `4` if quality is
stable and latency still needs more headroom. The runtime log will include
`selector_status=enabled:torch:tp=...:score_layers=max8,stride1`.

`TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND` prevents very small compactions
such as `2175 -> 2048` from running on NPU. With `--block-size 128`, the sparse
default `16` means compression waits until it can reclaim about 2048 KV tokens,
which better amortizes scoring, KV movement, and scheduler/worker
synchronization. Pure fast-recency keeps the cheaper `8` block interval.

`TRIATTN_RUNTIME_PREFILL_MIN_RECLAIM_BLOCKS_ON_ASCEND` applies only to
scheduled prefill chunks. The default `32` requires roughly 4096 reclaimable KV
tokens with `--block-size 128`, so small prefill chunks such as
`4096 -> 2048` are not repeatedly scored and compacted. The companion
`TRIATTN_RUNTIME_PREFILL_MAX_COMPRESSIONS_ON_ASCEND=1` allows at most one
prefill compaction per request; set it to `0` to disable prefill compaction
while keeping decode-time compaction enabled.

For maximum TTFT improvement on very long prompts, validate prefill compression
after confirming the build id above:

```bash
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=0
export TRIATTN_RUNTIME_PREFILL_MIN_RECLAIM_BLOCKS_ON_ASCEND=32
export TRIATTN_RUNTIME_PREFILL_MAX_COMPRESSIONS_ON_ASCEND=1
```

This allows one early compression before later prefill chunks, so the remaining
prompt prefill can benefit from the shorter KV without repeatedly paying the
sparse scoring cost. Keep it disabled if your target vLLM-Ascend build shows
any quality regression.

To isolate selector overhead from the NPU attention speedup, run one benchmark
with the recency-only selector and explicitly disable the sparse accuracy
guard:

```bash
export TRIATTN_RUNTIME_FAST_RECENCY_ONLY=1
export TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=0
```

This skips sparse-stat scoring and keeps the newest `KV_BUDGET` tokens. Without
the explicit guard disable, packaged or explicit sparse stats keep the accuracy
guard enabled and use sparse-stat TriAttention selection, which emits
`group_pipeline_enter` and `selector_scoring_enter` instead of the recency-only
zero-copy tail path. The long-context guard defaults off so diagnostic runs
still enter the runtime hook on 20k+ prompts; set
`TRIATTN_RUNTIME_FAST_RECENCY_LONG_CONTEXT_GUARD=1` only when you explicitly
want to suppress pure-recency compression above the guard threshold.

When `KV_BUDGET` is a multiple of `--block-size`, vLLM-Ascend uses a zero-copy
tail block remap by default instead of copying KV tensors; the expected
compression reason is `kv_compacted:zero_copy_tail` or a compression log with
`selector=enabled:recency_only reclaim=remap_tail`. To compare against the older
copy path, set `TRIATTN_RUNTIME_ENABLE_ZERO_COPY_RECENCY=0`.

On Ascend, `TRIATTN_RUNTIME_ZERO_COPY_RECENCY_ONLY_ON_ASCEND=1` is also enabled
by default. If a prefill or decode step cannot safely remap the newest blocks
yet, the runtime skips that compression attempt and waits for the next
zero-copy opportunity instead of falling back to `reclaim=truncate_tail`.

For correctness on long prompts, keep
`TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1`, so packaged or explicit sparse
stats drive sparse-stat TriAttention selection instead of pure recency. If
`FAST_RECENCY_ONLY=1` is set and an explicitly configured stats file is missing,
the runtime stays on the pure-recency diagnostic path instead of failing during
Ascend worker init. Pure recency is a performance diagnostic and can degrade
quality on 20k+ prompts.

The async compression boundary is disabled by default because it can repeatedly
block vLLM's batch-queue lookahead during generation. Re-enable it only for
debugging with `TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY=1`.

On Ascend, the runner proxy and input patches are installed during worker
initialization by default. This avoids first-request patch installation and ACL
graph replay in the measured request path. To compare against the older lazy
behavior, set `TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND=0` and
`TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH=0`.

`TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND` is disabled by default. It
is an experimental slot-mapping micro-optimization and should only be enabled
after validating output quality for the target vLLM-Ascend build.

`TRIATTN_RUNTIME_FORCE_EAGER_MULTI_REQ_ON_ASCEND_EFFECTIVE_OVERRIDES=1` is
enabled by default. It only applies to multi-request batches after a request has
compressed KV state and needs effective seq-len/slot-position overrides. Those
batches are run outside vLLM-Ascend graph/compiled mode to avoid ACL graph
replay/update instability; single-request batches keep the normal graph-mode
path. Set it to `0` only for targeted graph-mode experiments on a validated
vLLM-Ascend build.

`TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE=0` is the default. The effective
seq-len and slot-mapping overrides already describe the compressed KV view, so
the Ascend V1 block table keeps its full preallocated column shape for
multi-request serving stability. Set this to `1` only for targeted
single-request microbenchmarks after validating the target vLLM-Ascend build.

## Calibration Stats

TriAttention still requires a model-specific statistics file produced by
`scripts/calibrate.py`. The stats should match the model architecture and RoPE
layout used for serving. See [Calibration Guide](calibration.md).

## Current Limits

- Dense attention KV caches are the primary supported Ascend path.
- Tensor parallel serving is supported by slicing calibration statistics to the
  local TP head shard before TopK selection.
- Ascend sparse attention or MLA layouts may attach extra tensors after
  `(k_cache, v_cache)`. The current compaction path moves the dense K/V tensors
  and should be validated before using those model families in production.
- `NPUWorker310` and `XliteWorker` init hooks are patched on a best-effort basis,
  but the main validated target is `vllm_ascend.worker.worker.NPUWorker`.
