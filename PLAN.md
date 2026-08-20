# PLAN — kvpress-ascend & SqueezeAttention-ascend monkeypatch adaptation to vLLM-Ascend v0.23.0

## 1. Mission

Adapt two external KV-cache compression tools to `vllm-ascend-releases-v0.23.0`
**without touching any vLLM-Ascend source code**:

1. `kvpress-main` → new pip-installable package **`kvpress-ascend/`** (module `kvpress_ascend`)
2. `SqueezeAttention-main` → new pip-installable package **`SqueezeAttention-ascend/`** (module `squeezeattention_ascend`)

Both packages are **vLLM general plugins** (`vllm.general_plugins` entry point) that
monkeypatch vLLM/vLLM-Ascend classes at import time. Activation is env-gated:

```bash
export KVPRESS_ENABLE=1     # kvpress-ascend   (alias: export KVPRESS=1)
export SQUEEZE_ENABLE=1     # SqueezeAttention-ascend (alias: export SQUEEZE=1)
```

Each package also carries a **per-inference logging switch** (`*_PROBE=1`) that proves on
every model step that the patch enters its own core code and prints the core parameters.

Reference implementation for every "mechanism conversion" question:
`tri_3_5-fix-partial-rope-qwen35-v0.23.0` (working TriAttention→vLLM-Ascend 0.23.0 port).

## 2. Target deployment (user launch command)

```bash
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name qwen3.5 --host 0.0.0.0 --port 1144 \
  --data-parallel-size 1 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-batched-tokens 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[...], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code --async-scheduling --allowed-local-media-path / \
  --quantization ascend --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{...qwen3.5 rope/mrope...}'
```

Consequences that shape the design (all handled by the port):

| Launch feature | Port impact |
|---|---|
| `qwen3_5_mtp` spec decode (3 tokens) | Multiple KV-cache groups (`MultiGroupBlockTable`, per-group block sizes); only full-attention groups are compressible; speculative tokens must be excluded from compaction accounting |
| `--async-scheduling` | Compression must happen at a **boundary**; scheduler/worker KV state must stay in sync; optional batch-queue barrier |
| `FULL_DECODE_ONLY` cudagraph + `enforce_eager` | v1 model runner (`use_v2_model_runner=False` on vLLM 0.23); compression runs outside graph replay; input overrides must be applied in `_prepare_inputs` (v1 path) |
| `--no-enable-prefix-caching` | No prefix-cache hash invalidation worries; block reclaim stays purely physical |
| `--max-model-len 262144`, TP=4 | Long-context; tensor-parallel head shards — scoring must use the local TP head shard only |

## 3. Mechanism inventory (what each tool does, and its Ascend equivalent)

### 3.1 kvpress (HF-level mechanism)

| # | kvpress mechanism (HF transformers) | vLLM-Ascend conversion (this port) |
|---|---|---|
| K1 | `BasePress.__call__(model)` registers **forward hooks on every `self_attn` layer** | `AttentionHooks` installs hooks on vLLM decoder-layer attention modules (`layer.self_attn` / its `attention` backend layer) to capture per-layer **post-RoPE query** (and key/value when needed) + per-layer hidden-states capture for window presses |
| K2 | `forward_hook` reads KV from **HF `DynamicCache`** (`cache.layers[i].keys`, dense `[bsz, H, T, D]`) | `kv_layout.gather_request_kv_dense()` gathers the request's dense `[1, H, T, D]` K/V from the **Ascend paged block cache** (`[num_blocks, block_size, H, D]` split or `[2, ...]` combined; layout hint resolution; consecutive-block fast path) |
| K3 | `ScorerPress.compress()` computes scores, `scores.topk(n_kept, dim=-1)` → per-head keep indices, then **gathers kept K/V into a new dense tensor** | `press_bridge.select_keep_indices()` runs the same scoring math natively (Knorm / StreamingLLM / Random / SnapKV / TOVA / ObservedAttention / DecodingPress), then `kv_layout.compact_request_kv_in_place_per_head()` **permutes kept entries into the first slots of the request's existing blocks in-place** (per-head sets, uniform count) |
| K4 | Compressed cache written back as `cache_layer.keys = keys` (physical shrink of the dense tensor) | Block-row shrink: `num_blocks_per_row` reduced to `ceil(retained/block_size)`; freed blocks returned to the KV cache manager on the scheduler side; worker block-table row truncated |
| K5 | `cache_position` / `position_ids` continue from compressed length | **Effective-length bookkeeping**: scheduler effective-len tracker + `allocate_slots` hook; `_prepare_inputs` patch rewrites `seq_lens`/`seq_lens_np`/`positions`/`slot_mapping` so attention reads only the kept prefix and new tokens land right after it |
| K6 | `is_prefilling(cache_position, q_len)` phase detection | `is_prefill_step` from scheduler output (`num_computed_tokens < prefill_len`) |
| K7 | `DecodingPress` hidden-states buffer + `compression_interval`/`target_size` | Per-layer hidden-states capture buffer with identical interval/target semantics |
| K8 | `attention_patch` fake-key masking for head-wise methods (AdaKV…) | **Not physically expressible in Ascend kernels** → adapter rejects mask-based presses at startup with a clear log; per-head *different sets, same count* presses are fully supported via per-head compaction |
| K9 | `patch_attention_functions()` global HF patch | Not needed (vLLM does not call HF attention functions); the port patches **vLLM-Ascend paths only** |

### 3.2 SqueezeAttention (HF-level mechanism)

| # | SqueezeAttention mechanism (HF) | vLLM-Ascend conversion (this port) |
|---|---|---|
| S1 | Replaces decoder layers with `LlamaAttention_squeeze`/`LlamaDecoderLayer_squeeze` that drop KV **per layer** during streaming | Per-layer **recency keep sets** executed as block-level compaction of the request KV row; per-layer sets `[0, start_size) ∪ last(budget_L − start_size)` |
| S2 | `hiddlayer=True` prefill pass records per-layer **cosine similarity** of layer input vs output hidden states | Decoder-layer forward hooks (input/output hidden states) during prefill steps; per-token cosine similarity, averaged per layer |
| S3 | **KMeans (3 classes)** on per-layer importance → per-layer budgets `sliding_windows[idx]`; total budget conserved: `a = (N·ini_size − n3·percent)/(n1+n2)` | Same math ported (`budgets.py`), run once after prefill; budgets logged per layer |
| S4 | Per-step drop when `past_len > sliding_windows[idx]` | Compression triggered at block boundaries when request length exceeds `K` (see constraint below) |
| S5 | StreamingLLM `start_size` sink tokens always kept | Same (kept in every keep set) |

**Hard constraint discovered during analysis (documented honestly):** vLLM-Ascend shares one
block-table row and one uniform `seq_lens` across all layers of a KV group, and its
attention kernels do not accept per-position masks. Per-layer budgets with *different token
counts* are therefore **not physically expressible**. The port therefore:
- keeps the full SqueezeAttention mechanism: layer importance → KMeans → per-layer budgets;
- compacts **all layers to the same block-aligned count K** with per-layer recency keep sets
  (default `SQUEEZE_MODE=uniform`, correct);
- offers `SQUEEZE_MODE=class_weighted` (experimental): per-layer counts with **fake-key
  padding** of short-budget layers' tail slots, computed per decode step from the current
  query (kvpress-style hyperplane), which keeps the layer-wise budget semantics at a
  per-step CPU/NPU cost; gated behind `SQUEEZE_FAKE_KEY_PADDING=1`.

## 4. Shared runtime skeleton (each package is self-contained)

Both packages follow the tri_3_5 integration architecture, adapted with tool-specific env
prefixes (`KVPRESS_*` / `SQUEEZE_*`):

```
plugin.py                          # vllm.general_plugins entry; env gate; install
envs.py                            # config dataclass from env (budget, press, thresholds...)
logging_control.py                 # master LOGGING switch + PROBE switch + markers
runtime/
  monkeypatch.py                   # Scheduler.__init__/schedule/update_from_output,
                                   # KVCacheManager.allocate_slots, EngineCore boundary (opt),
                                   # NPUWorker.init_device/execute_model,
                                   # NPUModelRunner._prepare_inputs (v1) input patch
  scheduler_hooks.py               # compression signals, effective-len tracker, block reclaim
  worker_hooks.py                  # NPUWorker proxy install (early/lazy)
  runner_proxy.py                  # proxy around NPUModelRunner; execute_model interception;
                                   # trigger evaluation; compression execution; event attach
  compression_engine.py            # per-request: gather dense -> select -> compact -> reclaim
  kv_layout.py                     # cache layout split, dense gather, per-head compaction,
                                   # zero-copy tail remap (optional)
  block_sync.py                    # worker block-table shrink + scheduler-side block free
  input_patch_v1.py                # effective seq_lens/positions/slot-mapping on _prepare_inputs
  attention_hooks.py               # per-layer query/hidden-state capture (both tools)
  state.py                         # per-request state (prefill_len, compression_count, cache len)
  signals.py                       # CompressionSignal
```

vLLM-Ascend surfaces actually patched (identical list to tri_3_5's validated set):

- `vllm.v1.core.sched.scheduler.Scheduler` (`__init__`, `schedule`, `update_from_output`)
- `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots`
- `vllm_ascend.worker.worker.NPUWorker` (`init_device`, `execute_model`)
- `vllm_ascend.worker.model_runner_v1.NPUModelRunner._prepare_inputs` (v1 path, used by the
  target launch config) — plus best-effort v2 hooks
- `vllm_ascend.worker.block_table.BlockTable.get_device_tensor` (optional trim, default off)
- relaxed KV-cache memory check (`vllm.v1.core.kv_cache_utils`) so `--max-model-len 262144`
  can start when compression will keep real usage below the physical cache

## 5. kvpress-ascend specifics

Env switches (all optional):

| Env | Default | Meaning |
|---|---|---|
| `KVPRESS_ENABLE` / `KVPRESS` | off | plugin activation |
| `KVPRESS_PRESS` | `KnormPress` | press name: `KnormPress, StreamingLLMPress, RandomPress, SnapKVPress, TOVAPress, ObservedAttentionPress, DecodingPress` |
| `KVPRESS_COMPRESSION_RATIO` | 0.5 | `ScorerPress.compression_ratio` |
| `KVPRESS_TARGET_SIZE` | 0 | decode target KV tokens (DecodingPress `target_size`); 0 → ratio-driven |
| `KVPRESS_WINDOW_SIZE` | 64 | SnapKV window |
| `KVPRESS_SINK_TOKENS` | 4 | StreamingLLM n_sink |
| `KVPRESS_COMPRESSION_INTERVAL` | 512 | DecodingPress interval |
| `KVPRESS_KV_BUDGET` | 0 | absolute per-request KV budget (overrides ratio when >0) |
| `KVPRESS_MIN_RECLAIM_BLOCKS` | 1 | minimum freed blocks to run a compaction |
| `KVPRESS_RUNTIME_LOGGING` | 1 | master logging |
| `KVPRESS_PROBE` | 1 | **per-step core-entry probe logs** |
| `KVPRESS_LOG_ATTENTION_HOOK` | 0 | per-layer hook capture logs |
| `KVPRESS_USE_INSTALLED` | auto | prefer installed `kvpress` for press classes, else vendored |
| `KVPRESS_DEFER_PREFILL_COMPRESSION` | 1 | compress only after full prefill (stable Ascend mode) |
| `KVPRESS_MAX_COMPRESSIONS_PER_STEP` | 1 | guard |

Per-inference probe log (every model step, request-scoped):

```
[KVPRESS-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  press=KnormPress ratio=0.5 seq_len=4096 budget=2048 keep=2048 reclaimed_blocks=8
  compress_events=1 last_event=applied
```

`core_entered=1` means the runner-proxy `execute_model` interception ran;
`hook_entered=1` means the attention hook captured at least one layer's query this step.
If both are 0 the patch is not active — that is the user's "is it really working" signal.

## 6. SqueezeAttention-ascend specifics

Env switches:

| Env | Default | Meaning |
|---|---|---|
| `SQUEEZE_ENABLE` / `SQUEEZE` | off | plugin activation |
| `SQUEEZE_INI_SIZE` | 0.21 | initial per-layer KV budget (fraction of prompt) |
| `SQUEEZE_CLASS3_SIZE` | 0.08 | budget fraction for cluster class 3 layers |
| `SQUEEZE_START_SIZE` | 4 | StreamingLLM sink tokens |
| `SQUEEZE_MODE` | `uniform` | `uniform` (correct, shared K) or `class_weighted` (experimental fake-key) |
| `SQUEEZE_KMEANS_CLUSTERS` | 3 | cluster count (paper uses 3) |
| `SQUEEZE_KV_BUDGET` | 0 | absolute budget override |
| `SQUEEZE_RUNTIME_LOGGING` | 1 | master logging |
| `SQUEEZE_PROBE` | 1 | **per-step core-entry probe logs** |
| `SQUEEZE_LOG_BUDGETS` | 1 | per-layer budget table after clustering |

Probe log:

```
[SQUEEZE-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  layers=36 budgets_ready=1 K=1536 start=4 ini=0.21 class3=0.08
  keep=1536 reclaimed_blocks=10 compress_events=1
[SQUEEZE-ASCEND][CLUSTER] labels=[0,1,1,...] class1=12 class2=12 class3=12
  budgets=[a*L, ..., percent*L, ...]
```

## 7. Deliverables & file layout

```
try_5/
├── PLAN.md                     (this file)
├── TODO.md                     (execution checklist)
├── kvpress-ascend/
│   ├── setup.py  pyproject.toml  README.md
│   ├── kvpress_ascend/...        (plugin + runtime + core + vendored presses fallback)
│   └── tests/                    (simulated-debug suite, stub vllm/vllm_ascend)
└── SqueezeAttention-ascend/
    ├── setup.py  pyproject.toml  README.md
    ├── squeezeattention_ascend/... (plugin + runtime + core)
    └── tests/                    (simulated-debug suite)
```

Install on the target machine:

```bash
pip install ./kvpress-ascend
pip install ./SqueezeAttention-ascend
export KVPRESS_ENABLE=1        # or: export KVPRESS=1
export SQUEEZE_ENABLE=1        # or: export SQUEEZE=1
vllm serve ...                 # unchanged command
```

## 8. Testing strategy (no NPU on this machine → simulated debug)

Because vllm / vllm-ascend / transformers are absent locally, tests ship **stub modules**
that mirror vllm-ascend v0.23.0's real interfaces (NPUWorker, NPUModelRunner with
input_batch/block_table/seq_lens buffers, Scheduler, KVCacheManager, KV-cache group specs,
MTP multi-group block table) and run the real patch code against them on CPU. Coverage:

1. plugin env-gate & patch installation logs (KVPRESS_ENABLE=0 → no-op; =1 → installed)
2. scheduler compression signals at length threshold; effective-len tracking
3. runner-proxy interception; per-step `core_entered/hook_entered` probe logs
4. kv_layout: dense gather + per-head in-place compaction content correctness
   (identifiable per-token values; assert kept prefix after compaction)
5. block reclaim: worker row shrink + scheduler-side free of excess blocks
6. input patch: `seq_lens_np`/`positions`/`slot_mapping` rewritten to the compressed view
7. press bridge: Knorm / StreamingLLM / Random / SnapKV / TOVA selection math vs direct
   torch reference implementation
8. SqueezeAttention: KMeans budget math (deterministic labels), per-layer keep sets,
   uniform-mode K computation, fake-key padding math (experimental)
9. logging switches: PROBE logs per step; disabled → no probe output
10. MTP group handling: only full-attention group compacted; draft group untouched

## 9. Risks / limits (documented, matching tri_3_5's limits section)

- Head-wise mask-based presses (AdaKV, CriticalKV, PyramidKV per-head budgets…) cannot be
  physically realized in Ascend kernels → rejected with explicit log at startup.
- SqueezeAttention per-layer *different* budgets collapse to a shared K in `uniform` mode
  (block-table constraint); `class_weighted` mode is experimental.
- Dense/block attention layouts are the validated target; MLA/linear-attn layers are skipped.
- No hardware validation possible here; tests are simulated-debug only. First real-NPU run
  should use `KVPRESS_RUNTIME_LOGGING=1` / `SQUEEZE_RUNTIME_LOGGING=1` and check the
  `[PROBE] core_entered=1` lines and `[COMPRESS]/[CLUSTER]` events.
