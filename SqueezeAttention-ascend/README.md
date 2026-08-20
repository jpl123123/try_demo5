# SqueezeAttention-ascend: SqueezeAttention monkeypatch adapter for vLLM-Ascend v0.23.0

Monkeypatch integration that converts the SqueezeAttention 2D KV-cache management
mechanism (layer-wise optimal budgets × streaming token eviction) to vLLM-Ascend
block-cache compaction **without modifying any vLLM-Ascend source**. See the root
`PLAN.md` / `TODO.md` for the full design and mechanism-conversion table.

## Install

```bash
pip install ./SqueezeAttention-ascend
```

## Activation

```bash
export SQUEEZE_ENABLE=1        # or: export SQUEEZE=1
vllm serve <model> ...         # your normal vLLM-Ascend launch, unchanged
```

> Note: `export squeeze` is not valid shell syntax; use `export SQUEEZE_ENABLE=1`
> (or the `SQUEEZE=1` alias) before launching. With the switch off the plugin is
> a no-op, so it can stay installed alongside kvpress-ascend.

## Per-inference verification switch (is the patch really live?)

```bash
export SQUEEZE_PROBE=1         # default on
export SQUEEZE_RUNTIME_LOGGING=1
```

Every model step prints one probe line per scheduled request:

```
[SQUEEZE-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  mode=uniform layers=36 budgets_ready=1 K=1536 start=4 ini=0.210 class3=0.080
  seq_len=1536 keep=1536 reclaimed_blocks=10 compress_events=1 last_event=applied
[SQUEEZE-ASCEND][CLUSTER] budgets req=req-1 layers=36 prompt_len=8192 ...
  class_sizes={0:12,1:12,2:12} budgets=[...]
```

- `core_entered=1` — the runner-proxy interception ran.
- `hook_entered=1` — the layer hooks captured cosine similarities this step.
- `budgets_ready=1` — the KMeans layer-wise budgets were finalized after prefill
  (the `[CLUSTER]` line prints the per-layer budgets).

## Mechanism conversion summary

| SqueezeAttention (HF) | This adapter (vLLM-Ascend) |
|---|---|
| `LlamaAttention_squeeze` per-layer recency drop | per-layer recency keep sets executed as in-place block-cache compaction (`core/selection.py`) |
| `hiddlayer` per-layer cosine-similarity capture during prefill | decoder-layer attention hooks (`runtime/attention_hooks.py`) |
| KMeans 3-class budgets with total-budget invariant | `core/budgets.py` (same math, sklearn or torch fallback) |
| per-step drop when `past_len > sliding_windows[idx]` | boundary compression via scheduler signals + worker validation |
| `start_size` sink tokens | kept in every keep set |

**Documented constraint:** vLLM-Ascend shares one block-table row and a uniform
`seq_lens` across all layers of a KV group, and its attention kernels accept no
per-position masks. Per-layer budgets with *different token counts* are therefore
not physically expressible:

- `SQUEEZE_MODE=uniform` (default, correct): all layers compact to the shared
  keep count `K = max(per-layer budget)`, with per-layer recency keep sets.
- `SQUEEZE_MODE=class_weighted` (experimental): per-layer counts with fake-key
  padding (`SQUEEZE_FAKE_KEY_PADDING=1`) of short-budget layers' tail slots,
  computed per decode step from the current query (kvpress-style hyperplane).

Compile safety: the hooks are fully transparent to `torch.compile` — they only
store tensor references inside the forward (no tensor ops) and skip capture
while `torch.compiler.is_compiling()` is true; the cosine-similarity math runs
in the runner proxy after the forward. npugraph_ex / AOT artifacts stay
identical to an un-patched run. Note: layer-importance capture (hidd_data)
therefore requires the prefill step to run eager; under compiled prefill the
budgets fall back to the uniform initial budget (`SQUEEZE_KV_BUDGET` still
fixes K).

## Environment variables

| Env | Default | Meaning |
|---|---|---|
| `SQUEEZE_ENABLE` / `SQUEEZE` | off | plugin activation |
| `SQUEEZE_INI_SIZE` | 0.21 | initial per-layer KV budget (fraction of prompt) |
| `SQUEEZE_CLASS3_SIZE` | 0.08 | budget fraction for class 3 (most important) layers |
| `SQUEEZE_START_SIZE` | 4 | StreamingLLM sink tokens |
| `SQUEEZE_MODE` | `uniform` | `uniform` \| `class_weighted` |
| `SQUEEZE_FAKE_KEY_PADDING` | 0 | fake-key padding in class_weighted mode |
| `SQUEEZE_KMEANS_CLUSTERS` | 3 | cluster count |
| `SQUEEZE_KMEANS_SEED` | none | deterministic clustering |
| `SQUEEZE_KV_BUDGET` | 0 | absolute budget override |
| `SQUEEZE_MIN_RECLAIM_BLOCKS` | 1 | minimum freed blocks to run a compaction |
| `SQUEEZE_MAX_COMPRESSIONS_PER_STEP` | 1 | guard |
| `SQUEEZE_DEFER_PREFILL_COMPRESSION` | 1 | compress only after full prefill |
| `SQUEEZE_RUNTIME_LOGGING` | 1 | master logging |
| `SQUEEZE_PROBE` | 1 | per-step core-entry probe logs |
| `SQUEEZE_LOG_BUDGETS` | 1 | `[CLUSTER]` per-layer budget table |

## Simulated-debug tests (no NPU needed)

```bash
pip install pytest scikit-learn
cd SqueezeAttention-ascend/tests && python -m pytest ../
```

The suite ships stub modules mirroring the vllm-ascend v0.23.0 surfaces and runs
the real patch code on CPU: plugin activation, KMeans budget math (total-budget
conservation), recency selection, layer-importance capture, end-to-end
prefill→cluster→compress with probe logs, uniform/class_weighted modes, fake-key
padding math, compaction content correctness, and coexistence with kvpress-ascend.
