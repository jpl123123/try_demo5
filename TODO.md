# TODO — kvpress-ascend & SqueezeAttention-ascend execution checklist

Status legend: `[ ]` pending · `[x]` done · `[~]` in progress

## Phase 0 — Planning & reading (done)
- [x] Read tri_3_5 reference integration (plugin, integration_monkeypatch, runner proxy,
      kv_compaction, worker_reclaim_sync, scheduler events, input patches v1/v2)
- [x] Read vllm-ascend v0.23.0 surfaces to patch (NPUWorker, NPUModelRunner._prepare_inputs,
      BlockTable/MultiGroupBlockTable, attention backends, kv_cache groups)
- [x] Read kvpress mechanism (BasePress/ScorerPress/DecodingPress hooks, DynamicCache,
      attention_patch, supported presses)
- [x] Read SqueezeAttention mechanism (LlamaAttention_squeeze, hiddlayer cosine-sim,
      KMeans budgets, streaming recency drop)
- [x] Write PLAN.md

## Phase 1 — kvpress-ascend package (done)
- [x] `setup.py` + `pyproject.toml` with `vllm.general_plugins` entry point
- [x] `envs.py` — `KVPRESS_*` config (press, ratio, budget, thresholds, logging)
- [x] `logging_control.py` — master logging + per-step `PROBE` switch
- [x] `core/kv_layout.py` — Ascend block-cache split/gather/per-head in-place compaction
- [x] `core/press_bridge.py` — Ascend-native scoring bridge (Knorm/StreamingLLM/Random/
      SnapKV/TOVA/ObservedAttention/Decoding) + vendored fallback presses
- [x] `runtime/` — signals, state, attention hooks, compression engine, block sync,
      input patch v1 (`_prepare_inputs` overrides), scheduler hooks (signals +
      effective-len tracker + block reclaim), worker hooks (NPUWorker proxy),
      runner proxy (boundary flow + probe), monkeypatch installer, output bridge
- [x] `plugin.py` — env gate (`KVPRESS_ENABLE` / `KVPRESS` alias)
- [x] `README.md` — activation, probe switch, env table, mechanism-conversion table
- [x] tests: plugin activation, compaction content, press math, input patch,
      probe/logging switches, end-to-end loop, MTP multi-group, coexistence —
      **31 passed**

## Phase 2 — SqueezeAttention-ascend package (done)
- [x] `setup.py` + `pyproject.toml` with entry point
- [x] `envs.py` — `SQUEEZE_*` config (ini/class3/start/mode/clustering)
- [x] `logging_control.py` — master logging + `PROBE` + `CLUSTER` streams
- [x] `core/kv_layout.py` (self-contained copy)
- [x] `core/budgets.py` — cosine-sim accumulator + KMeans budgets (total-budget
      invariant, degenerate-cluster fallback)
- [x] `core/selection.py` — recency keep sets, per-head expansion, fake-key
      hyperplane padding (experimental class_weighted)
- [x] `runtime/` — same skeleton as Phase 1 (scheduler/worker/runner/input patch/
      block sync/attention hooks/state/output bridge)
- [x] `plugin.py` — env gate (`SQUEEZE_ENABLE` / `SQUEEZE` alias)
- [x] `README.md` — activation, probe switch, env table, constraint documentation
- [x] tests: plugin activation, budget math, selection, layer-importance capture,
      end-to-end prefill→cluster→compress, class_weighted, compaction content,
      probe switches — **19 passed**

## Phase 3 — Debug & verification (simulated) (done)
- [x] kvpress-ascend suite green (31 tests, CPU stubs)
- [x] SqueezeAttention-ascend suite green (19 tests, CPU stubs)
- [x] Probe logs verified: `[PROBE] core_entered=1 hook_entered=1` per step with
      core parameters; switches off suppress per-step lines
- [x] Both plugins coexist in one environment; each activates via its own env switch
- [x] `pip install` verified for both packages (entry points resolve:
      `vllm.general_plugins` → `register_kvpress_backend` /
      `register_squeezeattention_backend`)
- [x] READMEs updated with launch instructions and the `export` note
- [x] TODO.md finalized

## On-machine (NPU) first-run checklist (for the user)
- [ ] `pip install ./kvpress-ascend ./SqueezeAttention-ascend`
- [ ] `export KVPRESS_ENABLE=1` (or `export KVPRESS=1`) for kvpress
- [ ] `export SQUEEZE_ENABLE=1` (or `export SQUEEZE=1`) for SqueezeAttention
- [ ] Launch the unchanged `vllm serve` command
- [ ] Confirm startup logs: `Installed kvpress/SqueezeAttention monkeypatch integration`
      and `Worker injected ... runner proxy`
- [ ] Confirm per-step `[PROBE] core_entered=1 hook_entered=1` lines and
      `COMPRESS` / `CLUSTER` events
- [ ] If memory headroom allows, compare with `KVPRESS_RUNTIME_LOGGING=0` /
      `SQUEEZE_RUNTIME_LOGGING=0` for quiet performance runs
