<div align="center">

# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2604.04921)
[![Project Page](https://img.shields.io/badge/Project-Page-teal)](https://weianmao.github.io/tri-attention-project-page/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/downloads/)

*Compress KV cache by 10.7x and boost throughput by 2.5x on long reasoning tasks -- with no accuracy loss.*

[Weian Mao](https://scholar.google.com/citations?user=Qu-QXTsAAAAJ)<sup>1*</sup>,
[Xi Lin](https://profile.erix025.me/)<sup>3*</sup>,
[Wei Huang](https://aaron-weihuang.com/)<sup>2*</sup>,
Yuxin Xie<sup>1</sup>,
Tianfu Fu<sup>1</sup>,
[Bohan Zhuang](https://bohanzhuang.github.io)<sup>3</sup>,
[Song Han](http://songhan.mit.edu/)<sup>1,2</sup>,
[Yukang Chen](https://yukangchen.com/)<sup>2</sup>

<sup>1</sup>MIT, <sup>2</sup>NVIDIA, <sup>3</sup>ZJU &nbsp;&nbsp; <sup>*</sup>Equal contribution

</div>




https://github.com/user-attachments/assets/768e59bb-897e-41bf-81b8-e7376aa72056








## News

- **[2026-04-21]** SGLang backend support added — TriAttention now runs on SGLang in addition to vLLM. See [SGLang Integration](docs/sglang.md).
- **[2026-04-14]** Community DGX Spark (GB10/sm-121) enablement by [@dscain](https://github.com/dscain) — vLLM support merged, non-vLLM path in progress.
- **[2026-04-12]** TriAttention now supports AR video generation with KV cache compression. See [LongLive README](longlive/README.md).
- **[2026-04-11]** Community C/ggml port for llama.cpp (HIP/ROCm) by [@domvox](https://github.com/domvox) — enables TriAttention on AMD GPUs via llama.cpp, with ~6.8× KV reduction when composed with TurboQuant. See [triattention-ggml](https://github.com/domvox/triattention-ggml).
- **[2026-04-09]** Experimental MLX and TurboQuant support for Apple Silicon (M1/M2/M3/M4) — thanks to [@DeadByDawn101](https://github.com/DeadByDawn101) (RavenX AI) for proposing and contributing this feature.

## Highlights

- **2.5x throughput** on AIME25 long reasoning while matching Full Attention accuracy (40.8 vs 40.8)
- **10.7x KV memory reduction** with trigonometric frequency-domain compression
- **OpenClaw compatible** — enables local deployment on 24GB RTX 4090

<p align="center">
  <img src="docs/assets/tradeoff.png" width="80%">
</p>
<p align="center"><i>TriAttention achieves 2.5x higher throughput and 10.7x KV memory reduction on AIME25 while matching Full Attention accuracy.</i></p>

## How It Works

Pre-RoPE Q/K vectors in long reasoning models concentrate around fixed centers that determine distance preferences via a trigonometric series. TriAttention scores keys using these centers and norms instead of requiring representative query selection, enabling accurate KV cache compression without the overhead of existing attention-based methods.

## Documentation

- [OpenClaw](docs/openclaw.md) -- OpenClaw manual configuration
- [Reproduction Guide](docs/reproduction.md) -- full experiment commands for all benchmarks
- [Calibration Guide](docs/calibration.md) -- generating custom Q/K statistics
- [MLX Support](docs/mlx.md) -- supporting Apple Silicon Macs (M1/M2/M3/M4) via the MLX
- [SGLang Integration](docs/sglang.md) -- deploying TriAttention with SGLang as the inference backend
- [vLLM-Ascend Integration](docs/vllm_ascend.md) -- deploying TriAttention on Huawei Ascend NPUs through vLLM-Ascend
- [Video Generation](longlive/README.md) -- KV cache compression for long video generation with LongLive
- [Full Results](docs/results.md) -- complete tables, figures, and analysis

## Deploy with OpenClaw

TriAttention's vLLM server exposes an OpenAI-compatible API, which means you can use it directly as a custom provider in [OpenClaw](https://github.com/openclaw/openclaw).

### Quick Setup

1. Follow the [Installation](#installation) instructions, then start a vLLM server with the recommended settings below.
2. In OpenClaw, add a custom provider pointing to your vLLM server (e.g. `http://localhost:8000/v1`).

For manual configuration or troubleshooting, see the [OpenClaw Manual Configuration Guide](docs/openclaw.md).

### Recommended Server Settings for Chat

Interactive chat workloads differ from offline benchmarks — conversations are long-running and prefill chunks can trigger compression at unexpected points. We recommend the following adjustments:

```bash
# Required: path to precomputed frequency statistics
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/vllm/stats/qwen3_32b_int4_stats.pt

# Use a larger KV budget for multi-turn chat (default: 2048)
export TRIATTN_RUNTIME_KV_BUDGET=12000

vllm serve <model_path> \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code \
    --enable-prefix-caching false \
    --max-num-batched-tokens 1024
```

**Key differences from the default server mode:**

- **`--enable-prefix-caching false`** — Prefix caching is incompatible with KV compression currently; disable it to avoid incorrect cache hits on compressed entries.
- **`--max-num-batched-tokens 1024`** — Limits the prefill chunk size. Large chunks can overshoot the KV budget in a single step before compression has a chance to trigger, leading to OOM.
- **`TRIATTN_RUNTIME_KV_BUDGET=12000`** — Chat sessions accumulate context across many turns; a larger budget (e.g. 12k) keeps more history available and avoids aggressive eviction.

## Installation

```bash
git clone https://github.com/WeianMao/triattention.git
cd triattention
pip install -e .
pip install flash-attn --no-build-isolation  # recommended (takes 105m in DGX Spark / GB10)
```

## Quick Start

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```

## Datasets

Benchmark datasets (AIME 2024, AIME 2025, MATH-500) are automatically downloaded from HuggingFace on first run -- no manual data preparation is needed. The evaluation scripts handle downloading, caching, and formatting transparently.

## Supported Models

| Model | HuggingFace ID | Status |
|-------|---------------|--------|
| Qwen3-8B | `Qwen/Qwen3-8B` | Verified |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Verified |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Verified |

## Results

### AIME24 / AIME25 (KV Budget = 2048, DS-Llama = 512)

| Method | Qwen3-8B | DS-Llama-8B | DS-Qwen-7B | GPT-OSS-20B |
|--------|----------|-------------|-------------|-------------|
| Full Attention | 57.1 / 40.8 | 50.4 / 31.4 | 43.8 / 34.2 | 69.2 / 60.0 |
| SnapKV | 34.6 / 20.0 | 5.0 / 6.7 | 34.6 / 25.0 | 48.3 / 36.7 |
| R-KV | 25.4 / 17.5 | 25.8 / 11.2 | 34.6 / 23.3 | 49.6 / 39.2 |
| **TriAttention** | **42.1 / 32.9** | **33.8 / 19.6** | **42.5 / 30.0** | **59.2 / 49.2** |

### Throughput (Qwen3-8B, tokens/sec)

| Benchmark | TriAttn Budget | Full Acc | TriAttn Acc | Full Throughput | TriAttn Throughput | Speedup |
|-----------|---------------|----------|-------------|-----------------|-------------------|---------|
| MATH-500 | 1024 | 69.6 | 68.4 | 222.8 | 1405.2 | **6.3x** |
| AIME24 | 4096 | 57.1 | 54.6 | 222.8 | 413.9 | **1.9x** |
| AIME25 | 3072 | 40.8 | 40.8 | 222.8 | 563.5 | **2.5x** |

See [docs/results.md](docs/results.md) for complete results including MATH-500 accuracy table, accuracy vs. budget curves, and DFS memory retention analysis.

## vLLM Integration

TriAttention includes a vLLM plugin that enables transparent KV cache compression for production deployment. After installation, vLLM automatically discovers and activates the plugin -- no code changes required.

### Server Mode (OpenAI-Compatible API)

```bash
# Set compression parameters
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/vllm/stats/qwen3_32b_int4_stats.pt

# Launch vLLM server -- TriAttention activates automatically. Set `ENABLE_TRIATTENTION=0` to disable.
vllm serve <model_path> \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code \
    --enable-prefix-caching false

# Use the standard OpenAI-compatible API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "<model_path>", "messages": [{"role": "user", "content": "Solve: ..."}]}'
```

#### vLLM with DGX Spark / GB10

To enable vLLM in DGX Spark / GB10, run these installation steps instead:

```bash
uv venv
. .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
uv pip install \
  https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0-cp38-abi3-manylinux_2_31_aarch64.whl \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
uv pip install -e .

export TRITON_CACHE_DIR=~/.cache/.triton-cache
mkdir -p $TRITON_CACHE_DIR

PY_SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # Or adjust as needed to your environment
export LD_LIBRARY_PATH="$PY_SITE/torch/lib:$PY_SITE/nvidia/cu13/lib:/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"

vllm serve Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7
```

Verify the first vLLM log line is `[TriAttention] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True`.
```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3-8B","prompt":"hello","max_tokens":16}'
```

### Python API

```python
from triattention.vllm.runtime.integration_monkeypatch import (
    install_vllm_integration_monkeypatches,
)

# Install patches before creating the LLM instance
install_vllm_integration_monkeypatches(patch_scheduler=True, patch_worker=True)

# Standard vLLM API -- compression happens transparently
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model_path>",
    dtype="bfloat16",
    max_model_len=32768,
    enforce_eager=True,
    trust_remote_code=True,
)

outputs = llm.generate(["Your prompt here"], SamplingParams(temperature=0.6, top_p=0.95))
print(outputs[0].outputs[0].text)
```

### Runtime Logging

For performance tests that should not emit TriAttention runtime or profile
diagnostics, use the logging master switch:

```bash
export TRIATTN_RUNTIME_LOGGING=0
```

This suppresses TriAttention startup, scheduler/worker decision, compression
event, execution-path trace, and `TRIATTN_PERF` / `TRIATTN_E2E_PERF` /
`TRIATTN_PHASE_PERF` profile logs. Error and safety warning logs are still
emitted.

For debugging, leave logging enabled and turn on only the streams you need:

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

To keep only the core runtime path markers while suppressing repetitive
boundary and expected-skip logs, use:

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

For one-off deep debugging of the selector and per-layer compaction path, add:

```bash
export TRIATTN_RUNTIME_LOG_CORE_TRACE=1
export TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG=1
```

These two switches can produce many INFO lines and large selector payloads on
multi-rank runs, so keep them off for latency or throughput comparisons.

### Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRIATTN_RUNTIME_LOGGING` | `true` | Master switch for TriAttention runtime/profile diagnostic logs; set `0` for quiet performance tests |
| `TRIATTN_RUNTIME_LOG_DECISIONS` | `true` | Emit scheduler/worker compression decisions when logging is enabled |
| `TRIATTN_RUNTIME_LOG_EXECUTION_PATH` | `true` | Emit `TRIATTN_EXEC_PATH` markers that prove the request reached the TriAttention runner/hook path, including guard reasons before a hook is entered and selector/scoring markers after a real compression boundary |
| `TRIATTN_RUNTIME_LOG_EXECUTION_PATH_CORE_ONLY` | `false` | With execution-path logging enabled, suppress high-frequency boundary/context and expected-skip logs so core markers such as `hook_installed`, `worker_hook_enter`, `group_pipeline_enter`, `selector_scoring_enter`, and applied/unexpected results stand out |
| `TRIATTN_RUNTIME_LOG_CORE_TRACE` | `false` | Emit verbose `TRIATTN_CORE_TRACE` enter/exit markers around the group pipeline, selector bridge, and compaction internals |
| `TRIATTN_RUNTIME_LOG_SELECTOR_DEBUG` | `false` | Include nested selector debug payloads such as score layer indices, group aggregation metadata, and selector path details in hook results/logs |
| `TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS` | `false` | Emit compression event logs on every worker instead of rank 0 only when logging is enabled |
| `TRIATTN_RUNTIME_PERF_PROFILE` | `false` | Emit aggregated `TRIATTN_PERF` counters when logging is enabled |
| `TRIATTN_RUNTIME_E2E_PROFILE` | `false` | Emit aggregated `TRIATTN_E2E_PERF` counters when logging is enabled |
| `TRIATTN_RUNTIME_PHASE_PROFILE` | `false` | Emit deeper `TRIATTN_PHASE_PERF` timing counters when logging is enabled |
| `TRIATTN_RUNTIME_KV_BUDGET` | `2048` | Maximum tokens retained in KV cache per request |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | `128` | Compression trigger interval (every N new tokens) |
| `TRIATTN_RUNTIME_WINDOW_SIZE` | `128` | Recent tokens always preserved |
| `TRIATTN_RUNTIME_PRUNING_MODE` | `per_head` | Token selection strategy (`per_head` or `per_layer_per_head`) |
| `TRIATTN_RUNTIME_SCORING_BACKEND` | `auto` | Scoring backend (`auto`, `triton`, `torch`/`pytorch`); `auto` uses PyTorch/torch_npu on vLLM-Ascend |
| `TRIATTN_RUNTIME_FAST_RECENCY_ONLY` | `false` | Diagnostic low-overhead selector that keeps the most recent budget tokens without sparse-stat scoring |
| `TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD` | `true` | Prefer sparse TriAttention selection over pure recency for long-context correctness. When no stats path is set, the runtime uses packaged stats when available so core `group_pipeline_enter` / `selector_scoring_enter` markers are reached; explicitly missing stats paths still fall back to pure-recency diagnostics. |
| `TRIATTN_RUNTIME_FAST_RECENCY_LONG_CONTEXT_GUARD` | `false` | Optional safety gate for pure-recency diagnostics on very long prompts; defaults off so the vLLM-Ascend path still enters the TriAttention hook for validation |
| `TRIATTN_RUNTIME_ENABLE_ZERO_COPY_RECENCY` | `true` | On vLLM-Ascend, use block-table tail remap for `FAST_RECENCY_ONLY` when the budget is block-aligned |
| `TRIATTN_RUNTIME_ZERO_COPY_RECENCY_ONLY_ON_ASCEND` | `true` | On Ascend fast-recency runs, wait for zero-copy tail remap instead of falling back to copy-based recency compaction |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | packaged stats when available | Path to precomputed frequency statistics `.pt` file |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | `false` | Protect initial prompt tokens from eviction |
| `TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND` | `true` | On vLLM-Ascend, wait until full prompt prefill finishes before first KV compaction |
| `TRIATTN_RUNTIME_MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND` | `0` | Optional Ascend-only grace window after prefill before the first decode compression; default `0` enters the TriAttention core at the first eligible decode boundary |
| `TRIATTN_RUNTIME_SCORE_MAX_LAYERS` | `0` | Maximum number of layers to score before cross-layer aggregation (`0` means all layers unless the Ascend limit below is explicitly set) |
| `TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND` | effective default `8` | Ascend-only layer cap used when `SCORE_MAX_LAYERS=0`; set `0` explicitly to score all layers |
| `TRIATTN_RUNTIME_SCORE_LAYER_STRIDE` | `1` | Score one layer every N layers before optional max-layer sampling |
| `TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND` | effective sparse default `16` (`8` for fast-recency) | On Ascend, wait until at least this many KV blocks can be reclaimed before triggering compression |
| `TRIATTN_RUNTIME_PREFILL_MIN_RECLAIM_BLOCKS_ON_ASCEND` | `32` | On Ascend prefill steps, require a larger reclaim window before compression to amortize sparse scoring |
| `TRIATTN_RUNTIME_PREFILL_MAX_COMPRESSIONS_ON_ASCEND` | `1` | Maximum number of compression actions allowed during Ascend prefill for one request (`0` disables prefill compression) |
| `TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY` | `false` | Force an async batch-queue boundary around compression; normally keep disabled on Ascend for better TPOT |
| `TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND` | `false` | Experimental Ascend slot-mapping micro-optimization; keep disabled unless target output quality is validated |
| `TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND` | `true` | Install the TriAttention runner proxy during Ascend worker init so patching happens before measured requests |
| `TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH` | `true` | Install input patches when the runner proxy is created instead of waiting for the first compressed request |
| `TRIATTN_RUNTIME_FORCE_EAGER_MULTI_REQ_ON_ASCEND_EFFECTIVE_OVERRIDES` | `false` | On vLLM-Ascend, keep multi-request compressed-KV effective overrides graph-eligible by default for serving throughput; set `1` to force those batches outside graph/compiled mode when isolating ACL graph replay/update instability |
| `TRIATTN_RUNTIME_MAX_COMPRESSIONS_PER_STEP_ON_ASCEND` | `4` | On vLLM-Ascend, limit how many requests may run expensive sparse compression in one model step so large concurrent batches do not synchronize 16/32 selector+KV compactions before decode |
| `TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE` | `false` | Experimental Ascend V1 metadata micro-optimization; keep disabled for multi-request compressed batches so the full preallocated block table shape is preserved while seq-len and slot-mapping overrides carry the effective KV view |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION` | `true` | Enable in-place KV cache compaction |
| `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM` | `true` | Enable freed block reclamation |
| `ENABLE_TRIATTENTION` | `true` | Master switch to enable/disable the plugin |

### Precomputed Statistics

TriAttention requires precomputed Q/K frequency statistics for scoring. We provide pre-calibrated stats for supported models in `triattention/vllm/stats/`. See the [Calibration Guide](docs/calibration.md) for generating stats for custom models.

## Roadmap

- [x] vLLM integration
- [ ] SGLang integration
- [ ] Ollama integration
- [ ] Support for more model architectures

## Community Implementations

Independent ports and integrations maintained by the community:

| Project | Stack | Maintainer | Notes |
|---------|-------|------------|-------|
| [triattention-ggml](https://github.com/domvox/triattention-ggml) | C/ggml, llama.cpp (HIP/ROCm) | [@domvox](https://github.com/domvox) | AMD GPU support; composes with TurboQuant (~6.8× KV reduction). Includes pre-built calibration stats for Qwen3 family. |

> **Note:** Community projects are independently maintained and not officially supported. Please direct questions and issues to each project's own issue tracker.

## Citation

```bibtex
@article{mao2026triattention,
    title={TriAttention: Efficient Long Reasoning with Trigonometric KV Compression},
    author={Weian Mao and Xi Lin and Wei Huang and Yuxin Xie and Tianfu Fu and Bohan Zhuang and Song Han and Yukang Chen},
    year={2026},
    eprint={2604.04921},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgements

We thank the following projects for their contributions and inspiration:
[R-KV](https://github.com/Microsoft/R-KV) | [SnapKV](https://github.com/FasterDecoding/SnapKV)

[@DeadByDawn101](https://github.com/DeadByDawn101) (RavenX AI) — MLX port for Apple Silicon

[@kishan5111](https://github.com/kishan5111) — GPT-OSS-120B model integration

[@dscain](https://github.com/dscain) — DGX Spark (GB10) enablement for vLLM and non-vLLM paths

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
