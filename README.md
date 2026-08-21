# try_demo5 — KV-Cache 压缩工具 × vLLM-Ascend v0.23.0 一体化部署

本仓库把三个 KV cache 压缩方案统一适配到 `vllm-ascend-releases-v0.23.0`
（Qwen3.5-27B-w8a8-mtp 场景，TP=4，MTP 推测解码），**不修改任何 vLLM-Ascend 源码**：

| 目录 | 说明 |
|---|---|
| `kvpress-ascend/` | kvpress 的 monkeypatch 适配包（pip 安装，`vllm.general_plugins` 插件） |
| `SqueezeAttention-ascend/` | SqueezeAttention 的 monkeypatch 适配包（pip 安装，插件） |
| `tri_3_5-fix-partial-rope-qwen35-v0.23.0/` | TriAttention 参考实现（vLLM-Ascend 0.23.0 已验证，机制转换的参照物） |
| `kvpress-main/` | kvpress 原始工具源码 |
| `SqueezeAttention-main/` | SqueezeAttention 原始工具源码 |
| `vllm-ascend-releases-v0.23.0/` | 目标 vLLM-Ascend（只读参照，未被修改） |
| `PLAN.md` / `TODO.md` | 完整设计文档 / 执行清单 |

> 三者是**同一运行时可替换**的方案：一次只激活一个（`export KVPRESS_ENABLE=1`
> 或 `export SQUEEZE_ENABLE=1`，TriAttention 用 `export ENABLE_TRIATTENTION=1`），
> 但安装可以全部共存。

---

## 1. 参考运行点（实测口径）

- 输入长度：**32K（32768 tokens）**
- KV budget：**13446 tokens**（TriAttention 实测取值，即保留约 41% 的 KV）

### budget 语义说明（per-token vs per-layer）

**13446 是"每个 layer 每行保留的 token 数"**（行级 KV 长度），三个工具在该口径下
的内存占用都是 `L × 13446`（L = 层数）：

| 工具 | 参数 | 取值 | 语义 |
|---|---|---|---|
| TriAttention（参考） | `TRIATTN_RUNTIME_KV_BUDGET` | `13446` | 每层行保留 13446 token |
| kvpress-ascend | `KVPRESS_KV_BUDGET` | `13446` | 每层行保留 13446 token（与 TriAttention 同口径） |
| SqueezeAttention-ascend（方式 A） | `SQUEEZE_KV_BUDGET` | `13446` | uniform 模式共享 K=13446，内存精确对齐 |
| SqueezeAttention-ascend（方式 B，推荐） | `SQUEEZE_INI_SIZE=0.21` + `SQUEEZE_CLASS3_SIZE=0.41` | K≈13434 | 由逐层机制决定运行点：class3（重要层）预算 = 0.41×32768 ≈ 13446，K = max(预算) 自动落在 13446 附近 |

SqueezeAttention 的预算**本身就是 per-layer 的**（`sliding_windows[L]`），论文口径是
总量守恒 `Σ_L budget_L = L × ini_size × prompt_len`。但 vLLM-Ascend 的块表行和
`seq_lens` 跨层共享，物理行长度必须统一 → uniform 模式取 `K = max(budget_L)`：
- 方式 A：`SQUEEZE_KV_BUDGET=13446` 直接固定 K，内存与 TriAttention/kvpress 完全一致
  （per-layer 差异体现在 `[CLUSTER]` 预算日志中）；
- 方式 B：不设绝对 budget，让 KMeans 决定 K —— 例：ini=0.21、class3=0.41、36 层
  12/12/12 分类时，class1/2 预算 = 0.11×32768 ≈ 3604，class3 = 13434，
  `K = max = 13434 ≈ 13446`，运行点与 TriAttention 一致，同时 `[CLUSTER]` 日志保留
  了"哪些层需要更多预算"的逐层信息。

对应的等价压缩比 ≈ `1 - 13446/32768 ≈ 0.59`。

---

## 2. 一起 pip 安装

```bash
cd try_demo5

# 安装 kvpress-ascend（插件入口：vllm.general_plugins -> register_kvpress_backend）
pip install ./kvpress-ascend

# 安装 SqueezeAttention-ascend（插件入口：vllm.general_plugins -> register_squeezeattention_backend）
pip install ./SqueezeAttention-ascend

# （可选）kvpress 原始库，供 KVPRESS_USE_INSTALLED=1 时复用其 press 类
# pip install ./kvpress-main
```

安装后两个插件即注册到 vLLM；**开关默认关闭**，不设置环境变量时是纯 no-op，
不影响普通 vllm serve。

---

## 3. 拉起 vLLM-Ascend

### 3.0 Combo 模式（两个工具叠加，推荐 —— 只驱逐一次）

同时启用 kvpress 与 SqueezeAttention 时，**组合模式**把它们叠成一条流水线，
每个压缩边界只做**一次物理驱逐**：

- **layer 维度**：SqueezeAttention 的逐层重要性（hidd_data）→ KMeans 逐层预算
  （哪些层重要、该留多少）；
- **token 维度**：kvpress press 逐层打分（每个 layer 留**哪些** token）；
- **一次驱逐**：每个压缩边界 = 一次逐层原地压缩 + 一次块行收缩 + 一次调度器回收事件
  （不会出现双重 proxy / 双重压缩 / 重复释放）。

```bash
# ---- 先升级到 v0.2.0（修复了 layers=0 / seq_len=0 / 无驱逐 的问题）----
pip install ./kvpress-ascend
pip install ./SqueezeAttention-ascend

# ---- 组合模式（两个工具都启用，KVPRESS_COMBO=1 是开关）----
export KVPRESS_ENABLE=1
export SQUEEZE_ENABLE=1
export KVPRESS_COMBO=1                # 组合模式（SqueezeAttention 插件自动跳过独立安装）
export KVPRESS_PRESS=KnormPress       # token 维度：kvpress press
export KVPRESS_KV_BUDGET=13446        # 每层行保留 13446 token（与 TriAttention 同口径）
export KVPRESS_MIN_RECLAIM_BLOCKS=16  # 块大小 128 时约 2048 token 才压缩一次
export KVPRESS_DEFER_PREFILL_COMPRESSION=1
export KVPRESS_RUNTIME_LOGGING=1
export KVPRESS_PROBE=1                # combo 探针走 KVPRESS 日志通道
export SQUEEZE_INI_SIZE=0.21          # layer 维度：逐层预算基准（总预算守恒）
export SQUEEZE_CLASS3_SIZE=0.41       # 高重要性层保留占比 ≈ 13446/32768
export SQUEEZE_START_SIZE=4
export SQUEEZE_KMEANS_SEED=42
# （组合模式下 SQUEEZE_KV_BUDGET / SQUEEZE_PROBE / SQUEEZE_RUNTIME_LOGGING
#   不再生效：K 由 KVPRESS_KV_BUDGET 决定，探针走 KVPRESS_PROBE）
```

组合模式探针（每步每请求一行，两个维度都在）：

```
[KVPRESS-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  mode=combo layers=36 budgets_ready=1 K=13446 start=4 press=KnormPress
  ini=0.210 class3=0.410 seq_len=13446 keep=13446
  reclaimed_blocks=32 compress_events=1 last_event=applied
[KVPRESS-ASCEND][CLUSTER] combo budgets req=req-1 ... class_sizes={0:12,1:12,2:12}
```

注意：combo 模式下 SqueezeAttention 插件会自动跳过独立安装（日志会打印
`combo mode active ... standalone install skipped`），确保只有一条调度/压缩链路。

### 3.1 方式 A：kvpress-ascend（推荐参数，32K / budget 13446）

```bash
export KVPRESS_ENABLE=1                 # 或 export KVPRESS=1
export KVPRESS_PRESS=KnormPress         # 或 StreamingLLMPress / SnapKVPress / TOVAPress ...
export KVPRESS_KV_BUDGET=13446          # 与 TriAttention 实测口径一致
export KVPRESS_MIN_RECLAIM_BLOCKS=16    # 块大小 128 时约 2048 token 才压缩一次
export KVPRESS_DEFER_PREFILL_COMPRESSION=1
export KVPRESS_RUNTIME_LOGGING=1        # 实时有效性开关
export KVPRESS_PROBE=1                  # 每次推理打核心进入日志
```

### 3.2 方式 B：SqueezeAttention-ascend（推荐参数，32K / budget 13446）

```bash
export SQUEEZE_ENABLE=1                 # 或 export SQUEEZE=1
export SQUEEZE_MODE=uniform             # 正确模式（共享块表约束下的 K = max 预算）
# 二选一：
#   方式 A（内存与 TriAttention 精确对齐）：export SQUEEZE_KV_BUDGET=13446
#   方式 B（由逐层机制决定运行点，推荐）：
export SQUEEZE_INI_SIZE=0.21            # 逐层初始预算占比（总预算守恒基准）
export SQUEEZE_CLASS3_SIZE=0.41         # 高重要性层保留占比 ≈ 13446/32768，K≈13434
export SQUEEZE_START_SIZE=4             # StreamingLLM sink tokens
export SQUEEZE_KMEANS_SEED=42           # 聚类可复现
export SQUEEZE_RUNTIME_LOGGING=1
export SQUEEZE_PROBE=1
```

### 3.3 vllm serve 命令（两种方式共用，与平时完全一致）

```bash
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name "qwen3.5" \
  --host 0.0.0.0 \
  --port 1144 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code \
  --async-scheduling \
  --allowed-local-media-path / \
  --quantization ascend \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

---

## 4. 实时有效性验证（每次推理的核心进入日志）

### kvpress-ascend 每步探针

```
[KVPRESS-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  press=KnormPress ratio=0.590 seq_len=32768 budget=13446 keep=13446
  reclaimed_blocks=32 compress_events=1 last_event=applied
```

- `core_entered=1`：runner proxy 拦截生效（压缩边界流程活着）
- `hook_entered=1`：attention hook 本步捕获到 query（评分输入活着）
- 两者同时为 0 ⇒ patch 未生效，检查 `KVPRESS_ENABLE` 与启动日志

### SqueezeAttention-ascend 每步探针

```
[SQUEEZE-ASCEND][PROBE] step=42 req=req-1 core_entered=1 hook_entered=1
  mode=uniform layers=36 budgets_ready=1 K=13446 start=4 ini=0.210 class3=0.410
  seq_len=13446 keep=13446 reclaimed_blocks=32 compress_events=1 last_event=applied
[SQUEEZE-ASCEND][CLUSTER] budgets req=req-1 layers=36 prompt_len=32768 ...
  class_sizes={0:12,1:12,2:12} budgets=[...]
```

---

## 5. 机制转换要点（一句话版）

- **kvpress**：HF `DynamicCache` 稠密 KV ↔ Ascend 分块缓存（`gather_request_kv_dense` +
  按 head 原地压缩到行前缀）；`self_attn` hook ↔ vLLM 解码层 attention hook（捕获
  post-RoPE query）；`cache_position` 重映射 ↔ 调度器有效长度跟踪 + `_prepare_inputs`
  覆盖（seq_lens / positions / slot mapping）；块行收缩 + 调度器侧块回收。
- **SqueezeAttention**：`hiddlayer` 余弦相似度 ↔ 解码层 hook 捕获；KMeans 3 类逐层
  预算（总预算守恒）↔ `core/budgets.py`；逐层 streaming 丢弃 ↔ 逐层 recency keep
  set 的块级压缩。
- **组合模式（KVPRESS_COMBO=1）**：layer 维度预算（SqueezeAttention KMeans）×
  token 维度选择（kvpress press）叠成一条流水线，每个压缩边界只做**一次**物理驱逐
  （一次逐层压缩 + 一次行收缩 + 一次回收），避免双插件时的双重驱逐/重复释放。
- **已知约束**：vLLM-Ascend 块表/seq_len 跨层共享，逐层不同 token 数的预算无法物理
  表达 → `SQUEEZE_MODE=uniform` 用共享 K=max(预算)；`class_weighted` 为实验模式
  （fake-key 填充）。kvpress 的 mask 类 press（AdaKV 等）在 Ascend kernel 上不可
  物理实现，启动时明确拒绝并报错。

## 6. 测试（本机无 NPU，模拟调试）

```bash
pip install pytest scikit-learn
cd kvpress-ascend && python -m pytest tests/        # 38 项（含 combo、namedtuple 注册、编译透明）
cd ../SqueezeAttention-ascend && python -m pytest tests/   # 21 项
```

测试用 stub 镜像 vllm-ascend v0.23.0 的接口（NPUWorker / NPUModelRunner /
BlockTable / V1 调度器 / KV cache manager），在 CPU 上运行真实 patch 代码：
插件激活、压缩内容正确性、块回收、输入覆盖、探针开关、MTP 多 group、两插件共存。

## 7. 故障排查

### 探针全 0（`layers=0 hook_entered=0 seq_len=0 keep=0`，无驱逐）

日志里 `[PROBE]` 全 0 说明两个问题（v0.2.0 已修复，重新 `pip install` 两个包）：

1. `layers=0`：hook 没找到解码层 —— Qwen3.5 的层在 `language_model.model.layers`
   （Qwen3Next 在 `model.model.layers`），旧版只查了一层。新版用递归解析器定位
   层容器（跳过 draft/MTP 模块），并只接受"成员带 self_attn/attention"的容器。
2. `seq_len=0` + 无驱逐：请求没有注册成功 —— 真实 vLLM 的 `scheduled_new_reqs`
   是 `NewScheduledRequest` namedtuple（字段 `req_id, request, num_computed_tokens,
   num_prompt_tokens, num_scheduled_tokens`），旧版 `item[-1]` 取到的是数字。
   新版按字段名解析两种形态。另外新增了**首次压缩 worker 自触发**兜底：即使
   engine-core 的调度器信号缺失/滞后，worker 侧也能按自身 `num_computed_tokens`
   触发第一次压缩。

升级后验证：`[PROBE]` 应出现 `layers=36 ... budgets_ready=1 ... seq_len>0`，
且出现 `COMPRESS` / `[CLUSTER]` 事件行。

### `ValueError: too many values to unpack (expected 24)`（启动 profile_run 阶段）

报错出现在 `determine_available_memory -> profile_run -> _dummy_run` 的编译模型
forward 内部（npugraph_ex / AOT 编译产物）。日志中如果出现
`Loaded npugraph_ex compilation cache ... / Loaded AOT compilation from path ...`，
说明命中了**旧的编译缓存**，缓存产物的输入/输出数量与当前图不一致时就会
unpack 数量不匹配。按顺序排查：

```bash
# 1) 清掉编译缓存（最可能的原因），重跑
rm -rf /root/.cache/vllm /root/.cache/torch

# 2) 隔离验证：先关掉两个插件，确认报错是否与插件无关
unset KVPRESS_ENABLE KVPRESS SQUEEZE_ENABLE SQUEEZE
vllm serve <同上命令>

# 3) 若只有开插件时才报错：我们的 attention hook 在编译期是完全透明的
#    （torch.compiler.is_compiling() 时直接短路，且 hook 内不做任何张量运算），
#    仍出现则提供完整日志（含 [KVPRESS-ASCEND]/[SQUEEZE-ASCEND] 启动行）反馈。
```

注意：SqueezeAttention 的逐层重要性捕获（hidd_data）需要 prefill 走 eager。
若 prefill 被 npugraph_ex 编译（本配置下 profile 阶段即编译 1..4096 range），
hook 在编译路径下不会捕获 → 预算退化为 uniform 初始预算（`SQUEEZE_KV_BUDGET`
仍可固定 K）。需要逐层聚类生效时，可临时加 `--enforce-eager` 验证。

## 8. 文档索引

- `PLAN.md` — 完整设计（机制对照表、patch 面、参数表、限制）
- `TODO.md` — 执行清单 + 上机首跑检查项
- `kvpress-ascend/README.md`、`SqueezeAttention-ascend/README.md` — 各自完整参数表
- `tri_3_5-fix-partial-rope-qwen35-v0.23.0/docs/vllm_ascend.md` — TriAttention 的
  vLLM-Ascend 集成说明（机制转换参照物）
