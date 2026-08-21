---
name: kv-cache-compression-vllm-ascend
description: Use when adapting KV-cache compression tools (kvpress, SqueezeAttention, TriAttention, or new ones) to vLLM-Ascend — designing scheduling/eviction across multiple compression dimensions, or delivering the final combined repo with vllm serve launch instructions and pushing to a provided GitHub link. Always organize around the KV memory formula and TriAttention's proven Ascend scheduling philosophy.
---

# KV-Cache Compression × vLLM-Ascend

## 1. 总纲：一切优化围绕 KV 内存公式

```
memory(KV) = 2 × N_tokens × N_layer × N_head × d_head × sizeof(dtype)
```

任何压缩特性必须能回答：**它削减公式里的哪一项、削减多少**。

| 维度 | 削减的项 | 代表工具 |
|---|---|---|
| token 维度（行级 eviction） | `N_tokens`（每层行保留 K 个 token） | kvpress press（Knorm/StreamingLLM/SnapKV/TOVA…）、TriAttention recency |
| layer 维度（逐层预算） | `N_layer × N_tokens` 的分配（重要层多留、次要层少留） | SqueezeAttention（KMeans 逐层预算） |
| head 维度（逐头预算） | `N_head × N_tokens` 的分配 | kvpress AdaKV 类（Ascend kernel 不可物理实现，拒绝） |
| 量化 | `sizeof(dtype)` | vLLM-Ascend 自带 cache_dtype（不在本 skill 范围） |

在 vLLM-Ascend 上，**块表行和 `seq_lens` 跨层共享**、attention kernel 不接受
per-position mask，因此：

- 物理行长度必须统一 → 逐层不同 token 数的预算无法直接表达
  （uniform 用 `K = max(预算)`；逐层差异只体现在 keep set 与预算日志里）；
- 逐头不同保留数无法物理实现（mask 类 press 启动即拒绝）；
- 物理内存 = `2 × L × K × H × d × sizeof(dtype)`（K = 共享行保留数）。

## 2. 调度顺序与驱逐规则（多维度兼容 vllm-ascend 的核心）

### 2.1 调度顺序（先排维度，再排执行）

多维度叠加时，**先确定每个维度"决定什么"，再确定"谁执行物理驱逐"**：

1. **layer 维度先行**（prefill 完成后）：逐层重要性（hidd_data）→ KMeans 逐层预算
   `sliding_windows[L]`；无 hidd_data（编译 prefill / 0 层 hook）时退化为 uniform
   `ini_size` 预算。
2. **token 维度次之**（每个压缩边界）：逐层打分（keys-only 直接 gather 块缓存 K，
   query 类 press 无 query 时退化为 recency）→ 每层 keep set。
3. **物理驱逐最后，且只做一次**：逐层原地压缩（kept 条目排列到行前缀）→ 块行收缩
   → 调度器侧块回收 → 有效长度/输入覆盖更新。

### 2.2 驱逐规则（开几个特性 = 驱逐几次）

- **只开一个特性开关**（如只 `KVPRESS_ENABLE=1` 或只 `SQUEEZE_ENABLE=1`）：
  该特性自己完成驱逐（一次压缩 + 一次回收）。
- **同时开多个特性**（如 kvpress + SqueezeAttention）：**最后只驱逐一次**——
  各维度只贡献"决策"（layer 预算、token keep set），由一个组合 runner 合并成
  **一次**物理驱逐；严禁出现双 proxy → 双压缩 → 重复释放块。
- 实现方式（本仓库 `kvpress-ascend` 的 `KVPRESS_COMBO=1`）：
  单一调度器 patch + 单一 combo runner proxy；第二个插件检测 combo 激活后
  跳过独立安装（日志 `combo mode active ... standalone install skipped`）。
- 验收测试必须断言：每个压缩边界**恰好一个 applied 事件**、被释放块**无重复**。

## 3. Ascend 上的调度/驱逐逻辑：学 TriAttention

TriAttention（本仓库 `tri_3_5-fix-partial-rope-qwen35-v0.23.0`）是
**在 vllm-ascend v0.23.0 真机验证过、真实压缩**的参照实现。细节必须由 agent
自行阅读该仓库源码（`triattention/vllm/runtime/` 下的 scheduler.py、worker.py、
runner.py、runner_state_updates.py、prefill_phase.py、thresholds.py、
input_patch_*、kv_compaction.py、kv_group_resolver.py、docs/vllm_ascend.md）。
以下是被验证过的调度哲学摘要：

1. **worker 侧自足（最关键）**：不依赖 forward 内 hook 和 new-req 解析。
   压缩边界由 worker 从**块表真实容量**（`num_blocks_per_row × block_size`，
   `_get_actual_kv_from_model_runner` 同款）自推导长度并**自触发**；
   state 懒回填（`_ensure_state_for_existing_request`，从
   `base_runner.requests` / `input_batch.num_prompt_tokens` 现取）。
   即使 engine-core 调度器信号缺失/滞后，压缩依然执行。
2. **信号与阈值**：调度器在 `schedule()` 里按
   `budget + max(min_reclaim_blocks × block_size)` 产生候选信号；worker 再按
   真实容量验证。阈值语义：`KV_BUDGET` 是**每层行保留 token 数**（内存
   `2×L×K×…`），等价压缩比 ≈ `1 − K/prompt_len`。
3. **prefill 判定用逻辑进度，绝不用压缩后的有效长度**：
   - 请求还在 `scheduled_new_reqs`（chunked prefill 期间一直保留）→ prefill；
   - `num_scheduled_tokens > 1` 但 `scheduled_spec_decode_tokens` 非空
     （MTP 推测解码，1 目标 + N draft）→ **不是** prefill（否则每一步解码
     都被跳过，永远不压缩——本仓库踩过的坑，见 v0.3.1）；
   - 仅普通解码且已知 prompt 长度才用 `num_computed < prefill_len` 兜底。
4. **key 归一化**：`num_scheduled_tokens` / `scheduled_spec_decode_tokens` 的
   key 可能是 request 对象或 MTP tuple，统一归一化为 req_id
   （`req_id_from_scheduled_key` 同款）。
5. **物理压缩**：从块缓存 gather 稠密 K/V（`[num_blocks, block_size, H, D]`
   或 `[2, …]`），按 per-head keep set 原地排列到行前缀（`prefix_only`），
   行收缩到 `ceil(retained/block_size)`，调度器侧把多余块归还
   （`req_to_blocks` 截断 + `block_pool.free_blocks`），并同步
   `_prepare_inputs` 的 seq_lens / positions / slot mapping（有效长度覆盖）。
6. **禁止脆弱依赖**：
   - 不做 forward 内张量运算的 hook（会被 torch.compile/dynamo 追踪进图，
     导致 npugraph_ex/AOT 产物输出数变化 → `too many values to unpack`）；
     hook 要么只在非编译期存引用（`torch.compiler.is_compiling()` 短路），
     要么干脆不用 hook（keys-only 评分直接 gather 缓存 K）。
   - 评分、触发、长度推导都应有"无 hook / 无信号 / 无注册"的退化路径，
     并用测试验证剥离全部通道后仍能压缩。
7. **MTP/混合模型**：`kv_cache_config.kv_cache_groups` 有多个 KV group，
   只压缩 full-attention group（linear/mamba group 的层名带标记即跳过），
   draft group 的块行保持不动；块回收按 per-gid 计划执行。

## 4. 产出物：总 git 仓库 + README + 直推

1. 最终交付是一个**总 git 仓库**，包含：
   - 每个适配包（`*-ascend`，pip 可安装，`vllm.general_plugins` 入口，
     env 开关激活：`export XXX_ENABLE=1`，别名 `export XXX=1`）；
   - 参考实现目录（原工具源码 + 验证过的 tri_3_5）；
   - 总 `README.md`：**怎么一起 pip install、怎么拉起 vllm serve**
     （含用户实际启动命令：TP/DP、compilation-config、speculative_config、
     hf-overrides 等，原样保留）+ 每个特性/组合模式的 env 参数表 +
     每步推理的探针日志样例（`core_entered=1 hook_entered=1 …`）+ 故障排查。
2. 探针开关是硬性要求：每次推理打印 patch 是否进入核心代码及核心参数
   （`seq_len/budget/keep/reclaimed_blocks/compress_events`），
   并加调度器心跳日志（engine-core 每 N 步一行）证明调度 patch 活着。
3. 测试：本机无 NPU 时用 stub 镜像 vllm-ascend 接口（NPUWorker /
   NPUModelRunner / BlockTable / V1 Scheduler / KV cache manager），
   CPU 上跑真实 patch 代码做模拟调试；覆盖插件激活、压缩内容正确性、
   块回收、输入覆盖、探针开关、MTP 多 group、多插件共存、
   "剥离全部调度通道仍压缩"、MTP spec-decode 门控。
4. 完成后等用户提供 GitHub 链接，**直接 push**（SSH 可用时用
   `ssh://git@github.com/<owner>/<repo>.git` 绕过 https 代理重写；
   push 前 `git ls-remote` 确认远端存在且为空）。
5. 版本管理：每次修复 bump patch 版本并在 README 强调重装，避免
   "改了代码但机器上还是旧包"的排查死循环。
