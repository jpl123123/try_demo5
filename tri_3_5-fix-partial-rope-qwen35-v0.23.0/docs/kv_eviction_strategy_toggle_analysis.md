# KV Eviction Strategy Toggle Analysis

生成时间：2026-06-16  
当前分支：`tri_zxj_version0615`  
当前提交：`312a132c7f759763e71ea64b91e821b7402861bb`

本文只分析当前代码行为，不包含代码改动建议的实现。

## 结论先行

`triattention/vllm/runtime/config.py` 第 64、65、67 行不是一个完整的“关闭 KV 驱逐总开关”。

它们当前语义分别是：

| 行号 | 参数 | 当前默认 | 真实含义 | 是否能单独关闭 KV 驱逐 |
|---:|---|---:|---|---|
| 58 | `disable_compression` | `False` | 停止 scheduler 对未压缩请求发出长度触发压缩信号 | 接近“停止未来压缩”的主开关，但不会回滚已压缩请求 |
| 60 | `enable_kv_usage_trigger` | `False` | 是否额外启用 KV usage 压力触发 | 只能关 usage 触发，关不掉 length threshold 触发 |
| 64 | `enable_experimental_kv_compaction` | `True` | 控制 worker hook 是否真正执行 KV tensor compaction；runner 侧 worker self-trigger 也依赖它 | 不能清理已压缩状态；也不是 scheduler 触发总开关 |
| 65 | `enable_experimental_block_reclaim` | `True` | 控制 hook/scheduler 是否生成并执行 block reclaim payload | 不能关闭所有“有效长度缩短”；worker 侧还有 event-driven sync |
| 67 | `require_physical_reclaim` | `True` | 如果预计应物理 reclaim 却没有 reclaim，就报错 | 不是关闭开关，只是严格性校验 |

所以，把第 64、65、67 行设为 `False` 后仍看到 KV usage 很低，常见原因是：

1. 已经发生过的 compression/reclaim 不能被配置回滚。释放掉的 block 不会因为开关变 false 自动重新分配回来。
2. 当前请求只要进入过压缩态，`RequestStateStore` 和 scheduler 的 `EffectiveCacheLenTracker` 会继续让后续 decode 使用 shorter effective length，直到请求结束。
3. 第 64/65/67 行不控制 scheduler 的 length-threshold signal。真正让 scheduler 不给未压缩请求发新压缩信号的是 `disable_compression=True` 且 `enable_kv_usage_trigger=False`。
4. 配置由 `TriAttentionRuntimeConfig.from_env()` 读取，环境变量会覆盖源码默认值。运行中的 vLLM 进程也不会自动吃到你改源码后的默认值，通常需要重启。
5. 当前 worker 侧的 `apply_worker_block_reclaim_events()` 是 event-driven，没有读取 `enable_experimental_block_reclaim` 作为入口开关。只要 pending event 是 `status="applied"` 且带 `cache_len_after`，它就可能缩短 worker block table。

如果目标是“开启/关闭当前驱逐策略并观察 KV usage 高低差异”，当前代码里最接近的安全使用方式是：在新进程、新请求开始前设置：

```bash
TRIATTN_RUNTIME_DISABLE_COMPRESSION=1
TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER=0
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=0
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0
TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM=0
```

但严格说，这仍然只是“从启动起不再触发/执行新的压缩驱逐”。它不能恢复已经被驱逐的 active request，也不能保证老进程里 pending 的 compression event 不被消费。

## 当前 KV 驱逐链路

当前实现分成四层：

1. Scheduler 产生 compression signal。
2. Runner/worker hook 执行 KV compaction，并生成 compression event。
3. Event 通过 worker -> engine_core 的通道回传。
4. Scheduler 和 worker 两侧根据 event 更新 block table、effective length、slot mapping。

简化链路如下：

```text
Scheduler.schedule()
  -> triattention_signals
  -> TriAttentionModelRunner.execute_model()
      -> execute_runner_compression_actions()
          -> base_runner.triattention_apply_compression()
              -> hook_impl: compact / zero-copy-remap / block_reclaim payload
      -> apply_worker_block_reclaim_events()
      -> execute_base_model_with_effective_overrides()
      -> attach compression events to ModelRunnerOutput
  -> Scheduler.update_from_output()
      -> read events
      -> _apply_compression_events()
      -> scheduler block_pool free_blocks()
      -> refresh kv_cache_usage stats
```

## 本次 KV 驱逐修复的真实功能改动记录

### 1. `runner_output_bridge.py`

文件：`triattention/vllm/runtime/runner_output_bridge.py`

| 行号 | 功能改动 |
|---:|---|
| 30-45 | 新增 `_TriattentionEventBag`。它是一个 module-level、可 pickle 的事件容器，用来承载 compression events。放在 module scope 是为了跨进程反序列化时能按名字找到类。 |
| 35-36 | 构造时把传入 events 转成 list，避免引用外部可变对象。 |
| 38-39 | 实现 `__reduce__`，让 pickle/cloudpickle 能重建 `_TriattentionEventBag(events)`。 |
| 41-45 | 实现 `__getstate__` / `__setstate__`，保证事件列表能稳定序列化。 |
| 271-284 | `execute_model()` 返回 `None` 时，不再丢弃 pending events。它仍挂到 `scheduler_output.triattention_compression_events` 作为同进程 fallback，但返回值继续保留 `pending_events`，等待后续 `sample_tokens()` 产出真实 ModelRunnerOutput。 |
| 285 | 当 `execute_model()` 已经拿到 output 时，先调用 `_attach_triattention_events_via_kv_cache_events()` 写入 vLLM 声明字段。 |
| 286-296 | 保留旧的 `output.triattention_compression_events` 动态属性作为本进程兼容路径；如果 setattr 失败，pending events 不清空，留给 fallback。 |
| 299-323 | 新增 `_attach_triattention_events_via_kv_cache_events()`，把 events 写入 `output.kv_connector_output.kv_cache_events`。这是 vLLM V1 async worker -> engine_core 能跨 pickle 的字段。 |
| 307-313 | 如果 output 是 async wrapper，不直接写 wrapper，而是找 `model_runner_output` 或 `_model_runner_output` 中真正的 ModelRunnerOutput。 |
| 314-320 | 如果 `kv_connector_output` 缺失，则创建 `vllm.v1.outputs.KVConnectorOutput()`，再把 `_TriattentionEventBag` 写到 `kv_cache_events`。 |
| 326-339 | 新增 `_read_triattention_events_from_kv_cache_events()`，作为 engine_core 侧读取 `kv_connector_output.kv_cache_events.events` 的反向 helper。 |
| 342-352 | `sample_tokens()` fallback 也先写 `kv_connector_output.kv_cache_events`，再写旧动态属性。这样 vLLM async 路径终于能把 worker 侧 applied events 送回 scheduler。 |

影响：

- 以前 async 路径里，动态属性可能在 worker -> engine_core pickle 过程中丢失，scheduler 收不到 applied event，因此无法真实 free scheduler block pool。
- 当前版本把 event 放进 vLLM 声明字段后，scheduler 可以收到 event，所以 KV eviction / block reclaim 开始真正生效，KV usage 会明显降低。

### 2. `integration_monkeypatch.py`

文件：`triattention/vllm/runtime/integration_monkeypatch.py`

| 行号 | 功能改动 |
|---:|---|
| 28 | 从 `runner_output_bridge` 引入 `_read_triattention_events_from_kv_cache_events`。 |
| 210-215 | 在 scheduler `update_from_output()` 中，优先从 `model_runner_output.kv_connector_output.kv_cache_events` 读取 compression events。 |
| 216-222 | 如果官方字段没有事件，再回退读取 `model_runner_output.triattention_compression_events` 动态属性。 |
| 223-230 | 如果 model_runner_output 也没有，再回退读取 `scheduler_output.triattention_compression_events`。 |
| 231-240 | 只要读到 events，就调用 `TriAttentionScheduler._apply_compression_events()`，并用 scheduler 当前 `kv_cache_manager.usage` 刷新输出统计。 |

影响：

- 这是让“worker 已压缩/驱逐”影响 scheduler block pool 的关键入口。
- 如果这里读不到 event，worker 侧可能已经改了自己的 block table，但 scheduler block pool 不会 free，KV usage 不会正确下降。
- 当前修复后，scheduler 能稳定收到 event，所以你看到 KV usage 低，反而说明事件通路和 scheduler reclaim 已经在工作。

### 3. `input_patch_vllm_v1_backend.py`

文件：`triattention/vllm/runtime/input_patch_vllm_v1_backend.py`

| 行号 | 功能改动 |
|---:|---|
| 336-358 | 新增 `_effective_block_table_capacity()`，从 worker 本地 `input_batch.block_table` 读取每个 request row 当前容量。多 group 时取最小容量，避免某个 group 越界。 |
| 344-355 | 遍历 inner block tables，读取 block size 与 `num_blocks_per_row[row]`，计算当前 row 的 token capacity。 |
| 346-351 | 如果 table 上没有 block size，回退读取 `input_batch.cache_config.block_size`。 |
| 361-375 | 新增 `_clamp_effective_base_to_capacity()`。当 `effective_base + num_scheduled` 会超过 worker 本地容量时，把 effective base clamp 到 `capacity - num_scheduled`。 |
| 395-405 | 单请求 `ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE` 路径中，在重建 slot positions 前应用 capacity clamp。 |
| 407-435 | 多请求 sparse effective base 路径中，对每个 row 单独应用 capacity clamp，再写入 slot positions。 |
| 485-494 | 单请求 seq_len override 路径中，把 `base + scheduled` clamp 到 worker 本地 block table capacity。 |
| 503-512 | 多请求 sparse seq_len override 路径中，对每个 row 的 new seq_len 做同样 clamp。 |

影响：

- 这个文件不是“触发驱逐”的地方，而是“已经驱逐后如何适配 worker 输入”的地方。
- 当前修复解决的是：scheduler/worker async 边界上，effective base 可能比 worker 本地 block table 看到的容量更靠前，导致 slot position 或 seq_len OOB。
- clamp 后，驱逐策略可以在 vLLM-Ascend V1 async 路径里继续跑，而不因为边界 step 的本地容量滞后崩掉。

### 4. 测试改动

文件：

- `triattention/tests/test_runner_output_bridge.py`
- `triattention/tests/test_ascend_input_patch.py`

| 文件 | 行为覆盖 |
|---|---|
| `test_runner_output_bridge.py` | 覆盖 `_TriattentionEventBag` pickle 往返，以及 `sample_tokens()` wrapper 内部 ModelRunnerOutput 的 `kv_cache_events` 挂载/读取。 |
| `test_ascend_input_patch.py` | 把原来 “effective seq 超容量直接 OOB” 的预期改成 “clamp 到 block table 最后一个合法 slot 并提交 slot mapping”。 |

## 为什么第 64/65/67 行设 false 仍可能 KV usage 很低

### 原因 1：第 64 行不是 scheduler 总开关

`config.py:64`

```python
enable_experimental_kv_compaction: bool = True
```

它主要影响：

- `hook_impl.py:193-207`：如果为 false，worker hook 不执行真实 KV compaction，返回 `plan_only` 且 `applied=False`。
- `runner.py:689-690`：如果为 false，runner 不做 worker self-trigger。

但 scheduler 的 length threshold signal 不是由它关闭的。只要 `disable_compression=False`，scheduler 仍可能在 `_build_signals()` 中生成 `should_compress=True` 的 signal。

### 原因 2：第 65 行不是所有 worker reclaim 的唯一入口

`config.py:65`

```python
enable_experimental_block_reclaim: bool = True
```

它影响：

- `hook_group_pipeline.py:471-474`：是否把 group outcome 的 kept block ids 写回，并收集 reclaim groups。
- `hook_group_pipeline.py:526-540`：是否生成 `block_reclaim` payload。
- `scheduler.py:645-646`：scheduler 收到 event 后，如果这个开关为 false，会跳过 scheduler-side physical block reclaim。

但当前 `runner.py:1304` 每 step 都会调用：

```python
apply_worker_block_reclaim_events()
```

而 `worker_reclaim_sync.py:99-124` 的入口只检查 debug env 和 events，不检查 `enable_experimental_block_reclaim`。之后 `worker_reclaim_sync.py:172-232` 对所有 `status="applied"` 的 event，根据 `cache_len_after` / `retained_cache_len` 缩短 worker block table。

所以如果已经有 applied event，worker 侧仍可能表现为短 KV/effective length。scheduler 侧 free block pool 由 `enable_experimental_block_reclaim` 控制，但 worker 侧 block table sync 当前是 event-driven。

### 原因 3：第 67 行只是严格校验，不是关闭 reclaim

`config.py:67`

```python
require_physical_reclaim: bool = True
```

它在 `layout_engine.py:213-226` 里使用：如果 expected removed blocks 没有实际产生，才报 `physical_reclaim_missing`。设为 false 的效果是“不要求必须物理 reclaim 成功”，不是“不执行 reclaim”。

### 原因 4：已经压缩过的 request 会持续使用 effective overrides

`state.py:101-128` 中，成功 applied 后会：

- `compression_count += 1`
- 写入 `cache_len_after_last_compression`
- 写入 `nct_at_last_compression`
- 把 req_id 放进 `_compressed_req_ids`

之后 `effective_overrides.py:275-290` 会检查是否存在 active compressed requests；如果存在，就继续进入 override 构造。

`effective_overrides.py:312-369` 会基于 `cache_len_after_last_compression` 和 `nct_at_last_compression` 计算稳定 delta，并为压缩过的 request 生成 shorter seq base / pos delta。

这意味着：

- 开关变 false 后，已经压缩过的请求不会自动回到原始全长 KV 语义。
- 已释放的物理 block 也不会自动恢复。
- 观察同一个 active request 的 KV usage，可能仍然很低。

### 原因 5：async pending events 可能滞后到配置切换之后才被 scheduler 消费

当前修复让 event 通过 `kv_connector_output.kv_cache_events` 跨进程传回来。

路径是：

- `runner_output_bridge.py:271-284`：`execute_model()` 返回 None 时，pending event 继续保留。
- `runner_output_bridge.py:342-352`：`sample_tokens()` 产出真实 output 后再挂载 event。
- `integration_monkeypatch.py:210-240`：engine_core 下一侧 `update_from_output()` 消费 event。

因此，如果你在已有请求运行中修改开关，之前某一步产生的 applied event 仍可能在后续 output 中才被 scheduler 消费。现象上就像“我关了开关但还在驱逐”。

### 原因 6：源码默认值可能被环境变量覆盖

`config.py:218-245` 会从环境变量读取：

- `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION`
- `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM`
- `TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM`
- `TRIATTN_RUNTIME_DISABLE_COMPRESSION`
- `TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER`

如果运行脚本或 shell 里设置了 env，源码默认值不会生效。并且 vLLM 服务进程启动后，修改 `.py` 文件默认值不会改变已加载进程里的 config。

## 当前开关语义分层

### A. 停止未来新压缩

目标：新请求不再触发 compression/reclaim。

当前最接近的组合：

```bash
TRIATTN_RUNTIME_DISABLE_COMPRESSION=1
TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER=0
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=0
```

注意：

- 必须在进程启动前设置。
- 对已经压缩过的 active request 不会回滚。

### B. 关闭 scheduler-side physical block reclaim

目标：scheduler 不 free block pool。

当前相关开关：

```bash
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0
TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM=0
```

注意：

- 这会关闭 scheduler `_apply_compression_events()` 里的 physical block reclaim 分支。
- 但只要仍产生 applied event，worker-side `apply_worker_block_reclaim_events()` 当前仍可能根据 event 缩短 worker block table。
- 因此它不是完整的“关闭驱逐策略”。

### C. 允许 logical compaction 但不强制 physical reclaim

目标：可以 compact KV tensor/effective length，但不要因为 missing physical reclaim 报错。

组合：

```bash
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=1
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0
TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM=0
```

注意：

- 这不是“恢复高 KV usage”的配置。
- logical effective length 仍可能变短。

### D. 观察高 KV usage baseline

目标：完整禁用 TriAttention KV 驱逐，让 KV usage 随输入长度/输出长度按 vLLM 原生方式上涨。

建议实验方式：

1. 停止当前 vLLM 服务进程。
2. 清理 shell/env 中所有 `TRIATTN_RUNTIME_*` 相关变量，或者明确设置：

```bash
TRIATTN_RUNTIME_DISABLE_COMPRESSION=1
TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER=0
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=0
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0
TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM=0
```

3. 启动新进程。
4. 用新请求测试，不要复用已经运行过 compression 的 active request。
5. 从启动日志确认实际配置。应重点看类似字段：

```text
disable_compression=True
kv_usage_trigger_enabled=False
block_reclaim_enabled=False
compaction=False
```

## 当前设计里适合补一个总开关的点

如果后续要实现一个真正的“当前驱逐策略总开关”，建议不要只复用第 64/65/67 行，而是新增一个语义更清晰的开关，例如：

```python
enable_kv_eviction_strategy: bool = True
```

它至少需要同时控制：

1. Scheduler 是否生成新的 `triattention_signals`。
2. Runner 是否补 worker self-trigger。
3. Runner 是否执行 `execute_runner_compression_actions()`。
4. Runner 是否执行 `apply_worker_block_reclaim_events()`。
5. Runner 是否为已经压缩过的 request 继续生成 effective overrides。
6. Scheduler `update_from_output()` 是否消费 pending compression events。

但这里有一个安全边界：对于已经压缩并释放 block 的 active request，不能简单“关闭 effective overrides”，否则 attention 会按原始 token position 访问已经不存在或已重排的 KV。真正的 hard-off 只能保证“新请求不进入压缩态”，不能无损恢复已压缩请求。

所以更合理的设计是两个级别：

| 级别 | 语义 | 可否运行中切换 |
|---|---|---|
| soft-off | 不再触发新的 compression；已有 compressed request 继续用 effective overrides 直到结束 | 可以，但 pending events 仍需 drain |
| hard-off | 从进程启动开始完全不装/不用 eviction strategy | 应只在新进程、新请求前启用 |

## 当前现象的最可能解释

你现在把第 64、65、67 行设 false 后 KV usage 仍低，我认为优先排查顺序是：

1. 是否改的是运行进程实际加载的源码路径。
2. 是否有环境变量覆盖了源码默认值。
3. 是否没有重启 vLLM 服务。
4. 是否测试的是已经被压缩过的 active request。
5. 是否有 async pending event 在切换后才被 `update_from_output()` 消费。
6. 是否只关了 64/65/67，但没有设置 `disable_compression=True`。

最关键的一点：第 64/65/67 行不是完整关闭开关。尤其第 67 行完全不是关闭逻辑；第 65 行也只控制部分 physical reclaim；第 64 行不清理既有压缩状态，也不单独关闭 scheduler 的 length-threshold trigger。

