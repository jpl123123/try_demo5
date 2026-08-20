"""Stub: vllm_ascend.worker.model_runner_v1.NPUModelRunner.

A CPU-executable stand-in that mirrors the vllm-ascend v0.23.0 V1 model
runner surfaces used by the patches:

- ``_prepare_inputs(scheduler_output, num_scheduled_tokens)``: builds
  positions / seq_lens / slot mapping from the input batch;
- ``execute_model``: preprocess -> prepare -> forward (attention layers write
  K/V into the block cache through the slot mapping) -> ModelRunnerOutput;
- buffers: ``positions``/``_positions_np_buf``, ``seq_lens`` +
  ``optimistic_seq_lens_cpu``, ``query_start_loc``/``query_pos`` (CpuGpuBuffer),
  ``arange_np``, ``input_batch``, ``requests``, ``cache_config``,
  ``kv_cache_config``, ``compilation_config``, ``model``, ``kv_caches``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .block_table import BlockTable, CpuGpuBuffer, MultiGroupBlockTable


class StubInputBatch:
    def __init__(
        self,
        req_ids: list[str],
        num_computed_tokens_cpu: np.ndarray,
        num_prompt_tokens: np.ndarray,
        block_table: Any,
        max_num_reqs: int,
    ):
        self.req_ids = req_ids
        self.req_id_to_index: dict[str, int] = {
            rid: i for i, rid in enumerate(req_ids)
        }
        self.num_computed_tokens_cpu = num_computed_tokens_cpu
        self.num_prompt_tokens = num_prompt_tokens
        self.block_table = block_table
        self.num_reqs = len(req_ids)
        self.num_tokens = 0
        self.num_tokens_after_padding = 0


class StubAttentionMeta:
    def __init__(self, slot_mapping: np.ndarray, seq_lens: np.ndarray):
        self.slot_mapping = slot_mapping
        self.seq_lens = seq_lens


class StubAttentionModule(torch.nn.Module):
    """vLLM-style attention layer: forward(positions, query, key, value,
    kv_cache, attn_metadata) writes K/V into the block cache and returns
    attention output (identity-ish, CPU-simulated)."""

    def __init__(self, num_heads: int, head_dim: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

    def forward(self, positions, query, key, value, kv_cache, attn_metadata):
        if isinstance(kv_cache, (list, tuple)):
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        else:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        slots = np.asarray(attn_metadata.slot_mapping)
        query = query.float()
        key = key.float()
        value = value.float()
        num_tokens = int(key.shape[0])
        for t in range(num_tokens):
            slot = int(slots[t])
            if slot < 0:
                continue
            block_id = slot // k_cache.shape[1]
            offset = slot % k_cache.shape[1]
            k_cache[block_id, offset] = key[t]
            v_cache[block_id, offset] = value[t]
        # Simulated attention output: weighted sum of V over the row.
        return value.sum(dim=0).unsqueeze(0).expand(num_tokens, -1, -1)


class StubDecoderLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, hidden: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden = hidden
        self.self_attn = StubAttentionModule(num_heads, head_dim, layer_idx)
        self.qkv = torch.nn.Linear(hidden, (num_heads * head_dim) * 3, bias=False)
        self.o = torch.nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.attention = self.self_attn  # alias for hook discovery

    def forward(self, hidden_states, positions=None, attn_metadata=None, kv_cache=None):
        qkv = self.qkv(hidden_states)
        h = qkv.shape[-1] // 3
        q, k, v = qkv.split(h, dim=-1)
        num_heads = self.self_attn.num_heads
        head_dim = self.self_attn.head_dim
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_heads, head_dim)
        v = v.view(-1, num_heads, head_dim)
        attn_out = self.self_attn(positions, q, k, v, kv_cache, attn_metadata)
        return hidden_states + self.o(attn_out.flatten(1))


class StubDecoderModel(torch.nn.Module):
    def __init__(self, num_layers: int, hidden: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed = torch.nn.Embedding(32, hidden)
        self.layers = torch.nn.ModuleList(
            [
                StubDecoderLayer(i, hidden, num_heads, head_dim)
                for i in range(num_layers)
            ]
        )
        self.lm_head = torch.nn.Linear(hidden, 32, bias=False)

    def forward(self, input_ids, positions=None, attn_metadata=None, kv_caches=None):
        hidden = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden = layer(hidden, positions, attn_metadata, kv)
        return self.lm_head(hidden)


class StubKVGroupSpec:
    def __init__(self, name: str, layer_names: list[str]):
        self.name = name
        self.layer_names = layer_names
        self.kv_cache_spec = StubAttentionSpec(name)


class StubAttentionSpec:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"StubAttentionSpec({self.name})"


class StubKVGroup:
    def __init__(self, gid: int, name: str, layer_names: list[str]):
        self.gid = gid
        self.name = name
        self.layer_names = layer_names
        self.kv_cache_spec = StubAttentionSpec(name)


class StubKVCacheConfig:
    def __init__(self, block_size: int, groups: list[StubKVGroup]):
        self.block_size = block_size
        self.kv_cache_groups = groups
        self.cache_dtype = "auto"


class StubCompilationConfig:
    def __init__(self, static_forward_context: dict[str, Any]):
        self.static_forward_context = static_forward_context


class StubCacheConfig:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.cache_dtype = "auto"


class StubModelConfig:
    def __init__(self):
        self.enforce_eager = True
        self.max_model_len = 262144
        self.model = "stub-model"


class StubVllmConfig:
    def __init__(self):
        self.model_config = StubModelConfig()
        self.use_v2_model_runner = False


class NPUModelRunner:
    """CPU stand-in for vllm_ascend.worker.model_runner_v1.NPUModelRunner."""

    def __init__(
        self,
        *,
        num_layers: int = 4,
        hidden: int = 16,
        num_heads: int = 4,
        head_dim: int = 4,
        block_size: int = 8,
        max_num_reqs: int = 16,
        max_num_blocks_per_req: int = 64,
        num_groups: int = 1,
    ):
        self.device = torch.device("cpu")
        self.model = StubDecoderModel(num_layers, hidden, num_heads, head_dim)
        self.cache_config = StubCacheConfig(block_size)
        self.model_config = StubModelConfig()
        self.vllm_config = StubVllmConfig()
        self.max_num_reqs = max_num_reqs
        self.block_size = block_size
        self.num_kv_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups

        layer_names = [
            f"model.layers.{i}.self_attn" for i in range(num_layers)
        ]
        groups = [StubKVGroup(g, f"group{g}", layer_names) for g in range(num_groups)]
        self.kv_cache_config = StubKVCacheConfig(block_size, groups)

        num_blocks = max_num_blocks_per_req * max_num_reqs
        # Per-layer block caches (group gid -> layer -> cache tensors).
        self.kv_caches: list[Any] = []
        static_forward_context: dict[str, Any] = {}
        for i in range(num_layers):
            k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
            v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
            self.kv_caches.append((k_cache, v_cache))
            layer_holder = StubLayerHolder((k_cache, v_cache))
            static_forward_context[layer_names[i]] = layer_holder
        self.compilation_config = StubCompilationConfig(static_forward_context)

        tables = [
            BlockTable(block_size, max_num_reqs, max_num_blocks_per_req)
            for _ in range(num_groups)
        ]
        self.block_table: Any = (
            tables[0] if num_groups == 1 else MultiGroupBlockTable(tables)
        )
        self.input_batch: Optional[StubInputBatch] = None
        self.requests: dict[str, Any] = {}
        self.seq_lens = CpuGpuBuffer(np.zeros(max_num_reqs, dtype=np.int32))
        self.optimistic_seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.positions = torch.zeros(max_num_reqs * max_num_blocks_per_req * block_size, dtype=torch.int64)
        self._positions_np_buf = self.positions.numpy()
        self.query_start_loc = CpuGpuBuffer(np.zeros(max_num_reqs + 1, dtype=np.int64))
        self.query_pos = CpuGpuBuffer(np.zeros(max_num_reqs * max_num_blocks_per_req * block_size, dtype=np.int64))
        self.arange_np = np.arange(max_num_reqs, dtype=np.int64)

    def _build_input_batch(self, scheduler_output) -> StubInputBatch:
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        n = len(req_ids)
        num_computed = np.zeros(self.max_num_reqs, dtype=np.int64)
        num_prompt = np.zeros(self.max_num_reqs, dtype=np.int64)
        for i, rid in enumerate(req_ids):
            req = self.requests.get(rid)
            if req is not None:
                num_computed[i] = int(getattr(req, "num_computed_tokens", 0) or 0)
                num_prompt[i] = int(getattr(req, "num_prompt_tokens", 0) or 0)
        batch = StubInputBatch(req_ids, num_computed, num_prompt, self.block_table, self.max_num_reqs)
        # Rebuild block rows from the request states.
        for i, rid in enumerate(req_ids):
            req = self.requests.get(rid)
            if req is not None:
                self.block_table.add_row(list(req.block_ids or []), i)
        self.input_batch = batch
        return batch

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        """Mirror of NPUModelRunner._prepare_inputs (vllm-ascend v0.23.0)."""
        batch = self._build_input_batch(scheduler_output)
        num_reqs = batch.num_reqs
        total = int(scheduler_output.total_num_scheduled_tokens)
        self.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu = np.zeros(num_reqs + 1, dtype=np.int64)
        cu[1:] = np.cumsum(num_scheduled_tokens[:num_reqs])
        self.query_start_loc.np[: num_reqs + 1] = cu
        self.query_pos.np[:total] = np.concatenate(
            [np.arange(int(n)) for n in num_scheduled_tokens[:num_reqs]]
        )
        positions_np = self._positions_np_buf[:total]
        np.add(
            batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[:total],
            out=positions_np,
        )
        self.seq_lens.np[:num_reqs] = batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
        self.optimistic_seq_lens_cpu[:num_reqs] = torch.from_numpy(self.seq_lens.np[:num_reqs])
        positions_gpu = torch.from_numpy(positions_np).long()
        qsl_gpu = torch.from_numpy(cu).long()
        self.block_table.compute_slot_mapping(num_reqs, qsl_gpu, positions_gpu)
        return None, None, total

    def _model_forward(self, scheduler_output, total, input_ids_np, positions_np):
        batch = self.input_batch
        num_reqs = batch.num_reqs
        slot_table = self.block_table
        if hasattr(slot_table, "block_tables"):
            slot_table = slot_table.block_tables[0]
        slot_mapping = slot_table.slot_mapping.np[: slot_table.num_slots]
        meta = StubAttentionMeta(slot_mapping, self.seq_lens.np[:num_reqs])
        input_ids = torch.from_numpy(input_ids_np).long()
        positions = torch.from_numpy(positions_np).long()
        return self.model(input_ids, positions, meta, self.kv_caches)

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        num_scheduled_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[rid]
                for rid in scheduler_output.num_scheduled_tokens
            ],
            dtype=np.int64,
        )
        total = int(scheduler_output.total_num_scheduled_tokens)
        batch = self._build_input_batch(scheduler_output)
        self._prepare_inputs(scheduler_output, num_scheduled_tokens)
        input_ids_np = np.random.randint(0, 32, size=total)
        logits = self._model_forward(scheduler_output, total, input_ids_np, self._positions_np_buf[:total])

        from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

        output = ModelRunnerOutput(kv_connector_output=KVConnectorOutput())
        output.req_id_to_index = dict(batch.req_id_to_index)
        output.req_ids = list(batch.req_ids)
        output.output_tokens = torch.zeros(total, dtype=torch.long)
        output.logprobs = logits
        output.num_scheduled_tokens = num_scheduled_tokens
        # NOTE: request num_computed_tokens bookkeeping is done by the
        # SimLoop driver (mirroring vLLM scheduler _update_states); the runner
        # must NOT advance it here or the counter double-advances.
        return output

    def sample_tokens(self, grammar_output=None):
        from vllm.v1.outputs import ModelRunnerOutput

        output = ModelRunnerOutput(kv_connector_output=None)
        output.output_tokens = torch.zeros(1, dtype=torch.long)
        return output


class StubLayerHolder:
    """Stand-in for a static-forward-context layer exposing its kv_cache."""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
