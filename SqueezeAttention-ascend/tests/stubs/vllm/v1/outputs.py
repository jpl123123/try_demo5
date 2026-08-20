"""Stub: vllm.v1.outputs (ModelRunnerOutput / KVConnectorOutput)."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class KVConnectorOutput:
    kv_cache_events: Any = None


@dataclass
class ModelRunnerOutput:
    kv_connector_output: Optional[KVConnectorOutput] = None
    req_id_to_index: Optional[dict] = None
    req_ids: Optional[list] = None
    scheduler_stats: Any = None
    output_tokens: Any = None
    logprobs: Any = None
    prompt_logprobs: Any = None
    encrypted_token_ids: Any = None
    num_scheduled_tokens: Any = None
    num_sampled_tokens: Any = None
    last_hidden_states: Any = None
