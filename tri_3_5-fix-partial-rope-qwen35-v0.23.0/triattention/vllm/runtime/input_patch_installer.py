"""Installer for vLLM runtime input patch hooks used by TriAttention runtime."""
from __future__ import annotations

import os
from typing import Any, Callable

from vllm.logger import logger

from .logging_control import runtime_logging_enabled

from . import input_patch_state as _patch_state
from .input_patch_ascend_backend import (
    make_patched_ascend_v2_build_attn_metadata,
    make_patched_ascend_v2_compute_slot_mappings,
    make_patched_ascend_v2_update_seq_lens_cpu,
)
from .input_patch_vllm_backend import (
    make_patched_compute_slot_mappings,
    make_patched_prepare_pos_seq_lens,
)
from .input_patch_vllm_v1_backend import make_patched_v1_prepare_inputs
from .phase_profile import make_timed_wrapper

_PATCH_INSTALLED = False
_ORIGINAL_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None
_ORIGINAL_V1_PREPARE_INPUTS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_PREPARE_INPUTS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_EXECUTE_MODEL: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_SAMPLE_TOKENS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_BUILD_ATTENTION_METADATA: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_MODEL_FORWARD: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_SAMPLE: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_BOOKKEEPING_SYNC: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_DETERMINE_BATCH: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_PREPROCESS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_SYNC_BATCH_ACROSS_DP: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_EXECUTE_MODEL: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_PREPARE_INPUTS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_POSTPROCESS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_PREPARE_ATTN: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_BLOCK_TABLE_GET_DEVICE_TENSOR: Callable[..., Any] | None = None


def _debug_disable_v1_override_path() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_V1_OVERRIDE_PATH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ascend_v1_block_table_trim_enabled() -> bool:
    if _env_bool("TRIATTN_DEBUG_DISABLE_ASCEND_BLOCK_TABLE_TRIM", False):
        return False
    return _env_bool("TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE", False)


def _is_triattention_patched(func: Any) -> bool:
    return bool(getattr(func, "_triattention_patched", False))


def _install_timed_method(
    *,
    cls: Any,
    method_name: str,
    phase: str,
    storage_name: str,
    details_fn: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any] | None]
    | None,
    patched_targets: list[str],
    target_label: str,
) -> bool:
    if globals().get(storage_name) is not None:
        return False
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    globals()[storage_name] = original
    setattr(cls, method_name, make_timed_wrapper(phase, original, details_fn))
    patched_targets.append(target_label)
    return True


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _scheduler_output_details(scheduler_output: Any) -> dict[str, Any]:
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    num_reqs = len(num_scheduled) if isinstance(num_scheduled, dict) else None
    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total_tokens is None and isinstance(num_scheduled, dict):
        try:
            total_tokens = sum(int(v) for v in num_scheduled.values())
        except Exception:
            total_tokens = None
    return {
        "num_reqs": num_reqs,
        "total_tokens": _safe_int(total_tokens),
    }


def _array_max(value: Any) -> int | None:
    try:
        return int(value.max(initial=0))
    except TypeError:
        try:
            return int(value.max())
        except Exception:
            return None
    except Exception:
        return None


def _numel(value: Any) -> int | None:
    try:
        return int(value.numel())
    except Exception:
        try:
            return int(value.size)
        except Exception:
            return None


def _shape0(value: Any) -> int | None:
    try:
        return int(value.shape[0])
    except Exception:
        return None


def _shape1(value: Any) -> int | None:
    try:
        return int(value.shape[1])
    except Exception:
        return None


def _ceil_div(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


def make_patched_ascend_v1_block_table_get_device_tensor(
    original_get_device_tensor: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_get_device_tensor(self):
        tensor = original_get_device_tensor(self)
        if (
            not _ascend_v1_block_table_trim_enabled()
            or not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED
        ):
            return tensor

        max_seq_len = _patch_state.ACTIVE_EFFECTIVE_MAX_SEQ_LEN
        block_size = _safe_int(getattr(self, "block_size", None))
        original_cols = _shape1(tensor)
        if (
            max_seq_len is None
            or block_size is None
            or block_size <= 0
            or original_cols is None
        ):
            return tensor

        trim_cols = max(1, _ceil_div(int(max_seq_len), int(block_size)))
        trim_cols = min(int(original_cols), int(trim_cols))
        _patch_state.set_active_block_table_trim_observation(
            block_size=int(block_size),
            original_cols=int(original_cols),
            effective_cols=int(trim_cols),
        )
        if trim_cols >= int(original_cols):
            return tensor
        view = tensor[:, :trim_cols]
        if _env_bool("TRIATTN_RUNTIME_ASCEND_V1_BLOCK_TABLE_TRIM_CONTIGUOUS", False):
            return view.contiguous()
        return view

    setattr(_patched_get_device_tensor, "_triattention_patched", True)
    return _patched_get_device_tensor


def _execute_model_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    scheduler_output = kwargs.get("scheduler_output")
    if scheduler_output is None and len(args) >= 2:
        scheduler_output = args[1]
    return _scheduler_output_details(scheduler_output)


def _sample_tokens_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del kwargs
    self_obj = args[0] if args else None
    input_batch = getattr(self_obj, "input_batch", None)
    return {
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "result_type": type(result).__name__ if result is not None else "None",
    }


def _prepare_inputs_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    scheduler_output = kwargs.get("scheduler_output")
    batch_desc = kwargs.get("batch_desc")
    if scheduler_output is None and len(args) >= 2:
        scheduler_output = args[1]
    if batch_desc is None and len(args) >= 3:
        batch_desc = args[2]
    details = _scheduler_output_details(scheduler_output)
    details.update(
        {
            "batch_tokens": _safe_int(getattr(batch_desc, "num_tokens", None)),
            "batch_reqs": _safe_int(getattr(batch_desc, "num_reqs", None)),
            "cg_mode": getattr(batch_desc, "cg_mode", None),
        }
    )
    if result is not None:
        details.update(
            {
                "out_reqs": _safe_int(getattr(result, "num_reqs", None)),
                "out_tokens": _safe_int(getattr(result, "num_tokens", None)),
                "out_tokens_padded": _safe_int(
                    getattr(result, "num_tokens_after_padding", None)
                ),
                "attn_state": getattr(result, "attn_state", None),
            }
        )
    return details


def _v1_attention_metadata_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    self_obj = args[0] if args else None
    num_tokens = kwargs.get("num_tokens")
    num_tokens_padded = kwargs.get("num_tokens_padded")
    num_reqs = kwargs.get("num_reqs")
    num_reqs_padded = kwargs.get("num_reqs_padded")
    max_query_len = kwargs.get("max_query_len")
    if len(args) >= 2 and num_tokens is None:
        num_tokens = args[1]
    if len(args) >= 3 and num_reqs is None:
        num_reqs = args[2]
    if len(args) >= 4 and max_query_len is None:
        max_query_len = args[3]
    seq_lens = getattr(self_obj, "seq_lens", None)
    seq_lens_np = getattr(seq_lens, "np", None)
    return {
        "num_reqs": _safe_int(num_reqs),
        "num_reqs_padded": _safe_int(num_reqs_padded),
        "num_tokens": _safe_int(num_tokens),
        "num_tokens_padded": _safe_int(num_tokens_padded),
        "max_query_len": _safe_int(max_query_len),
        "seq_lens_np_max": _array_max(
            seq_lens_np[: int(num_reqs)] if seq_lens_np is not None and num_reqs else seq_lens_np
        ),
        "effective_max_seq_len": _patch_state.ACTIVE_EFFECTIVE_MAX_SEQ_LEN,
        "block_table_block_size": _patch_state.ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE,
        "block_table_cols": _patch_state.ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS,
        "effective_block_table_cols": (
            _patch_state.ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS
        ),
        "use_spec_decode": int(bool(kwargs.get("use_spec_decode", False))),
    }


def _v1_model_forward_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    num_tokens_padded = kwargs.get("num_tokens_padded")
    input_ids = kwargs.get("input_ids")
    positions = kwargs.get("positions")
    if len(args) >= 2 and num_tokens_padded is None:
        num_tokens_padded = args[1]
    if len(args) >= 3 and input_ids is None:
        input_ids = args[2]
    if len(args) >= 4 and positions is None:
        positions = args[3]
    return {
        "num_tokens_padded": _safe_int(num_tokens_padded),
        "input_ids": _numel(input_ids),
        "positions": _numel(positions),
    }


def _v1_sample_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    logits = kwargs.get("logits")
    spec_decode_metadata = kwargs.get("spec_decode_metadata")
    if len(args) >= 2 and logits is None:
        logits = args[1]
    if len(args) >= 3 and spec_decode_metadata is None:
        spec_decode_metadata = args[2]
    return {
        "logits_rows": _shape0(logits),
        "spec_decode": int(spec_decode_metadata is not None),
    }


def _v1_bookkeeping_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    scheduler_output = kwargs.get("scheduler_output")
    sampler_output = kwargs.get("sampler_output")
    if scheduler_output is None and len(args) >= 2:
        scheduler_output = args[1]
    if sampler_output is None and len(args) >= 3:
        sampler_output = args[2]
    sampled = getattr(sampler_output, "sampled_token_ids", None)
    details = _scheduler_output_details(scheduler_output)
    details.update({"sampled_rows": _shape0(sampled)})
    return details


def _v1_determine_batch_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    num_tokens = kwargs.get("num_tokens")
    num_reqs = kwargs.get("num_reqs")
    max_num_scheduled_tokens = kwargs.get("max_num_scheduled_tokens")
    if len(args) >= 2 and num_tokens is None:
        num_tokens = args[1]
    if len(args) >= 3 and num_reqs is None:
        num_reqs = args[2]
    if len(args) >= 4 and max_num_scheduled_tokens is None:
        max_num_scheduled_tokens = args[3]
    cudagraph_mode = None
    batch_desc = None
    should_ubatch = None
    if isinstance(result, tuple):
        if len(result) > 0:
            cudagraph_mode = result[0]
        if len(result) > 1:
            batch_desc = result[1]
        if len(result) > 2:
            should_ubatch = result[2]
    return {
        "num_reqs": _safe_int(num_reqs),
        "num_tokens": _safe_int(num_tokens),
        "max_scheduled": _safe_int(max_num_scheduled_tokens),
        "cg_mode": cudagraph_mode,
        "batch_tokens": _safe_int(getattr(batch_desc, "num_tokens", None)),
        "batch_reqs": _safe_int(getattr(batch_desc, "num_reqs", None)),
        "ubatch": int(bool(should_ubatch)) if should_ubatch is not None else None,
    }


def _v1_preprocess_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    scheduler_output = kwargs.get("scheduler_output")
    num_tokens = kwargs.get("num_tokens")
    if scheduler_output is None and len(args) >= 2:
        scheduler_output = args[1]
    if len(args) >= 3 and num_tokens is None:
        num_tokens = args[2]
    details = _scheduler_output_details(scheduler_output)
    details.update({"num_tokens_arg": _safe_int(num_tokens)})
    if isinstance(result, tuple):
        details["result_parts"] = len(result)
    return details


def _v1_sync_batch_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    num_tokens_padded = kwargs.get("num_tokens_padded")
    cudagraph_mode = kwargs.get("cudagraph_mode")
    if len(args) >= 2 and num_tokens_padded is None:
        num_tokens_padded = args[1]
    if len(args) >= 3 and cudagraph_mode is None:
        cudagraph_mode = args[2]
    return {
        "num_tokens_padded": _safe_int(num_tokens_padded),
        "cg_mode": cudagraph_mode,
        "result_type": type(result).__name__ if result is not None else "None",
    }


def _postprocess_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del kwargs, result
    input_batch = args[1] if len(args) >= 2 else None
    return {
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_tokens": _safe_int(getattr(input_batch, "num_tokens", None)),
        "num_tokens_padded": _safe_int(
            getattr(input_batch, "num_tokens_after_padding", None)
        ),
        "attn_state": getattr(input_batch, "attn_state", None),
    }


def _prepare_attn_details(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    del result
    self_obj = args[0] if args else None
    input_batch = kwargs.get("input_batch")
    cudagraph_mode = kwargs.get("cudagraph_mode")
    block_tables = kwargs.get("block_tables")
    if input_batch is None and len(args) >= 2:
        input_batch = args[1]
    if cudagraph_mode is None and len(args) >= 3:
        cudagraph_mode = args[2]
    if block_tables is None and len(args) >= 4:
        block_tables = args[3]
    return {
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_tokens": _safe_int(getattr(input_batch, "num_tokens", None)),
        "num_tokens_padded": _safe_int(
            getattr(input_batch, "num_tokens_after_padding", None)
        ),
        "max_model_len": _safe_int(getattr(self_obj, "max_model_len", None)),
        "cg_mode": cudagraph_mode,
        "attn_state": getattr(input_batch, "attn_state", None),
        "kv_groups": len(block_tables) if isinstance(block_tables, tuple) else None,
    }


def install_runtime_input_patch_hooks() -> bool:
    """Patch vLLM GPU input prep once.

    Returns True when the patch is active (including repeated calls).
    """
    global _PATCH_INSTALLED, _ORIGINAL_PREPARE_POS_SEQ_LENS, _ORIGINAL_COMPUTE_SLOT_MAPPINGS
    global _ORIGINAL_V1_PREPARE_INPUTS, _ORIGINAL_ASCEND_V1_PREPARE_INPUTS
    global _ORIGINAL_ASCEND_V1_EXECUTE_MODEL, _ORIGINAL_ASCEND_V1_SAMPLE_TOKENS
    global _ORIGINAL_ASCEND_V1_BUILD_ATTENTION_METADATA
    global _ORIGINAL_ASCEND_V1_MODEL_FORWARD, _ORIGINAL_ASCEND_V1_SAMPLE
    global _ORIGINAL_ASCEND_V1_BOOKKEEPING_SYNC, _ORIGINAL_ASCEND_V1_DETERMINE_BATCH
    global _ORIGINAL_ASCEND_V1_PREPROCESS, _ORIGINAL_ASCEND_V1_SYNC_BATCH_ACROSS_DP
    global _ORIGINAL_ASCEND_V2_EXECUTE_MODEL, _ORIGINAL_ASCEND_V2_PREPARE_INPUTS
    global _ORIGINAL_ASCEND_V2_POSTPROCESS
    global _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS
    global _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU
    global _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS
    global _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA
    global _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA
    global _ORIGINAL_ASCEND_V2_PREPARE_ATTN
    global _ORIGINAL_ASCEND_V1_BLOCK_TABLE_GET_DEVICE_TENSOR
    patched_any = False
    patched_targets: list[str] = []

    try:
        import vllm.v1.worker.gpu.block_table as gpu_block_table
        import vllm.v1.worker.gpu.model_runner as gpu_model_runner
    except Exception:
        gpu_block_table = None
        gpu_model_runner = None

    if gpu_block_table is not None and gpu_model_runner is not None:
        original = getattr(gpu_model_runner, "prepare_pos_seq_lens", None)
        compute_slot_mappings = getattr(gpu_block_table.BlockTables, "compute_slot_mappings", None)
        if (
            original is not None
            and compute_slot_mappings is not None
            and _ORIGINAL_PREPARE_POS_SEQ_LENS is None
            and _ORIGINAL_COMPUTE_SLOT_MAPPINGS is None
        ):
            _ORIGINAL_PREPARE_POS_SEQ_LENS = original
            _ORIGINAL_COMPUTE_SLOT_MAPPINGS = compute_slot_mappings
            gpu_model_runner.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
                _ORIGINAL_PREPARE_POS_SEQ_LENS
            )
            gpu_block_table.BlockTables.compute_slot_mappings = make_patched_compute_slot_mappings(
                _ORIGINAL_COMPUTE_SLOT_MAPPINGS
            )
            patched_any = True
            patched_targets.append("vllm.v1.worker.gpu")

    if not _debug_disable_v1_override_path():
        try:
            import vllm.v1.worker.gpu_model_runner as gpu_model_runner_v1
        except Exception:
            gpu_model_runner_v1 = None
        if gpu_model_runner_v1 is not None:
            original_v1_prepare_inputs = getattr(gpu_model_runner_v1.GPUModelRunner, "_prepare_inputs", None)
            if (
                original_v1_prepare_inputs is not None
                and _ORIGINAL_V1_PREPARE_INPUTS is None
            ):
                _ORIGINAL_V1_PREPARE_INPUTS = original_v1_prepare_inputs
                gpu_model_runner_v1.GPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(
                    _ORIGINAL_V1_PREPARE_INPUTS
                )
                patched_any = True
                patched_targets.append("vllm.v1.worker.gpu_model_runner.GPUModelRunner")

        try:
            import vllm_ascend.worker.model_runner_v1 as ascend_model_runner_v1
        except Exception:
            ascend_model_runner_v1 = None
        if ascend_model_runner_v1 is not None:
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="execute_model",
                phase="ascend_v1_execute_model",
                storage_name="_ORIGINAL_ASCEND_V1_EXECUTE_MODEL",
                details_fn=_execute_model_details,
                patched_targets=patched_targets,
                target_label="vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model",
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="sample_tokens",
                phase="ascend_v1_sample_tokens",
                storage_name="_ORIGINAL_ASCEND_V1_SAMPLE_TOKENS",
                details_fn=_sample_tokens_details,
                patched_targets=patched_targets,
                target_label="vllm_ascend.worker.model_runner_v1.NPUModelRunner.sample_tokens",
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_build_attention_metadata",
                phase="ascend_v1_build_attention_metadata",
                storage_name="_ORIGINAL_ASCEND_V1_BUILD_ATTENTION_METADATA",
                details_fn=_v1_attention_metadata_details,
                patched_targets=patched_targets,
                target_label=(
                    "vllm_ascend.worker.model_runner_v1.NPUModelRunner."
                    "_build_attention_metadata"
                ),
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_model_forward",
                phase="ascend_v1_model_forward",
                storage_name="_ORIGINAL_ASCEND_V1_MODEL_FORWARD",
                details_fn=_v1_model_forward_details,
                patched_targets=patched_targets,
                target_label="vllm_ascend.worker.model_runner_v1.NPUModelRunner._model_forward",
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_sample",
                phase="ascend_v1_sample",
                storage_name="_ORIGINAL_ASCEND_V1_SAMPLE",
                details_fn=_v1_sample_details,
                patched_targets=patched_targets,
                target_label="vllm_ascend.worker.model_runner_v1.NPUModelRunner._sample",
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_bookkeeping_sync",
                phase="ascend_v1_bookkeeping_sync",
                storage_name="_ORIGINAL_ASCEND_V1_BOOKKEEPING_SYNC",
                details_fn=_v1_bookkeeping_details,
                patched_targets=patched_targets,
                target_label=(
                    "vllm_ascend.worker.model_runner_v1.NPUModelRunner."
                    "_bookkeeping_sync"
                ),
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_determine_batch_execution_and_padding",
                phase="ascend_v1_determine_batch_execution",
                storage_name="_ORIGINAL_ASCEND_V1_DETERMINE_BATCH",
                details_fn=_v1_determine_batch_details,
                patched_targets=patched_targets,
                target_label=(
                    "vllm_ascend.worker.model_runner_v1.NPUModelRunner."
                    "_determine_batch_execution_and_padding"
                ),
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_preprocess",
                phase="ascend_v1_preprocess",
                storage_name="_ORIGINAL_ASCEND_V1_PREPROCESS",
                details_fn=_v1_preprocess_details,
                patched_targets=patched_targets,
                target_label="vllm_ascend.worker.model_runner_v1.NPUModelRunner._preprocess",
            ):
                patched_any = True
            if _install_timed_method(
                cls=ascend_model_runner_v1.NPUModelRunner,
                method_name="_sync_batch_across_dp",
                phase="ascend_v1_sync_batch_across_dp",
                storage_name="_ORIGINAL_ASCEND_V1_SYNC_BATCH_ACROSS_DP",
                details_fn=_v1_sync_batch_details,
                patched_targets=patched_targets,
                target_label=(
                    "vllm_ascend.worker.model_runner_v1.NPUModelRunner."
                    "_sync_batch_across_dp"
                ),
            ):
                patched_any = True

            original_ascend_v1_prepare_inputs = getattr(
                ascend_model_runner_v1.NPUModelRunner,
                "_prepare_inputs",
                None,
            )
            if original_ascend_v1_prepare_inputs is not None:
                if _ORIGINAL_ASCEND_V1_PREPARE_INPUTS is None:
                    _ORIGINAL_ASCEND_V1_PREPARE_INPUTS = original_ascend_v1_prepare_inputs
                    ascend_model_runner_v1.NPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(
                        _ORIGINAL_ASCEND_V1_PREPARE_INPUTS
                    )
                    patched_any = True
                    patched_targets.append("vllm_ascend.worker.model_runner_v1.NPUModelRunner")

        try:
            import vllm_ascend.worker.block_table as ascend_block_table_v1
        except Exception:
            ascend_block_table_v1 = None
        if ascend_block_table_v1 is not None:
            original_get_device_tensor = getattr(
                ascend_block_table_v1.BlockTable,
                "get_device_tensor",
                None,
            )
            if (
                original_get_device_tensor is not None
                and _ORIGINAL_ASCEND_V1_BLOCK_TABLE_GET_DEVICE_TENSOR is None
                and not _is_triattention_patched(original_get_device_tensor)
            ):
                _ORIGINAL_ASCEND_V1_BLOCK_TABLE_GET_DEVICE_TENSOR = (
                    original_get_device_tensor
                )
                ascend_block_table_v1.BlockTable.get_device_tensor = (
                    make_patched_ascend_v1_block_table_get_device_tensor(
                        _ORIGINAL_ASCEND_V1_BLOCK_TABLE_GET_DEVICE_TENSOR
                    )
                )
                patched_any = True
                patched_targets.append(
                    "vllm_ascend.worker.block_table.BlockTable.get_device_tensor"
                )

    try:
        import vllm_ascend.worker.v2.model_runner as ascend_model_runner_v2
    except Exception:
        ascend_model_runner_v2 = None
    if ascend_model_runner_v2 is not None:
        original_execute_model = getattr(
            ascend_model_runner_v2.NPUModelRunner,
            "execute_model",
            None,
        )
        if (
            original_execute_model is not None
            and _ORIGINAL_ASCEND_V2_EXECUTE_MODEL is None
        ):
            _ORIGINAL_ASCEND_V2_EXECUTE_MODEL = original_execute_model
            ascend_model_runner_v2.NPUModelRunner.execute_model = make_timed_wrapper(
                "ascend_v2_execute_model",
                _ORIGINAL_ASCEND_V2_EXECUTE_MODEL,
                _execute_model_details,
            )
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_runner.NPUModelRunner.execute_model"
            )

        original_prepare_inputs = getattr(
            ascend_model_runner_v2.NPUModelRunner,
            "prepare_inputs",
            None,
        )
        if (
            original_prepare_inputs is not None
            and _ORIGINAL_ASCEND_V2_PREPARE_INPUTS is None
        ):
            _ORIGINAL_ASCEND_V2_PREPARE_INPUTS = original_prepare_inputs
            ascend_model_runner_v2.NPUModelRunner.prepare_inputs = make_timed_wrapper(
                "ascend_v2_prepare_inputs",
                _ORIGINAL_ASCEND_V2_PREPARE_INPUTS,
                _prepare_inputs_details,
            )
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_runner.NPUModelRunner.prepare_inputs"
            )

        original_postprocess = getattr(
            ascend_model_runner_v2.NPUModelRunner,
            "postprocess",
            None,
        )
        if (
            original_postprocess is not None
            and _ORIGINAL_ASCEND_V2_POSTPROCESS is None
        ):
            _ORIGINAL_ASCEND_V2_POSTPROCESS = original_postprocess
            ascend_model_runner_v2.NPUModelRunner.postprocess = make_timed_wrapper(
                "ascend_v2_postprocess",
                _ORIGINAL_ASCEND_V2_POSTPROCESS,
                _postprocess_details,
            )
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_runner.NPUModelRunner.postprocess"
            )

        original_prepare_pos_seq_lens = getattr(
            ascend_model_runner_v2,
            "prepare_pos_seq_lens",
            None,
        )
        if (
            original_prepare_pos_seq_lens is not None
            and _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS is None
        ):
            _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS = original_prepare_pos_seq_lens
            ascend_model_runner_v2.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
                _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens")

        original_update_seq_lens_cpu = getattr(
            ascend_model_runner_v2.NPUModelRunner,
            "_update_seq_lens_cpu",
            None,
        )
        if (
            original_update_seq_lens_cpu is not None
            and _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU is None
        ):
            _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU = original_update_seq_lens_cpu
            ascend_model_runner_v2.NPUModelRunner._update_seq_lens_cpu = (
                make_patched_ascend_v2_update_seq_lens_cpu(
                    _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU
                )
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.model_runner.NPUModelRunner")

    try:
        import vllm_ascend.worker.v2.attn_utils as ascend_attn_utils_v2
    except Exception:
        ascend_attn_utils_v2 = None
    patched_ascend_build_attn_metadata = None
    if ascend_attn_utils_v2 is not None:
        original_build_attn_metadata = getattr(
            ascend_attn_utils_v2,
            "build_attn_metadata",
            None,
        )
        if _is_triattention_patched(original_build_attn_metadata):
            patched_ascend_build_attn_metadata = original_build_attn_metadata
        elif (
            original_build_attn_metadata is not None
            and _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA is None
        ):
            _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA = original_build_attn_metadata
            patched_ascend_build_attn_metadata = (
                make_patched_ascend_v2_build_attn_metadata(
                    _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA
                )
            )
            ascend_attn_utils_v2.build_attn_metadata = patched_ascend_build_attn_metadata
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.attn_utils.build_attn_metadata")

    try:
        import vllm_ascend.worker.v2.model_states.default as ascend_default_state_v2
    except Exception:
        ascend_default_state_v2 = None
    if ascend_default_state_v2 is not None:
        original_prepare_attn = getattr(
            ascend_default_state_v2.AscendModelState,
            "prepare_attn",
            None,
        )
        if (
            original_prepare_attn is not None
            and _ORIGINAL_ASCEND_V2_PREPARE_ATTN is None
        ):
            _ORIGINAL_ASCEND_V2_PREPARE_ATTN = original_prepare_attn
            ascend_default_state_v2.AscendModelState.prepare_attn = make_timed_wrapper(
                "ascend_v2_prepare_attn",
                _ORIGINAL_ASCEND_V2_PREPARE_ATTN,
                _prepare_attn_details,
            )
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_states.default.AscendModelState.prepare_attn"
            )

        original_default_build_attn_metadata = getattr(
            ascend_default_state_v2,
            "build_attn_metadata",
            None,
        )
        if _is_triattention_patched(original_default_build_attn_metadata):
            if patched_ascend_build_attn_metadata is None:
                patched_ascend_build_attn_metadata = original_default_build_attn_metadata
        elif (
            original_default_build_attn_metadata is not None
            and _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA is None
        ):
            _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA = (
                original_default_build_attn_metadata
            )
            if patched_ascend_build_attn_metadata is None:
                patched_ascend_build_attn_metadata = (
                    make_patched_ascend_v2_build_attn_metadata(
                        _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA
                    )
                )
            ascend_default_state_v2.build_attn_metadata = patched_ascend_build_attn_metadata
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_states.default.build_attn_metadata"
            )

    try:
        import vllm_ascend.worker.v2.block_table as ascend_block_table_v2
    except Exception:
        ascend_block_table_v2 = None
    if ascend_block_table_v2 is not None:
        original_ascend_compute_slot_mappings = getattr(
            ascend_block_table_v2.AscendBlockTables,
            "compute_slot_mappings",
            None,
        )
        if (
            original_ascend_compute_slot_mappings is not None
            and _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS is None
        ):
            _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS = (
                original_ascend_compute_slot_mappings
            )
            ascend_block_table_v2.AscendBlockTables.compute_slot_mappings = (
                make_patched_ascend_v2_compute_slot_mappings(
                    _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS
                )
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.block_table.AscendBlockTables")

    _PATCH_INSTALLED = _PATCH_INSTALLED or patched_any
    if patched_targets and runtime_logging_enabled():
        logger.info(
            "Installed TriAttention runtime input patches: %s",
            ", ".join(patched_targets),
        )
    return _PATCH_INSTALLED
