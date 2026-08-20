"""NPUWorker patches for SqueezeAttention-ascend.

Converts SqueezeAttention's HF mechanism (compression hooks registered on the model
before generation) into a lazy/early runner-proxy install on the vLLM-Ascend
worker: after ``init_device`` creates the NPUModelRunner, the worker's
``model_runner`` is wrapped by ``SqueezeAttentionModelRunner`` so every subsequent
``execute_model`` runs the SqueezeAttention boundary flow.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..envs import SqueezeRuntimeConfig
from ..logging_control import log_info, log_warning

_ORIG_ASCEND_WORKER_METHODS: dict[type, dict[str, Callable[..., Any]]] = {}


def _looks_like_ascend_runtime(worker: Any) -> bool:
    for candidate in (
        worker,
        getattr(worker, "model_runner", None),
        getattr(getattr(worker, "device_config", None), "device_type", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, str):
            text = candidate
        else:
            text = (
                f"{getattr(type(candidate), '__module__', '')} "
                f"{getattr(type(candidate), '__qualname__', '')}"
            )
        lowered = text.lower()
        if "vllm_ascend" in lowered or "ascend" in lowered or "npu" in lowered:
            return True
    return False


def should_early_install_proxy(worker: Any, config: SqueezeRuntimeConfig) -> bool:
    if not bool(config.early_install_proxy):
        return False
    if _looks_like_ascend_runtime(worker):
        return True
    return bool(getattr(worker, "model_runner", None))


def _install_runner_proxy(worker: Any) -> None:
    if getattr(worker, "_squeeze_runner_proxy_installed", False):
        return
    from .runner_proxy import SqueezeAttentionModelRunner

    base_runner = getattr(worker, "model_runner", None)
    if base_runner is None:
        log_warning("SqueezeAttention worker: model_runner not available yet")
        return
    if isinstance(base_runner, SqueezeAttentionModelRunner):
        worker._squeeze_runner_proxy_installed = True
        return
    config = getattr(worker, "_squeeze_runtime_config", None) or SqueezeRuntimeConfig.from_env()
    worker.model_runner = SqueezeAttentionModelRunner(base_runner=base_runner, config=config)
    worker._squeeze_runner_proxy_installed = True
    if config.logging_enabled:
        log_info(
            "Worker injected SqueezeAttention runner proxy: mode=%s ini_size=%.3f "
            "start_size=%d defer_prefill=%s build=%s",
            config.mode,
            config.ini_size,
            config.start_size,
            config.defer_prefill_compression,
            config.build_id,
        )


def _resolve_original_worker_method(worker: Any, method_name: str) -> Callable[..., Any]:
    for cls in type(worker).__mro__:
        methods = _ORIG_ASCEND_WORKER_METHODS.get(cls)
        if methods is not None and method_name in methods:
            return methods[method_name]
    raise RuntimeError(f"missing_original_ascend_worker_method:{method_name}")


def _patched_ascend_worker_init_device(self):
    _resolve_original_worker_method(self, "init_device")(self)
    config = SqueezeRuntimeConfig.from_env()
    self._squeeze_runtime_config = config
    if should_early_install_proxy(self, config):
        _install_runner_proxy(self)


def _patched_ascend_worker_execute_model(self, scheduler_output):
    if not getattr(self, "_squeeze_runner_proxy_installed", False):
        signals = getattr(scheduler_output, "squeeze_signals", None)
        if signals:
            _install_runner_proxy(self)
    return _resolve_original_worker_method(self, "execute_model")(self, scheduler_output)


def _patch_worker_class(
    worker_cls: type,
    *,
    patch_init: bool,
    patch_execute: bool,
) -> bool:
    if worker_cls in _ORIG_ASCEND_WORKER_METHODS:
        return False
    methods: dict[str, Callable[..., Any]] = {}
    if patch_init:
        init_device = getattr(worker_cls, "init_device", None)
        if callable(init_device):
            methods["init_device"] = init_device
            worker_cls.init_device = _patched_ascend_worker_init_device
    if patch_execute:
        execute_model = getattr(worker_cls, "execute_model", None)
        if callable(execute_model):
            methods["execute_model"] = execute_model
            worker_cls.execute_model = _patched_ascend_worker_execute_model
    if not methods:
        return False
    _ORIG_ASCEND_WORKER_METHODS[worker_cls] = methods
    return True


def install_worker_patches() -> None:
    patched: list[str] = []
    try:
        import vllm_ascend.worker.worker as ascend_worker_mod

        worker_cls = ascend_worker_mod.NPUWorker
        if _patch_worker_class(worker_cls, patch_init=True, patch_execute=True):
            patched.append("vllm_ascend.worker.worker.NPUWorker")
    except Exception:
        log_warning(
            "could not import vllm_ascend.worker.worker.NPUWorker for SqueezeAttention "
            "worker patches",
            exc_info=True,
        )

    optional_workers = (
        ("vllm_ascend._310p.worker_310p", "NPUWorker310"),
        ("vllm_ascend.xlite.xlite_worker", "XliteWorker"),
    )
    for module_name, class_name in optional_workers:
        try:
            module = __import__(module_name, fromlist=[class_name])
            worker_cls = getattr(module, class_name)
            if _patch_worker_class(worker_cls, patch_init=True, patch_execute=False):
                patched.append(f"{module_name}.{class_name}")
        except Exception:
            continue

    if patched:
        log_info(
            "Installed SqueezeAttention runtime worker patches for Ascend: %s",
            ", ".join(patched),
        )
