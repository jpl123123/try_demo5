"""Cross-process compression event bridge.

vLLM V1 workers run in a separate process from the engine core; dynamic
attributes set on ``scheduler_output`` do not survive the round trip. Events
are therefore attached through the declared ``kv_connector_output.kv_cache_events``
field of ``ModelRunnerOutput`` (picklable bag), with a same-process fallback on
``scheduler_output`` for the async (``execute_model`` returns ``None``) path.
"""

from __future__ import annotations

from typing import Any


class _EventBag:
    """Picklable carrier for compression events."""

    __slots__ = ("events",)

    def __init__(self, events: list[dict[str, Any]]):
        self.events = list(events)

    def __reduce__(self):
        return (_EventBag, (list(self.events),))

    def __getstate__(self):
        return {"events": list(self.events)}

    def __setstate__(self, state):
        self.events = list(state.get("events", []))


def attach_events_via_kv_connector_output(output: Any, events: list[dict[str, Any]]) -> bool:
    """Attach events through the vLLM declared cross-process output field."""
    if output is None or not events:
        return False
    try:
        target = output
        if not hasattr(target, "kv_connector_output"):
            for attr_name in ("model_runner_output", "_model_runner_output"):
                candidate = getattr(target, attr_name, None)
                if candidate is not None and hasattr(candidate, "kv_connector_output"):
                    target = candidate
                    break
        kco = getattr(target, "kv_connector_output", None)
        if kco is None:
            try:
                from vllm.v1.outputs import KVConnectorOutput

                kco = KVConnectorOutput()
                setattr(target, "kv_connector_output", kco)
            except Exception:
                return False
        kco.kv_cache_events = _EventBag(events)
        return True
    except Exception:
        return False


def read_events_from_kv_connector_output(model_runner_output: Any) -> list[dict[str, Any]] | None:
    """Read compression events from the vLLM declared cross-process field."""
    if model_runner_output is None:
        return None
    kco = getattr(model_runner_output, "kv_connector_output", None)
    if kco is None:
        return None
    bag = getattr(kco, "kv_cache_events", None)
    if bag is None:
        return None
    events = getattr(bag, "events", None)
    if not isinstance(events, list):
        return None
    return events


def attach_events_to_output(
    output: Any,
    pending_events: list[dict[str, Any]],
    scheduler_output: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Attach compression events to the execute_model/sample_tokens output.

    Returns ``(output, remaining_pending_events)``.
    """
    if output is None:
        if scheduler_output is not None and pending_events:
            setattr(scheduler_output, "kvpress_compression_events", pending_events)
            return output, pending_events
        return output, pending_events
    attach_events_via_kv_connector_output(output, pending_events)
    try:
        setattr(output, "kvpress_compression_events", pending_events)
    except Exception:
        return output, pending_events
    return output, []


def read_events_from_output(model_runner_output: Any) -> list[dict[str, Any]] | None:
    """Read compression events from a model runner output (worker->engine core)."""
    events = read_events_from_kv_connector_output(model_runner_output)
    if events is not None:
        return events
    raw = getattr(model_runner_output, "kvpress_compression_events", None)
    if isinstance(raw, list):
        return raw
    return None
