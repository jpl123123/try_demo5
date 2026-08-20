"""Patch helpers for vLLM-Ascend graph/compile mode guards."""

from __future__ import annotations

from typing import Any, Callable

from .graph_mode_guard import force_ascend_eager_and_skip_compiled_active


def make_patched_ascend_forward_context(
    original_set_forward_context: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_set_ascend_forward_context(*args, **kwargs):
        if force_ascend_eager_and_skip_compiled_active():
            if len(args) >= 12:
                args_list = list(args)
                args_list[11] = True
                args = tuple(args_list)
            else:
                kwargs = dict(kwargs)
                kwargs["skip_compiled"] = True
        return original_set_forward_context(*args, **kwargs)

    setattr(_patched_set_ascend_forward_context, "_triattention_patched", True)
    return _patched_set_ascend_forward_context
