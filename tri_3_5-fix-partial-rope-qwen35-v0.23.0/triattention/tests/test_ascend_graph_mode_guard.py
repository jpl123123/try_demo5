import sys
import types


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.SimpleNamespace()
if "vllm.logger" not in sys.modules:
    sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())
if "vllm.v1.outputs" not in sys.modules:
    sys.modules["vllm.v1.outputs"] = types.SimpleNamespace(ModelRunnerOutput=object)
try:
    import torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(
            Tensor=object,
            is_tensor=lambda value: False,
        )

from triattention.vllm.runtime.graph_mode_guard import (
    force_ascend_eager_and_skip_compiled,
)
from triattention.vllm.runtime.ascend_graph_mode_patch import (
    make_patched_ascend_forward_context,
)


def test_patched_ascend_forward_context_sets_skip_compiled_when_guarded():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "context"

    patched = make_patched_ascend_forward_context(original)

    with force_ascend_eager_and_skip_compiled(True):
        assert patched("attn", skip_compiled=False) == "context"

    assert calls == [(("attn",), {"skip_compiled": True})]


def test_patched_ascend_forward_context_preserves_skip_compiled_without_guard():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "context"

    patched = make_patched_ascend_forward_context(original)

    assert patched("attn", skip_compiled=False) == "context"

    assert calls == [(("attn",), {"skip_compiled": False})]
