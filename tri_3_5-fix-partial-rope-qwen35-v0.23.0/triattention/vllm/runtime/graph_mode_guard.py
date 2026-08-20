"""Runtime guard for graph/compile modes around effective input overrides."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_FORCE_ASCEND_EAGER_AND_SKIP_COMPILED: ContextVar[bool] = ContextVar(
    "triattention_force_ascend_eager_and_skip_compiled",
    default=False,
)


def force_ascend_eager_and_skip_compiled_active() -> bool:
    return bool(_FORCE_ASCEND_EAGER_AND_SKIP_COMPILED.get())


@contextmanager
def force_ascend_eager_and_skip_compiled(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    token = _FORCE_ASCEND_EAGER_AND_SKIP_COMPILED.set(True)
    try:
        yield
    finally:
        _FORCE_ASCEND_EAGER_AND_SKIP_COMPILED.reset(token)
