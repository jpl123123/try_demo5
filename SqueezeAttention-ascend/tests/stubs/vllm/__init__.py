"""Stub `vllm` package for simulated-debug tests (no real vLLM installed)."""

import logging

logger = logging.getLogger("vllm.stub")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
