"""Stub `vllm_ascend` package for simulated-debug tests."""

import logging

logger = logging.getLogger("vllm_ascend.stub")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
