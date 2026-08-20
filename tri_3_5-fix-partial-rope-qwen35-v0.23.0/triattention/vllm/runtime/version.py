"""Runtime build fingerprint for deployment/log verification."""

from __future__ import annotations

from pathlib import Path

RUNTIME_BUILD_ID = "ascend-core-entry-v27-20260605"


def runtime_build_info() -> str:
    return f"{RUNTIME_BUILD_ID} source={Path(__file__).resolve()}"
