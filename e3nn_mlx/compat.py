"""Compatibility helpers for optional MLX imports."""

from __future__ import annotations

import os
import subprocess
import sys
import functools


class _UnavailableMLXModule:
    """Import-only fallback used when MLX cannot initialize."""

    def __init__(self) -> None:
        pass


@functools.lru_cache(maxsize=1)
def mlx_runtime_available() -> bool:
    env = dict(os.environ)
    probe = [sys.executable, "-c", "import mlx.core as mx; print(int(mx.is_available(mx.cpu)))"]
    result = subprocess.run(probe, capture_output=True, text=True, env=env, check=False)
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def mlx_module_base() -> type:
    """Return ``mlx.nn.Module`` without making import-only environments crash."""

    if not mlx_runtime_available():
        return _UnavailableMLXModule
    _, nn = require_mlx()
    return nn.Module


def require_mlx() -> tuple[object, object]:
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        import mlx.nn as nn  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLX is required for e3nn_mlx numerical operations. Install mlx to enable this backend."
        ) from exc
    return mx, nn
