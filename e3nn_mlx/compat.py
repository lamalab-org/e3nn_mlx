"""Compatibility helpers for optional MLX imports."""

from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys


class _UnavailableMLXModule:
    """Import-only fallback used when MLX cannot initialize."""

    def __init__(self) -> None:
        pass


@functools.lru_cache(maxsize=1)
def mlx_runtime_available() -> bool:
    # The runtime probe must not discover packages that are absent from this
    # interpreter (for example when it was started with ``python -S``).
    if importlib.util.find_spec("mlx") is None:
        return False
    env = dict(os.environ)
    probe = [sys.executable, "-c", "import mlx.core as mx; print(int(mx.is_available(mx.cpu)))"]
    result = subprocess.run(probe, capture_output=True, text=True, env=env, check=False)
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def mlx_metal_available() -> bool:
    """Return whether this interpreter can execute MLX Metal kernels.

    MLX also runs on CPU-only platforms, where ``mx.fast.metal_kernel`` can be
    constructed but fails when evaluated. Keep the capability check separate
    from :func:`mlx_runtime_available`, which intentionally accepts CPU MLX.
    """

    if sys.platform != "darwin" or not mlx_runtime_available():
        return False
    mx, _ = require_mlx()
    try:
        return bool(mx.is_available(mx.gpu))
    except (AttributeError, RuntimeError, ValueError):
        return False


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
