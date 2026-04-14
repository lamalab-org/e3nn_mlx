"""Spherical harmonics MLX helpers."""

from __future__ import annotations

from math import sqrt

from e3nn_core.irreps import Irreps
from .compat import require_mlx


def _normalize_vectors(vectors, normalize: bool):
    mx, _ = require_mlx()
    if not normalize:
        return vectors
    norms = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))
    eps = mx.array(1e-12, dtype=vectors.dtype)
    return vectors / mx.maximum(norms, eps)


def _normalization_scale(l: int, normalization: str) -> float:
    if normalization == "component":
        return 1.0
    if normalization == "norm":
        return 1.0 / sqrt(2 * l + 1)
    if normalization == "integral":
        return 1.0 / sqrt(4.0 * 3.141592653589793)
    raise ValueError(f"unknown normalization {normalization!r}")


def _spherical_harmonics_single(l: int, vectors, normalization: str):
    mx, _ = require_mlx()
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    if l == 0:
        out = mx.ones((*vectors.shape[:-1], 1), dtype=vectors.dtype)
    elif l == 1:
        out = mx.stack([x, y, z], axis=-1)
    elif l == 2:
        out = mx.stack(
            [
                sqrt(3.0 / 2.0) * (z * z - 1.0 / 3.0),
                sqrt(2.0) * x * y,
                sqrt(2.0) * y * z,
                sqrt(2.0) * z * x,
                (x * x - y * y) / sqrt(2.0),
            ],
            axis=-1,
        )
    else:
        raise NotImplementedError("spherical_harmonics is implemented for l <= 2 in the MVP")
    return (out * _normalization_scale(l, normalization)).astype(vectors.dtype)


def spherical_harmonics(ls, vectors, *, normalize: bool = True, normalization: str = "component"):
    mx, _ = require_mlx()
    vectors = _normalize_vectors(vectors, normalize=normalize)
    if isinstance(ls, int):
        return _spherical_harmonics_single(ls, vectors, normalization)
    irreps = Irreps(ls)
    outputs = []
    for part in irreps:
        block = _spherical_harmonics_single(part.ir.l, vectors, normalization)
        if part.mul > 1:
            block = mx.concatenate([block] * part.mul, axis=-1)
        outputs.append(block)
    if not outputs:
        shape = (*vectors.shape[:-1], 0)
        return mx.zeros(shape, dtype=vectors.dtype)
    return mx.concatenate(outputs, axis=-1)
