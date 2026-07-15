"""Reduction helpers such as norm, dot, and cross."""

from __future__ import annotations

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import require_mlx
from .irreps_array import IrrepsArray


def norm(array: IrrepsArray, *, per_irrep: bool = True, squared: bool = False) -> IrrepsArray:
    mx, _ = require_mlx()
    if per_irrep:
        if not array.irreps:
            return IrrepsArray(Irreps(), mx.zeros((*array.leading_shape, 0), dtype=array.dtype))
        outputs = []
        out_irreps = []
        for part, chunk in zip(array.irreps, array.chunk_arrays(), strict=True):
            block = chunk.reshape(*array.leading_shape, part.mul, part.ir.dim)
            reduced = mx.sum(block * block, axis=-1, keepdims=True)
            if not squared:
                reduced = mx.where(
                    reduced > 0,
                    mx.sqrt(mx.maximum(reduced, mx.array(1e-12, dtype=reduced.dtype))),
                    mx.zeros_like(reduced),
                )
            outputs.append(reduced.reshape(*array.leading_shape, part.mul))
            out_irreps.append(MulIrrep(part.mul, Irrep(0, 1)))
        return IrrepsArray(Irreps(out_irreps), mx.concatenate(outputs, axis=-1))
    reduced = mx.sum(array.array * array.array, axis=-1, keepdims=True)
    if not squared:
        reduced = mx.where(
            reduced > 0,
            mx.sqrt(mx.maximum(reduced, mx.array(1e-12, dtype=reduced.dtype))),
            mx.zeros_like(reduced),
        )
    return IrrepsArray("0e", reduced)


def dot(left: IrrepsArray, right: IrrepsArray, *, per_irrep: bool = True) -> IrrepsArray:
    mx, _ = require_mlx()
    if left.irreps != right.irreps:
        raise ValueError("dot expects matching irreps")
    if per_irrep:
        outputs = []
        out_irreps = []
        for part, left_chunk, right_chunk in zip(left.irreps, left.chunk_arrays(), right.chunk_arrays(), strict=True):
            left_block = left_chunk.reshape(*left.leading_shape, part.mul, part.ir.dim)
            right_block = right_chunk.reshape(*right.leading_shape, part.mul, part.ir.dim)
            outputs.append(mx.sum(left_block * right_block, axis=-1).reshape(*left.leading_shape, part.mul))
            out_irreps.append(MulIrrep(part.mul, Irrep(0, 1)))
        return IrrepsArray(Irreps(out_irreps), mx.concatenate(outputs, axis=-1))
    reduced = mx.sum(left.array * right.array, axis=-1, keepdims=True)
    return IrrepsArray("0e", reduced)


def cross(left: IrrepsArray, right: IrrepsArray) -> IrrepsArray:
    mx, _ = require_mlx()
    if len(left.irreps) != len(right.irreps):
        raise ValueError("cross expects matching chunk structure")
    outputs = []
    out_irreps = []
    for left_part, right_part, left_chunk, right_chunk in zip(
        left.irreps, right.irreps, left.chunk_arrays(), right.chunk_arrays(), strict=True
    ):
        if left_part.mul != right_part.mul or left_part.ir.l != 1 or right_part.ir.l != 1:
            raise ValueError("cross expects matching l=1 blocks")
        left_block = left_chunk.reshape(*left.leading_shape, left_part.mul, 3)
        right_block = right_chunk.reshape(*right.leading_shape, right_part.mul, 3)
        out = _cross_last_dim(left_block, right_block)
        outputs.append(out.reshape(*left.leading_shape, left_part.mul * 3))
        out_irreps.append(MulIrrep(left_part.mul, Irrep(1, left_part.ir.p * right_part.ir.p)))
    return IrrepsArray(Irreps(out_irreps), mx.concatenate(outputs, axis=-1))


def _cross_last_dim(left, right):
    mx, _ = require_mlx()
    return mx.stack(
        [
            left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
            left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
            left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
        ],
        axis=-1,
    )
