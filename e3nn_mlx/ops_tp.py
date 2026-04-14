"""Tensor product MLX helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from e3nn_core.instructions import TensorProductInstruction, generate_tensor_product_instructions
from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import require_mlx
from .irreps_array import IrrepsArray
from .ops_basic import compile_or_identity, get_extension


@dataclass(frozen=True, slots=True)
class TensorProductPlan:
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    instructions: tuple[TensorProductInstruction, ...]
    weighted: bool
    weight_numel: int
    weight_slices: tuple[slice, ...]


def tensor_product_plan(
    irreps_in1: Irreps | str,
    irreps_in2: Irreps | str,
    irreps_out: Irreps | str | None = None,
    *,
    weighted: bool = False,
) -> TensorProductPlan:
    left = Irreps(irreps_in1).simplify()
    right = Irreps(irreps_in2).simplify()
    instructions = generate_tensor_product_instructions(left, right, irreps_out)
    if weighted:
        out = Irreps(irreps_out)
        start = 0
        weight_slices = []
        for inst in instructions:
            size = inst.path_shape[2] * inst.path_shape[0] * inst.path_shape[1]
            weight_slices.append(slice(start, start + size))
            start += size
        return TensorProductPlan(left, right, out, instructions, True, start, tuple(weight_slices))

    inferred_parts = [MulIrrep(inst.path_shape[0] * inst.path_shape[1], inst.ir_out) for inst in instructions]
    out = Irreps(inferred_parts).regroup()
    if irreps_out is not None:
        requested = Irreps(irreps_out).simplify()
        if requested != out:
            raise ValueError(f"unweighted tensor product infers irreps {out}, got explicit {requested}")
    return TensorProductPlan(left, right, out, instructions, False, 0, tuple())


def _reshape_chunk(chunk: Any, mul: int, dim: int) -> Any:
    return chunk.reshape(*chunk.shape[:-1], mul, dim)


def _pairwise_scalar_mul(a: Any, b: Any) -> Any:
    mx, _ = require_mlx()
    return mx.einsum("...u,...v->...uv", a[..., 0], b[..., 0])[..., None]


def _couple_blocks(inst: TensorProductInstruction, left: Any, right: Any) -> Any:
    mx, _ = require_mlx()
    l1 = inst.ir_in1.l
    l2 = inst.ir_in2.l
    l_out = inst.ir_out.l

    if l1 == 0 and l2 == 0 and l_out == 0:
        return _pairwise_scalar_mul(left, right).astype(left.dtype)

    if l1 == 0 and l2 == l_out:
        return mx.einsum("...u,...vd->...uvd", left[..., 0], right).astype(left.dtype)

    if l2 == 0 and l1 == l_out:
        return mx.einsum("...ud,...v->...uvd", left, right[..., 0]).astype(left.dtype)

    if l1 == 1 and l2 == 1 and l_out == 0:
        dot = mx.einsum("...ua,...va->...uv", left, right) / sqrt(3.0)
        return dot[..., None].astype(left.dtype)

    if l1 == 1 and l2 == 1 and l_out == 1:
        ax, ay, az = left[..., 0], left[..., 1], left[..., 2]
        bx, by, bz = right[..., 0], right[..., 1], right[..., 2]
        out = mx.stack(
            [
                (ay[..., :, None] * bz[..., None, :] - az[..., :, None] * by[..., None, :]) / sqrt(2.0),
                (az[..., :, None] * bx[..., None, :] - ax[..., :, None] * bz[..., None, :]) / sqrt(2.0),
                (ax[..., :, None] * by[..., None, :] - ay[..., :, None] * bx[..., None, :]) / sqrt(2.0),
            ],
            axis=-1,
        )
        return out.astype(left.dtype)

    if l1 == 1 and l2 == 1 and l_out == 2:
        ax, ay, az = left[..., 0], left[..., 1], left[..., 2]
        bx, by, bz = right[..., 0], right[..., 1], right[..., 2]
        return mx.stack(
            [
                (2.0 * az[..., :, None] * bz[..., None, :] - ax[..., :, None] * bx[..., None, :] - ay[..., :, None] * by[..., None, :]) / sqrt(6.0),
                (ax[..., :, None] * by[..., None, :] + ay[..., :, None] * bx[..., None, :]) / sqrt(2.0),
                (ay[..., :, None] * bz[..., None, :] + az[..., :, None] * by[..., None, :]) / sqrt(2.0),
                (az[..., :, None] * bx[..., None, :] + ax[..., :, None] * bz[..., None, :]) / sqrt(2.0),
                (ax[..., :, None] * bx[..., None, :] - ay[..., :, None] * by[..., None, :]) / sqrt(2.0),
            ],
            axis=-1,
        ).astype(left.dtype)

    raise NotImplementedError(
        f"tensor_product currently supports scalar couplings and 1 x 1 -> 0/1/2, got {inst.ir_in1} x {inst.ir_in2} -> {inst.ir_out}"
    )


def _group_instruction_indices(plan: TensorProductPlan) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, inst in enumerate(plan.instructions):
        groups.setdefault(inst.output_index, []).append(index)
    return groups


def _output_block_cursors(plan: TensorProductPlan) -> list[int]:
    return [0 for _ in plan.irreps_out]


def _numerical_tensor_product(
    plan: TensorProductPlan,
    left: IrrepsArray,
    right: IrrepsArray,
    *,
    weights: Any | None = None,
    extension: str | None = None,
) -> IrrepsArray:
    mx, _ = require_mlx()
    if extension is not None and (kernel := get_extension(extension)) is not None:
        return kernel.fn(plan, left, right, weights=weights)
    if plan.weighted != (weights is not None):
        raise ValueError("weights presence does not match plan.weighted")
    if left.irreps != plan.irreps_in1 or right.irreps != plan.irreps_in2:
        raise ValueError("input irreps do not match tensor product plan")

    left_chunks = left.chunk_arrays()
    right_chunks = right.chunk_arrays()
    output_blocks = [mx.zeros((*left.leading_shape, part.mul, part.ir.dim), dtype=left.array.dtype) for part in plan.irreps_out]
    grouped_outputs: list[list[Any]] = [[] for _ in plan.irreps_out]
    for instruction_index, inst in enumerate(plan.instructions):
        block_left = _reshape_chunk(left_chunks[inst.input1_index], inst.path_shape[0], inst.ir_in1.dim)
        block_right = _reshape_chunk(right_chunks[inst.input2_index], inst.path_shape[1], inst.ir_in2.dim)
        pair = _couple_blocks(inst, block_left, block_right) * inst.normalization.scale()
        output_index = inst.output_index
        if weights is None:
            grouped_outputs[output_index].append(pair.reshape(*left.leading_shape, inst.path_shape[0] * inst.path_shape[1], inst.ir_out.dim))
            continue
        raw_weight = weights[plan.weight_slices[instruction_index]]
        weight = raw_weight.reshape(inst.path_shape[2], inst.path_shape[0], inst.path_shape[1]).astype(pair.dtype)
        contribution = mx.einsum("wuv,...uvd->...wd", weight, pair)
        output_blocks[output_index] = output_blocks[output_index] + contribution

    if weights is None:
        output_blocks = [
            mx.concatenate(blocks, axis=-2) if blocks else mx.zeros((*left.leading_shape, part.mul, part.ir.dim), dtype=left.array.dtype)
            for part, blocks in zip(plan.irreps_out, grouped_outputs, strict=True)
        ]
    if not output_blocks:
        return IrrepsArray(plan.irreps_out, mx.zeros((*left.leading_shape, 0), dtype=left.array.dtype))
    array = mx.concatenate([block.reshape(*left.leading_shape, -1) for block in output_blocks], axis=-1)
    return IrrepsArray(plan.irreps_out, array)


def tensor_product(
    left: IrrepsArray | Irreps | str,
    right: IrrepsArray | Irreps | str,
    irreps_out: Irreps | str | None = None,
    *,
    weights: Any | None = None,
    plan: TensorProductPlan | None = None,
    extension: str | None = None,
) -> TensorProductPlan | IrrepsArray:
    if isinstance(left, IrrepsArray) and isinstance(right, IrrepsArray):
        resolved_plan = plan or tensor_product_plan(left.irreps, right.irreps, irreps_out, weighted=weights is not None)
        return _numerical_tensor_product(resolved_plan, left, right, weights=weights, extension=extension)
    if isinstance(left, IrrepsArray) or isinstance(right, IrrepsArray):
        raise TypeError("tensor_product expects either two IrrepsArray inputs or two irreps specs")
    return plan or tensor_product_plan(left, right, irreps_out, weighted=weights is not None)


def compile_tensor_product(plan: TensorProductPlan, *, extension: str | None = None):
    def compiled(left_array: Any, right_array: Any, weights: Any | None = None) -> Any:
        left = IrrepsArray(plan.irreps_in1, left_array)
        right = IrrepsArray(plan.irreps_in2, right_array)
        return _numerical_tensor_product(plan, left, right, weights=weights, extension=extension).array

    return compile_or_identity(compiled)
