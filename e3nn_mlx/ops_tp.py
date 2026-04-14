"""Tensor product MLX helpers."""

from __future__ import annotations

from dataclasses import dataclass
import functools
from math import prod
from typing import Any

from e3nn_core.cg import clebsch_gordan
from e3nn_core.instructions import TensorProductInstruction, generate_tensor_product_instructions, make_tensor_product_instructions
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
    weighted_instruction_indices: tuple[int, ...]
    weight_slices: tuple[slice, ...]


@dataclass(frozen=True, slots=True)
class WeightedInstruction:
    instruction_index: int
    instruction: TensorProductInstruction
    weight_slice: slice


def tensor_product_plan(
    irreps_in1: Irreps | str,
    irreps_in2: Irreps | str,
    irreps_out: Irreps | str | None = None,
    *,
    weighted: bool = False,
    mode: str = "uvw",
) -> TensorProductPlan:
    left = Irreps(irreps_in1).simplify()
    right = Irreps(irreps_in2).simplify()
    instructions = generate_tensor_product_instructions(left, right, irreps_out, mode=mode)
    if weighted:
        out = Irreps(irreps_out)
        start = 0
        weight_slices = []
        for inst in instructions:
            size = prod(inst.path_shape)
            weight_slices.append(slice(start, start + size))
            start += size
        return TensorProductPlan(left, right, out, instructions, True, start, tuple(range(len(instructions))), tuple(weight_slices))

    inferred_parts = []
    for inst in instructions:
        if inst.mode == "uvw":
            out_mul = inst.path_shape[0] * inst.path_shape[1]
        elif inst.mode == "uvu":
            out_mul = inst.path_shape[0]
        elif inst.mode == "uvv":
            out_mul = inst.path_shape[1]
        elif inst.mode == "uuw":
            out_mul = 1
        elif inst.mode == "uuu":
            out_mul = inst.path_shape[0]
        elif inst.mode == "uvuv":
            out_mul = inst.path_shape[0] * inst.path_shape[1]
        elif inst.mode == "uvu<v":
            out_mul = inst.path_shape[0]
        elif inst.mode == "u<vw":
            out_mul = inst.path_shape[1]
        else:
            raise ValueError(f"unsupported tensor product mode {inst.mode!r}")
        inferred_parts.append(MulIrrep(out_mul, inst.ir_out))
    out = Irreps(inferred_parts)
    if irreps_out is not None:
        requested = Irreps(irreps_out).simplify()
        if requested != out:
            raise ValueError(f"unweighted tensor product infers irreps {out}, got explicit {requested}")
    return TensorProductPlan(left, right, out, instructions, False, 0, tuple(), tuple())


def _reshape_chunk(chunk: Any, mul: int, dim: int) -> Any:
    return chunk.reshape(*chunk.shape[:-1], mul, dim)


@functools.lru_cache(maxsize=None)
def _cg_array_data(ir1: Irrep, ir2: Irrep, ir_out: Irrep) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return clebsch_gordan(ir1, ir2, ir_out)


def _couple_blocks(inst: TensorProductInstruction, left: Any, right: Any) -> Any:
    mx, _ = require_mlx()
    if inst.ir_in1.p * inst.ir_in2.p != inst.ir_out.p:
        raise ValueError(f"parity mismatch in tensor product instruction {inst}")
    cg = mx.array(_cg_array_data(inst.ir_in1, inst.ir_in2, inst.ir_out), dtype=left.dtype)
    return mx.einsum("...ua,...vb,abc->...uvc", left, right, cg).astype(left.dtype)


def _diagonal_pair(pair: Any) -> Any:
    mx, _ = require_mlx()
    size = pair.shape[-3]
    indices = mx.arange(size)
    return pair[..., indices, indices, :]


def _strict_upper_pair(pair: Any) -> Any:
    mx, _ = require_mlx()
    size = pair.shape[-3]
    rows = []
    cols = []
    for i in range(size):
        for j in range(i + 1, size):
            rows.append(i)
            cols.append(j)
    row_idx = mx.array(rows, dtype=mx.int32)
    col_idx = mx.array(cols, dtype=mx.int32)
    return pair[..., row_idx, col_idx, :]


def _apply_connection_mode(inst: TensorProductInstruction, pair: Any, weights: Any | None, dtype: Any) -> Any:
    mx, _ = require_mlx()
    mode = inst.mode
    if mode == "uvw":
        if weights is None:
            return pair.reshape(*pair.shape[:-3], inst.path_shape[0] * inst.path_shape[1], inst.ir_out.dim)
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[2], inst.path_shape[0], inst.path_shape[1]).astype(dtype)
        return mx.einsum("...wuv,...uvd->...wd", weight, pair)
    if mode == "uvu":
        if weights is None:
            return mx.sum(pair, axis=-2)
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0], inst.path_shape[1]).astype(dtype)
        return mx.einsum("...uv,...uvd->...ud", weight, pair)
    if mode == "uvv":
        if weights is None:
            return mx.sum(pair, axis=-3)
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0], inst.path_shape[1]).astype(dtype)
        return mx.einsum("...uv,...uvd->...vd", weight, pair)
    if mode == "uuw":
        diag = _diagonal_pair(pair)
        if weights is None:
            return mx.sum(diag, axis=-2, keepdims=True)
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0], inst.path_shape[1]).astype(dtype)
        return mx.einsum("...uw,...ud->...wd", weight, diag)
    if mode == "uuu":
        diag = _diagonal_pair(pair)
        if weights is None:
            return diag
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0]).astype(dtype)
        return mx.einsum("...u,...ud->...ud", weight, diag)
    if mode == "uvuv":
        scaled = (
            pair
            if weights is None
            else weights.reshape(*weights.shape[:-1], inst.path_shape[0], inst.path_shape[1]).astype(dtype)[..., None] * pair
        )
        return scaled.reshape(*scaled.shape[:-3], inst.path_shape[0] * inst.path_shape[1], inst.ir_out.dim)
    if mode == "uvu<v":
        upper = _strict_upper_pair(pair)
        if weights is None:
            return upper
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0]).astype(dtype)
        return mx.einsum("...q,...qd->...qd", weight, upper)
    if mode == "u<vw":
        upper = _strict_upper_pair(pair)
        if weights is None:
            return mx.sum(upper, axis=-2, keepdims=True)
        weight = weights.reshape(*weights.shape[:-1], inst.path_shape[0], inst.path_shape[1]).astype(dtype)
        return mx.einsum("...qw,...qd->...wd", weight, upper)
    raise ValueError(f"unsupported tensor product mode {mode!r}")


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
    weighted_lookup = {instruction_index: weight_slice for instruction_index, weight_slice in zip(plan.weighted_instruction_indices, plan.weight_slices, strict=True)}
    for instruction_index, inst in enumerate(plan.instructions):
        left_mul = plan.irreps_in1[inst.input1_index].mul
        right_mul = plan.irreps_in2[inst.input2_index].mul
        block_left = _reshape_chunk(left_chunks[inst.input1_index], left_mul, inst.ir_in1.dim)
        block_right = _reshape_chunk(right_chunks[inst.input2_index], right_mul, inst.ir_in2.dim)
        pair = _couple_blocks(inst, block_left, block_right) * inst.normalization.scale()
        output_index = inst.output_index
        raw_weight = None
        if weights is not None:
            weight_slice = weighted_lookup.get(instruction_index)
            if weight_slice is not None:
                raw_weight = weights[..., weight_slice]
        contribution = _apply_connection_mode(inst, pair, raw_weight, pair.dtype)
        if weights is None:
            grouped_outputs[output_index].append(contribution)
        else:
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
    mode: str = "uvw",
) -> TensorProductPlan | IrrepsArray:
    if isinstance(left, IrrepsArray) and isinstance(right, IrrepsArray):
        resolved_plan = plan or tensor_product_plan(left.irreps, right.irreps, irreps_out, weighted=weights is not None, mode=mode)
        return _numerical_tensor_product(resolved_plan, left, right, weights=weights, extension=extension)
    if isinstance(left, IrrepsArray) or isinstance(right, IrrepsArray):
        raise TypeError("tensor_product expects either two IrrepsArray inputs or two irreps specs")
    return plan or tensor_product_plan(left, right, irreps_out, weighted=weights is not None, mode=mode)


def compile_tensor_product(plan: TensorProductPlan, *, extension: str | None = None):
    def compiled(left_array: Any, right_array: Any, weights: Any | None = None) -> Any:
        left = IrrepsArray(plan.irreps_in1, left_array)
        right = IrrepsArray(plan.irreps_in2, right_array)
        return _numerical_tensor_product(plan, left, right, weights=weights, extension=extension).array

    return compile_or_identity(compiled)


class TensorProduct:
    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        instructions: list[tuple[int, int, int, str, bool] | tuple[int, int, int, str, bool, float]],
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: bool | None = None,
        shared_weights: bool = True,
        compile_left_right: bool = True,
    ) -> None:
        mx, _ = require_mlx()
        self.irreps_in1 = Irreps(irreps_in1).simplify()
        self.irreps_in2 = Irreps(irreps_in2).simplify()
        self.irreps_out = Irreps(irreps_out).simplify()
        self.instructions = make_tensor_product_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )
        self.shared_weights = shared_weights
        self.internal_weights = any(ins[4] for ins in instructions) if internal_weights is None else internal_weights
        if self.internal_weights and not self.shared_weights:
            raise ValueError("internal_weights=True requires shared_weights=True")
        self._weighted_instruction_meta: list[WeightedInstruction] = []
        start = 0
        for index, (raw_instruction, normalized) in enumerate(zip(instructions, self.instructions, strict=True)):
            has_weight = raw_instruction[4]
            if not has_weight:
                continue
            size = prod(normalized.path_shape)
            self._weighted_instruction_meta.append(
                WeightedInstruction(
                    instruction_index=index,
                    instruction=normalized,
                    weight_slice=slice(start, start + size),
                )
            )
            start += size
        self.weight_numel = start
        self.weight = mx.random.normal(shape=(self.weight_numel,)) if self.internal_weights and self.weight_numel else mx.zeros((0,))
        self.output_mask = self._build_output_mask(mx)
        self._compiled = compile_or_identity(self._call_arrays, enabled=compile_left_right)

    def _build_output_mask(self, mx):
        chunks = []
        active = {ins.output_index for ins in self.instructions if ins.normalization.coefficient != 0.0 and 0 not in ins.path_shape}
        for index, part in enumerate(self.irreps_out):
            value = 1.0 if index in active else 0.0
            chunks.append(mx.full((part.mul * part.ir.dim,), value))
        return mx.concatenate(chunks) if chunks else mx.zeros((0,))

    def __repr__(self) -> str:
        path_count = sum(prod(ins.path_shape) for ins in self.instructions)
        return f"TensorProduct({self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {path_count} paths | {self.weight_numel} weights)"

    def _plan(self, weighted: bool) -> TensorProductPlan:
        weight_slices = tuple(meta.weight_slice for meta in self._weighted_instruction_meta)
        weighted_instruction_indices = tuple(meta.instruction_index for meta in self._weighted_instruction_meta)
        return TensorProductPlan(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=self.instructions,
            weighted=weighted,
            weight_numel=self.weight_numel,
            weighted_instruction_indices=weighted_instruction_indices,
            weight_slices=weight_slices,
        )

    def _call_arrays(self, left_array: Any, right_array: Any, weight: Any | None = None) -> Any:
        left = IrrepsArray(self.irreps_in1, left_array)
        right = IrrepsArray(self.irreps_in2, right_array)
        return _numerical_tensor_product(self._plan(weight is not None), left, right, weights=weight).array

    def _get_weight(self, weight: Any | None) -> Any | None:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights is False")
            return self.weight if self.weight_numel > 0 else None
        if self.shared_weights:
            if tuple(weight.shape) != (self.weight_numel,):
                raise ValueError(f"Expected shared weight shape {(self.weight_numel,)}, got {tuple(weight.shape)}")
        else:
            if weight.shape[-1] != self.weight_numel:
                raise ValueError(f"Expected unshared weight shape (..., {self.weight_numel}), got {tuple(weight.shape)}")
        return weight

    def __call__(self, left: IrrepsArray, right: IrrepsArray, weight: Any | None = None) -> IrrepsArray:
        resolved_weight = self._get_weight(weight)
        return IrrepsArray(self.irreps_out, self._compiled(left.array, right.array, resolved_weight))

    def right(self, right: IrrepsArray, weight: Any | None = None) -> Any:
        mx, _ = require_mlx()
        if right.irreps != self.irreps_in2:
            raise ValueError("input irreps do not match TensorProduct.irreps_in2")
        resolved_weight = self._get_weight(weight)
        eye = mx.eye(self.irreps_in1.dim, dtype=right.array.dtype)
        basis_outputs = []
        for index in range(self.irreps_in1.dim):
            basis = mx.broadcast_to(eye[index], (*right.leading_shape, self.irreps_in1.dim))
            left = IrrepsArray(self.irreps_in1, basis)
            out = self(left, right, weight=resolved_weight)
            basis_outputs.append(out.array)
        return mx.stack(basis_outputs, axis=-2)

    def weight_view_for_instruction(self, instruction_index: int, weight: Any | None = None) -> Any:
        resolved_weight = self._get_weight(weight)
        for meta in self._weighted_instruction_meta:
            if meta.instruction_index == instruction_index:
                return resolved_weight[..., meta.weight_slice].reshape(*resolved_weight.shape[:-1], *meta.instruction.path_shape)
        raise ValueError(f"Instruction {instruction_index} has no weights.")

    def weight_views(self, weight: Any | None = None):
        for meta in self._weighted_instruction_meta:
            yield self.weight_view_for_instruction(meta.instruction_index, weight=weight)
