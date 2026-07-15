"""Tensor product MLX helpers."""

from __future__ import annotations

from dataclasses import dataclass
import functools
from math import prod
from typing import Any

from e3nn_core.cg import clebsch_gordan
from e3nn_core.instructions import TensorProductInstruction, generate_tensor_product_instructions, make_tensor_product_instructions
from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
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
        weight = weights.reshape(*weights.shape[:-1], *inst.path_shape).astype(dtype)
        return mx.einsum("...uvw,...uvd->...wd", weight, pair)
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
    if left.dtype != right.dtype:
        raise TypeError(f"tensor product inputs must have matching dtypes, got {left.dtype} and {right.dtype}")
    if weights is not None:
        if weights.ndim < 1 or weights.shape[-1] != plan.weight_numel:
            raise ValueError(f"expected weight shape (..., {plan.weight_numel}), got {tuple(weights.shape)}")
        weight_leading_shape = tuple(int(size) for size in weights.shape[:-1])
    else:
        weight_leading_shape = ()
    try:
        leading_shape = mx.broadcast_shapes(left.leading_shape, right.leading_shape, weight_leading_shape)
    except ValueError as exc:
        raise ValueError(
            f"tensor product leading shapes are not broadcastable: {left.leading_shape}, "
            f"{right.leading_shape}, {weight_leading_shape}"
        ) from exc
    if left.leading_shape != leading_shape:
        left = IrrepsArray(left.irreps, mx.broadcast_to(left.array, (*leading_shape, left.irreps.dim)))
    if right.leading_shape != leading_shape:
        right = IrrepsArray(right.irreps, mx.broadcast_to(right.array, (*leading_shape, right.irreps.dim)))

    left_chunks = left.chunk_arrays()
    right_chunks = right.chunk_arrays()
    output_blocks = [mx.zeros((*leading_shape, part.mul, part.ir.dim), dtype=left.array.dtype) for part in plan.irreps_out]
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
            mx.concatenate(blocks, axis=-2) if blocks else mx.zeros((*leading_shape, part.mul, part.ir.dim), dtype=left.array.dtype)
            for part, blocks in zip(plan.irreps_out, grouped_outputs, strict=True)
        ]
    if not output_blocks:
        return IrrepsArray(plan.irreps_out, mx.zeros((*leading_shape, 0), dtype=left.array.dtype))
    array = mx.concatenate([block.reshape(*leading_shape, -1) for block in output_blocks], axis=-1)
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


class TensorProduct(mlx_module_base()):
    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        instructions: list[tuple[int, int, int, str, bool] | tuple[int, int, int, str, bool, float]],
        *,
        in1_var: list[float] | None = None,
        in2_var: list[float] | None = None,
        out_var: list[float] | None = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: bool | None = None,
        shared_weights: bool = True,
        compile_left_right: bool = True,
    ) -> None:
        super().__init__()
        mx, _ = require_mlx()
        self.irreps_in1 = Irreps(irreps_in1).remove_zero_multiplicities()
        self.irreps_in2 = Irreps(irreps_in2).remove_zero_multiplicities()
        self.irreps_out = Irreps(irreps_out).remove_zero_multiplicities()
        self.instructions = make_tensor_product_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
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
            has_weight = normalized.has_weight
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
        self.weight = mx.random.normal(shape=(self.weight_numel,)) if self.internal_weights and self.weight_numel else None
        self._output_mask = self._build_output_mask(mx)
        self._compiled = compile_or_identity(self._call_arrays, enabled=compile_left_right)

    @property
    def output_mask(self):
        return self._output_mask

    def _build_output_mask(self, mx):
        chunks = []
        active = {ins.output_index for ins in self.instructions if ins.normalization.coefficient != 0.0 and 0 not in ins.path_shape}
        for index, part in enumerate(self.irreps_out):
            value = 1.0 if index in active else 0.0
            chunks.append(mx.full((part.mul * part.ir.dim,), value))
        return mx.concatenate(chunks) if chunks else mx.zeros((0,))

    def __repr__(self) -> str:
        path_count = sum(prod(ins.path_shape) for ins in self.instructions)
        return f"{self.__class__.__name__}({self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out} | {path_count} paths | {self.weight_numel} weights)"

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
        if left.irreps != self.irreps_in1:
            raise ValueError(f"left input irreps {left.irreps} do not match {self.irreps_in1}")
        if right.irreps != self.irreps_in2:
            raise ValueError(f"right input irreps {right.irreps} do not match {self.irreps_in2}")
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


def _as_irreps_list(irreps: Irreps | str) -> list[MulIrrep]:
    return list(Irreps(irreps).simplify())


def _split_elementwise_inputs(irreps_in1: Irreps, irreps_in2: Irreps) -> tuple[Irreps, Irreps]:
    parts1 = list(irreps_in1)
    parts2 = list(irreps_in2)
    if irreps_in1.num_irreps != irreps_in2.num_irreps:
        raise ValueError("ElementwiseTensorProduct requires irreps_in1.num_irreps == irreps_in2.num_irreps")
    i = 0
    while i < len(parts1):
        part1 = parts1[i]
        part2 = parts2[i]
        if part1.mul < part2.mul:
            parts2[i] = MulIrrep(part1.mul, part2.ir)
            parts2.insert(i + 1, MulIrrep(part2.mul - part1.mul, part2.ir))
        elif part2.mul < part1.mul:
            parts1[i] = MulIrrep(part2.mul, part1.ir)
            parts1.insert(i + 1, MulIrrep(part1.mul - part2.mul, part1.ir))
        i += 1
    return Irreps(parts1), Irreps(parts2)


def _group_output_irreps(irreps: Irreps) -> tuple[Irreps, tuple[tuple[int, ...], ...]]:
    grouped: dict[Irrep, list[int]] = {}
    multiplicities: dict[Irrep, int] = {}
    for index, part in enumerate(irreps):
        grouped.setdefault(part.ir, []).append(index)
        multiplicities[part.ir] = multiplicities.get(part.ir, 0) + part.mul
    ordered_irreps = sorted(grouped, key=lambda ir: (ir.l, ir.p))
    grouped_irreps = Irreps(MulIrrep(multiplicities[ir], ir) for ir in ordered_irreps)
    index_groups = tuple(tuple(grouped[ir]) for ir in ordered_irreps)
    return grouped_irreps, index_groups


def _regroup_output_array(array: IrrepsArray, irreps_out: Irreps, index_groups: tuple[tuple[int, ...], ...]) -> IrrepsArray:
    mx, _ = require_mlx()
    chunks = array.chunk_arrays()
    regrouped_chunks = []
    for part, indices in zip(irreps_out, index_groups, strict=True):
        merged = [
            _reshape_chunk(chunks[index], array.irreps[index].mul, part.ir.dim)
            for index in indices
        ]
        regrouped = mx.concatenate(merged, axis=-2) if len(merged) > 1 else merged[0]
        regrouped_chunks.append(regrouped.reshape(*array.leading_shape, -1))
    return IrrepsArray.from_chunks(irreps_out, regrouped_chunks, backend=mx)


def _explicit_mul_repr(irreps: Irreps) -> str:
    return "+".join(f"{part.mul}x{part.ir}" for part in irreps)


class FullyConnectedTensorProduct(TensorProduct):
    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: bool | None = None,
        shared_weights: bool = True,
        in1_var: list[float] | None = None,
        in2_var: list[float] | None = None,
        out_var: list[float] | None = None,
        compile_left_right: bool = True,
    ) -> None:
        left = Irreps(irreps_in1).simplify()
        right = Irreps(irreps_in2).simplify()
        output = Irreps(irreps_out).simplify()
        instructions = [
            (i1, i2, i_out, "uvw", True, 1.0)
            for i1, part1 in enumerate(left)
            for i2, part2 in enumerate(right)
            for i_out, part_out in enumerate(output)
            if part_out.ir in (part1.ir * part2.ir)
        ]
        super().__init__(
            left,
            right,
            output,
            instructions,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            compile_left_right=compile_left_right,
        )


class ElementwiseTensorProduct(TensorProduct):
    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        filter_ir_out: list[Irrep | str] | None = None,
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        compile_left_right: bool = True,
    ) -> None:
        left = Irreps(irreps_in1).simplify()
        right = Irreps(irreps_in2).simplify()
        left, right = _split_elementwise_inputs(left, right)
        if filter_ir_out is not None:
            filter_ir_out = [Irrep.parse(ir) for ir in filter_ir_out]
        output_parts: list[MulIrrep] = []
        instructions: list[tuple[int, int, int, str, bool]] = []
        for i, (part1, part2) in enumerate(zip(left, right, strict=True)):
            for ir_out in part1.ir * part2.ir:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue
                i_out = len(output_parts)
                output_parts.append(MulIrrep(part1.mul, ir_out))
                instructions.append((i, i, i_out, "uuu", False))
        super().__init__(
            left,
            right,
            Irreps(output_parts),
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=False,
            shared_weights=True,
            compile_left_right=compile_left_right,
        )


class FullTensorProduct(TensorProduct):
    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        filter_ir_out: list[Irrep | str] | None = None,
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        compile_left_right: bool = True,
    ) -> None:
        left = Irreps(irreps_in1).simplify()
        right = Irreps(irreps_in2).simplify()
        self._execution_irreps_out: Irreps | None = None
        self._output_index_groups: tuple[tuple[int, ...], ...] | None = None
        if filter_ir_out is not None:
            filter_ir_out = [Irrep.parse(ir) for ir in filter_ir_out]
        output_parts: list[MulIrrep] = []
        instructions: list[tuple[int, int, int, str, bool]] = []
        for i1, part1 in enumerate(left):
            for i2, part2 in enumerate(right):
                for ir_out in part1.ir * part2.ir:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    i_out = len(output_parts)
                    output_parts.append(MulIrrep(part1.mul * part2.mul, ir_out))
                    instructions.append((i1, i2, i_out, "uvuv", False))
        output = Irreps(output_parts)
        super().__init__(
            left,
            right,
            output,
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=False,
            shared_weights=True,
            compile_left_right=compile_left_right,
        )
        self._execution_irreps_out = self.irreps_out
        grouped_out, index_groups = _group_output_irreps(self._execution_irreps_out)
        self.irreps_out = grouped_out
        self._output_index_groups = index_groups
        mx, _ = require_mlx()
        grouped_mask = _regroup_output_array(
            IrrepsArray(self._execution_irreps_out, self.output_mask[None, :]),
            self.irreps_out,
            self._output_index_groups,
        )
        self._output_mask = mx.maximum(grouped_mask.array[0], 0.0)

    def __repr__(self) -> str:
        path_count = sum(prod(ins.path_shape) for ins in self.instructions)
        return (
            f"FullTensorProduct({_explicit_mul_repr(self.irreps_in1)} x {_explicit_mul_repr(self.irreps_in2)} -> "
            f"{_explicit_mul_repr(self.irreps_out)} | {path_count} paths | {self.weight_numel} weights)"
        )

    def _plan(self, weighted: bool) -> TensorProductPlan:
        plan = super()._plan(weighted)
        return TensorProductPlan(
            irreps_in1=plan.irreps_in1,
            irreps_in2=plan.irreps_in2,
            irreps_out=self._execution_irreps_out,
            instructions=plan.instructions,
            weighted=plan.weighted,
            weight_numel=plan.weight_numel,
            weighted_instruction_indices=plan.weighted_instruction_indices,
            weight_slices=plan.weight_slices,
        )

    def __call__(self, left: IrrepsArray, right: IrrepsArray, weight: Any | None = None) -> IrrepsArray:
        out = IrrepsArray(self._execution_irreps_out, self._compiled(left.array, right.array, None))
        return _regroup_output_array(out, self.irreps_out, self._output_index_groups)


def _tensor_square_full_instructions(
    irreps_in: Irreps,
    filter_ir_out: list[Irrep] | None,
    irrep_normalization: str,
) -> tuple[Irreps, list[tuple[int, int, int, str, bool, float]]]:
    output_parts: list[MulIrrep] = []
    instructions: list[tuple[int, int, int, str, bool, float]] = []
    for i1, part1 in enumerate(irreps_in):
        for i2, part2 in enumerate(irreps_in):
            for ir_out in part1.ir * part2.ir:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue
                alpha = _tensor_square_alpha(part1.ir, part2.ir, ir_out, irrep_normalization)
                if i1 < i2:
                    i_out = len(output_parts)
                    output_parts.append(MulIrrep(part1.mul * part2.mul, ir_out))
                    instructions.append((i1, i2, i_out, "uvuv", False, alpha))
                elif i1 == i2:
                    mul = part1.mul
                    if mul > 1:
                        i_out = len(output_parts)
                        output_parts.append(MulIrrep(mul * (mul - 1) // 2, ir_out))
                        instructions.append((i1, i1, i_out, "uvu<v", False, alpha))
                    if ir_out.l % 2 == 0:
                        even_alpha = _tensor_square_diagonal_alpha(part1.ir, ir_out, irrep_normalization)
                        i_out = len(output_parts)
                        output_parts.append(MulIrrep(mul, ir_out))
                        instructions.append((i1, i1, i_out, "uuu", False, even_alpha))
    output = Irreps(output_parts)
    return output, instructions


def _tensor_square_fc_instructions(
    irreps_in: Irreps,
    irreps_out: Irreps,
    irrep_normalization: str,
) -> list[tuple[int, int, int, str, bool, float]]:
    instructions: list[tuple[int, int, int, str, bool, float]] = []
    for i1, part1 in enumerate(irreps_in):
        for i2, part2 in enumerate(irreps_in):
            for i_out, part_out in enumerate(irreps_out):
                if part_out.ir not in (part1.ir * part2.ir):
                    continue
                alpha = _tensor_square_alpha(part1.ir, part2.ir, part_out.ir, irrep_normalization)
                if i1 < i2:
                    instructions.append((i1, i2, i_out, "uvw", True, alpha))
                elif i1 == i2:
                    if part1.mul > 1:
                        instructions.append((i1, i1, i_out, "u<vw", True, alpha))
                    if part_out.ir.l % 2 == 0:
                        even_alpha = _tensor_square_diagonal_alpha(part1.ir, part_out.ir, irrep_normalization)
                        instructions.append((i1, i1, i_out, "uuw", True, even_alpha))
    return instructions


def _tensor_square_alpha(ir1: Irrep, ir2: Irrep, ir_out: Irrep, irrep_normalization: str) -> float:
    if irrep_normalization == "component":
        return float(ir_out.dim)
    if irrep_normalization == "norm":
        return float(ir1.dim * ir2.dim)
    if irrep_normalization == "none":
        return 1.0
    raise ValueError(f"unsupported irrep_normalization {irrep_normalization!r}")


def _tensor_square_diagonal_alpha(ir: Irrep, ir_out: Irrep, irrep_normalization: str) -> float:
    if irrep_normalization == "component":
        if ir_out.l == 0:
            return float(ir_out.dim / (ir.dim + 2))
        return float(ir_out.dim / 2)
    if irrep_normalization == "norm":
        if ir_out.l == 0:
            return float(ir_out.dim * ir.dim)
        return float(ir.dim * (ir.dim + 2) / 2)
    if irrep_normalization == "none":
        return 1.0
    raise ValueError(f"unsupported irrep_normalization {irrep_normalization!r}")


class TensorSquare(TensorProduct):
    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str | None = None,
        filter_ir_out: list[Irrep | str] | None = None,
        *,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: bool | None = None,
        shared_weights: bool = True,
        compile_left_right: bool = True,
    ) -> None:
        irreps_in = Irreps(irreps_in).simplify()
        parsed_filter = None if filter_ir_out is None else [Irrep.parse(ir) for ir in filter_ir_out]
        self._execution_irreps_out: Irreps | None = None
        self._output_index_groups: tuple[tuple[int, ...], ...] | None = None
        if irreps_out is None:
            output, instructions = _tensor_square_full_instructions(irreps_in, parsed_filter, irrep_normalization)
            super().__init__(
                irreps_in,
                irreps_in,
                output,
                instructions,
                irrep_normalization="none",
                path_normalization=path_normalization,
                internal_weights=False,
                shared_weights=True,
                compile_left_right=compile_left_right,
            )
            self._execution_irreps_out = self.irreps_out
            grouped_out, index_groups = _group_output_irreps(self._execution_irreps_out)
            self.irreps_out = grouped_out
            self._output_index_groups = index_groups
            mx, _ = require_mlx()
            grouped_mask = _regroup_output_array(
                IrrepsArray(self._execution_irreps_out, self.output_mask[None, :]),
                self.irreps_out,
                self._output_index_groups,
            )
            self._output_mask = mx.maximum(grouped_mask.array[0], 0.0)
        else:
            if parsed_filter is not None:
                raise ValueError("Both irreps_out and filter_ir_out were provided")
            output = Irreps(irreps_out).simplify()
            instructions = _tensor_square_fc_instructions(irreps_in, output, irrep_normalization)
            super().__init__(
                irreps_in,
                irreps_in,
                output,
                instructions,
                irrep_normalization="none",
                path_normalization=path_normalization,
                internal_weights=internal_weights,
                shared_weights=shared_weights,
                compile_left_right=compile_left_right,
            )
            self._execution_irreps_out = self.irreps_out
        self.irreps_in = irreps_in

    def __repr__(self) -> str:
        path_count = sum(prod(ins.path_shape) for ins in self.instructions)
        return (
            f"TensorSquare({_explicit_mul_repr(self.irreps_in)} -> {_explicit_mul_repr(self.irreps_out)} | "
            f"{path_count} paths | {self.weight_numel} weights)"
        )

    def _plan(self, weighted: bool) -> TensorProductPlan:
        plan = super()._plan(weighted)
        if self._execution_irreps_out is None:
            return plan
        return TensorProductPlan(
            irreps_in1=plan.irreps_in1,
            irreps_in2=plan.irreps_in2,
            irreps_out=self._execution_irreps_out,
            instructions=plan.instructions,
            weighted=plan.weighted,
            weight_numel=plan.weight_numel,
            weighted_instruction_indices=plan.weighted_instruction_indices,
            weight_slices=plan.weight_slices,
        )

    def __call__(self, array: IrrepsArray, weight: Any | None = None) -> IrrepsArray:
        resolved_weight = self._get_weight(weight)
        out = IrrepsArray(self._execution_irreps_out, self._compiled(array.array, array.array, resolved_weight))
        if self._output_index_groups is None:
            return out
        return _regroup_output_array(out, self.irreps_out, self._output_index_groups)
