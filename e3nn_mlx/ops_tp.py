"""Tensor product MLX helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import functools
from math import prod
from typing import Any

from e3nn_core.cg import clebsch_gordan
from e3nn_core.instructions import TensorProductInstruction, generate_tensor_product_instructions, make_tensor_product_instructions
from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_metal_available, mlx_module_base, require_mlx
from .irreps_array import IrrepsArray
from ._metal_tp import (
    build_channel_metadata,
    build_metadata,
    make_channel_operation,
    make_operation,
)
from .ops_basic import compile_or_identity, get_extension


# Empirical crossover/metadata limits for the scalar-path Metal dispatch.
# Keep these named and centralized so benchmark-driven tuning does not require
# modifying the selection control flow.
_METAL_SCALAR_PATH_MAX_TERMS = 2_000_000
_METAL_SCALAR_PATH_MAX_BATCH = 512
_METAL_SCALAR_PATH_MAX_WORK = 2_000_000


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


@functools.lru_cache(maxsize=None)
def _combined_cg_array_data(
    specifications: tuple[tuple[Irrep, Irrep, Irrep, float], ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Concatenate normalized CG bases sharing the same pair of input irreps."""
    if not specifications:
        return ()
    tensors = [
        (_cg_array_data(ir1, ir2, ir_out), scale)
        for ir1, ir2, ir_out, scale in specifications
    ]
    dim1 = specifications[0][0].dim
    dim2 = specifications[0][1].dim
    return tuple(
        tuple(
            tuple(
                value * scale
                for tensor, scale in tensors
                for value in tensor[index1][index2]
            )
            for index2 in range(dim2)
        )
        for index1 in range(dim1)
    )


def _couple_blocks(inst: TensorProductInstruction, left: Any, right: Any) -> Any:
    mx, _ = require_mlx()
    if inst.ir_in1.p * inst.ir_in2.p != inst.ir_out.p:
        raise ValueError(f"parity mismatch in tensor product instruction {inst}")
    data = _cg_array_data(inst.ir_in1, inst.ir_in2, inst.ir_out)
    if (
        inst.mode != "uvw"
        and inst.ir_in1.dim == 1
        and inst.ir_out.dim == inst.ir_in2.dim
    ):
        diagonal = mx.array(
            tuple(data[0][index][index] for index in range(inst.ir_in2.dim)),
            dtype=left.dtype,
        )
        return left[..., :, None, 0, None] * right[..., None, :, :] * diagonal
    if (
        inst.mode != "uvw"
        and inst.ir_in2.dim == 1
        and inst.ir_out.dim == inst.ir_in1.dim
    ):
        diagonal = mx.array(
            tuple(data[index][0][index] for index in range(inst.ir_in1.dim)),
            dtype=left.dtype,
        )
        return left[..., :, None, :] * right[..., None, :, 0, None] * diagonal
    cg = mx.array(data, dtype=left.dtype)
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


def _couple_instruction_group(
    instructions: tuple[TensorProductInstruction, ...], left: Any, right: Any
) -> Any:
    """Couple one input-block pair to several output irreps in one transform."""
    mx, _ = require_mlx()
    specifications = tuple(
        (
            instruction.ir_in1,
            instruction.ir_in2,
            instruction.ir_out,
            instruction.normalization.scale(),
        )
        for instruction in instructions
    )
    data = _combined_cg_array_data(specifications)
    dim1 = instructions[0].ir_in1.dim
    dim2 = instructions[0].ir_in2.dim
    output_dim = sum(instruction.ir_out.dim for instruction in instructions)

    # Coupling to or from a scalar is a diagonal scaling in the canonical basis.
    if dim1 == 1 and output_dim == dim2:
        diagonal = mx.array(
            tuple(data[0][index][index] for index in range(dim2)),
            dtype=left.dtype,
        )
        return (
            left[..., :, None, 0, None]
            * right[..., None, :, :]
            * diagonal
        )
    if dim2 == 1 and output_dim == dim1:
        diagonal = mx.array(
            tuple(data[index][0][index] for index in range(dim1)),
            dtype=left.dtype,
        )
        return (
            left[..., :, None, :]
            * right[..., None, :, 0, None]
            * diagonal
        )

    coefficients = mx.array(data, dtype=left.dtype).reshape(
        dim1 * dim2, output_dim
    )
    outer = (
        left[..., :, None, :, None] * right[..., None, :, None, :]
    ).reshape(*left.shape[:-2], left.shape[-2], right.shape[-2], dim1 * dim2)
    return outer @ coefficients


def _grouped_weighted_tensor_product(
    plan: TensorProductPlan,
    left_chunks: tuple[Any, ...],
    right_chunks: tuple[Any, ...],
    weights: Any,
    leading_shape: tuple[int, ...],
    weighted_lookup: dict[int, slice],
    dtype: Any,
) -> IrrepsArray:
    """Execute compatible weighted paths without repeated coupling tensors."""
    mx, _ = require_mlx()
    groups: dict[
        tuple[int, int, str], list[tuple[int, TensorProductInstruction]]
    ] = {}
    for index, instruction in enumerate(plan.instructions):
        groups.setdefault(
            (
                instruction.input1_index,
                instruction.input2_index,
                instruction.mode,
            ),
            [],
        ).append((index, instruction))

    output_blocks = [
        mx.zeros((*leading_shape, part.mul, part.ir.dim), dtype=dtype)
        for part in plan.irreps_out
    ]
    for (input1, input2, mode), indexed_instructions in groups.items():
        instructions = tuple(instruction for _, instruction in indexed_instructions)
        part1 = plan.irreps_in1[input1]
        part2 = plan.irreps_in2[input2]
        block1 = _reshape_chunk(left_chunks[input1], part1.mul, part1.ir.dim)
        block2 = _reshape_chunk(right_chunks[input2], part2.mul, part2.ir.dim)
        coupled = _couple_instruction_group(instructions, block1, block2)
        start = 0
        for instruction_index, instruction in indexed_instructions:
            stop = start + instruction.ir_out.dim
            weight_slice = weighted_lookup[instruction_index]
            weight = weights[..., weight_slice].reshape(
                *weights.shape[:-1], *instruction.path_shape
            ).astype(dtype)
            selected = coupled[..., start:stop]
            if mode == "uvu":
                contribution = mx.sum(selected * weight[..., None], axis=-2)
            elif mode == "uvv":
                contribution = mx.sum(selected * weight[..., None], axis=-3)
            else:  # guarded by the caller
                raise ValueError(f"unsupported grouped weighted mode {mode!r}")
            output_blocks[instruction.output_index] = (
                output_blocks[instruction.output_index] + contribution
            )
            start = stop

    if not output_blocks:
        return IrrepsArray(
            plan.irreps_out, mx.zeros((*leading_shape, 0), dtype=dtype)
        )
    array = mx.concatenate(
        [
            block.reshape(*leading_shape, part.dim)
            for block, part in zip(output_blocks, plan.irreps_out, strict=True)
        ],
        axis=-1,
    )
    return IrrepsArray(plan.irreps_out, array)


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
        if weights.dtype != left.dtype:
            raise TypeError(
                f"tensor product weights must match input dtype {left.dtype}, "
                f"got {weights.dtype}"
            )
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
    # A one-dimensional weight vector is shared across every leading input
    # dimension.  Keep it compact and let the contraction broadcast it
    # implicitly: materializing ``(*leading_shape, weight_numel)`` makes
    # reverse mode construct one weight-gradient row per item before reducing
    # it back to the shared parameter, which is especially expensive for dense
    # ``uvw`` products.  Non-empty leading dimensions identify batched
    # (unshared) weights and still need explicit broadcasting when compatible.
    if (
        weights is not None
        and weight_leading_shape
        and weight_leading_shape != leading_shape
    ):
        weights = mx.broadcast_to(
            weights,
            (*leading_shape, plan.weight_numel),
        )

    left_chunks = left.chunk_arrays()
    right_chunks = right.chunk_arrays()
    weighted_lookup = {
        instruction_index: weight_slice
        for instruction_index, weight_slice in zip(
            plan.weighted_instruction_indices, plan.weight_slices, strict=True
        )
    }
    if weights is not None and plan.instructions and all(
        instruction.mode in ("uvu", "uvv")
        and instruction.has_weight
        and index in weighted_lookup
        for index, instruction in enumerate(plan.instructions)
    ):
        return _grouped_weighted_tensor_product(
            plan,
            left_chunks,
            right_chunks,
            weights,
            leading_shape,
            weighted_lookup,
            left.array.dtype,
        )
    output_blocks = [mx.zeros((*leading_shape, part.mul, part.ir.dim), dtype=left.array.dtype) for part in plan.irreps_out]
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
        output_blocks[output_index] = output_blocks[output_index] + contribution
    if not output_blocks:
        return IrrepsArray(plan.irreps_out, mx.zeros((*leading_shape, 0), dtype=left.array.dtype))
    array = mx.concatenate(
        [block.reshape(*leading_shape, part.dim) for block, part in zip(output_blocks, plan.irreps_out, strict=True)],
        axis=-1,
    )
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
        use_custom_kernel: bool = True,
    ) -> None:
        super().__init__()
        mx, _ = require_mlx()
        original_irreps_in1 = Irreps(irreps_in1)
        original_irreps_in2 = Irreps(irreps_in2)
        original_irreps_out = Irreps(irreps_out)
        self.irreps_in1 = original_irreps_in1.remove_zero_multiplicities()
        self.irreps_in2 = original_irreps_in2.remove_zero_multiplicities()
        self.irreps_out = original_irreps_out.remove_zero_multiplicities()
        self.instructions = make_tensor_product_instructions(
            original_irreps_in1,
            original_irreps_in2,
            original_irreps_out,
            instructions,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )
        self.shared_weights = shared_weights
        self.use_custom_kernel = bool(use_custom_kernel)
        if internal_weights is None:
            self.internal_weights = any(ins[4] for ins in instructions) and shared_weights
        else:
            self.internal_weights = internal_weights
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
        self._metal_dummy_weight = mx.zeros((0,), dtype=mx.float32)
        self._metal_operation = None
        self._configure_metal()
        self._compiled = compile_or_identity(self._call_arrays, enabled=compile_left_right)

    @property
    def output_mask(self):
        return self._output_mask

    def __deepcopy__(self, memo):
        """Copy MLX module and Python state without copying runtime callables.

        ``mlx.nn.Module`` stores registered arrays/lists/dicts in the inherited
        dictionary and ordinary Python attributes in ``__dict__``; those stores
        are disjoint by construction. MLX compiled functions and custom
        functions are process-local runtime objects, so the copy explicitly
        resets them and initially uses the general array implementation.
        """

        duplicate = self.__class__.__new__(self.__class__)
        dict.__init__(duplicate)
        memo[id(self)] = duplicate
        for name, value in dict.items(self):
            dict.__setitem__(duplicate, name, copy.deepcopy(value, memo))
        for name, value in vars(self).items():
            if name in {"_metal_operation", "_compiled"}:
                continue
            object.__setattr__(duplicate, name, copy.deepcopy(value, memo))
        object.__setattr__(duplicate, "_metal_operation", None)
        object.__setattr__(duplicate, "_compiled", duplicate._call_arrays)
        return duplicate

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

    def _configure_metal(
        self,
        *,
        output_irreps: Irreps | None = None,
        output_maps: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if (
            not self.use_custom_kernel
            or not self.instructions
            or not mlx_metal_available()
        ):
            self._metal_operation = None
            return
        weighted_flags = {instruction.has_weight for instruction in self.instructions}
        if len(weighted_flags) != 1:
            self._metal_operation = None
            return
        plan = self._plan(self.weight_numel > 0)
        weighted_lookup = {
            index: weight_slice
            for index, weight_slice in zip(
                plan.weighted_instruction_indices,
                plan.weight_slices,
                strict=True,
            )
        }
        def general(first, second, value):
            return self._general_call_arrays(
                first,
                second,
                value if self.weight_numel > 0 else None,
            )
        if output_maps is None:
            channel_metadata = build_channel_metadata(
                plan.irreps_in1,
                plan.irreps_in2,
                output_irreps or plan.irreps_out,
                plan.instructions,
                weighted_lookup,
            )
            if channel_metadata is not None:
                self._metal_operation = make_channel_operation(
                    channel_metadata,
                    self.irreps_in1.dim,
                    self.irreps_in2.dim,
                    general,
                )
                self._metal_kernel_kind = "channel_uvu"
                return

        estimated_terms = 0
        for instruction in self.instructions:
            connection_count = prod(instruction.path_shape)
            coefficient_count = sum(
                value != 0.0
                for plane in _cg_array_data(
                    instruction.ir_in1,
                    instruction.ir_in2,
                    instruction.ir_out,
                )
                for row in plane
                for value in row
            )
            estimated_terms += connection_count * coefficient_count
        self._metal_path_count = estimated_terms
        # Direct scalar paths are ideal for sparse edge contractions, but dense
        # multiplicity mixing belongs in MLX's matrix kernels. Guard before
        # allocating padded metadata, which can otherwise be much larger than
        # the learned parameter tensor for high-multiplicity uvw products.
        if estimated_terms > _METAL_SCALAR_PATH_MAX_TERMS:
            self._metal_operation = None
            return
        metadata = build_metadata(
            plan.irreps_in1,
            plan.irreps_in2,
            output_irreps or plan.irreps_out,
            plan.instructions,
            weighted_lookup,
            output_maps=output_maps,
        )
        self._metal_operation = make_operation(
            metadata,
            self.irreps_in1.dim,
            self.irreps_in2.dim,
            general,
        )
        self._metal_kernel_kind = "scalar_paths"

    def _metal_dispatch_kind(
        self, left_array: Any, right_array: Any, weight: Any | None
    ) -> str | None:
        mx, _ = require_mlx()
        if (
            self._metal_operation is None
            or left_array.dtype != mx.float32
            or right_array.dtype != mx.float32
            or left_array.ndim != 2
            or right_array.ndim != 2
            or left_array.shape[0] != right_array.shape[0]
        ):
            return None
        # The scalar sparse dispatch removes Python/array intermediates at small
        # batch sizes, but sufficiently large dense products are faster as
        # batched MLX matrix contractions. Channel-local kernels remain faster
        # at large edge counts and are not subject to this crossover guard.
        if (
            getattr(self, "_metal_kernel_kind", None) == "scalar_paths"
            and (
                left_array.shape[0] > _METAL_SCALAR_PATH_MAX_BATCH
                or self._metal_path_count * left_array.shape[0]
                > _METAL_SCALAR_PATH_MAX_WORK
            )
        ):
            return None
        if self.weight_numel > 0:
            if weight is None or weight.dtype != mx.float32:
                return None
            if weight.ndim == 1:
                if not self.shared_weights:
                    return None
            elif weight.ndim == 2:
                if self.shared_weights or weight.shape[0] != left_array.shape[0]:
                    return None
            else:
                return None
        return getattr(self, "_metal_kernel_kind", None)

    def _try_metal(
        self, left_array: Any, right_array: Any, weight: Any | None
    ) -> Any | None:
        if self._metal_dispatch_kind(left_array, right_array, weight) is None:
            return None
        metal_weight = weight if self.weight_numel > 0 else self._metal_dummy_weight
        return self._metal_operation(left_array, right_array, metal_weight)

    def _general_call_arrays(
        self, left_array: Any, right_array: Any, weight: Any | None = None
    ) -> Any:
        left = IrrepsArray(self.irreps_in1, left_array)
        right = IrrepsArray(self.irreps_in2, right_array)
        return _numerical_tensor_product(self._plan(weight is not None), left, right, weights=weight).array

    def _call_arrays(self, left_array: Any, right_array: Any, weight: Any | None = None) -> Any:
        metal = self._try_metal(left_array, right_array, weight)
        if metal is not None:
            return metal
        return self._general_call_arrays(left_array, right_array, weight)

    def differentiable_arrays(
        self, left_array: Any, right_array: Any, weight: Any | None = None
    ) -> Any:
        """General MLX fallback for JVP and arbitrary transform nesting."""

        return self._general_call_arrays(left_array, right_array, weight)

    def _get_weight(self, weight: Any | None) -> Any | None:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights is False")
            return self.weight if self.weight_numel > 0 else None
        if isinstance(weight, (list, tuple)):
            weight = self._flatten_weight_list(weight)
        if self.shared_weights:
            if tuple(weight.shape) != (self.weight_numel,):
                raise ValueError(f"Expected shared weight shape {(self.weight_numel,)}, got {tuple(weight.shape)}")
        else:
            if weight.ndim < 2:
                raise ValueError(
                    "shared_weights=False requires weights with a batch dimension"
                )
            if weight.shape[-1] != self.weight_numel:
                raise ValueError(f"Expected unshared weight shape (..., {self.weight_numel}), got {tuple(weight.shape)}")
        return weight

    def _flatten_weight_list(self, weights: list[Any] | tuple[Any, ...]) -> Any:
        mx, _ = require_mlx()
        if len(weights) != len(self._weighted_instruction_meta):
            raise ValueError(
                f"expected {len(self._weighted_instruction_meta)} per-instruction weights, got {len(weights)}"
            )
        if not weights:
            return mx.zeros((0,))
        leading_shapes = []
        for value, meta in zip(weights, self._weighted_instruction_meta, strict=True):
            path_shape = meta.instruction.path_shape
            if self.shared_weights:
                if tuple(value.shape) != path_shape:
                    raise ValueError(f"expected per-instruction weight shape {path_shape}, got {tuple(value.shape)}")
                leading_shapes.append(())
            else:
                if value.ndim < len(path_shape) or tuple(value.shape[-len(path_shape) :]) != path_shape:
                    raise ValueError(
                        f"expected per-instruction weight shape (..., {', '.join(map(str, path_shape))}), got {tuple(value.shape)}"
                    )
                leading_shapes.append(tuple(value.shape[: -len(path_shape)]))
        leading_shape = mx.broadcast_shapes(*leading_shapes)
        flattened = []
        for value, meta in zip(weights, self._weighted_instruction_meta, strict=True):
            path_shape = meta.instruction.path_shape
            value = mx.broadcast_to(value, (*leading_shape, *path_shape))
            flattened.append(value.reshape(*leading_shape, prod(path_shape)))
        return mx.concatenate(flattened, axis=-1)

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
        weight_leading_shape = (
            tuple(int(size) for size in resolved_weight.shape[:-1])
            if resolved_weight is not None and not self.shared_weights
            else ()
        )
        try:
            leading_shape = mx.broadcast_shapes(
                right.leading_shape, weight_leading_shape
            )
        except ValueError as exc:
            raise ValueError(
                "TensorProduct.right leading shapes are not broadcastable: "
                f"{right.leading_shape}, {weight_leading_shape}"
            ) from exc

        # Evaluate every left basis vector as one additional batch dimension.
        # This produces (..., irreps_in1.dim, irreps_out.dim) directly and lets
        # MLX compile the complete right-operator construction as one graph,
        # instead of launching the tensor product once per input component.
        left_array = mx.broadcast_to(
            mx.eye(self.irreps_in1.dim, dtype=right.array.dtype),
            (*leading_shape, self.irreps_in1.dim, self.irreps_in1.dim),
        )
        right_array = mx.broadcast_to(
            right.array, (*leading_shape, self.irreps_in2.dim)
        )[..., None, :]
        if resolved_weight is not None and not self.shared_weights:
            resolved_weight = mx.broadcast_to(
                resolved_weight, (*leading_shape, self.weight_numel)
            )[..., None, :]
        return self._compiled(left_array, right_array, resolved_weight)

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
    if not irreps_out:
        return IrrepsArray(irreps_out, mx.zeros((*array.leading_shape, 0), dtype=array.dtype))
    chunks = array.chunk_arrays()
    regrouped_chunks = []
    for part, indices in zip(irreps_out, index_groups, strict=True):
        merged = [
            _reshape_chunk(chunks[index], array.irreps[index].mul, part.ir.dim)
            for index in indices
        ]
        regrouped = mx.concatenate(merged, axis=-2) if len(merged) > 1 else merged[0]
        regrouped_chunks.append(regrouped.reshape(*array.leading_shape, part.dim))
    return IrrepsArray.from_chunks(irreps_out, regrouped_chunks, backend=mx)


def _explicit_mul_repr(irreps: Irreps) -> str:
    return "+".join(f"{part.mul}x{part.ir}" for part in irreps)


def _block_offsets_for_kernel(irreps: Irreps) -> tuple[int, ...]:
    offsets = []
    cursor = 0
    for part in irreps:
        offsets.append(cursor)
        cursor += part.dim
    return tuple(offsets)


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
        use_custom_kernel: bool = True,
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
            use_custom_kernel=use_custom_kernel,
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
        use_custom_kernel: bool = True,
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
            use_custom_kernel=use_custom_kernel,
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
        use_custom_kernel: bool = True,
    ) -> None:
        left = Irreps(irreps_in1).simplify()
        right = Irreps(irreps_in2).simplify()
        self._full_instruction_groups = None
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
        self._execution_irreps_out = output
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
            use_custom_kernel=use_custom_kernel,
        )
        self._execution_irreps_out = self.irreps_out
        if filter_ir_out is None:
            grouped_instructions: dict[
                tuple[int, int], list[TensorProductInstruction]
            ] = {}
            for instruction in self.instructions:
                grouped_instructions.setdefault(
                    (instruction.input1_index, instruction.input2_index), []
                ).append(instruction)
            self._full_instruction_groups = tuple(
                (input1, input2, tuple(group))
                for (input1, input2), group in grouped_instructions.items()
            )
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
        execution_maps: dict[int, tuple[int, ...]] = {}
        final_offsets = _block_offsets_for_kernel(self.irreps_out)
        for final_index, execution_indices in enumerate(self._output_index_groups):
            multiplicity_cursor = 0
            final_part = self.irreps_out[final_index]
            for execution_index in execution_indices:
                execution_part = self._execution_irreps_out[execution_index]
                execution_maps[execution_index] = tuple(
                    final_offsets[final_index]
                    + (multiplicity_cursor + mul) * final_part.ir.dim
                    + component
                    for mul in range(execution_part.mul)
                    for component in range(final_part.ir.dim)
                )
                multiplicity_cursor += execution_part.mul
        if self._full_instruction_groups is not None:
            self._configure_metal(
                output_irreps=self.irreps_out,
                output_maps=tuple(
                    execution_maps[instruction.output_index]
                    for instruction in self.instructions
                ),
            )
        else:
            self._metal_operation = None

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

    def _general_call_arrays(self, left_array: Any, right_array: Any, weight: Any | None = None) -> Any:
        if self._full_instruction_groups is None:
            return super()._general_call_arrays(left_array, right_array, weight)
        if weight is not None:
            raise ValueError("FullTensorProduct does not accept weights")
        if left_array.dtype != right_array.dtype:
            raise TypeError(
                "tensor product inputs must have matching dtypes, got "
                f"{left_array.dtype} and {right_array.dtype}"
            )
        mx, _ = require_mlx()
        left_leading = tuple(int(size) for size in left_array.shape[:-1])
        right_leading = tuple(int(size) for size in right_array.shape[:-1])
        try:
            leading_shape = mx.broadcast_shapes(left_leading, right_leading)
        except ValueError as exc:
            raise ValueError(
                "tensor product leading shapes are not broadcastable: "
                f"{left_leading}, {right_leading}"
            ) from exc
        if left_leading != leading_shape:
            left_array = mx.broadcast_to(
                left_array, (*leading_shape, self.irreps_in1.dim)
            )
        if right_leading != leading_shape:
            right_array = mx.broadcast_to(
                right_array, (*leading_shape, self.irreps_in2.dim)
            )

        left_chunks = IrrepsArray(self.irreps_in1, left_array).chunk_arrays()
        right_chunks = IrrepsArray(self.irreps_in2, right_array).chunk_arrays()
        output_blocks: dict[Irrep, list[Any]] = {
            part.ir: [] for part in self.irreps_out
        }
        for input1, input2, instructions in self._full_instruction_groups:
            part1 = self.irreps_in1[input1]
            part2 = self.irreps_in2[input2]
            block1 = _reshape_chunk(left_chunks[input1], part1.mul, part1.ir.dim)
            block2 = _reshape_chunk(right_chunks[input2], part2.mul, part2.ir.dim)
            specifications = tuple(
                (
                    instruction.ir_in1,
                    instruction.ir_in2,
                    instruction.ir_out,
                    instruction.normalization.scale(),
                )
                for instruction in instructions
            )
            coefficients = mx.array(
                _combined_cg_array_data(specifications), dtype=left_array.dtype
            )
            outer = (
                block1[..., :, None, :, None]
                * block2[..., None, :, None, :]
            ).reshape(
                *leading_shape,
                part1.mul,
                part2.mul,
                part1.ir.dim * part2.ir.dim,
            )
            coupled = outer @ coefficients.reshape(
                part1.ir.dim * part2.ir.dim, -1
            )
            start = 0
            for instruction in instructions:
                stop = start + instruction.ir_out.dim
                output_blocks[instruction.ir_out].append(
                    coupled[..., start:stop].reshape(
                        *leading_shape,
                        part1.mul * part2.mul,
                        instruction.ir_out.dim,
                    )
                )
                start = stop

        grouped = []
        for part in self.irreps_out:
            blocks = output_blocks[part.ir]
            block = mx.concatenate(blocks, axis=-2) if len(blocks) > 1 else blocks[0]
            grouped.append(block.reshape(*leading_shape, part.dim))
        if not grouped:
            return mx.zeros((*leading_shape, 0), dtype=left_array.dtype)
        return mx.concatenate(grouped, axis=-1)

    def _call_arrays(self, left_array: Any, right_array: Any, weight: Any | None = None) -> Any:
        metal = self._try_metal(left_array, right_array, weight)
        if metal is not None:
            return metal
        return self._general_call_arrays(left_array, right_array, weight)

    def __call__(self, left: IrrepsArray, right: IrrepsArray, weight: Any | None = None) -> IrrepsArray:
        if left.irreps.simplify() != self.irreps_in1:
            raise ValueError(f"left input irreps {left.irreps} do not match {self.irreps_in1}")
        if right.irreps.simplify() != self.irreps_in2:
            raise ValueError(f"right input irreps {right.irreps} do not match {self.irreps_in2}")
        array = self._compiled(left.array, right.array, None)
        if self._full_instruction_groups is not None:
            return IrrepsArray(self.irreps_out, array)
        out = IrrepsArray(self._execution_irreps_out, array)
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
        use_custom_kernel: bool = True,
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
                use_custom_kernel=use_custom_kernel,
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
                use_custom_kernel=use_custom_kernel,
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
