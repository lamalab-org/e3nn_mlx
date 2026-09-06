"""Symbolic tensor-product instruction generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from .irreps import Irrep, Irreps, MulIrrep
from .normalization import NormalizationMetadata
from .typing import TensorProductMode


@dataclass(frozen=True, slots=True)
class TensorProductInstruction:
    input1_index: int
    input2_index: int
    output_index: int
    ir_in1: Irrep
    ir_in2: Irrep
    ir_out: Irrep
    mode: TensorProductMode
    path_shape: tuple[int, ...]
    normalization: NormalizationMetadata
    has_weight: bool = False
    path_weight: float = 1.0

    @property
    def i_in1(self) -> int:
        return self.input1_index

    @property
    def i_in2(self) -> int:
        return self.input2_index

    @property
    def i_out(self) -> int:
        return self.output_index

    @property
    def signature(self) -> tuple[Irrep, Irrep, Irrep, TensorProductMode]:
        return (self.ir_in1, self.ir_in2, self.ir_out, self.mode)


def _resolve_output_parts(output_irreps: Irreps, ir_out: Irrep) -> tuple[tuple[int, MulIrrep], ...]:
    return tuple((index, part) for index, part in enumerate(output_irreps) if part.ir == ir_out and part.mul > 0)


def _infer_output_mul(mode: TensorProductMode, mul1: int, mul2: int, explicit_out_mul: int | None) -> int:
    if mode == "uvw":
        return explicit_out_mul if explicit_out_mul is not None else mul1 * mul2
    if mode == "uvu":
        return mul1
    if mode == "uvv":
        return mul2
    if mode == "uuw":
        return explicit_out_mul if explicit_out_mul is not None else 1
    if mode == "uuu":
        return mul1
    if mode == "uvuv":
        return mul1 * mul2
    if mode == "uvu<v":
        return mul1 * (mul1 - 1) // 2
    if mode == "u<vw":
        return explicit_out_mul if explicit_out_mul is not None else mul1 * (mul1 - 1) // 2
    raise ValueError(f"unsupported tensor product mode {mode!r}")


def _path_shape(mode: TensorProductMode, mul1: int, mul2: int, mul_out: int) -> tuple[int, ...]:
    if mode == "uvw":
        return (mul1, mul2, mul_out)
    if mode in ("uvu", "uvv"):
        return (mul1, mul2)
    if mode == "uuw":
        return (mul1, mul_out)
    if mode == "uuu":
        return (mul1,)
    if mode == "uvuv":
        return (mul1, mul2)
    if mode == "uvu<v":
        return (mul1 * (mul1 - 1) // 2,)
    if mode == "u<vw":
        return (mul1 * (mul1 - 1) // 2, mul_out)
    raise ValueError(f"unsupported tensor product mode {mode!r}")


def _num_elements(mode: TensorProductMode, mul1: int, mul2: int, _mul_out: int) -> int:
    if mode == "uvw":
        return mul1 * mul2
    if mode == "uvu":
        return mul2
    if mode == "uvv":
        return mul1
    if mode == "uuw":
        return mul1
    if mode == "uuu":
        return 1
    if mode == "uvuv":
        return 1
    if mode == "uvu<v":
        return 1
    if mode == "u<vw":
        return mul1 * (mul1 - 1) // 2
    raise ValueError(f"unsupported tensor product mode {mode!r}")


def _validate_mode(mode: TensorProductMode, mul1: int, mul2: int, mul_out: int) -> None:
    if mode == "uvu" and mul_out != mul1:
        raise ValueError("mode 'uvu' requires output multiplicity to equal input1 multiplicity")
    if mode == "uvv" and mul_out != mul2:
        raise ValueError("mode 'uvv' requires output multiplicity to equal input2 multiplicity")
    if mode == "uuw" and mul1 != mul2:
        raise ValueError("mode 'uuw' requires equal input multiplicities")
    if mode == "uuu" and not (mul1 == mul2 == mul_out):
        raise ValueError("mode 'uuu' requires equal input and output multiplicities")
    if mode == "uvuv" and mul_out != mul1 * mul2:
        raise ValueError("mode 'uvuv' requires output multiplicity to equal mul1 * mul2")
    if mode == "uvu<v":
        if mul1 != mul2:
            raise ValueError("mode 'uvu<v' requires equal input multiplicities")
        if mul_out != mul1 * (mul1 - 1) // 2:
            raise ValueError("mode 'uvu<v' requires output multiplicity n(n-1)/2")
    if mode == "u<vw":
        if mul1 != mul2:
            raise ValueError("mode 'u<vw' requires equal input multiplicities")


def generate_tensor_product_instructions(
    irreps_in1: Irreps | str,
    irreps_in2: Irreps | str,
    irreps_out: Irreps | str | None = None,
    *,
    mode: TensorProductMode = "uvw",
    irrep_normalization: str = "component",
    path_normalization: str = "element",
) -> tuple[TensorProductInstruction, ...]:
    left = Irreps(irreps_in1).remove_zero_multiplicities()
    right = Irreps(irreps_in2).remove_zero_multiplicities()
    output = None if irreps_out is None else Irreps(irreps_out).remove_zero_multiplicities()

    instructions: list[TensorProductInstruction] = []
    if path_normalization == "component":
        path_normalization = "element"

    for i_in1, part1 in enumerate(left):
        for i_in2, part2 in enumerate(right):
            candidates = part1.ir * part2.ir
            for ir_out in candidates:
                resolved_parts: tuple[tuple[int, MulIrrep] | None, ...]
                if output is None:
                    resolved_parts = (None,)
                else:
                    resolved_parts = _resolve_output_parts(output, ir_out)
                if not resolved_parts:
                    continue
                for resolved in resolved_parts:
                    out_index = len(instructions) if resolved is None else resolved[0]
                    explicit_out_mul = None if resolved is None else resolved[1].mul
                    out_mul = _infer_output_mul(mode, part1.mul, part2.mul, explicit_out_mul)
                    _validate_mode(mode, part1.mul, part2.mul, out_mul)
                    out_part = MulIrrep(out_mul, ir_out)
                    instructions.append(
                        TensorProductInstruction(
                            input1_index=i_in1,
                            input2_index=i_in2,
                            output_index=out_index,
                            ir_in1=part1.ir,
                            ir_in2=part2.ir,
                            ir_out=ir_out,
                            mode=mode,
                            path_shape=_path_shape(mode, part1.mul, part2.mul, out_part.mul),
                            normalization=NormalizationMetadata(
                                irrep_normalization=irrep_normalization,
                                path_normalization=path_normalization,
                                num_paths=part1.mul * part2.mul,
                                num_elements=_num_elements(mode, part1.mul, part2.mul, out_part.mul),
                            ),
                        ),
                    )

    path_counts = {instruction.output_index: 0 for instruction in instructions}
    path_elements = {instruction.output_index: 0 for instruction in instructions}
    for instruction in instructions:
        path_counts[instruction.output_index] += 1
        path_elements[instruction.output_index] += instruction.normalization.num_elements

    normalized: list[TensorProductInstruction] = []
    for instruction in instructions:
        if instruction.normalization.irrep_normalization == "component":
            alpha = instruction.ir_out.dim
        elif instruction.normalization.irrep_normalization == "norm":
            alpha = instruction.ir_in1.dim * instruction.ir_in2.dim
        elif instruction.normalization.irrep_normalization == "none":
            alpha = 1.0
        else:
            raise ValueError(f"unsupported irrep normalization {instruction.normalization.irrep_normalization!r}")

        if instruction.normalization.path_normalization == "element":
            divisor = path_elements[instruction.output_index]
        elif instruction.normalization.path_normalization == "path":
            divisor = instruction.normalization.num_elements * path_counts[instruction.output_index]
        elif instruction.normalization.path_normalization == "none":
            divisor = 1
        else:
            raise ValueError(f"unsupported path normalization {instruction.normalization.path_normalization!r}")
        # Upstream leaves alpha undivided when the divisor is zero rather than zeroing the path.
        coefficient = sqrt(alpha / divisor) if divisor > 0 else sqrt(alpha)
        normalized.append(
            TensorProductInstruction(
                input1_index=instruction.input1_index,
                input2_index=instruction.input2_index,
                output_index=instruction.output_index,
                ir_in1=instruction.ir_in1,
                ir_in2=instruction.ir_in2,
                ir_out=instruction.ir_out,
                mode=instruction.mode,
                path_shape=instruction.path_shape,
                normalization=NormalizationMetadata(
                    irrep_normalization=instruction.normalization.irrep_normalization,
                    path_normalization=instruction.normalization.path_normalization,
                    num_paths=path_counts[instruction.output_index],
                    num_elements=instruction.normalization.num_elements,
                    coefficient=coefficient,
                ),
                has_weight=instruction.has_weight,
                path_weight=instruction.path_weight,
            )
        )
    return tuple(normalized)


def make_tensor_product_instructions(
    irreps_in1: Irreps | str,
    irreps_in2: Irreps | str,
    irreps_out: Irreps | str,
    instructions: Iterable[tuple[int, int, int, TensorProductMode, bool] | tuple[int, int, int, TensorProductMode, bool, float]],
    *,
    irrep_normalization: str = "component",
    path_normalization: str = "element",
    in1_var: Iterable[float] | None = None,
    in2_var: Iterable[float] | None = None,
    out_var: Iterable[float] | None = None,
) -> tuple[TensorProductInstruction, ...]:
    original_left = Irreps(irreps_in1)
    original_right = Irreps(irreps_in2)
    original_output = Irreps(irreps_out)
    left = original_left.remove_zero_multiplicities()
    right = original_right.remove_zero_multiplicities()
    output = original_output.remove_zero_multiplicities()
    left_indices = {
        original: compact
        for compact, original in enumerate(
            index for index, part in enumerate(original_left) if part.mul > 0
        )
    }
    right_indices = {
        original: compact
        for compact, original in enumerate(
            index for index, part in enumerate(original_right) if part.mul > 0
        )
    }
    output_indices = {
        original: compact
        for compact, original in enumerate(
            index for index, part in enumerate(original_output) if part.mul > 0
        )
    }
    raw: list[TensorProductInstruction] = []
    path_weights: list[float] = []

    if path_normalization == "component":
        path_normalization = "element"

    def compact_variances(values, original, compact, name):
        if values is None:
            return [1.0 for _ in compact]
        parsed = [float(value) for value in values]
        if len(parsed) == len(original):
            return [
                value
                for value, part in zip(parsed, original, strict=True)
                if part.mul > 0
            ]
        if len(parsed) == len(compact):
            return parsed
        raise ValueError(
            f"len({name}) must match the irreps before or after removing "
            "zero multiplicities"
        )

    in1_var_list = compact_variances(in1_var, original_left, left, "in1_var")
    in2_var_list = compact_variances(in2_var, original_right, right, "in2_var")
    out_var_list = compact_variances(out_var, original_output, output, "out_var")

    for instruction in instructions:
        if len(instruction) == 5:
            i_in1, i_in2, i_out, mode, has_weight = instruction
            path_weight = 1.0
        else:
            i_in1, i_in2, i_out, mode, has_weight, path_weight = instruction
        if not isinstance(i_in1, int) or not isinstance(i_in2, int) or not isinstance(i_out, int):
            raise TypeError("tensor-product instruction indices must be integers")
        if not isinstance(has_weight, bool):
            raise TypeError("tensor-product instruction weight flag must be bool")
        if mode == "uvw" and not has_weight:
            raise ValueError("mode 'uvw' requires weights")
        if not isinstance(path_weight, (int, float)):
            raise TypeError("tensor-product path_weight must be numeric")
        if path_weight < 0:
            raise ValueError("tensor-product path_weight must be non-negative")
        if (
            not 0 <= i_in1 < len(original_left)
            or not 0 <= i_in2 < len(original_right)
            or not 0 <= i_out < len(original_output)
        ):
            raise IndexError(f"tensor-product instruction index out of range: {instruction!r}")
        if (
            original_left[i_in1].mul == 0
            or original_right[i_in2].mul == 0
            or original_output[i_out].mul == 0
        ):
            raise ValueError(
                "tensor-product instructions cannot reference zero-multiplicity irreps"
            )
        i_in1 = left_indices[i_in1]
        i_in2 = right_indices[i_in2]
        i_out = output_indices[i_out]
        part1 = left[i_in1]
        part2 = right[i_in2]
        part_out = output[i_out]
        if part1.ir.p * part2.ir.p != part_out.ir.p:
            raise ValueError("parity mismatch in explicit tensor-product instruction")
        if part_out.ir not in (part1.ir * part2.ir):
            raise ValueError("output irrep is not allowed by the selection rule")
        _validate_mode(mode, part1.mul, part2.mul, part_out.mul)
        raw.append(
            TensorProductInstruction(
                input1_index=i_in1,
                input2_index=i_in2,
                output_index=i_out,
                ir_in1=part1.ir,
                ir_in2=part2.ir,
                ir_out=part_out.ir,
                mode=mode,
                path_shape=_path_shape(mode, part1.mul, part2.mul, part_out.mul),
                normalization=NormalizationMetadata(
                    irrep_normalization=irrep_normalization,
                    path_normalization=path_normalization,
                    num_elements=_num_elements(mode, part1.mul, part2.mul, part_out.mul),
                ),
                has_weight=has_weight,
                path_weight=float(path_weight),
            )
        )
        path_weights.append(path_weight)

    path_counts = {instruction.output_index: 0 for instruction in raw}
    path_elements = {instruction.output_index: 0 for instruction in raw}
    for instruction in raw:
        path_counts[instruction.output_index] += 1
        path_elements[instruction.output_index] += instruction.normalization.num_elements

    out: list[TensorProductInstruction] = []
    for instruction, path_weight in zip(raw, path_weights, strict=True):
        if instruction.normalization.irrep_normalization == "component":
            alpha = instruction.ir_out.dim
        elif instruction.normalization.irrep_normalization == "norm":
            alpha = instruction.ir_in1.dim * instruction.ir_in2.dim
        elif instruction.normalization.irrep_normalization == "none":
            alpha = 1.0
        else:
            raise ValueError(f"unsupported irrep normalization {instruction.normalization.irrep_normalization!r}")

        if instruction.normalization.path_normalization == "element":
            divisor = sum(
                in1_var_list[other.input1_index] * in2_var_list[other.input2_index] * other.normalization.num_elements
                for other in raw
                if other.output_index == instruction.output_index
            )
        elif instruction.normalization.path_normalization == "path":
            divisor = (
                in1_var_list[instruction.input1_index]
                * in2_var_list[instruction.input2_index]
                * instruction.normalization.num_elements
                * path_counts[instruction.output_index]
            )
        elif instruction.normalization.path_normalization == "none":
            divisor = 1
        else:
            raise ValueError(f"unsupported path normalization {instruction.normalization.path_normalization!r}")

        alpha *= out_var_list[instruction.output_index]
        # Upstream leaves alpha undivided when the divisor is zero rather than zeroing the path.
        coefficient = sqrt(alpha * path_weight / divisor) if divisor > 0 else sqrt(alpha * path_weight)
        out.append(
            TensorProductInstruction(
                input1_index=instruction.input1_index,
                input2_index=instruction.input2_index,
                output_index=instruction.output_index,
                ir_in1=instruction.ir_in1,
                ir_in2=instruction.ir_in2,
                ir_out=instruction.ir_out,
                mode=instruction.mode,
                path_shape=instruction.path_shape,
                normalization=NormalizationMetadata(
                    irrep_normalization=instruction.normalization.irrep_normalization,
                    path_normalization=instruction.normalization.path_normalization,
                    num_paths=path_counts[instruction.output_index],
                    num_elements=instruction.normalization.num_elements,
                    coefficient=coefficient,
                ),
                has_weight=instruction.has_weight,
                path_weight=instruction.path_weight,
            )
        )
    return tuple(out)
