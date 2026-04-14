"""Symbolic tensor-product instruction generation."""

from __future__ import annotations

from dataclasses import dataclass

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
    path_shape: tuple[int, int, int]
    normalization: NormalizationMetadata

    @property
    def signature(self) -> tuple[Irrep, Irrep, Irrep, TensorProductMode]:
        return (self.ir_in1, self.ir_in2, self.ir_out, self.mode)


def _resolve_output_part(output_irreps: Irreps, ir_out: Irrep) -> tuple[int, MulIrrep] | None:
    for index, part in enumerate(output_irreps):
        if part.ir == ir_out and part.mul > 0:
            return index, part
    return None


def generate_tensor_product_instructions(
    irreps_in1: Irreps | str,
    irreps_in2: Irreps | str,
    irreps_out: Irreps | str | None = None,
    *,
    mode: TensorProductMode = "uvw",
    irrep_normalization: str = "component",
    path_normalization: str = "component",
) -> tuple[TensorProductInstruction, ...]:
    left = Irreps(irreps_in1).simplify()
    right = Irreps(irreps_in2).simplify()
    output = None if irreps_out is None else Irreps(irreps_out).simplify()

    instructions: list[TensorProductInstruction] = []
    for i_in1, part1 in enumerate(left):
        for i_in2, part2 in enumerate(right):
            candidates = part1.ir * part2.ir
            for ir_out in candidates:
                resolved = (None if output is None else _resolve_output_part(output, ir_out))
                if output is not None and resolved is None:
                    continue
                out_index = len(instructions) if resolved is None else resolved[0]
                out_part = MulIrrep(1, ir_out) if resolved is None else resolved[1]
                instructions.append(
                    TensorProductInstruction(
                        input1_index=i_in1,
                        input2_index=i_in2,
                        output_index=out_index,
                        ir_in1=part1.ir,
                        ir_in2=part2.ir,
                        ir_out=ir_out,
                        mode=mode,
                        path_shape=(part1.mul, part2.mul, out_part.mul),
                        normalization=NormalizationMetadata(
                            irrep_normalization=irrep_normalization,
                            path_normalization=path_normalization,
                            num_paths=part1.mul * part2.mul,
                        ),
                    )
                )
    return tuple(instructions)
