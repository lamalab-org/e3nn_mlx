"""Representation-aware feature extraction."""

from __future__ import annotations

from typing import Sequence

from e3nn_core.irreps import Irrep, Irreps

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class Extract(mlx_module_base()):
    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_outs: Sequence[Irreps | str],
        instructions: Sequence[Sequence[int]],
        squeeze_out: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        self.irreps_outs = tuple(Irreps(value).remove_zero_multiplicities() for value in irreps_outs)
        self.instructions = tuple(tuple(int(index) for index in instruction) for instruction in instructions)
        self.squeeze_out = bool(squeeze_out)
        if len(self.irreps_outs) != len(self.instructions):
            raise ValueError("one extraction instruction is required per output")
        for output, instruction in zip(self.irreps_outs, self.instructions, strict=True):
            if len(output) != len(instruction):
                raise ValueError("each extraction instruction must have one input index per output block")
            for output_part, input_index in zip(output, instruction, strict=True):
                if input_index < 0 or input_index >= len(self.irreps_in):
                    raise IndexError(f"input block index {input_index} is out of range")
                if output_part != self.irreps_in[input_index]:
                    raise ValueError("output block does not match the selected input block")

    def __call__(self, features: IrrepsArray):
        if features.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Extract.irreps_in")
        mx, _ = require_mlx()
        chunks = features.chunk_arrays()
        outputs = []
        for irreps_out, instruction in zip(self.irreps_outs, self.instructions, strict=True):
            selected = [chunks[index] for index in instruction]
            array = mx.concatenate(selected, axis=-1) if selected else mx.zeros((*features.leading_shape, 0), dtype=features.dtype)
            outputs.append(IrrepsArray(irreps_out, array))
        if self.squeeze_out and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class ExtractIr(Extract):
    def __init__(self, irreps_in: Irreps | str, ir: Irrep | str) -> None:
        irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        target = Irrep.parse(ir)
        selected = tuple(index for index, part in enumerate(irreps_in) if part.ir == target)
        self.irreps_out = Irreps(irreps_in[index] for index in selected)
        super().__init__(irreps_in, [self.irreps_out], [selected], squeeze_out=True)
