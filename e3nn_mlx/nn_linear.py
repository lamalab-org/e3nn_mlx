"""MLX Linear module."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from e3nn_core.irreps import Irreps

from .compat import require_mlx
from .compat import mlx_module_base
from .irreps_array import IrrepsArray
from .ops_basic import compile_or_identity, get_extension


@dataclass(frozen=True, slots=True)
class _LinearInstruction:
    input_index: int
    output_index: int
    in_mul: int
    out_mul: int
    dim: int
    weight_slice: slice


class Linear(mlx_module_base()):
    def __init__(self, irreps_in: Irreps | str, irreps_out: Irreps | str, *, bias: bool = True, compile: bool = False) -> None:
        super().__init__()
        mx, _ = require_mlx()
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        self.irreps_out = Irreps(irreps_out).remove_zero_multiplicities()
        instructions = []
        start = 0
        for output_index, out_part in enumerate(self.irreps_out):
            for input_index, in_part in enumerate(self.irreps_in):
                if in_part.ir == out_part.ir:
                    size = out_part.mul * in_part.mul
                    instructions.append(
                        _LinearInstruction(
                            input_index=input_index,
                            output_index=output_index,
                            in_mul=in_part.mul,
                            out_mul=out_part.mul,
                            dim=out_part.ir.dim,
                            weight_slice=slice(start, start + size),
                        )
                    )
                    start += size
        self.instructions = tuple(instructions)
        weight_chunks = []
        for instruction in instructions:
            fan_in = sum(part.mul for part in self.irreps_in if part.ir == self.irreps_out[instruction.output_index].ir)
            weight_chunks.append(
                mx.random.normal(shape=(instruction.weight_slice.stop - instruction.weight_slice.start,))
                / sqrt(max(1, fan_in))
            )
        self.weight = mx.concatenate(weight_chunks) if weight_chunks else None
        parameter_dtype = self.weight.dtype if self.weight is not None else mx.float32
        scalar_bias = sum(part.mul for part in self.irreps_out if part.ir.is_scalar()) if bias else 0
        self.bias = mx.zeros((scalar_bias,), dtype=parameter_dtype) if scalar_bias else None
        self._compiled_impl = compile_or_identity(self._apply_arrays, enabled=compile)

    def _apply_arrays(self, array: Any, weight: Any, bias: Any | None) -> Any:
        mx, _ = require_mlx()
        input_array = IrrepsArray(self.irreps_in, array)
        input_chunks = input_array.chunk_arrays()
        outputs = [mx.zeros((*input_array.leading_shape, part.mul, part.ir.dim), dtype=array.dtype) for part in self.irreps_out]
        bias_cursor = 0
        for output_index, out_part in enumerate(self.irreps_out):
            for instruction in self.instructions:
                if instruction.output_index != output_index:
                    continue
                chunk = input_chunks[instruction.input_index].reshape(*input_array.leading_shape, instruction.in_mul, instruction.dim)
                matrix = weight[instruction.weight_slice].reshape(instruction.out_mul, instruction.in_mul).astype(array.dtype)
                outputs[output_index] = outputs[output_index] + mx.einsum("oi,...id->...od", matrix, chunk)
            if bias is not None and out_part.ir.is_scalar():
                width = out_part.mul
                bias_shape = (*((1,) * len(input_array.leading_shape)), width, 1)
                outputs[output_index] = outputs[output_index] + bias[bias_cursor : bias_cursor + width].reshape(*bias_shape)
                bias_cursor += width
        return mx.concatenate([out.reshape(*input_array.leading_shape, -1) for out in outputs], axis=-1)

    def __call__(self, array: IrrepsArray, *, weight: Any | None = None, bias: Any | None = None, extension: str | None = None) -> IrrepsArray:
        if array.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Linear.irreps_in")
        if extension is not None and (kernel := get_extension(extension)) is not None:
            return kernel.fn(self, array, weight=weight, bias=bias)
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        out = self._compiled_impl(array.array, weight, bias)
        return IrrepsArray(self.irreps_out, out)
