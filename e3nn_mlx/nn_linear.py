"""MLX Linear module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
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
    path_weight: float

    @property
    def i_in(self) -> int:
        return self.input_index

    @property
    def i_out(self) -> int:
        return self.output_index

    @property
    def path_shape(self) -> tuple[int, int]:
        return self.in_mul, self.out_mul


class Linear(mlx_module_base()):
    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        *,
        bias: bool | Sequence[bool] = True,
        biases: bool | Sequence[bool] | None = None,
        instructions: Sequence[tuple[int, int]] | None = None,
        internal_weights: bool = True,
        shared_weights: bool = True,
        path_normalization: str = "element",
        compile: bool = False,
    ) -> None:
        super().__init__()
        mx, _ = require_mlx()
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        self.irreps_out = Irreps(irreps_out).remove_zero_multiplicities()
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)
        if self.internal_weights and not self.shared_weights:
            raise ValueError("internal_weights=True requires shared_weights=True")
        if path_normalization not in ("element", "path"):
            raise ValueError("path_normalization must be 'element' or 'path'")

        if instructions is None:
            path_indices = [
                (input_index, output_index)
                for output_index, out_part in enumerate(self.irreps_out)
                for input_index, in_part in enumerate(self.irreps_in)
                if in_part.ir == out_part.ir
            ]
        else:
            path_indices = []
            for input_index, output_index in instructions:
                if input_index < 0 or input_index >= len(self.irreps_in):
                    raise IndexError(f"input instruction index {input_index} is out of range")
                if output_index < 0 or output_index >= len(self.irreps_out):
                    raise IndexError(f"output instruction index {output_index} is out of range")
                if self.irreps_in[input_index].ir != self.irreps_out[output_index].ir:
                    raise ValueError("Linear instructions can only connect identical irreps")
                path_indices.append((input_index, output_index))

        normalization_denominators = []
        for input_index, output_index in path_indices:
            denominator = sum(
                self.irreps_in[candidate_input if path_normalization == "element" else input_index].mul
                for candidate_input, candidate_output in path_indices
                if candidate_output == output_index
            )
            normalization_denominators.append(max(1, denominator))

        normalized_instructions = []
        start = 0
        for (input_index, output_index), denominator in zip(path_indices, normalization_denominators, strict=True):
            in_part = self.irreps_in[input_index]
            out_part = self.irreps_out[output_index]
            size = out_part.mul * in_part.mul
            normalized_instructions.append(
                _LinearInstruction(
                    input_index=input_index,
                    output_index=output_index,
                    in_mul=in_part.mul,
                    out_mul=out_part.mul,
                    dim=out_part.ir.dim,
                    weight_slice=slice(start, start + size),
                    path_weight=denominator**-0.5,
                )
            )
            start += size
        self.instructions = tuple(normalized_instructions)
        self.weight_numel = start

        weight_chunks = []
        for instruction in self.instructions:
            weight_chunks.append(
                mx.random.normal(shape=(instruction.weight_slice.stop - instruction.weight_slice.start,))
            )
        initial_weight = mx.concatenate(weight_chunks) if weight_chunks else None
        self.weight = initial_weight if self.internal_weights else None

        requested_biases = bias if biases is None else biases
        if isinstance(requested_biases, bool):
            bias_flags = tuple(requested_biases and part.ir.is_scalar() for part in self.irreps_out)
        else:
            bias_flags = tuple(bool(value) for value in requested_biases)
            if len(bias_flags) != len(self.irreps_out):
                raise ValueError(f"expected {len(self.irreps_out)} bias flags, got {len(bias_flags)}")
            for flag, part in zip(bias_flags, self.irreps_out, strict=True):
                if flag and not part.ir.is_scalar():
                    raise ValueError("biases are only allowed for invariant even scalar outputs")
        self._bias_output_indices = tuple(index for index, flag in enumerate(bias_flags) if flag)
        bias_numel = sum(self.irreps_out[index].mul for index in self._bias_output_indices)
        parameter_dtype = initial_weight.dtype if initial_weight is not None else mx.float32
        self.bias = mx.zeros((bias_numel,), dtype=parameter_dtype) if bias_numel else None

        active_outputs = {instruction.output_index for instruction in self.instructions}
        mask_chunks = [
            mx.full((part.dim,), index in active_outputs or index in self._bias_output_indices, dtype=mx.bool_)
            for index, part in enumerate(self.irreps_out)
        ]
        self._output_mask = mx.concatenate(mask_chunks) if mask_chunks else mx.zeros((0,), dtype=mx.bool_)
        self._compiled_impl = compile_or_identity(self._apply_arrays, enabled=compile)

    @property
    def output_mask(self):
        return self._output_mask

    def _get_weight(self, weight: Any | None) -> Any | None:
        if self.weight_numel == 0:
            if weight is not None and (weight.ndim < 1 or weight.shape[-1] != 0):
                raise ValueError("this Linear has no weighted instructions")
            return None
        resolved = self.weight if weight is None else weight
        if resolved is None:
            raise RuntimeError("Weights must be provided when internal_weights is False")
        if self.shared_weights:
            if tuple(resolved.shape) != (self.weight_numel,):
                raise ValueError(f"expected shared weight shape {(self.weight_numel,)}, got {tuple(resolved.shape)}")
        elif resolved.ndim < 1 or resolved.shape[-1] != self.weight_numel:
            raise ValueError(f"expected unshared weight shape (..., {self.weight_numel}), got {tuple(resolved.shape)}")
        return resolved

    def weight_view_for_instruction(self, instruction_index: int, weight: Any | None = None):
        try:
            instruction = self.instructions[instruction_index]
        except IndexError as exc:
            raise IndexError(f"instruction index {instruction_index} is out of range") from exc
        resolved = self._get_weight(weight)
        if resolved is None:
            raise ValueError(f"instruction {instruction_index} has no weights")
        return resolved[..., instruction.weight_slice].reshape(
            *resolved.shape[:-1], instruction.in_mul, instruction.out_mul
        )

    def weight_views(self, weight: Any | None = None, *, yield_instruction: bool = False):
        for index, instruction in enumerate(self.instructions):
            view = self.weight_view_for_instruction(index, weight)
            yield (index, instruction, view) if yield_instruction else view

    def _apply_arrays(self, array: Any, weight: Any | None, bias: Any | None) -> Any:
        mx, _ = require_mlx()
        input_array = IrrepsArray(self.irreps_in, array)
        input_chunks = input_array.chunk_arrays()
        outputs = [
            mx.zeros((*input_array.leading_shape, part.mul, part.ir.dim), dtype=array.dtype)
            for part in self.irreps_out
        ]
        bias_cursor = 0
        for output_index, out_part in enumerate(self.irreps_out):
            for instruction in self.instructions:
                if instruction.output_index != output_index:
                    continue
                chunk = input_chunks[instruction.input_index].reshape(
                    *input_array.leading_shape, instruction.in_mul, instruction.dim
                )
                matrix = weight[..., instruction.weight_slice].reshape(
                    *weight.shape[:-1], instruction.in_mul, instruction.out_mul
                ).astype(array.dtype)
                outputs[output_index] = outputs[output_index] + instruction.path_weight * mx.einsum(
                    "...io,...id->...od", matrix, chunk
                )
            if bias is not None and output_index in self._bias_output_indices:
                width = out_part.mul
                bias_shape = (*((1,) * len(input_array.leading_shape)), width, 1)
                outputs[output_index] = outputs[output_index] + bias[bias_cursor : bias_cursor + width].reshape(*bias_shape)
                bias_cursor += width
        if not outputs:
            return mx.zeros((*input_array.leading_shape, 0), dtype=array.dtype)
        return mx.concatenate(
            [
                out.reshape(*input_array.leading_shape, part.dim)
                for out, part in zip(outputs, self.irreps_out, strict=True)
            ],
            axis=-1,
        )

    def __call__(
        self,
        array: IrrepsArray,
        weight: Any | None = None,
        *,
        bias: Any | None = None,
        extension: str | None = None,
    ) -> IrrepsArray:
        if array.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Linear.irreps_in")
        if extension is not None and (kernel := get_extension(extension)) is not None:
            return kernel.fn(self, array, weight=weight, bias=bias)
        weight = self._get_weight(weight)
        bias = self.bias if bias is None else bias
        out = self._compiled_impl(array.array, weight, bias)
        return IrrepsArray(self.irreps_out, out)
