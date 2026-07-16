"""Norm-based equivariant activation."""

from __future__ import annotations

from typing import Any, Callable

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class NormActivation(mlx_module_base()):
    def __init__(
        self,
        irreps_in: Irreps | str,
        scalar_nonlinearity: Callable[[Any], Any],
        normalize: bool = True,
        epsilon: float | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        self.irreps_out = self.irreps_in
        if epsilon is None and normalize:
            epsilon = 1e-8
        elif epsilon is not None and not normalize:
            raise ValueError("epsilon cannot be used when normalize is False")
        if epsilon is not None and epsilon <= 0:
            raise ValueError("epsilon must be strictly positive")
        self.epsilon = epsilon
        self.scalar_nonlinearity = scalar_nonlinearity
        self.normalize = bool(normalize)
        self.use_bias = bool(bias)
        mx, _ = require_mlx()
        self.biases = mx.zeros((self.irreps_in.num_irreps,)) if bias else None

    def __call__(self, features: IrrepsArray) -> IrrepsArray:
        if features.irreps != self.irreps_in:
            raise ValueError("input irreps do not match NormActivation.irreps_in")
        mx, _ = require_mlx()
        outputs = []
        bias_cursor = 0
        for part, chunk in zip(self.irreps_in, features.chunk_arrays(), strict=True):
            field = chunk.reshape(*features.leading_shape, part.mul, part.ir.dim)
            squared = mx.sum(field * field, axis=-1)
            if self.epsilon is None:
                norms = mx.sqrt(squared)
            else:
                norms = mx.sqrt(mx.maximum(squared, self.epsilon * self.epsilon))
            argument = norms
            if self.biases is not None:
                shape = (*((1,) * len(features.leading_shape)), part.mul)
                argument = argument + self.biases[bias_cursor : bias_cursor + part.mul].reshape(*shape)
            scaling = self.scalar_nonlinearity(argument)
            if self.normalize:
                scaling = scaling / norms
            outputs.append((field * scaling[..., None]).reshape(*features.leading_shape, part.dim))
            bias_cursor += part.mul
        output = mx.concatenate(outputs, axis=-1) if outputs else mx.zeros_like(features.array)
        return IrrepsArray(self.irreps_out, output)
