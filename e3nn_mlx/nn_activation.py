"""Parity-aware normalized scalar activations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Sequence

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


@dataclass(frozen=True, slots=True)
class _ActivationPath:
    activation: Callable[[Any], Any] | None
    scale: float


def _activation_properties(activation):
    import numpy as np

    mx, _ = require_mlx()
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    samples = mx.array(sqrt(2.0) * nodes, dtype=mx.float32)
    values = activation(samples)
    second_moment = float(mx.sum(mx.array(weights, dtype=mx.float32) * values * values) / sqrt(np.pi))
    if not second_moment > 0:
        raise ValueError("activation must have a positive finite second moment")
    probe = mx.linspace(0.0, 10.0, 257)
    positive, negative = activation(probe), activation(-probe)
    even_error = float(mx.max(mx.abs(positive - negative)))
    odd_error = float(mx.max(mx.abs(positive + negative)))
    parity = 1 if even_error < 1e-5 else (-1 if odd_error < 1e-5 else 0)
    return second_moment**-0.5, parity


class Activation(mlx_module_base()):
    def __init__(self, irreps_in: Irreps | str, acts: Sequence[Callable[[Any], Any] | None]) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in).remove_zero_multiplicities()
        if len(self.irreps_in) != len(acts):
            raise ValueError(
                f"irreps and activation counts differ: {len(self.irreps_in)} irreps blocks, {len(acts)} activations"
            )
        paths = []
        outputs = []
        for part, activation in zip(self.irreps_in, acts, strict=True):
            if activation is None:
                paths.append(_ActivationPath(None, 1.0))
                outputs.append(part)
                continue
            if part.ir.l != 0:
                raise ValueError("activation functions can only be applied to scalar irreps")
            scale, activation_parity = _activation_properties(activation)
            output_parity = part.ir.p if part.ir.p == 1 else activation_parity
            if output_parity == 0:
                raise ValueError("an odd scalar input requires an even or odd activation")
            paths.append(_ActivationPath(activation, scale))
            outputs.append(MulIrrep(part.mul, Irrep(0, output_parity)))
        self._paths = tuple(paths)
        self.irreps_out = Irreps(outputs)

    def __repr__(self) -> str:
        active = "".join("x" if path.activation is not None else " " for path in self._paths)
        return f"Activation [{active}] ({self.irreps_in} -> {self.irreps_out})"

    def __call__(self, features: IrrepsArray) -> IrrepsArray:
        if features.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Activation.irreps_in")
        mx, _ = require_mlx()
        outputs = []
        for chunk, path in zip(features.chunk_arrays(), self._paths, strict=True):
            outputs.append(chunk if path.activation is None else path.scale * path.activation(chunk))
        output = mx.concatenate(outputs, axis=-1) if outputs else mx.zeros_like(features.array)
        return IrrepsArray(self.irreps_out, output)
