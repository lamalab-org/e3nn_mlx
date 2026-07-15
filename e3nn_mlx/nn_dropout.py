"""Equivariant dropout shared across each irrep's components."""

from __future__ import annotations

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class Dropout(mlx_module_base()):
    def __init__(self, irreps: Irreps | str, p: float) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("dropout probability must be between zero and one")
        self.irreps = Irreps(irreps).remove_zero_multiplicities()
        self.irreps_in = self.irreps_out = self.irreps
        self.p = float(p)

    def __repr__(self) -> str:
        return f"Dropout ({self.irreps}, p={self.p})"

    def __call__(self, features: IrrepsArray) -> IrrepsArray:
        if features.irreps != self.irreps:
            raise ValueError("input irreps do not match Dropout.irreps")
        if not self.training or self.p <= 0:
            return features
        mx, _ = require_mlx()
        if self.p >= 1:
            return IrrepsArray(self.irreps, mx.zeros_like(features.array))
        if not features.leading_shape:
            raise ValueError("Dropout requires a batch dimension")
        batch = features.leading_shape[0]
        extra = (1,) * (len(features.leading_shape) - 1)
        noises = []
        for part in self.irreps:
            keep = mx.random.uniform(shape=(batch, part.mul)) >= self.p
            keep = keep.astype(features.dtype) / (1.0 - self.p)
            keep = keep.reshape(batch, *extra, part.mul, 1)
            noises.append(mx.broadcast_to(keep, (*features.leading_shape, part.mul, part.ir.dim)).reshape(*features.leading_shape, part.dim))
        noise = mx.concatenate(noises, axis=-1) if noises else mx.zeros_like(features.array)
        return IrrepsArray(self.irreps, features.array * noise)
