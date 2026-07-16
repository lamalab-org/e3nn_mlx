"""Pointwise nonlinear activation of band-limited signals on SO(3)."""

from __future__ import annotations

from typing import Any, Callable

from .compat import mlx_module_base
from .nn_activation import _activation_properties
from .ops_so3 import SO3Grid, so3_irreps


class SO3Activation(mlx_module_base()):
    r"""Map coefficients to SO(3), activate pointwise, and project back."""

    def __init__(
        self,
        lmax_in: int,
        lmax_out: int,
        act: Callable[[Any], Any],
        resolution: int,
        *,
        normalization: str = "component",
        aspect_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out
        self.irreps_in = so3_irreps(lmax_in)
        self.irreps_out = so3_irreps(lmax_out)
        self.grid_in = SO3Grid(
            lmax_in,
            resolution,
            normalization=normalization,
            aspect_ratio=aspect_ratio,
        )
        self.grid_out = (
            self.grid_in
            if lmax_in == lmax_out
            else SO3Grid(
                lmax_out,
                resolution,
                normalization=normalization,
                aspect_ratio=aspect_ratio,
            )
        )
        self.act = act
        self.activation_scale, _ = _activation_properties(act)

    def __repr__(self) -> str:
        return f"SO3Activation ({self.lmax_in} -> {self.lmax_out})"

    def __call__(self, features):
        grid = self.grid_in.to_grid(features)
        return self.grid_out.from_grid(self.activation_scale * self.act(grid))
