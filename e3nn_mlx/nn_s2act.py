"""Pointwise nonlinear activation of band-limited signals on the sphere."""

from __future__ import annotations

from typing import Any, Callable

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .compat import mlx_module_base, require_mlx
from .nn_activation import _activation_properties
from .ops_rotations import irreps_wigner_d, rand_angles
from .ops_s2 import FromS2Grid, ToS2Grid


class S2Activation(mlx_module_base()):
    r"""Apply a normalized pointwise nonlinearity to a signal on :math:`S^2`."""

    def __init__(
        self,
        irreps: Irreps | str,
        act: Callable[[Any], Any],
        res: int | tuple[int, int],
        normalization: str = "component",
        lmax_out: int | None = None,
        random_rot: bool = False,
    ) -> None:
        super().__init__()
        parsed = Irreps(irreps).simplify()
        if not parsed:
            raise ValueError("irreps must contain at least one degree")
        if any(part.mul != 1 for part in parsed):
            raise ValueError("S2Activation requires multiplicity one for every degree")
        lmax = parsed[-1].ir.l
        if [part.ir.l for part in parsed] != list(range(lmax + 1)):
            raise ValueError("S2Activation requires consecutive degrees from zero through lmax")

        p_val = parsed[0].ir.p
        if all(part.ir.p == p_val for part in parsed):
            p_arg = 1
        elif all(part.ir.p == p_val * (-1) ** part.ir.l for part in parsed):
            p_arg = -1
        else:
            raise ValueError("the parity of the input spherical signal is not well defined")

        scale, _ = _activation_properties(act)
        mx, _ = require_mlx()
        probe = mx.linspace(0.0, 10.0, 256)
        positive, negative = act(probe), act(-probe)
        magnitude = float(mx.max(mx.abs(positive)))
        even_error = float(mx.max(mx.abs(positive - negative)))
        odd_error = float(mx.max(mx.abs(positive + negative)))
        activation_parity = (
            1
            if even_error < magnitude * 1e-10
            else (-1 if odd_error < magnitude * 1e-10 else 0)
        )
        if lmax_out is None:
            lmax_out = lmax
        if lmax_out < 0:
            raise ValueError("lmax_out must be non-negative")
        if p_val == 1:
            output_value_parity = 1
        elif activation_parity in (-1, 1):
            output_value_parity = activation_parity
        else:
            raise ValueError("an odd-valued spherical signal requires an even or odd activation")

        self.irreps_in = parsed
        self.irreps_out = Irreps(
            [MulIrrep(1, Irrep(l, output_value_parity * p_arg**l)) for l in range(lmax_out + 1)]
        )
        self.to_s2 = ToS2Grid(lmax, res, normalization=normalization)
        self.from_s2 = FromS2Grid(
            res, lmax_out, normalization=normalization, lmax_in=lmax
        )
        self.act = act
        self.activation_scale = scale
        self.random_rot = bool(random_rot)

    def __repr__(self) -> str:
        return f"S2Activation ({self.irreps_in} -> {self.irreps_out})"

    def __call__(self, features):
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(f"expected input dimension {self.irreps_in.dim}")
        mx, _ = require_mlx()
        angles = None
        if self.random_rot:
            angles = rand_angles(dtype=features.dtype)
            rotation = irreps_wigner_d(self.irreps_in, *angles)
            features = features @ mx.swapaxes(rotation, -1, -2)

        grid = self.to_s2(features)
        output = self.from_s2(self.activation_scale * self.act(grid))

        if angles is not None:
            rotation = irreps_wigner_d(self.irreps_out, *angles)
            output = output @ rotation
        return output
