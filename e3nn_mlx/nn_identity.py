"""Equivariant identity module."""

from __future__ import annotations

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .irreps_array import IrrepsArray


class Identity(mlx_module_base()):
    def __init__(self, irreps_in: Irreps | str, irreps_out: Irreps | str) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in).simplify()
        self.irreps_out = Irreps(irreps_out).simplify()
        if self.irreps_in != self.irreps_out:
            raise ValueError("Identity requires equal input and output irreps")
        mx, _ = require_mlx()
        self._output_mask = mx.ones((self.irreps_out.dim,))

    @property
    def output_mask(self):
        return self._output_mask

    def __repr__(self) -> str:
        return f"Identity({self.irreps_in} -> {self.irreps_out})"

    def __call__(self, array: IrrepsArray) -> IrrepsArray:
        if array.irreps.simplify() != self.irreps_in:
            raise ValueError("input irreps do not match Identity.irreps_in")
        return IrrepsArray(self.irreps_out, array.array)
