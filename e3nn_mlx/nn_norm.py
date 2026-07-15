"""MLX Norm module."""

from __future__ import annotations

from e3nn_core.irreps import Irrep, Irreps, MulIrrep

from .irreps_array import IrrepsArray
from .compat import mlx_module_base
from .ops_reduce import norm


class Norm(mlx_module_base()):
    def __init__(self, irreps_in: Irreps | str | None = None, *, per_irrep: bool = True, squared: bool = False) -> None:
        super().__init__()
        self.irreps_in = None if irreps_in is None else Irreps(irreps_in).remove_zero_multiplicities()
        self.per_irrep = per_irrep
        self.squared = squared
        if self.irreps_in is None:
            self.irreps_out = None
        elif per_irrep:
            self.irreps_out = Irreps(MulIrrep(part.mul, Irrep(0, 1)) for part in self.irreps_in)
        else:
            self.irreps_out = Irreps("0e")

    def __call__(self, array: IrrepsArray) -> IrrepsArray:
        if self.irreps_in is not None and array.irreps != self.irreps_in:
            raise ValueError("input irreps do not match Norm.irreps_in")
        return norm(array, per_irrep=self.per_irrep, squared=self.squared)
