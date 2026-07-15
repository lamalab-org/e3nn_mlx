"""MLX Norm module."""

from __future__ import annotations

from .irreps_array import IrrepsArray
from .compat import mlx_module_base
from .ops_reduce import norm


class Norm(mlx_module_base()):
    def __init__(self, *, per_irrep: bool = True, squared: bool = False) -> None:
        super().__init__()
        self.per_irrep = per_irrep
        self.squared = squared

    def __call__(self, array: IrrepsArray) -> IrrepsArray:
        return norm(array, per_irrep=self.per_irrep, squared=self.squared)
