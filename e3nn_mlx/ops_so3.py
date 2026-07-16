"""Dense Fourier transforms between SO(3) coefficients and Euler grids."""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt

import numpy as np

from e3nn_core.irreps import Irrep, Irreps, MulIrrep
from e3nn_core.wigner import so3_generators

from .compat import mlx_module_base, require_mlx
from .ops_s2 import _quadrature_beta_weights


def so3_irreps(lmax: int) -> Irreps:
    """Regular SO(3) representation through degree ``lmax``."""

    if not isinstance(lmax, int) or lmax < 0:
        raise ValueError("lmax must be a non-negative integer")
    return Irreps([MulIrrep(2 * l + 1, Irrep(l, 1)) for l in range(lmax + 1)])


def _matrix_exp_generator(generator: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Evaluate the same fixed-work exponential used by the MLX Wigner code."""

    scaled = np.remainder(angles, 2.0 * pi)[..., None, None] * generator / 64.0
    result = np.broadcast_to(np.eye(generator.shape[-1]), scaled.shape).copy() + scaled
    term = scaled.copy()
    for order in range(2, 19):
        term = (term @ scaled) / float(order)
        result += term
    for _ in range(6):
        result = result @ result
    return result


@lru_cache(maxsize=None)
def _so3_grid_data(lmax: int, resolution: int, aspect_ratio: int):
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    res_beta = 2 * resolution
    res_alpha = round(2 * aspect_ratio * resolution)
    alpha = np.arange(res_alpha, dtype=np.float64) * (2.0 * pi / res_alpha)
    beta = (np.arange(res_beta, dtype=np.float64) + 0.5) * (pi / res_beta)

    blocks = []
    for l in range(lmax + 1):
        generators = np.asarray(so3_generators(l), dtype=np.float64)
        alpha_matrix = _matrix_exp_generator(generators[1], alpha)
        beta_matrix = _matrix_exp_generator(generators[0], beta)
        matrix = np.einsum(
            "aij,bjk,ckl->abcil",
            alpha_matrix,
            beta_matrix,
            alpha_matrix,
            optimize=True,
        )
        blocks.append(sqrt(2 * l + 1) * matrix.reshape(res_alpha, res_beta, res_alpha, -1))
    basis = np.concatenate(blocks, axis=-1)

    # Normalized Haar measure: d(alpha)/(2pi) sin(beta)d(beta)/2
    # d(gamma)/(2pi).  The beta rule integrates sin(beta)d(beta).
    beta_weights = _quadrature_beta_weights(res_beta, np)
    weights = np.broadcast_to(
        beta_weights[None, :, None] / (2.0 * res_alpha**2),
        (res_alpha, res_beta, res_alpha),
    )
    return (
        basis.astype(np.float32),
        weights.astype(np.float32),
        alpha.astype(np.float32),
        beta.astype(np.float32),
    )


class SO3Grid(mlx_module_base()):
    """Transform regular-representation coefficients to and from an SO(3) grid."""

    def __init__(
        self,
        lmax: int,
        resolution: int,
        *,
        normalization: str = "component",
        aspect_ratio: int = 2,
    ) -> None:
        super().__init__()
        if normalization != "component":
            raise ValueError("SO3Grid currently supports normalization='component' only")
        self.irreps = so3_irreps(lmax)
        self.lmax = lmax
        self.res_beta = 2 * resolution
        self.res_alpha = round(2 * aspect_ratio * resolution)
        self.res_gamma = self.res_alpha
        basis, weights, alpha, beta = _so3_grid_data(lmax, resolution, aspect_ratio)
        mx, _ = require_mlx()
        self._basis = mx.array(basis.reshape(-1, self.irreps.dim))
        self._weights = mx.array(weights.reshape(-1))
        self._alpha = mx.array(alpha)
        self._beta = mx.array(beta)
        self._gamma = self._alpha

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    def __repr__(self) -> str:
        return f"SO3Grid ({self.lmax})"

    def to_grid(self, features):
        if features.shape[-1] != self.irreps.dim:
            raise ValueError(f"expected input dimension {self.irreps.dim}")
        values = features @ self._basis.T / sqrt(self.irreps.dim)
        return values.reshape(
            *features.shape[:-1], self.res_alpha, self.res_beta, self.res_gamma
        )

    def from_grid(self, features):
        expected = (self.res_alpha, self.res_beta, self.res_gamma)
        if tuple(features.shape[-3:]) != expected:
            raise ValueError(f"expected grid shape (..., {expected[0]}, {expected[1]}, {expected[2]})")
        flattened = features.reshape(*features.shape[:-3], -1)
        return (flattened * self._weights) @ self._basis * sqrt(self.irreps.dim)
