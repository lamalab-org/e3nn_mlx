"""Dense transforms between spherical tensors and midpoint S2 grids."""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt
from typing import Any

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .ops_rotations import angles_to_xyz
from .ops_sh import spherical_harmonics


def _complete_lmax_res(lmax, res_beta, res_alpha):
    if res_beta is None:
        if lmax is not None:
            res_beta = 2 * (lmax + 1)
        elif res_alpha is not None:
            res_beta = 2 * ((res_alpha + 1) // 2)
        else:
            raise ValueError("lmax, res_beta, and res_alpha cannot all be None")
    if res_alpha is None:
        if lmax is not None:
            res_alpha = max(2 * lmax + 1, res_beta - 1)
        else:
            res_alpha = res_beta - 1
    if lmax is None:
        lmax = min(res_beta // 2 - 1, (res_alpha - 1) // 2)
    if res_beta % 2 != 0:
        raise ValueError("res_beta must be even")
    if lmax < 0 or lmax + 1 > res_beta // 2:
        raise ValueError("res_beta is too small for lmax")
    if 2 * lmax + 1 > res_alpha:
        raise ValueError("res_alpha is too small for lmax")
    return int(lmax), int(res_beta), int(res_alpha)


def _resolve(lmax, res):
    if isinstance(res, int) or res is None:
        return _complete_lmax_res(lmax, res, None)
    if len(res) != 2:
        raise ValueError("res must be an integer or (res_beta, res_alpha)")
    return _complete_lmax_res(lmax, res[0], res[1])


def s2_grid(res_beta: int, res_alpha: int, *, dtype=None):
    mx, _ = require_mlx()
    dtype = mx.float32 if dtype is None else dtype
    betas = (mx.arange(res_beta, dtype=dtype) + 0.5) * (pi / res_beta)
    alphas = mx.arange(res_alpha, dtype=dtype) * (2.0 * pi / res_alpha)
    return betas, alphas


def _degree_scales(lmax: int, normalization: str | Any, np):
    if isinstance(normalization, str):
        if normalization == "component":
            per_degree = [sqrt(4.0 * pi / (2 * l + 1)) / sqrt(lmax + 1) for l in range(lmax + 1)]
        elif normalization == "norm":
            per_degree = [sqrt(4.0 * pi) / sqrt(lmax + 1) for _ in range(lmax + 1)]
        elif normalization == "integral":
            per_degree = [1.0 for _ in range(lmax + 1)]
        else:
            raise ValueError("normalization must be 'component', 'norm', 'integral', or an array")
    else:
        per_degree = list(normalization.tolist())
        if len(per_degree) != lmax + 1:
            raise ValueError(f"normalization array must have length {lmax + 1}")
    return np.asarray([scale for l, scale in enumerate(per_degree) for _ in range(2 * l + 1)], dtype=np.float64)


def _quadrature_beta_weights(res_beta: int, np):
    bandwidth = res_beta // 2
    k = np.arange(bandwidth, dtype=np.float64)
    weights = []
    for j in range(res_beta):
        angle = pi * (2 * j + 1) / (4 * bandwidth)
        value = (2.0 / bandwidth) * np.sin(angle) * np.sum(np.sin((2 * j + 1) * (2 * k + 1) * pi / (4 * bandwidth)) / (2 * k + 1))
        weights.append(value)
    return np.asarray(weights, dtype=np.float64)


@lru_cache(maxsize=None)
def _transform_data(lmax: int, res_beta: int, res_alpha: int, normalization: str):
    import numpy as np

    mx, _ = require_mlx()
    betas, alphas = s2_grid(res_beta, res_alpha)
    grid = angles_to_xyz(alphas[None, :], betas[:, None])
    harmonics = spherical_harmonics(
        list(range(lmax + 1)), grid, normalize=False, normalization="integral"
    )
    scales = _degree_scales(lmax, normalization, np)
    synthesis = np.asarray(harmonics.tolist(), dtype=np.float64).reshape(res_beta * res_alpha, -1) * scales[None, :]
    beta_weights = _quadrature_beta_weights(res_beta, np)
    weights = np.repeat(beta_weights, res_alpha)
    weighted = synthesis * weights[:, None]
    gram = synthesis.T @ weighted
    analysis = np.linalg.solve(gram, weighted.T)
    return (
        synthesis.astype(np.float32),
        analysis.astype(np.float32),
        np.asarray(betas.tolist(), dtype=np.float32),
        np.asarray(alphas.tolist(), dtype=np.float32),
        np.asarray(grid.tolist(), dtype=np.float32),
    )


class ToS2Grid(mlx_module_base()):
    def __init__(self, lmax=None, res=None, normalization: str = "component") -> None:
        super().__init__()
        self.lmax, self.res_beta, self.res_alpha = _resolve(lmax, res)
        synthesis, _, betas, alphas, grid = _transform_data(
            self.lmax, self.res_beta, self.res_alpha, normalization
        )
        mx, _ = require_mlx()
        self._synthesis = mx.array(synthesis)
        self._betas = mx.array(betas)
        self._alphas = mx.array(alphas)
        self._grid = mx.array(grid)
        self.irreps_in = Irreps.spherical_harmonics(self.lmax)

    @property
    def betas(self):
        return self._betas

    @property
    def alphas(self):
        return self._alphas

    @property
    def grid(self):
        return self._grid

    def __repr__(self) -> str:
        return f"ToS2Grid(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    def __call__(self, coefficients):
        if coefficients.shape[-1] != (self.lmax + 1) ** 2:
            raise ValueError(f"expected {(self.lmax + 1) ** 2} spherical coefficients")
        output = coefficients @ self._synthesis.T
        return output.reshape(*coefficients.shape[:-1], self.res_beta, self.res_alpha)


class FromS2Grid(mlx_module_base()):
    def __init__(self, res=None, lmax=None, normalization: str = "component", lmax_in=None) -> None:
        super().__init__()
        self.lmax, self.res_beta, self.res_alpha = _resolve(lmax, res)
        if lmax_in is not None and lmax_in < self.lmax:
            raise ValueError("lmax_in must be at least lmax")
        _, analysis, betas, alphas, grid = _transform_data(
            self.lmax, self.res_beta, self.res_alpha, normalization
        )
        mx, _ = require_mlx()
        self._analysis = mx.array(analysis)
        self._betas = mx.array(betas)
        self._alphas = mx.array(alphas)
        self._grid = mx.array(grid)
        self.irreps_out = Irreps.spherical_harmonics(self.lmax)

    @property
    def betas(self):
        return self._betas

    @property
    def alphas(self):
        return self._alphas

    @property
    def grid(self):
        return self._grid

    def __repr__(self) -> str:
        return f"FromS2Grid(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    def __call__(self, grid_values):
        if tuple(grid_values.shape[-2:]) != (self.res_beta, self.res_alpha):
            raise ValueError(f"expected grid shape (..., {self.res_beta}, {self.res_alpha})")
        flattened = grid_values.reshape(*grid_values.shape[:-2], self.res_beta * self.res_alpha)
        return flattened @ self._analysis.T
