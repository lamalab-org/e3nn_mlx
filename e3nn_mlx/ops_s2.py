"""Dense transforms between spherical tensors and midpoint S2 grids."""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt
from typing import Any

from e3nn_core.irreps import Irreps

from .compat import mlx_module_base, require_mlx
from .ops_rotations import angles_to_xyz
from .ops_sh import spherical_harmonics, spherical_harmonics_alpha


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
    if res_beta <= 0:
        raise ValueError("res_beta must be positive")
    if res_alpha <= 0:
        raise ValueError("res_alpha must be positive")
    if res_beta % 2 != 0:
        raise ValueError("res_beta must be even")
    if lmax < 0 or lmax + 1 > res_beta // 2:
        raise ValueError("res_beta is too small for lmax")
    # No res_alpha >= 2*lmax+1 check. A caller that supplies lmax explicitly may
    # intend an m-truncated grid: eSCN/UMA runs lmax=6 on five longitude samples
    # because it only ever uses coefficients with |m| <= mmax. Such a grid cannot
    # invert the full basis, which is the caller's contract to keep, not ours.
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


def _per_degree_scales(lmax: int, normalization: str | Any) -> list[float]:
    """Forward (synthesis) scale applied to every coefficient of each degree."""

    if isinstance(normalization, str):
        if normalization == "component":
            return [sqrt(4.0 * pi / (2 * l + 1)) / sqrt(lmax + 1) for l in range(lmax + 1)]
        if normalization == "norm":
            return [sqrt(4.0 * pi) / sqrt(lmax + 1) for _ in range(lmax + 1)]
        if normalization == "integral":
            return [1.0 for _ in range(lmax + 1)]
        raise ValueError("normalization must be 'component', 'norm', 'integral', or an array")
    per_degree = list(normalization.tolist())
    if len(per_degree) != lmax + 1:
        raise ValueError(f"normalization array must have length {lmax + 1}")
    return per_degree


def _to_degree_scale(l: int, lmax: int, normalization: str | Any) -> float:
    return _per_degree_scales(lmax, normalization)[l]


def _degree_scales(lmax: int, normalization: str | Any, np):
    per_degree = _per_degree_scales(lmax, normalization)
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
def _factorized_base_data(lmax: int, res_beta: int, res_alpha: int):
    """Split the spherical harmonics into a longitude and a latitude factor.

    Returns ``(betas, alphas, grid, sha, base_shb)`` where ``sha`` has shape
    ``(res_alpha, 2*lmax+1)`` and ``base_shb`` has shape
    ``(2*lmax+1, res_beta, (lmax+1)**2)``, both integral-normalized and before
    any ToS2Grid/FromS2Grid scaling.

    The beta factor is recovered by projecting the canonical spherical
    harmonics onto the alpha basis, evaluated on a *fully sampled* reference
    longitude grid of ``2*lmax+1`` points. That reference is independent of the
    requested ``res_alpha``: on a deliberately undersampled grid the alpha modes
    alias onto each other, and projecting there would mix them arbitrarily.
    """

    import numpy as np

    mx, _ = require_mlx()
    betas, alphas = s2_grid(res_beta, res_alpha)
    grid = angles_to_xyz(alphas[None, :], betas[:, None])

    reference_res_alpha = 2 * lmax + 1
    reference_alphas = mx.arange(reference_res_alpha, dtype=betas.dtype) * (
        2.0 * pi / reference_res_alpha
    )
    reference_grid = angles_to_xyz(reference_alphas[None, :], betas[:, None])
    harmonics = spherical_harmonics(
        list(range(lmax + 1)),
        reference_grid,
        normalize=False,
        normalization="integral",
        use_custom_kernel=False,
    )
    harmonics_np = np.asarray(harmonics.tolist(), dtype=np.float64)
    sha_reference = np.asarray(
        spherical_harmonics_alpha(lmax, reference_alphas).tolist(), dtype=np.float64
    )

    base_shb = np.zeros((2 * lmax + 1, res_beta, (lmax + 1) ** 2), dtype=np.float64)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            coefficient_index = l * l + (m + l)
            m_row = lmax + m
            alpha_basis = sha_reference[:, m_row]
            denominator = np.dot(alpha_basis, alpha_basis)
            values = harmonics_np[:, :, coefficient_index]
            # Each coefficient belongs to exactly one m, so only that row is
            # populated. The sparsity is the point: it keeps a truncated grid
            # from folding one m into another.
            base_shb[m_row, :, coefficient_index] = (values @ alpha_basis) / denominator

    sha = np.asarray(spherical_harmonics_alpha(lmax, alphas).tolist(), dtype=np.float64)
    return (
        np.asarray(betas.tolist(), dtype=np.float64),
        np.asarray(alphas.tolist(), dtype=np.float64),
        np.asarray(grid.tolist(), dtype=np.float64),
        sha,
        base_shb,
    )


def _to_s2_factors(lmax: int, res_beta: int, res_alpha: int, normalization):
    """``(sha, shb, synthesis)`` for ToS2Grid."""

    import numpy as np

    betas, alphas, grid, sha, base_shb = _factorized_base_data(lmax, res_beta, res_alpha)
    shb = base_shb.copy()
    for l in range(lmax + 1):
        scale = _to_degree_scale(l, lmax, normalization)
        for m in range(-l, l + 1):
            shb[:, :, l * l + (m + l)] *= scale
    dense = np.einsum("mbi,am->bai", shb, sha)
    synthesis = dense.reshape(res_beta * res_alpha, (lmax + 1) ** 2)
    return betas, alphas, grid, sha, shb, synthesis


def _from_s2_factors(lmax: int, res_beta: int, res_alpha: int, normalization, lmax_in: int):
    """``(sha, shb, analysis)`` for FromS2Grid, built by quadrature.

    A global inverse is unusable on an undersampled longitude grid, so the
    projection integrates against the quadrature weights instead. On a fully
    sampled grid the quadrature is exact and reproduces the inverse.
    """

    import numpy as np

    betas, alphas, grid, sha, base_shb = _factorized_base_data(lmax, res_beta, res_alpha)
    integration_weights = _quadrature_beta_weights(res_beta, np) * (2.0 * pi / res_alpha)
    # ``component`` and ``norm`` synthesis scale every degree by
    # 1/sqrt(lmax+1), so projection must undo the scaling of the *signal's*
    # bandwidth, which need not equal this transform's lmax. That correction is
    # uniform across degrees; the per-degree part still keys off l and lmax.
    # S2Activation drives lmax_in below lmax, so a per-degree lookup at lmax_in
    # would index past the end of the list.
    bandwidth_correction = 1.0
    if isinstance(normalization, str) and normalization in ("component", "norm"):
        bandwidth_correction = sqrt((lmax_in + 1) / (lmax + 1))
    shb = np.zeros_like(base_shb)
    for l in range(lmax + 1):
        inverse_scale = bandwidth_correction / _to_degree_scale(l, lmax, normalization)
        for m in range(-l, l + 1):
            coefficient_index = l * l + (m + l)
            m_row = lmax + m
            shb[m_row, :, coefficient_index] = (
                base_shb[m_row, :, coefficient_index] * integration_weights * inverse_scale
            )
    dense = np.einsum("am,mbi->bai", sha, shb)
    analysis = dense.reshape(res_beta * res_alpha, (lmax + 1) ** 2).T
    return betas, alphas, grid, sha, shb, analysis


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
        import numpy as np

        self.lmax, self.res_beta, self.res_alpha = _resolve(lmax, res)
        betas, alphas, grid, sha, shb, synthesis = _to_s2_factors(
            self.lmax, self.res_beta, self.res_alpha, normalization
        )
        mx, _ = require_mlx()
        # Underscored so MLX keeps them out of parameters(): these are constants.
        self._synthesis = mx.array(synthesis.astype(np.float32))
        self._sha = mx.array(sha.astype(np.float32))
        self._shb = mx.array(shb.astype(np.float32))
        self._betas = mx.array(betas.astype(np.float32))
        self._alphas = mx.array(alphas.astype(np.float32))
        self._grid = mx.array(grid.astype(np.float32))
        self.irreps_in = Irreps.spherical_harmonics(self.lmax)

    @property
    def sha(self):
        """Longitude factor, shape ``(res_alpha, 2*lmax+1)``."""

        return self._sha

    @property
    def shb(self):
        """Latitude factor, shape ``(2*lmax+1, res_beta, (lmax+1)**2)``."""

        return self._shb

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
        if lmax_in is None:
            lmax_in = self.lmax
        if lmax_in < 0:
            raise ValueError("lmax_in must be non-negative")
        import numpy as np

        betas, alphas, grid, sha, shb, analysis = _from_s2_factors(
            self.lmax, self.res_beta, self.res_alpha, normalization, int(lmax_in)
        )
        mx, _ = require_mlx()
        # Underscored so MLX keeps them out of parameters(): these are constants.
        self._analysis = mx.array(analysis.astype(np.float32))
        self._sha = mx.array(sha.astype(np.float32))
        self._shb = mx.array(shb.astype(np.float32))
        self._betas = mx.array(betas.astype(np.float32))
        self._alphas = mx.array(alphas.astype(np.float32))
        self._grid = mx.array(grid.astype(np.float32))
        self.lmax_in = int(lmax_in)
        self.irreps_out = Irreps.spherical_harmonics(self.lmax)

    @property
    def sha(self):
        """Longitude factor, shape ``(res_alpha, 2*lmax+1)``."""

        return self._sha

    @property
    def shb(self):
        """Latitude factor, shape ``(2*lmax+1, res_beta, (lmax+1)**2)``."""

        return self._shb

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
