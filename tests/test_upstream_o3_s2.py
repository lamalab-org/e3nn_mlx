"""MLX adaptation of upstream e3nn S2 grid transform tests."""

from __future__ import annotations

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


_RES_ALPHA = [11, 12, 13, 14, 15, 16, None]
_RES_BETA = [12, 14, 16, None]
_LMAX = [0, 1, 5, None]


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
@pytest.mark.parametrize("res_alpha", _RES_ALPHA)
@pytest.mark.parametrize("res_beta", _RES_BETA)
@pytest.mark.parametrize("lmax", _LMAX)
def test_upstream_s2_grid_projection_is_idempotent(lmax, res_beta, res_alpha) -> None:
    if lmax is None and res_beta is None and res_alpha is None:
        return
    mx = mlx_backend._require()
    from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax)
    to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha))
    values = mx.random.normal(shape=(from_grid.res_beta, from_grid.res_alpha))
    projected = to_grid(from_grid(values))
    assert _max_abs(to_grid(from_grid(projected)) - projected) < 4e-4


@pytest.mark.mlx
@pytest.mark.parametrize("res_alpha", _RES_ALPHA)
@pytest.mark.parametrize("res_beta", _RES_BETA)
@pytest.mark.parametrize("lmax", _LMAX)
def test_upstream_s2_spherical_coefficients_round_trip(lmax, res_beta, res_alpha) -> None:
    if lmax is None and res_beta is None and res_alpha is None:
        return
    mx = mlx_backend._require()
    from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax)
    to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha))
    coefficients = mx.random.normal(shape=((from_grid.lmax + 1) ** 2,))
    assert _max_abs(from_grid(to_grid(coefficients)) - coefficients) < 4e-4


@pytest.mark.mlx
@pytest.mark.parametrize("res_alpha", [100, 101])
@pytest.mark.parametrize("res_beta", [98, 100])
@pytest.mark.parametrize("lmax", [1, 5])
def test_upstream_s2_nonlinear_projection_is_equivariant(lmax: int, res_beta: int, res_alpha: int) -> None:
    mx = mlx_backend._require()
    from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax)
    to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha))
    coefficients = mx.random.normal(shape=(3, (lmax + 1) ** 2))

    def function(values):
        return from_grid(mx.exp(to_grid(values)))

    angles = o3.rand_angles()
    representation = o3.irreps_wigner_d(o3.Irreps.spherical_harmonics(lmax), *angles)
    actual = function(coefficients @ mx.swapaxes(representation, -1, -2))
    expected = function(coefficients) @ mx.swapaxes(representation, -1, -2)
    assert _max_abs(actual - expected) < 2e-3


@pytest.mark.mlx
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_upstream_s2_grid_metadata_normalizations_and_compile(normalization: str) -> None:
    mx = mlx_backend._require()
    to_grid = o3.ToS2Grid(3, (12, 13), normalization=normalization)
    from_grid = o3.FromS2Grid((12, 13), 3, normalization=normalization)
    assert to_grid.grid.shape == from_grid.grid.shape == (12, 13, 3)
    assert to_grid.betas.shape == (12,)
    assert to_grid.alphas.shape == (13,)
    assert to_grid.parameters() == from_grid.parameters() == {}

    coefficients = mx.random.normal(shape=(2, 16))
    compiled_to = mx.compile(to_grid)
    compiled_from = mx.compile(from_grid)
    assert _max_abs(compiled_to(coefficients) - to_grid(coefficients)) < 2e-6
    assert _max_abs(compiled_from(to_grid(coefficients)) - coefficients) < 4e-4
