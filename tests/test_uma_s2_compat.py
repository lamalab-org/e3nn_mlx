"""S2-grid surface required by eSCN/UMA-style models.

UMA drives ``ToS2Grid``/``FromS2Grid`` with a longitude grid that deliberately
resolves fewer Fourier modes than ``lmax`` implies, because it only ever uses
coefficients with ``|m| <= mmax``. These tests pin that contract.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from e3nn_mlx import o3
from e3nn_mlx.backend import mlx_backend


def uma_coefficient_indices(lmax: int, mmax: int) -> list[int]:
    """Flattened indices of the ``|m| <= mmax`` subspace UMA actually uses."""

    indices = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if abs(m) <= mmax:
                indices.append(l * l + (m + l))
    return indices


def uma_grid_resolution(lmax: int, mmax: int) -> tuple[int, int]:
    res_beta = 2 * (lmax + 1)
    res_alpha = 2 * (mmax + 1) + 1 if lmax == mmax else 2 * mmax + 1
    return res_beta, res_alpha


def _max_abs(array) -> float:
    return float(np.abs(np.asarray(array)).max())


def test_uma_coefficient_indices_count() -> None:
    assert len(uma_coefficient_indices(6, 2)) == 29
    assert len(uma_coefficient_indices(2, 2)) == 9


@pytest.mark.mlx
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (4, 2), (6, 2)])
def test_uma_grid_shapes_and_truncated_roundtrip(lmax: int, mmax: int) -> None:
    mx = mlx_backend._require()
    res_beta, res_alpha = uma_grid_resolution(lmax, mmax)

    to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization="integral")
    from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax, normalization="integral")

    num_coefficients = (lmax + 1) ** 2
    num_m_modes = 2 * lmax + 1
    for transform in (to_grid, from_grid):
        assert transform.sha.shape == (res_alpha, num_m_modes)
        assert transform.shb.shape == (num_m_modes, res_beta, num_coefficients)

    # The exact contractions eSCN/UMA performs to build its grid matrices.
    to_matrix = mx.einsum("mbi,am->bai", to_grid.shb, to_grid.sha)
    from_matrix = mx.einsum("am,mbi->bai", from_grid.sha, from_grid.shb)
    assert to_matrix.shape == (res_beta, res_alpha, num_coefficients)
    assert from_matrix.shape == (res_beta, res_alpha, num_coefficients)

    # Only the |m| <= mmax subspace round-trips. res_alpha samples resolve
    # res_alpha independent Fourier modes, so the full basis cannot be
    # recovered, and is not expected to be.
    indices = mx.array(uma_coefficient_indices(lmax, mmax))
    restricted_to = mx.take(to_matrix, indices, axis=2)
    restricted_from = mx.take(from_matrix, indices, axis=2)

    samples, channels = 3, 4
    mx.random.seed(0)
    x = mx.random.normal((samples, len(uma_coefficient_indices(lmax, mmax)), channels))
    grid = mx.einsum("bai,zic->zbac", restricted_to, x)
    recovered = mx.einsum("bai,zbac->zic", restricted_from, grid)
    assert _max_abs(recovered - x) < 5e-4


@pytest.mark.mlx
def test_k16l6_undersampled_alpha_grid_is_accepted() -> None:
    """lmax=6 on five longitude samples: the configuration UMA K16L6 needs.

    This asserts the res_alpha >= 2*lmax+1 restriction stays removed.
    """

    to_grid = o3.ToS2Grid(6, (14, 5), normalization="integral")
    from_grid = o3.FromS2Grid((14, 5), 6, normalization="integral")

    assert to_grid.sha.shape == (5, 13)
    assert to_grid.shb.shape == (13, 14, 49)
    assert from_grid.sha.shape == (5, 13)
    assert from_grid.shb.shape == (13, 14, 49)
    assert to_grid.lmax == from_grid.lmax == 6
    assert to_grid.res_beta == from_grid.res_beta == 14
    assert to_grid.res_alpha == from_grid.res_alpha == 5


@pytest.mark.mlx
def test_s2_grid_transforms_expose_no_parameters() -> None:
    to_grid = o3.ToS2Grid(6, (14, 5), normalization="integral")
    from_grid = o3.FromS2Grid((14, 5), 6, normalization="integral")
    assert to_grid.parameters() == {}
    assert from_grid.parameters() == {}


@pytest.mark.mlx
@pytest.mark.parametrize("res", [(14, 13), (14, 5)])
def test_s2_grid_compiles(res) -> None:
    mx = mlx_backend._require()
    to_grid = o3.ToS2Grid(6, res, normalization="integral")
    from_grid = o3.FromS2Grid(res, 6, normalization="integral")

    mx.random.seed(1)
    coefficients = mx.random.normal((2, 49))
    eager = to_grid(coefficients)
    compiled = mx.compile(to_grid)(coefficients)
    assert _max_abs(compiled - eager) < 1e-5

    # The compiled forward must agree with the dense sha/shb construction.
    dense = mx.einsum("mbi,am->bai", to_grid.shb, to_grid.sha)
    expected = mx.einsum("bai,zi->zba", dense, coefficients)
    assert _max_abs(compiled - expected) < 1e-4

    grid_values = to_grid(coefficients)
    assert _max_abs(mx.compile(from_grid)(grid_values) - from_grid(grid_values)) < 1e-5


@pytest.mark.mlx
def test_spherical_harmonics_alpha_matches_closed_form() -> None:
    mx = mlx_backend._require()
    alpha = mx.array([0.0, 0.3, 1.0])

    expected = mx.stack(
        [
            sqrt(2) * mx.sin(2 * alpha),
            sqrt(2) * mx.sin(alpha),
            mx.ones_like(alpha),
            sqrt(2) * mx.cos(alpha),
            sqrt(2) * mx.cos(2 * alpha),
        ],
        axis=-1,
    )
    assert _max_abs(o3.spherical_harmonics_alpha(2, alpha) - expected) < 1e-6

    assert o3.spherical_harmonics_alpha(0, alpha).shape == (3, 1)
    assert o3.spherical_harmonics_alpha(1, alpha).shape == (3, 3)
    assert o3.spherical_harmonics_alpha(6, alpha).shape == (3, 13)
    with pytest.raises(ValueError):
        o3.spherical_harmonics_alpha(-1, alpha)


@pytest.mark.mlx
def test_uma_elementwise_scalar_mask() -> None:
    """UMA's equivariant dropout: one scalar gate per irrep instance."""

    mx = mlx_backend._require()
    irreps = o3.Irreps("2x0e + 3x1o + 1x2e")
    mask_irreps = o3.Irreps(f"{irreps.num_irreps}x0e")
    product = o3.ElementwiseTensorProduct(irreps, mask_irreps)

    mx.random.seed(2)
    x = mx.random.normal((8, irreps.dim))
    mask = mx.ones((8, irreps.num_irreps))
    y = product(x, mask)

    assert y.shape == x.shape
    assert _max_abs(y - x) < 1e-6
    assert irreps.num_irreps == 6
    assert o3.Irrep("0e").is_scalar() and not o3.Irrep("1o").is_scalar()


REFERENCE = (
    __import__("pathlib").Path(__file__).resolve().parent
    / "reference_data"
    / "uma_s2_reference.npz"
)


def _relative(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-30))


@pytest.mark.mlx
@pytest.mark.e3nn_reference
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (4, 2), (6, 2)])
def test_uma_grids_match_upstream_e3nn_fixtures(lmax: int, mmax: int) -> None:
    """dense_to and dense_from are the matrices UMA consumes: they are the gate."""

    mx = mlx_backend._require()
    reference = np.load(REFERENCE)
    res_beta, res_alpha = uma_grid_resolution(lmax, mmax)

    to_grid = o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization="integral")
    from_grid = o3.FromS2Grid((res_beta, res_alpha), lmax, normalization="integral")
    key = f"{lmax}_{mmax}"

    dense_to = mx.einsum("mbi,am->bai", to_grid.shb, to_grid.sha)
    dense_from = mx.einsum("am,mbi->bai", from_grid.sha, from_grid.shb)

    # float32 evaluation of the harmonics sets the floor here.
    assert _relative(dense_to, reference[f"{key}_dense_to"]) < 1e-5
    assert _relative(dense_from, reference[f"{key}_dense_from"]) < 1e-5
    assert _relative(to_grid.sha, reference[f"{key}_to_sha"]) < 1e-5
    assert _relative(to_grid.shb, reference[f"{key}_to_shb"]) < 1e-5
    assert _relative(from_grid.sha, reference[f"{key}_from_sha"]) < 1e-5
    assert _relative(from_grid.shb, reference[f"{key}_from_shb"]) < 1e-5


# (identifier, kind, kwargs) mirroring NORMALIZATION_CASES in the generator.
NORMALIZATION_CASES = [
    ("to_l3_component", "to", {"lmax": 3, "normalization": "component"}),
    ("to_l3_norm", "to", {"lmax": 3, "normalization": "norm"}),
    ("to_l2_integral", "to", {"lmax": 2, "normalization": "integral"}),
    ("from_l3_component", "from", {"lmax": 3, "normalization": "component"}),
    ("from_l2_in3_component", "from", {"lmax": 2, "lmax_in": 3, "normalization": "component"}),
    ("from_l2_in3_norm", "from", {"lmax": 2, "lmax_in": 3, "normalization": "norm"}),
    ("from_l3_in5_component", "from", {"lmax": 3, "lmax_in": 5, "normalization": "component"}),
]


@pytest.mark.mlx
@pytest.mark.e3nn_reference
@pytest.mark.parametrize("identifier,kind,kwargs", NORMALIZATION_CASES)
def test_normalization_factors_match_upstream_absolutely(identifier, kind, kwargs) -> None:
    """Pin synthesis and analysis scaling against upstream, not just round-trip.

    A round-trip cannot catch this class of bug: an error in the forward scale
    that is mirrored in the inverse cancels exactly. These cases compare the
    matrices themselves, and cover lmax_in != lmax, which drives the bandwidth
    correction in _from_s2_factors.
    """

    mx = mlx_backend._require()
    reference = np.load(REFERENCE)

    if kind == "to":
        grid = o3.ToS2Grid(**kwargs)
        dense = mx.einsum("mbi,am->bai", grid.shb, grid.sha)
    else:
        grid = o3.FromS2Grid(**kwargs)
        dense = mx.einsum("am,mbi->bai", grid.sha, grid.shb)

    assert grid.res_beta == reference[f"{identifier}_shb"].shape[1]
    assert _relative(grid.sha, reference[f"{identifier}_sha"]) < 1e-5
    assert _relative(grid.shb, reference[f"{identifier}_shb"]) < 1e-5
    assert _relative(dense, reference[f"{identifier}_dense"]) < 1e-5


@pytest.mark.mlx
@pytest.mark.e3nn_reference
def test_lmax_in_changes_the_inverse_scaling() -> None:
    """lmax_in must actually alter component/norm analysis, and not integral.

    Guards against a bandwidth correction that is silently dropped: without it
    these two would be identical and the parity test above would be the only
    thing standing between us and a regression.
    """

    baseline = o3.FromS2Grid(lmax=2, lmax_in=2, normalization="component")
    widened = o3.FromS2Grid(lmax=2, lmax_in=3, normalization="component")
    ratio = float(np.sqrt(4.0 / 3.0))
    assert _relative(widened.shb, np.asarray(baseline.shb, dtype=np.float64) * ratio) < 1e-5

    # integral carries no 1/sqrt(lmax+1) factor, so lmax_in must not matter.
    integral_baseline = o3.FromS2Grid(lmax=2, lmax_in=2, normalization="integral")
    integral_widened = o3.FromS2Grid(lmax=2, lmax_in=3, normalization="integral")
    assert _relative(integral_widened.shb, integral_baseline.shb) < 1e-6
