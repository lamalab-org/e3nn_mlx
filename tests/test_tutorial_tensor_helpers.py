from __future__ import annotations

import pytest

from e3nn_mlx.backend import mlx_backend
from tutorials import CartesianTensor, SphericalTensor


pytestmark = pytest.mark.mlx


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


def test_spherical_tensor_plot_and_geometry_helpers() -> None:
    mx = mlx_backend._require()
    coefficients = mx.zeros((16,)).at[6].add(1.0)
    tensor = SphericalTensor(coefficients)
    surface, values = tensor.plot(relu=False, radius=True, res=20)

    assert tensor.lmax == 3
    assert tensor.Rs == [(1, 0, 1), (1, 1, -1), (1, 2, 1), (1, 3, -1)]
    assert surface.shape == (20, 21, 3)
    assert values.shape == (20, 21)
    assert bool(mx.all(mx.isfinite(surface)))
    assert _max_abs(surface[:, 0] - surface[:, -1]) < 2e-6

    sphere, _ = tensor.plot(relu=False, radius=False, res=20)
    assert _max_abs(sphere[:, 0] - sphere[:, -1]) < 2e-6
    assert _max_abs(sphere[0] - mx.array([0.0, 1.0, 0.0])) < 2e-6
    assert _max_abs(sphere[-1] - mx.array([0.0, -1.0, 0.0])) < 2e-6

    geometry = SphericalTensor.from_geometry(
        mx.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0]]),
        4,
    )
    assert geometry.shape == (25,)
    assert geometry.lmax == 4


def test_cartesian_rank_two_round_trip_and_metadata() -> None:
    mx = mlx_backend._require()
    matrix = mx.arange(9, dtype=mx.float32).reshape(3, 3)
    tensor = CartesianTensor(matrix)
    representations, basis = tensor.to_irrep_transformation()
    converted = tensor.to_irrep_tensor()

    assert representations == [(1, 0, 1), (1, 1, 1), (1, 2, 1)]
    assert basis.shape == (9, 3, 3)
    assert converted.Rs == representations
    assert _max_abs(tensor.project_to_cartesian() - matrix) < 2e-5


def test_cartesian_symmetry_projection_and_spherical_conversion() -> None:
    mx = mlx_backend._require()
    raw = mx.arange(9, dtype=mx.float32).reshape(3, 3)
    symmetric = raw + mx.swapaxes(raw, 0, 1)
    tensor = CartesianTensor(symmetric, formula="ij=ji")
    converted = tensor.to_irrep_tensor()
    spherical = SphericalTensor.from_irrep_tensor(converted)

    assert tensor.Rs == [(1, 0, 1), (1, 2, 1)]
    assert tensor.change_of_basis.shape == (6, 3, 3)
    assert spherical.shape == (9,)
    assert _max_abs(tensor.project_to_cartesian() - symmetric) < 2e-5


def test_spherical_tensor_rejects_repeated_irrep_degrees() -> None:
    mx = mlx_backend._require()
    converted = CartesianTensor(mx.zeros((3, 3, 3))).to_irrep_tensor()
    assert converted.Rs == [
        (1, 0, -1),
        (3, 1, -1),
        (2, 2, -1),
        (1, 3, -1),
    ]
    with pytest.raises(ValueError, match="at most one copy"):
        SphericalTensor.from_irrep_tensor(converted)


def test_cartesian_higher_rank_tutorial_examples() -> None:
    mx = mlx_backend._require()
    symmetric_rank_three = CartesianTensor(
        mx.zeros((3, 3, 3)),
        formula="ijk=jik=ikj",
    )
    elasticity = CartesianTensor(
        mx.zeros((3, 3, 3, 3)),
        formula="ijkl=jikl=ijlk",
    )

    assert symmetric_rank_three.Rs == [(1, 1, -1), (1, 3, -1)]
    assert symmetric_rank_three.change_of_basis.shape == (10, 3, 3, 3)
    assert elasticity.Rs == [
        (2, 0, 1),
        (1, 1, 1),
        (3, 2, 1),
        (1, 3, 1),
        (1, 4, 1),
    ]
    assert elasticity.change_of_basis.shape == (36, 3, 3, 3, 3)
