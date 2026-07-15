from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.ops_rotations import rotation_matrix
from e3nn_mlx.ops_sh import spherical_harmonics


@pytest.mark.mlx
def test_spherical_harmonics_l1_component_matches_unit_vector() -> None:
    vectors = mlx_backend.asarray([[3.0, 0.0, 4.0]])
    y1 = spherical_harmonics(1, vectors, normalize=True, normalization="component")
    expected = mlx_backend.asarray([[math.sqrt(3.0) * 0.6, 0.0, math.sqrt(3.0) * 0.8]])
    assert abs(float((y1 - expected).abs().max())) < 1e-6


@pytest.mark.mlx
def test_spherical_harmonics_normalization_modes_scale_consistently() -> None:
    vectors = mlx_backend.asarray([[0.0, 0.0, 1.0]])
    component = spherical_harmonics(1, vectors, normalize=True, normalization="component")
    normed = spherical_harmonics(1, vectors, normalize=True, normalization="norm")
    integral = spherical_harmonics(1, vectors, normalize=True, normalization="integral")

    assert abs(float(component[0, 2]) - math.sqrt(3.0)) < 1e-6
    assert abs(float(normed[0, 2]) - 1.0) < 1e-6
    assert abs(float(integral[0, 2]) - math.sqrt(3.0 / (4.0 * math.pi))) < 1e-6


@pytest.mark.mlx
def test_spherical_harmonics_l1_is_equivariant() -> None:
    alpha = mlx_backend.asarray(0.4)
    beta = mlx_backend.asarray(-0.3)
    gamma = mlx_backend.asarray(0.2)
    rot = rotation_matrix(alpha, beta, gamma)
    vectors = mlx_backend.asarray([[0.2, -0.5, 0.7]])
    rotated_vectors = vectors @ rot.T

    lhs = spherical_harmonics(1, rotated_vectors, normalize=True, normalization="component")
    rhs = spherical_harmonics(1, vectors, normalize=True, normalization="component") @ rot.T
    assert abs(float((lhs - rhs).abs().max())) < 1e-5


@pytest.mark.mlx
def test_spherical_harmonics_l2_reference_values() -> None:
    vectors = mlx_backend.asarray([[1.0, 0.0, 0.0]])
    y2 = spherical_harmonics(2, vectors, normalize=True, normalization="component")
    expected = [0.0, 0.0, -math.sqrt(5.0) / 2.0, 0.0, -math.sqrt(15.0) / 2.0]
    assert max(abs(a - b) for a, b in zip(y2.tolist()[0], expected)) < 1e-6


@pytest.mark.mlx
def test_spherical_harmonics_gradient_exists() -> None:
    mx = mlx_backend._require()
    vectors = mlx_backend.asarray([[0.2, 0.3, 0.4]])

    def loss_fn(v):
        y = spherical_harmonics(2, v, normalize=True, normalization="component")
        return mx.sum(y * y)

    grad_fn = mx.grad(loss_fn)
    grad = grad_fn(vectors)
    mx.eval(grad)
    assert grad.shape == vectors.shape


@pytest.mark.mlx
def test_spherical_harmonics_supports_arbitrary_degrees_and_norms() -> None:
    mx = mlx_backend._require()
    vectors = mlx_backend.asarray([[0.2, -0.5, 0.7]])
    output = spherical_harmonics([0, 2, 4, 6], vectors, normalize=True, normalization="component")
    assert output.shape == (1, 1 + 5 + 9 + 13)
    cursor = 0
    for l in (0, 2, 4, 6):
        width = 2 * l + 1
        block = output[..., cursor : cursor + width]
        assert abs(float(mx.sum(block * block)) - width) < 5e-4
        cursor += width


@pytest.mark.mlx
def test_spherical_harmonics_homogeneity_without_normalization() -> None:
    vectors = mlx_backend.asarray([[0.2, -0.5, 0.7]])
    for l in range(7):
        original = spherical_harmonics(l, vectors, normalize=False)
        scaled = spherical_harmonics(l, 1.7 * vectors, normalize=False)
        assert abs(float((scaled - (1.7**l) * original).abs().max())) < 2e-4
