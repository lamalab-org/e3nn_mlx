from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.ops_rotations import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    irreps_wigner_d,
    irreps_wigner_d_from_matrix,
    matrix_to_angles,
    quaternion_to_matrix,
    rotation_matrix,
    wigner_d,
)


@pytest.mark.mlx
def test_rotation_matrix_is_orthogonal() -> None:
    alpha = mlx_backend.asarray(0.3)
    beta = mlx_backend.asarray(-0.2)
    gamma = mlx_backend.asarray(0.7)
    rot = rotation_matrix(alpha, beta, gamma)
    ident = rot @ rot.T
    eye = mlx_backend.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(float((ident - eye).abs().max())) < 1e-5


@pytest.mark.mlx
def test_wigner_d_l1_matches_rotation_matrix() -> None:
    alpha = mlx_backend.asarray(0.1)
    beta = mlx_backend.asarray(0.2)
    gamma = mlx_backend.asarray(-0.4)
    rot = rotation_matrix(alpha, beta, gamma)
    d1 = wigner_d(1, alpha, beta, gamma)
    assert abs(float((rot - d1).abs().max())) < 1e-6


@pytest.mark.mlx
def test_wigner_d_l0_is_scalar_one() -> None:
    d0 = wigner_d(0, mlx_backend.asarray(0.0), mlx_backend.asarray(math.pi / 3), mlx_backend.asarray(0.0))
    assert d0.tolist() == [[1.0]]


@pytest.mark.mlx
def test_wigner_d_identity_and_orthogonality_through_l6() -> None:
    mx = mlx_backend._require()
    zero = mlx_backend.asarray(0.0)
    for l in range(7):
        matrix = wigner_d(l, zero, zero, zero)
        eye = mx.eye(2 * l + 1, dtype=matrix.dtype)
        assert abs(float((matrix - eye).abs().max())) < 2e-5
        rotated = wigner_d(l, mlx_backend.asarray(0.2), mlx_backend.asarray(-0.4), mlx_backend.asarray(0.7))
        product = rotated @ mx.swapaxes(rotated, -1, -2)
        assert abs(float((product - eye).abs().max())) < 3e-5


@pytest.mark.mlx
def test_wigner_d_supports_batched_angles_and_direct_sums() -> None:
    alpha = mlx_backend.asarray([0.0, 0.2])
    beta = mlx_backend.asarray([0.1, -0.4])
    gamma = mlx_backend.asarray([0.3, 0.7])
    matrix = wigner_d(3, alpha, beta, gamma)
    assert matrix.shape == (2, 7, 7)
    direct_sum = irreps_wigner_d("0e+2x1o", alpha, beta, gamma)
    assert direct_sum.shape == (2, 7, 7)


@pytest.mark.mlx
def test_rotation_conversion_round_trips() -> None:
    mx = mlx_backend._require()
    axis = mlx_backend.asarray([[0.2, -0.4, 0.7]])
    angle = mlx_backend.asarray([0.8])
    matrix = axis_angle_to_matrix(axis, angle)
    quaternion_matrix = quaternion_to_matrix(axis_angle_to_quaternion(axis, angle))
    assert abs(float(mx.max(mx.abs(matrix - quaternion_matrix)))) < 2e-6

    original = rotation_matrix(mlx_backend.asarray(0.3), mlx_backend.asarray(0.5), mlx_backend.asarray(-0.7))
    alpha, beta, gamma = matrix_to_angles(original)
    reconstructed = rotation_matrix(alpha, beta, gamma)
    assert abs(float(mx.max(mx.abs(original - reconstructed)))) < 2e-6


@pytest.mark.mlx
def test_improper_rotation_applies_irrep_parity() -> None:
    mx = mlx_backend._require()
    inversion = -mx.eye(3, dtype=mx.float32)
    representation = irreps_wigner_d_from_matrix("0e+0o+1o+1e", inversion)
    diagonal = mx.diag(representation)
    expected = mlx_backend.asarray([1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert abs(float(mx.max(mx.abs(diagonal - expected)))) < 2e-6
