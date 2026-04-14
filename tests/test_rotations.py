from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.ops_rotations import rotation_matrix, wigner_d


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
