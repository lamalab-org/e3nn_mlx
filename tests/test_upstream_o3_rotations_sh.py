"""MLX adaptations of e3nn's upstream ``tests/o3`` rotation and SH tests."""

from __future__ import annotations

import math

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
def test_upstream_rotation_xyz_and_random_matrix() -> None:
    mx = mlx_backend._require()
    rotations = o3.rand_matrix(128)
    assert _max_abs(rotations @ mx.swapaxes(rotations, -1, -2) - mx.eye(3)) < 2e-5

    alpha, beta, _ = o3.matrix_to_angles(rotations)
    position = rotations @ mx.array([0.0, 1.0, 0.0])
    assert _max_abs(o3.angles_to_xyz(alpha, beta) - position) < 2e-5
    alpha2, beta2 = o3.xyz_to_angles(position)
    assert _max_abs(alpha - alpha2) < 2e-5
    assert _max_abs(beta - beta2) < 2e-5


@pytest.mark.mlx
def test_upstream_rotation_conversions_and_composition() -> None:
    original = o3.rand_matrix(256)
    axis, angle = o3.matrix_to_axis_angle(original)
    quaternion = o3.matrix_to_quaternion(original)

    assert _max_abs(o3.axis_angle_to_matrix(axis, angle) - original) < 2e-4
    assert _max_abs(o3.quaternion_to_matrix(quaternion) - original) < 2e-4
    assert _max_abs(o3.angles_to_matrix(*o3.matrix_to_angles(original)) - original) < 2e-4
    assert _max_abs(o3.axis_angle_to_matrix(*o3.quaternion_to_axis_angle(quaternion)) - original) < 2e-4
    assert _max_abs(o3.angles_to_matrix(*o3.axis_angle_to_angles(axis, angle)) - original) < 2e-4

    q1 = o3.rand_quaternion(64)
    q2 = o3.rand_quaternion(64)
    expected = o3.quaternion_to_matrix(q1) @ o3.quaternion_to_matrix(q2)
    assert _max_abs(o3.quaternion_to_matrix(o3.compose_quaternion(q1, q2)) - expected) < 2e-5
    a1 = o3.quaternion_to_angles(q1)
    a2 = o3.quaternion_to_angles(q2)
    assert _max_abs(o3.angles_to_matrix(*o3.compose_angles(*a1, *a2)) - expected) < 2e-5
    axis1, angle1 = o3.quaternion_to_axis_angle(q1)
    axis2, angle2 = o3.quaternion_to_axis_angle(q2)
    assert _max_abs(o3.axis_angle_to_matrix(*o3.compose_axis_angle(axis1, angle1, axis2, angle2)) - expected) < 2e-5


@pytest.mark.mlx
def test_upstream_rotation_inverse_and_coordinate_axes() -> None:
    mx = mlx_backend._require()
    angles = o3.rand_angles(32)
    inverse = o3.inverse_angles(*angles)
    identity = o3.angles_to_matrix(*o3.compose_angles(*angles, *inverse))
    assert _max_abs(identity - mx.eye(3)) < 2e-5

    vectors = mx.random.normal(shape=(100, 3))
    assert _max_abs((o3.matrix_x(mx.random.normal(shape=(100,))) @ vectors[..., None])[..., 0, 0] - vectors[:, 0]) < 2e-5
    assert _max_abs((o3.matrix_y(mx.random.normal(shape=(100,))) @ vectors[..., None])[..., 1, 0] - vectors[:, 1]) < 2e-5
    assert _max_abs((o3.matrix_z(mx.random.normal(shape=(100,))) @ vectors[..., None])[..., 2, 0] - vectors[:, 2]) < 2e-5


@pytest.mark.mlx
def test_upstream_angular_spherical_harmonics_equivariance_and_identity() -> None:
    mx = mlx_backend._require()
    for degree in range(8):
        alpha, beta, _ = o3.rand_angles()
        rot_alpha, rot_beta, rot_gamma = o3.rand_angles()
        composed = o3.compose_angles(rot_alpha, rot_beta, rot_gamma, alpha, beta, mx.array(0.0))
        lhs = o3.spherical_harmonics_alpha_beta(degree, composed[0], composed[1])
        harmonic = o3.spherical_harmonics_alpha_beta(degree, alpha, beta)
        rhs = o3.wigner_d(degree, rot_alpha, rot_beta, rot_gamma) @ harmonic
        assert _max_abs(lhs - rhs) < 2e-4

    for degree in range(5):
        alpha, beta, _ = o3.rand_angles()
        harmonic = o3.spherical_harmonics_alpha_beta(degree, alpha, beta)
        harmonic = harmonic * math.sqrt(4.0 * math.pi / (2 * degree + 1))
        representation = o3.wigner_d(degree, alpha, beta, mlx_backend.asarray(0.0))
        assert _max_abs(harmonic - representation[:, degree]) < 2e-4


@pytest.mark.mlx
def test_upstream_spherical_harmonics_general_calls_parity_and_zeros() -> None:
    mx = mlx_backend._require()
    vectors = mx.random.normal(shape=(2, 1, 2, 3))
    weird = o3.spherical_harmonics([4, 1, 2, 3, 3, 1, 0], vectors, normalize=False)
    assert weird.shape == (2, 1, 2, 35)

    requested = o3.Irreps("1x0e + 4x1o + 3x2e")
    output = o3.spherical_harmonics(requested, mx.random.normal(shape=(7, 3)), normalize=True)
    assert output.shape[-1] == requested.dim

    zeros = o3.spherical_harmonics([0, 1], mx.zeros((1, 3)), normalize=False, normalization="norm")
    assert _max_abs(zeros - mx.array([[1.0, 0.0, 0.0, 0.0]])) == 0.0

    vector = mx.random.normal(shape=(3,))
    for degree in range(12):
        lhs = ((-1) ** degree) * o3.spherical_harmonics(degree, vector, normalize=False)
        rhs = o3.spherical_harmonics(degree, -vector, normalize=False)
        assert _max_abs(lhs - rhs) < 3e-3


@pytest.mark.mlx
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [True, False])
def test_upstream_spherical_harmonics_module_and_compile(normalization: str, normalize: bool) -> None:
    mx = mlx_backend._require()
    irreps = o3.Irreps("0e + 1o + 3o")
    module = o3.SphericalHarmonics(irreps, normalize, normalization)
    xyz = mx.random.normal(shape=(11, 3))
    expected = o3.spherical_harmonics(irreps, xyz, normalize=normalize, normalization=normalization)
    assert _max_abs(module(xyz) - expected) < 2e-5
    assert _max_abs(mx.compile(module)(xyz) - expected) < 2e-5
    assert module.parameters() == {}


@pytest.mark.mlx
def test_upstream_angular_spherical_harmonics_module_and_gradient() -> None:
    mx = mlx_backend._require()
    module = o3.SphericalHarmonicsAlphaBeta([0, 1, 2])
    alpha = mx.random.normal(shape=(5, 4))
    beta = mx.random.normal(shape=(5, 4))
    assert _max_abs(module(alpha, beta) - mx.compile(module)(alpha, beta)) < 2e-5

    vector = mx.array([0.435, 0.7644, 0.023])
    gradient = mx.grad(
        lambda x: mx.sum(o3.spherical_harmonics([0, 1, 2, 3], x, normalize=False) ** 2)
    )(vector)
    mx.eval(gradient)
    assert gradient.shape == (3,)
    assert bool(mx.all(mx.isfinite(gradient)))
