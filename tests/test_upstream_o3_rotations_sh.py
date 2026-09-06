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
def test_matrix_to_axis_angle_is_stable_at_pi() -> None:
    mx = mlx_backend._require()
    axes = mx.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, -3.0],
            [-2.0, 3.0, 1.0],
        ]
    )
    axes = axes / mx.sqrt(mx.sum(axes * axes, axis=-1, keepdims=True))
    angles = mx.array([math.pi, math.pi, math.pi, math.pi, math.pi - 1e-6])
    original = o3.axis_angle_to_matrix(axes, angles)

    quaternion = o3.matrix_to_quaternion(original)
    recovered_axis, recovered_angle = o3.matrix_to_axis_angle(original)

    assert _max_abs(o3.quaternion_to_matrix(quaternion) - original) < 2e-6
    assert _max_abs(o3.axis_angle_to_matrix(recovered_axis, recovered_angle) - original) < 2e-6


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

    axis, angle = o3.rand_axis_angle(100_000)
    rotated = o3.axis_angle_to_matrix(axis, angle) @ mx.array([0.2, 0.5, 0.3])
    assert _max_abs(mx.mean(rotated, axis=0)) < 0.008


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
    with pytest.raises(ValueError, match="requires parity"):
        o3.SphericalHarmonics("0e + 1e + 2e", irreps_in="1o")
    with pytest.raises(ValueError, match="exactly one vector"):
        o3.SphericalHarmonics("0e + 1o", irreps_in="1o + 2e")

    zeros = o3.spherical_harmonics([0, 1], mx.zeros((1, 3)), normalize=False, normalization="norm")
    assert _max_abs(zeros - mx.array([[1.0, 0.0, 0.0, 0.0]])) == 0.0

    vector = mx.random.normal(shape=(3,))
    for degree in range(12):
        lhs = ((-1) ** degree) * o3.spherical_harmonics(degree, vector, normalize=False)
        rhs = o3.spherical_harmonics(degree, -vector, normalize=False)
        assert _max_abs(lhs - rhs) < 3e-3


@pytest.mark.mlx
@pytest.mark.parametrize("degree", range(11))
def test_upstream_spherical_harmonics_normalization_through_l10(degree: int) -> None:
    mx = mlx_backend._require()
    vector = mx.random.normal(shape=(3,))
    integral = o3.spherical_harmonics(degree, vector, normalize=True, normalization="integral")
    normed = o3.spherical_harmonics(degree, vector, normalize=True, normalization="norm")
    component = o3.spherical_harmonics(degree, vector, normalize=True, normalization="component")
    assert abs(float(mx.mean(integral**2)) - 1.0 / (4.0 * math.pi)) < 3e-4
    assert abs(float(mx.sqrt(mx.sum(normed**2))) - 1.0) < 2e-3
    assert abs(float(mx.mean(component**2)) - 1.0) < 3e-3


@pytest.mark.mlx
@pytest.mark.parametrize("degree", range(10))
def test_upstream_spherical_harmonics_recurrence_and_jacobian(degree: int) -> None:
    mx = mlx_backend._require()
    vector = mx.random.normal(shape=(3,))
    higher = o3.spherical_harmonics(degree + 1, vector, normalize=False)
    lower = o3.spherical_harmonics(degree, vector, normalize=False)
    coefficient = mx.array(o3.wigner_3j(degree + 1, degree, 1), dtype=vector.dtype)
    recurrence = mx.einsum("ijk,j,k->i", coefficient, lower, vector)
    alpha = mx.sqrt(mx.sum(recurrence**2)) / mx.sqrt(mx.sum(higher**2))
    higher_unit = higher / mx.sqrt(mx.sum(higher**2))
    recurrence_unit = recurrence / mx.sqrt(mx.sum(recurrence**2))
    assert _max_abs(higher_unit - recurrence_unit) < (4e-3 if degree >= 7 else 3e-4)

    jacobian = mx.stack(
        [
            mx.grad(lambda value, index=index: o3.spherical_harmonics(degree + 1, value, normalize=False)[index])(
                vector
            )
            for index in range(2 * (degree + 1) + 1)
        ],
        axis=0,
    )
    expected = (degree + 1) / alpha * mx.einsum("ijk,j->ik", coefficient, lower)
    # normalize=False makes these scale as |vector| ** (degree + 1), so a fixed
    # absolute bound is really a draw-dependent one: the same correct result
    # measured 2.4e-7 for a short vector and 3.8e-1 for a long one. The relative
    # error is flat at a few float32 ulps across every degree, so bound that.
    scale = max(float(mx.max(mx.abs(expected))), 1.0)
    assert _max_abs(jacobian - expected) < 5e-6 * scale


@pytest.mark.mlx
def test_upstream_spherical_harmonics_monte_carlo_closure() -> None:
    mx = mlx_backend._require()
    vectors = mx.random.normal(shape=(200_000, 3))
    harmonics = [
        o3.spherical_harmonics(degree, vectors, normalize=True, normalization="integral")
        for degree in range(4)
    ]
    for degree1, first in enumerate(harmonics):
        for degree2, second in enumerate(harmonics):
            gram = 4.0 * math.pi * mx.mean(first[..., :, None] * second[..., None, :], axis=0)
            if degree1 == degree2:
                assert _max_abs(gram - mx.eye(2 * degree1 + 1)) < 0.025
            else:
                assert _max_abs(gram) < 0.025


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
