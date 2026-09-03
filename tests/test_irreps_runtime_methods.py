"""Backend-dependent Irrep/Irreps methods installed by the MLX runtime."""

from __future__ import annotations

import numpy as np
import pytest

from e3nn_core.irreps import Irrep, Irreps

from e3nn_mlx.compat import require_mlx


def test_core_declares_the_runtime_hooks_without_an_array_backend() -> None:
    import e3nn_core

    assert set(e3nn_core.registered_runtimes()) >= {
        "irrep_D_from_angles",
        "irrep_D_from_matrix",
        "irreps_D_from_angles",
        "irreps_D_from_matrix",
        "irreps_randn",
    }


def test_irrep_d_from_angles_matches_the_free_function() -> None:
    mx, _ = require_mlx()
    from e3nn_mlx.ops_rotations import irrep_wigner_d, rand_angles

    alpha, beta, gamma = rand_angles(4)
    for spec in ("1o", "2e", "3o"):
        method = np.array(Irrep(spec).D_from_angles(alpha, beta, gamma))
        free = np.array(irrep_wigner_d(spec, alpha, beta, gamma))
        assert np.allclose(method, free)


def test_irrep_d_from_angles_applies_the_parity_factor() -> None:
    mx, _ = require_mlx()
    from e3nn_mlx.ops_rotations import rand_angles

    alpha, beta, gamma = rand_angles(4)
    even = np.array(Irrep("2e").D_from_angles(alpha, beta, gamma, 1))
    odd = np.array(Irrep("1o").D_from_angles(alpha, beta, gamma, 1))
    assert np.allclose(even, np.array(Irrep("2e").D_from_angles(alpha, beta, gamma)))
    assert np.allclose(odd, -np.array(Irrep("1o").D_from_angles(alpha, beta, gamma)))


def test_irreps_d_from_matrix_handles_improper_rotations() -> None:
    mx, _ = require_mlx()
    from e3nn_mlx.ops_rotations import rand_matrix

    matrix = rand_matrix(6)
    irreps = Irreps("2x0e+1x1o")
    proper = np.array(irreps.D_from_matrix(matrix))
    improper = np.array(irreps.D_from_matrix(-matrix))
    # The 1o block flips sign under inversion; the 0e blocks do not.
    assert np.allclose(proper[:, :2, :2], improper[:, :2, :2])
    assert np.allclose(proper[:, 2:, 2:], -improper[:, 2:, 2:])


def test_d_from_quaternion_and_axis_angle_agree_with_the_angle_path() -> None:
    mx, _ = require_mlx()
    from e3nn_mlx.ops_rotations import (
        axis_angle_to_angles,
        quaternion_to_angles,
        rand_axis_angle,
        rand_quaternion,
    )

    quaternion = rand_quaternion(4)
    axis, angle = rand_axis_angle(4)
    for spec in ("1o", "2e"):
        irrep = Irrep(spec)
        assert np.allclose(
            np.array(irrep.D_from_quaternion(quaternion)),
            np.array(irrep.D_from_angles(*quaternion_to_angles(quaternion))),
        )
        assert np.allclose(
            np.array(irrep.D_from_axis_angle(axis, angle)),
            np.array(irrep.D_from_angles(*axis_angle_to_angles(axis, angle))),
        )


def test_randn_component_normalization_shape_and_variance() -> None:
    mx, _ = require_mlx()
    irreps = Irreps("4x0e+3x1o+2x2e")
    sample = irreps.randn(4096, -1)
    assert sample.shape == (4096, irreps.dim)
    assert abs(float(mx.var(sample)) - 1.0) < 0.1


def test_randn_norm_normalization_gives_unit_irrep_norms() -> None:
    mx, _ = require_mlx()
    irreps = Irreps("4x0e+3x1o+2x2e")
    sample = np.array(irreps.randn(256, -1, normalization="norm"))
    for part, chunk in zip(irreps, irreps.slices()):
        block = sample[:, chunk].reshape(256, part.mul, part.ir.dim)
        assert np.allclose(np.linalg.norm(block, axis=-1), 1.0, atol=1e-5)


def test_randn_places_the_representation_on_the_marked_axis() -> None:
    require_mlx()
    irreps = Irreps("2x1o")
    assert irreps.randn(2, -1, 3).shape == (2, irreps.dim, 3)
    with pytest.raises(ValueError, match="exactly one dimension"):
        irreps.randn(2, 3)


def test_irreps_index_locates_a_part() -> None:
    irreps = Irreps("2x0e+3x1o")
    assert irreps.index("3x1o") == 1
    assert irreps.index((2, "0e")) == 0
    with pytest.raises(ValueError):
        irreps.index("9x4e")
