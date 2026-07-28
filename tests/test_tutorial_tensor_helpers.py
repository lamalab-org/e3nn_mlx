from __future__ import annotations

import pytest

from e3nn_mlx import IrrepsArray, o3
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
    points = mx.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )
    tetrahedron = SphericalTensor.from_geometry(points, 6)
    harmonics = o3.spherical_harmonics(
        list(range(7)),
        points,
        normalize=True,
        normalization="integral",
    )
    expected_radii = mx.linalg.norm(points, axis=-1)
    assert _max_abs(harmonics @ tetrahedron.array - expected_radii) < 2e-5


def test_spherical_tensor_accepts_historical_explicit_lmax() -> None:
    mx = mlx_backend._require()
    signal = mx.zeros((4,))

    tensor = SphericalTensor(signal, 1)

    assert tensor.lmax == 1
    assert tensor.shape == (4,)
    assert isinstance(tensor, IrrepsArray)
    assert tensor.as_irreps_array() is tensor
    norms = o3.Norm()(tensor)
    assert isinstance(norms, IrrepsArray)
    assert norms.shape == (2,)
    with pytest.raises(ValueError, match="requires 9 coefficients"):
        SphericalTensor(signal, 2)
    with pytest.raises(ValueError, match="non-negative integer"):
        SphericalTensor(signal, -1)


def test_spherical_tensor_descriptor_constructor_and_sum_of_diracs() -> None:
    mx = mlx_backend._require()
    descriptor = SphericalTensor(lmax=4, p_val=1, p_arg=-1)
    positions = mx.array([[1.0, 0.0, 0.0]])
    values = mx.array([2.0])

    coefficients = descriptor.sum_of_diracs(positions, values)
    harmonics = o3.spherical_harmonics(
        list(range(5)),
        positions,
        normalize=True,
        normalization="integral",
    )

    assert descriptor.Rs == [
        (1, 0, 1),
        (1, 1, -1),
        (1, 2, 1),
        (1, 3, -1),
        (1, 4, 1),
    ]
    assert coefficients.shape == (25,)
    assert abs(float(harmonics @ coefficients) - 2.0) < 2e-5
    assert SphericalTensor(4, 1, -1).Rs == descriptor.Rs

    batched = descriptor.sum_of_diracs(
        mx.ones((1, 3, 2, 3)),
        mx.ones((2, 1, 1)),
    )
    assert batched.shape == (2, 3, 25)

    angles = (mx.array(0.2), mx.array(-0.3), mx.array(0.4))
    rotation = descriptor.D_from_angles(*angles)
    expected = o3.irreps_wigner_d(descriptor.irreps, *angles)
    assert rotation.shape == (25, 25)
    assert _max_abs(rotation - expected) == 0.0


def test_spherical_tensor_with_peaks_at_fits_values_and_default_radii() -> None:
    mx = mlx_backend._require()
    descriptor = SphericalTensor(4, p_val=1, p_arg=-1)
    points = mx.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    harmonics = o3.spherical_harmonics(
        list(range(5)),
        points,
        normalize=True,
        normalization="integral",
    )

    values = mx.array([-1.5, 2.0])
    fitted = descriptor.with_peaks_at(points, values)
    assert fitted.shape == (25,)
    assert _max_abs(harmonics @ fitted - values) < 2e-5

    scaled_points = mx.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    scaled_harmonics = o3.spherical_harmonics(
        list(range(5)),
        scaled_points,
        normalize=True,
        normalization="integral",
    )
    fitted_radii = descriptor.with_peaks_at(scaled_points)
    assert _max_abs(scaled_harmonics @ fitted_radii - mx.array([1.0, 5.0])) < 2e-5

    with pytest.raises(ValueError, match="p_val=1"):
        SphericalTensor(4, p_val=-1, p_arg=-1).with_peaks_at(points)


def test_spherical_tensor_addition_aligns_bandwidths() -> None:
    mx = mlx_backend._require()
    degree_one = SphericalTensor(mx.arange(4, dtype=mx.float32), 1)
    degree_two = SphericalTensor(mx.ones((9,)), 2)

    result = degree_one + degree_two

    assert result.lmax == 2
    assert result.shape == (9,)
    assert result.signal is result.array
    assert _max_abs(result.array[:4] - (degree_one.array + 1.0)) == 0.0
    assert _max_abs(result.array[4:] - 1.0) == 0.0
    assert degree_two.change_lmax(1).shape == (4,)
    assert float(degree_one.dot(degree_one)) == 14.0
    assert float(degree_one.dot(SphericalTensor(mx.zeros((4,)), 1))) == 0.0


def test_spherical_tensor_full_product_preserves_tutorial_slice_order() -> None:
    mx = mlx_backend._require()
    signal_1 = mx.zeros((4,)).at[1].add(1.0)
    signal_2 = mx.zeros((4,)).at[3].add(1.0)

    product = SphericalTensor(signal_1, 1) @ SphericalTensor(signal_2, 1)

    assert product.Rs == [(2, 0, 0), (3, 1, 0), (1, 2, 0)]
    assert isinstance(product, IrrepsArray)
    assert product.shape == (16,)
    assert _max_abs(product.array[:2]) < 1e-6
    assert _max_abs(product.array[2:5]) > 0.1
    assert _max_abs(product.array[5:11]) < 1e-6
    assert _max_abs(product.array[11:]) > 0.1


def test_spherical_tensor_legacy_axes_follow_tutorial_l1_order() -> None:
    mx = mlx_backend._require()
    modern_peaks = []
    legacy_peaks = []
    for component in range(3):
        coefficients = mx.zeros((4,)).at[1 + component].add(1.0)
        tensor = SphericalTensor(coefficients)
        modern_surface, modern_values = tensor.plot(
            relu=False,
            radius=False,
            res=101,
        )
        legacy_surface, legacy_values = tensor.plot(
            relu=False,
            radius=False,
            res=101,
            legacy_axes=True,
        )
        maximum = int(mx.argmax(modern_values))
        row, column = divmod(maximum, modern_values.shape[1])
        modern_peaks.append(modern_surface[row, column])
        legacy_peaks.append(legacy_surface[row, column])
        assert _max_abs(legacy_values - modern_values) == 0.0

    modern = mx.stack(modern_peaks)
    legacy = mx.stack(legacy_peaks)
    assert _max_abs(modern - mx.eye(3)) < 0.02
    assert _max_abs(
        legacy
        - mx.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    ) < 0.02


def test_spherical_tensor_plot_uses_historical_unscaled_normalization() -> None:
    mx = mlx_backend._require()
    coefficients = mx.arange(16, dtype=mx.float32) / 10.0
    tensor = SphericalTensor(coefficients)
    sphere, values = tensor.plot(
        relu=False,
        radius=False,
        res=21,
    )
    expected = o3.spherical_harmonics(
        list(range(tensor.lmax + 1)),
        sphere,
        normalize=True,
        normalization="integral",
    ) @ coefficients

    assert _max_abs(values - expected) < 2e-5


def test_cartesian_rank_two_round_trip_and_metadata() -> None:
    mx = mlx_backend._require()
    matrix = mx.arange(9, dtype=mx.float32).reshape(3, 3)
    tensor = CartesianTensor(matrix)
    representations, basis = tensor.to_irrep_transformation()
    converted = tensor.to_irrep_tensor()

    assert isinstance(tensor.decomposition, o3.ReducedTensorProducts)
    assert isinstance(converted, IrrepsArray)
    assert representations == [(1, 0, 1), (1, 1, 1), (1, 2, 1)]
    assert basis.shape == (9, 3, 3)
    assert converted.Rs == representations
    assert _max_abs(tensor.project_to_cartesian() - matrix) < 2e-5


def test_cartesian_rank_two_legacy_basis_matches_original_tutorial_table() -> None:
    mx = mlx_backend._require()
    basis = CartesianTensor(mx.zeros((3, 3)), legacy_basis=True).change_of_basis
    diagonal_columns = mx.stack([basis[:, axis, axis] for axis in range(3)])
    expected = mx.array(
        [
            [0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.41, 0.0, -0.71],
            [0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.41, 0.0, 0.71],
            [0.58, 0.0, 0.0, 0.0, 0.0, 0.0, -0.82, 0.0, 0.0],
        ]
    )
    assert _max_abs(mx.round(diagonal_columns, decimals=2) - expected) < 1e-6

    for first, second, expected_index in ((0, 1, 2), (0, 2, 1), (1, 2, 3)):
        antisymmetric = mx.zeros((3, 3)).at[first, second].add(1.0)
        antisymmetric = antisymmetric.at[second, first].add(-1.0)
        converted = CartesianTensor(
            antisymmetric,
            legacy_basis=True,
        ).to_irrep_tensor().array
        nonzero = [
            index
            for index, value in enumerate(converted.tolist())
            if abs(value) > 1e-6
        ]
        assert nonzero == [expected_index]


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

    with pytest.raises(ValueError, match="rank-two"):
        CartesianTensor(mx.zeros((3, 3, 3)), legacy_basis=True)
