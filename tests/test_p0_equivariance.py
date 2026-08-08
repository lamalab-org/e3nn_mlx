from __future__ import annotations

import pytest

from e3nn_core.irreps import Irreps
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_rotations import irreps_wigner_d
from e3nn_mlx.ops_sh import spherical_harmonics
from e3nn_mlx.ops_tp import FullTensorProduct, FullyConnectedTensorProduct, TensorProduct


pytestmark = pytest.mark.mlx


def _rotate(array: IrrepsArray, alpha, beta, gamma) -> IrrepsArray:
    mx = mlx_backend._require()
    matrix = irreps_wigner_d(array.irreps, alpha, beta, gamma)
    return IrrepsArray(array.irreps, array.array @ mx.swapaxes(matrix, -1, -2))


def test_spherical_harmonics_equivariance_through_l6() -> None:
    mx = mlx_backend._require()
    alpha, beta, gamma = (mlx_backend.asarray(value) for value in (0.2, -0.4, 0.7))
    vector = IrrepsArray("1o", mlx_backend.asarray([[0.2, -0.5, 0.7]]))
    rotated_vector = _rotate(vector, alpha, beta, gamma)
    for l in range(7):
        expected = _rotate(spherical_harmonics(l, vector), alpha, beta, gamma)
        actual = spherical_harmonics(l, rotated_vector)
        assert abs(float(mx.max(mx.abs(actual.array - expected.array)))) < 8e-5


def test_full_tensor_product_is_equivariant() -> None:
    mx = mlx_backend._require()
    alpha, beta, gamma = (mlx_backend.asarray(value) for value in (0.3, 0.5, -0.2))
    left = IrrepsArray("0e+1o+2e", mx.arange(9, dtype=mx.float32)[None, :] / 7.0)
    right = IrrepsArray("1o", mlx_backend.asarray([[0.2, -0.4, 0.8]]))
    product = FullTensorProduct(left.irreps, right.irreps, compile_left_right=False)
    expected = _rotate(product(left, right), alpha, beta, gamma)
    actual = product(_rotate(left, alpha, beta, gamma), _rotate(right, alpha, beta, gamma))
    assert abs(float(mx.max(mx.abs(actual.array - expected.array)))) < 1e-4


def test_fully_connected_tensor_product_is_equivariant_with_external_weights() -> None:
    mx = mlx_backend._require()
    alpha, beta, gamma = (mlx_backend.asarray(value) for value in (-0.4, 0.25, 0.6))
    left = IrrepsArray("2x1o", mx.arange(6, dtype=mx.float32)[None, :] / 5.0)
    right = IrrepsArray("1o", mlx_backend.asarray([[0.3, 0.1, -0.7]]))
    product = FullyConnectedTensorProduct(
        left.irreps, right.irreps, "2x0e+2x1e+2x2e", internal_weights=False, compile_left_right=False
    )
    weight = mx.arange(product.weight_numel, dtype=mx.float32) / max(product.weight_numel, 1)
    expected = _rotate(product(left, right, weight=weight), alpha, beta, gamma)
    actual = product(
        _rotate(left, alpha, beta, gamma), _rotate(right, alpha, beta, gamma), weight=weight
    )
    assert abs(float(mx.max(mx.abs(actual.array - expected.array)))) < 1e-4


def test_tensor_product_broadcasts_all_leading_dimensions() -> None:
    left = IrrepsArray("0e", mlx_backend.asarray([[[2.0]], [[3.0]]]))
    right = IrrepsArray("0e", mlx_backend.asarray([[[5.0], [7.0], [11.0]]]))
    product = FullTensorProduct("0e", "0e", compile_left_right=False)
    output = product(left, right)
    assert output.shape == (2, 3, 1)
    assert output.array[..., 0].tolist() == [[10.0, 14.0, 22.0], [15.0, 21.0, 33.0]]


def test_tensor_product_rejects_wrong_metadata_even_when_dimensions_match() -> None:
    product = TensorProduct(
        "1o", "1o", "0e", [(0, 0, 0, "uvuv", False)], internal_weights=False, compile_left_right=False
    )
    wrong = IrrepsArray("3x0e", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    vector = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="left input irreps"):
        product(wrong, vector)


def test_duplicate_output_irreps_remain_distinct_chunks() -> None:
    product = TensorProduct(
        "1o",
        "1o",
        "0e+2x0e",
        [(0, 0, 0, "uvw", True), (0, 0, 1, "uvw", True)],
        internal_weights=False,
        compile_left_right=False,
    )
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[3.0, 1.0, -2.0]]))
    output = product(left, right, weight=mlx_backend.asarray([1.0, 2.0, 3.0]))
    assert output.irreps == Irreps("0e+2x0e")
    assert output.shape == (1, 3)
