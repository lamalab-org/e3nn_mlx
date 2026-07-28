from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_tp import ElementwiseTensorProduct, FullTensorProduct, FullyConnectedTensorProduct, TensorProduct, TensorSquare

pytestmark = pytest.mark.mlx


def _max_abs_diff(actual, expected) -> float:
    if isinstance(actual[0], list):
        return max(abs(a - b) for row_a, row_b in zip(actual, expected, strict=True) for a, b in zip(row_a, row_b, strict=True))
    return max(abs(a - b) for a, b in zip(actual, expected, strict=True))


def test_fully_connected_tensor_product_builds_weighted_paths() -> None:
    tp = FullyConnectedTensorProduct("1o", "1o", "0e + 1e + 2e", internal_weights=False)
    assert str(tp.irreps_out) == "0e+1e+2e"
    assert len(tp.instructions) == 3
    assert tp.weight_numel == 3


def test_full_tensor_product_builds_unweighted_full_output() -> None:
    tp = FullTensorProduct("2x0e", "3x0e")
    assert str(tp.irreps_out) == "6x0e"
    assert len(tp.instructions) == 1
    assert tp.weight_numel == 0


def test_full_tensor_product_regroups_public_output() -> None:
    tp = FullTensorProduct("2x1o", "1x0e + 1x1e")
    assert str(tp.irreps_out) == "2x0o+4x1o+2x2o"
    assert repr(tp) == "FullTensorProduct(2x1o x 1x0e+1x1e -> 2x0o+4x1o+2x2o | 8 paths | 0 weights)"


def test_elementwise_tensor_product_builds_elementwise_paths() -> None:
    tp = ElementwiseTensorProduct("2x0e + 1o", "2x0e + 1o")
    assert str(tp.irreps_out) == "2x0e+0e+1e+2e"
    assert [inst.mode for inst in tp.instructions] == ["uuu", "uuu", "uuu", "uuu"]


def test_tensor_square_symbolic_full_mode() -> None:
    tp = TensorSquare("2x0e")
    assert str(tp.irreps_out) == "3x0e"
    assert {inst.mode for inst in tp.instructions} == {"uvu<v", "uuu"}


def test_tensor_square_propagates_custom_kernel_setting() -> None:
    general = TensorSquare("2x0e", use_custom_kernel=False)
    requested = TensorSquare("2x0e", use_custom_kernel=True)

    assert general.use_custom_kernel is False
    assert general._metal_operation is None
    assert requested.use_custom_kernel is True


def test_tensor_square_symbolic_fully_connected_mode() -> None:
    tp = TensorSquare("2x0e", "1x0e", internal_weights=False)
    assert str(tp.irreps_out) == "0e"
    assert {inst.mode for inst in tp.instructions} == {"u<vw", "uuw"}


def test_tensor_square_symbolic_full_mode_regroups_public_output() -> None:
    tp = TensorSquare("5x1e + 2e")
    assert str(tp.irreps_out) == "16x0e+15x1e+21x2e+5x3e+4e"
    assert repr(tp) == "TensorSquare(5x1e+1x2e -> 16x0e+15x1e+21x2e+5x3e+1x4e | 58 paths | 0 weights)"


@pytest.mark.mlx
def test_full_tensor_product_numeric_scalar_blocks() -> None:
    tp = FullTensorProduct("2x0e", "3x0e")
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    out = tp(left, right)
    assert _max_abs_diff(out.array.tolist()[0], [10.0, 14.0, 22.0, 15.0, 21.0, 33.0]) < 1e-6


@pytest.mark.mlx
def test_full_tensor_product_numeric_regroups_output() -> None:
    tp = FullTensorProduct("1x0e", "1x0e + 1x0e")
    left = IrrepsArray("0e", mlx_backend.asarray([[2.0]]))
    right = IrrepsArray("0e+0e", mlx_backend.asarray([[3.0, 5.0]]))
    out = tp(left, right)
    assert str(out.irreps) == "2x0e"
    assert _max_abs_diff(out.array.tolist()[0], [6.0, 10.0]) < 1e-6


@pytest.mark.mlx
def test_elementwise_tensor_product_numeric_scalar_blocks() -> None:
    tp = ElementwiseTensorProduct("2x0e", "2x0e")
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[5.0, 7.0]]))
    out = tp(left, right)
    assert _max_abs_diff(out.array.tolist()[0], [10.0, 21.0]) < 1e-6


@pytest.mark.mlx
def test_fully_connected_tensor_product_numeric() -> None:
    tp = FullyConnectedTensorProduct("1o", "1o", "0e", internal_weights=False)
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    out = tp(left, right, weight=mlx_backend.asarray([2.0]))
    expected = [2.0 * (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0) / math.sqrt(3.0)]
    assert _max_abs_diff(out.array.tolist()[0], expected) < 1e-5


@pytest.mark.mlx
def test_tensor_square_full_numeric() -> None:
    tp = TensorSquare("2x0e")
    array = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    ref = TensorProduct(
        "2x0e",
        "2x0e",
        "0e+2x0e",
        [
            (0, 0, 0, "uvu<v", False, 1.0),
            (0, 0, 1, "uuu", False, 1.0 / 3.0),
        ],
        irrep_normalization="none",
        internal_weights=False,
    )
    out = tp(array)
    ref_out = ref(array, array)
    assert str(out.irreps) == "3x0e"
    assert _max_abs_diff(out.array.tolist()[0], ref_out.array.tolist()[0]) < 1e-6


@pytest.mark.mlx
def test_tensor_square_fully_connected_numeric() -> None:
    tp = TensorSquare("2x0e", "1x0e", internal_weights=False)
    array = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    weights = mlx_backend.asarray([1.0, -1.0, 0.5])
    ref = TensorProduct(
        "2x0e",
        "2x0e",
        "0e",
        [
            (0, 0, 0, "u<vw", True, 1.0),
            (0, 0, 0, "uuw", True, 1.0 / 3.0),
        ],
        irrep_normalization="none",
        internal_weights=False,
    )
    out = tp(array, weight=weights)
    ref_out = ref(array, array, weight=weights)
    assert _max_abs_diff(out.array.tolist()[0], ref_out.array.tolist()[0]) < 1e-6
