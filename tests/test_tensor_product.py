from __future__ import annotations

import math

import pytest

from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_tp import compile_tensor_product, tensor_product, tensor_product_plan


def _assert_close(actual, expected, tol: float = 1e-6) -> None:
    values = actual.tolist()
    if isinstance(values[0], list):
        flat_actual = values[0]
    else:
        flat_actual = values
    assert max(abs(a - b) for a, b in zip(flat_actual, expected, strict=True)) < tol


def test_tensor_product_symbolic_plan() -> None:
    plan = tensor_product("1o", "1o")
    assert str(plan.irreps_out) == "0e+1e+2e"
    assert [str(inst.ir_out) for inst in plan.instructions] == ["0e", "1e", "2e"]


def test_tensor_product_weighted_plan_tracks_weight_size() -> None:
    plan = tensor_product_plan("2x0e", "1o", "1o", weighted=True)
    assert str(plan.irreps_out) == "1o"
    assert plan.weight_numel == 2


@pytest.mark.mlx
def test_tensor_product_vector_vector_reference() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    result = tensor_product(left, right)
    expected = [
        (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0) / math.sqrt(3.0),
        (2.0 * 6.0 - 3.0 * 5.0) / math.sqrt(2.0),
        (3.0 * 4.0 - 1.0 * 6.0) / math.sqrt(2.0),
        (1.0 * 5.0 - 2.0 * 4.0) / math.sqrt(2.0),
        (2.0 * 3.0 * 6.0 - 1.0 * 4.0 - 2.0 * 5.0) / math.sqrt(6.0),
        (1.0 * 5.0 + 2.0 * 4.0) / math.sqrt(2.0),
        (2.0 * 6.0 + 3.0 * 5.0) / math.sqrt(2.0),
        (3.0 * 4.0 + 1.0 * 6.0) / math.sqrt(2.0),
        (1.0 * 4.0 - 2.0 * 5.0) / math.sqrt(2.0),
    ]
    _assert_close(result.array, expected)


@pytest.mark.mlx
def test_tensor_product_scalar_vector_weighted_reference() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    weights = mlx_backend.asarray([0.5, -1.0])
    result = tensor_product(left, right, "1o", weights=weights)
    scale = 0.5 * 2.0 + (-1.0) * 3.0
    _assert_close(result.array, [scale * 4.0, scale * 5.0, scale * 6.0])


@pytest.mark.mlx
def test_tensor_product_dtype_is_preserved() -> None:
    mx = mlx_backend._require()
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 0.0, 0.0]], dtype=mx.float16))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.0, 1.0, 0.0]], dtype=mx.float16))
    result = tensor_product(left, right)
    assert result.array.dtype == mx.float16


@pytest.mark.mlx
def test_tensor_product_gradient_exists() -> None:
    mx = mlx_backend._require()
    left = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.5, 0.6, 0.7]]))
    weights = mlx_backend.asarray([1.0, -0.5, 0.25])

    def loss_fn(w):
        out = tensor_product(left, right, "0e + 1e + 2e", weights=w)
        return mx.sum(out.array * out.array)

    grad = mx.grad(loss_fn)(weights)
    mx.eval(grad)
    assert grad.shape == weights.shape


@pytest.mark.mlx
def test_compiled_tensor_product_matches_eager() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.5, 0.6, 0.7]]))
    plan = tensor_product_plan("1o", "1o")
    compiled = compile_tensor_product(plan)
    eager = tensor_product(left, right).array
    compiled_out = compiled(left.array, right.array)
    mlx_backend._require().eval(compiled_out)
    assert abs(float((eager - compiled_out).abs().max())) < 1e-6
