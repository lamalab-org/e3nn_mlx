from __future__ import annotations

import math

import pytest

from e3nn_core.cg import clebsch_gordan
from e3nn_core.instructions import make_tensor_product_instructions
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray
from e3nn_mlx.ops_tp import TensorProduct, compile_tensor_product, tensor_product, tensor_product_plan


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


def test_tensor_product_weighted_mode_sizes() -> None:
    assert tensor_product_plan("2x0e", "3x0e", "2x0e", weighted=True, mode="uvu").weight_numel == 6
    assert tensor_product_plan("2x0e", "3x0e", "3x0e", weighted=True, mode="uvv").weight_numel == 6
    assert tensor_product_plan("2x0e", "2x0e", "4x0e", weighted=True, mode="uuw").weight_numel == 8
    assert tensor_product_plan("2x0e", "2x0e", "2x0e", weighted=True, mode="uuu").weight_numel == 2
    assert tensor_product_plan("2x0e", "3x0e", "6x0e", weighted=True, mode="uvuv").weight_numel == 6
    assert tensor_product_plan("3x0e", "3x0e", "3x0e", weighted=True, mode="uvu<v").weight_numel == 3
    assert tensor_product_plan("3x0e", "3x0e", "2x0e", weighted=True, mode="u<vw").weight_numel == 6


def test_tensor_product_symbolic_higher_l_selection_rules() -> None:
    plan = tensor_product("2e", "1o")
    assert str(plan.irreps_out) == "1o+2o+3o"
    assert [str(inst.ir_out) for inst in plan.instructions] == ["1o", "2o", "3o"]


def test_tensor_product_symbolic_connection_modes() -> None:
    assert str(tensor_product("2x0e", "3x0e", mode="uvu").irreps_out) == "2x0e"
    assert str(tensor_product("2x0e", "3x0e", mode="uvv").irreps_out) == "3x0e"
    assert str(tensor_product("2x0e", "2x0e", mode="uuw").irreps_out) == "0e"
    assert str(tensor_product("2x0e", "2x0e", mode="uuu").irreps_out) == "2x0e"
    assert str(tensor_product("2x0e", "3x0e", mode="uvuv").irreps_out) == "6x0e"
    assert str(tensor_product("3x0e", "3x0e", mode="uvu<v").irreps_out) == "3x0e"
    assert str(tensor_product("3x0e", "3x0e", "2x0e", mode="u<vw").irreps_out) == "2x0e"


@pytest.mark.mlx
def test_tensor_product_vector_vector_reference() -> None:
    left = IrrepsArray("1o", mlx_backend.asarray([[1.0, 2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    result = tensor_product(left, right)
    scalar, vector, rank2 = result.chunk_arrays()
    _assert_close(scalar, [(1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0) / math.sqrt(3.0)])
    _assert_close(
        vector,
        [
            (2.0 * 6.0 - 3.0 * 5.0) / math.sqrt(2.0),
            (3.0 * 4.0 - 1.0 * 6.0) / math.sqrt(2.0),
            (1.0 * 5.0 - 2.0 * 4.0) / math.sqrt(2.0),
        ],
    )
    expected_rank2 = []
    cg = clebsch_gordan("1o", "1o", "2e")
    scale = math.sqrt(5.0)
    for c in range(5):
        value = 0.0
        for a in range(3):
            for b in range(3):
                value += [1.0, 2.0, 3.0][a] * [4.0, 5.0, 6.0][b] * cg[a][b][c]
        expected_rank2.append(scale * value)
    _assert_close(rank2, expected_rank2, tol=1e-5)


@pytest.mark.mlx
def test_tensor_product_scalar_vector_weighted_reference() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[4.0, 5.0, 6.0]]))
    weights = mlx_backend.asarray([0.5, -1.0])
    result = tensor_product(left, right, "1o", weights=weights)
    scale = (0.5 * 2.0 + (-1.0) * 3.0) / math.sqrt(2.0)
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


@pytest.mark.mlx
def test_tensor_product_scalar_l2_preserves_block_up_to_scale() -> None:
    left = IrrepsArray("0e", mlx_backend.asarray([[2.0]]))
    right = IrrepsArray("2e", mlx_backend.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    result = tensor_product(left, right)
    expected = [2.0 * value for value in [1.0, 2.0, 3.0, 4.0, 5.0]]
    _assert_close(result.array, expected)


@pytest.mark.mlx
def test_tensor_product_higher_l_output_shape() -> None:
    left = IrrepsArray("2e", mlx_backend.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.0, 1.0, 0.0]]))
    result = tensor_product(left, right)
    assert str(result.irreps) == "1o+2o+3o"
    assert result.shape == (1, 3 + 5 + 7)


@pytest.mark.mlx
def test_tensor_product_connection_mode_uvu() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    result = tensor_product(left, right, mode="uvu")
    scale = 1.0 / math.sqrt(3.0)
    _assert_close(result.array, [scale * 2.0 * (5.0 + 7.0 + 11.0), scale * 3.0 * (5.0 + 7.0 + 11.0)], tol=1e-5)


@pytest.mark.mlx
def test_tensor_product_connection_mode_uvv() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    result = tensor_product(left, right, mode="uvv")
    scale = 1.0 / math.sqrt(2.0)
    _assert_close(result.array, [scale * 5.0 * (2.0 + 3.0), scale * 7.0 * (2.0 + 3.0), scale * 11.0 * (2.0 + 3.0)])


@pytest.mark.mlx
def test_tensor_product_connection_mode_uuw() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[5.0, 7.0]]))
    result = tensor_product(left, right, mode="uuw")
    _assert_close(result.array, [math.sqrt(0.5) * (2.0 * 5.0 + 3.0 * 7.0)])


@pytest.mark.mlx
def test_tensor_product_connection_mode_uuu() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[5.0, 7.0]]))
    result = tensor_product(left, right, mode="uuu")
    _assert_close(result.array, [2.0 * 5.0, 3.0 * 7.0])


@pytest.mark.mlx
def test_tensor_product_connection_mode_uvuv() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    result = tensor_product(left, right, mode="uvuv")
    _assert_close(result.array, [2.0 * 5.0, 2.0 * 7.0, 2.0 * 11.0, 3.0 * 5.0, 3.0 * 7.0, 3.0 * 11.0])


@pytest.mark.mlx
def test_tensor_product_weighted_connection_mode_uvu() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    weights = mlx_backend.asarray([1.0, 0.5, -1.0, 2.0, 0.0, -0.5])
    result = tensor_product(left, right, "2x0e", mode="uvu", weights=weights)
    scale = 1.0 / math.sqrt(3.0)
    expected = [
        scale * 2.0 * (1.0 * 5.0 + 0.5 * 7.0 - 1.0 * 11.0),
        scale * 3.0 * (2.0 * 5.0 + 0.0 * 7.0 - 0.5 * 11.0),
    ]
    _assert_close(result.array, expected, tol=1e-5)


@pytest.mark.mlx
def test_tensor_product_connection_mode_uvu_strict_upper() -> None:
    left = IrrepsArray("3x0e", mlx_backend.asarray([[2.0, 3.0, 5.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[7.0, 11.0, 13.0]]))
    result = tensor_product(left, right, mode="uvu<v")
    _assert_close(result.array, [2.0 * 11.0, 2.0 * 13.0, 3.0 * 13.0])


@pytest.mark.mlx
def test_tensor_product_connection_mode_u_lt_vw() -> None:
    left = IrrepsArray("3x0e", mlx_backend.asarray([[2.0, 3.0, 5.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[7.0, 11.0, 13.0]]))
    weights = mlx_backend.asarray([1.0, 0.0, -1.0, 0.5, 2.0, 0.0])
    result = tensor_product(left, right, "2x0e", mode="u<vw", weights=weights)
    scale = 1.0 / math.sqrt(3.0)
    expected = [
        scale * (1.0 * (2.0 * 11.0) + (-1.0) * (2.0 * 13.0) + 2.0 * (3.0 * 13.0)),
        scale * (0.0 * (2.0 * 11.0) + 0.5 * (2.0 * 13.0) + 0.0 * (3.0 * 13.0)),
    ]
    _assert_close(result.array, expected, tol=1e-5)


@pytest.mark.mlx
def test_torch_like_tensor_product_module_api() -> None:
    tp = TensorProduct(
        "2x0e",
        "2x0e",
        "2x0e",
        [
            (0, 0, 0, "uuu", True),
        ],
    )
    tp.weight = mlx_backend.asarray([2.0, -1.0])
    left = IrrepsArray("2x0e", mlx_backend.asarray([[3.0, 5.0]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[7.0, 11.0]]))
    out = tp(left, right)
    _assert_close(out.array, [2.0 * 3.0 * 7.0, -1.0 * 5.0 * 11.0])
    assert tp.weight_numel == 2
    assert tp.weight_view_for_instruction(0).tolist() == [2.0, -1.0]
    assert len(list(tp.weight_views())) == 1
    assert tp.output_mask.tolist() == [1.0, 1.0]


@pytest.mark.mlx
def test_tensor_product_module_external_weights() -> None:
    tp = TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", True),
        ],
        internal_weights=False,
    )
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    out = tp(left, right, weight=mlx_backend.asarray([1.0, 0.0, -1.0, 0.5, 0.5, 0.5]))
    scale = 1.0 / math.sqrt(3.0)
    _assert_close(out.array, [scale * 2.0 * (5.0 - 11.0), scale * 3.0 * (0.5 * (5.0 + 7.0 + 11.0))], tol=1e-5)


@pytest.mark.mlx
def test_tensor_product_module_unshared_weights() -> None:
    tp = TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0], [5.0, 7.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[11.0, 13.0, 17.0], [19.0, 23.0, 29.0]]))
    weights = mlx_backend.asarray(
        [
            [1.0, 0.0, -1.0, 0.5, 0.5, 0.5],
            [0.0, 1.0, 1.0, 2.0, 0.0, -1.0],
        ]
    )
    out = tp(left, right, weight=weights)
    scale = 1.0 / math.sqrt(3.0)
    expected = [
        [scale * 2.0 * (11.0 - 17.0), scale * 3.0 * (0.5 * (11.0 + 13.0 + 17.0))],
        [scale * 5.0 * (23.0 + 29.0), scale * 7.0 * (2.0 * 19.0 - 29.0)],
    ]
    assert max(abs(a - b) for row_a, row_b in zip(out.array.tolist(), expected, strict=True) for a, b in zip(row_a, row_b, strict=True)) < 2e-5
    assert tp.weight_view_for_instruction(0, weight=weights).shape == (2, 2, 3)


@pytest.mark.mlx
def test_tensor_product_right_matches_forward() -> None:
    tp = TensorProduct(
        "1o",
        "1o",
        "0e + 1e + 2e",
        [
            (0, 0, 0, "uvw", False),
            (0, 0, 1, "uvw", False),
            (0, 0, 2, "uvw", False),
        ],
        internal_weights=False,
    )
    left = IrrepsArray("1o", mlx_backend.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]))
    right = IrrepsArray("1o", mlx_backend.asarray([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]))
    operator = tp.right(right)
    forward = tp(left, right)
    mx = mlx_backend._require()
    via_right = mx.einsum("...io,...i->...o", operator, left.array)
    mx.eval(via_right)
    assert abs(float((via_right - forward.array).abs().max())) < 1e-5


@pytest.mark.mlx
def test_tensor_product_right_with_unshared_weights() -> None:
    tp = TensorProduct(
        "2x0e",
        "2x0e",
        "2x0e",
        [
            (0, 0, 0, "uuu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0], [5.0, 7.0]]))
    right = IrrepsArray("2x0e", mlx_backend.asarray([[11.0, 13.0], [17.0, 19.0]]))
    weights = mlx_backend.asarray([[1.0, -1.0], [0.5, 2.0]])
    operator = tp.right(right, weight=weights)
    mx = mlx_backend._require()
    via_right = mx.einsum("...io,...i->...o", operator, left.array)
    forward = tp(left, right, weight=weights)
    mx.eval(via_right)
    assert abs(float((via_right - forward.array).abs().max())) < 1e-6


def test_tensor_product_module_variance_arguments_affect_normalization() -> None:
    instructions_default = make_tensor_product_instructions(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", False),
        ],
    )
    instructions_scaled = make_tensor_product_instructions(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", False),
        ],
        in1_var=[4.0],
        in2_var=[9.0],
        out_var=[16.0],
    )
    coeff_default = instructions_default[0].normalization.coefficient
    coeff_scaled = instructions_scaled[0].normalization.coefficient
    assert abs(coeff_default - (1.0 / math.sqrt(3.0))) < 1e-12
    assert abs(coeff_scaled - (4.0 / math.sqrt(108.0))) < 1e-12


@pytest.mark.mlx
def test_tensor_product_module_variance_arguments_change_output_scale() -> None:
    left = IrrepsArray("2x0e", mlx_backend.asarray([[2.0, 3.0]]))
    right = IrrepsArray("3x0e", mlx_backend.asarray([[5.0, 7.0, 11.0]]))
    tp_default = TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", False),
        ],
        internal_weights=False,
    )
    tp_scaled = TensorProduct(
        "2x0e",
        "3x0e",
        "2x0e",
        [
            (0, 0, 0, "uvu", False),
        ],
        in1_var=[4.0],
        in2_var=[9.0],
        out_var=[16.0],
        internal_weights=False,
    )
    out_default = tp_default(left, right)
    out_scaled = tp_scaled(left, right)
    ratio = tp_scaled.instructions[0].normalization.coefficient / tp_default.instructions[0].normalization.coefficient
    expected = [[ratio * value for value in out_default.array.tolist()[0]]]
    assert max(abs(a - b) for row_a, row_b in zip(out_scaled.array.tolist(), expected, strict=True) for a, b in zip(row_a, row_b, strict=True)) < 1e-5
