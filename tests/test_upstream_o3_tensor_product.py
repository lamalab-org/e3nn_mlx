"""MLX adaptations of upstream e3nn tensor-product tests."""

from __future__ import annotations

import copy
import tempfile

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def _array(irreps, values):
    return o3.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


_MODE_CASES = [
    ("2x1o", "3x1e", "4x1o", "uvw", True),
    ("2x1o", "3x1e", "2x1o", "uvu", True),
    ("2x1o", "3x1e", "2x1o", "uvu", False),
    ("2x1o", "3x1e", "3x1o", "uvv", True),
    ("2x1o", "3x1e", "3x1o", "uvv", False),
    ("3x1o", "3x1e", "2x1o", "uuw", True),
    ("3x1o", "3x1e", "1x1o", "uuw", False),
    ("3x1o", "3x1e", "3x1o", "uuu", True),
    ("3x1o", "3x1e", "3x1o", "uuu", False),
    ("2x1o", "3x1e", "6x1o", "uvuv", True),
    ("2x1o", "3x1e", "6x1o", "uvuv", False),
]


def _make_mode_product(case, *, compile_left_right=False):
    irreps_in1, irreps_in2, irreps_out, mode, weighted = case
    return o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        [(0, 0, 0, mode, weighted)],
        internal_weights=False,
        compile_left_right=compile_left_right,
    )


@pytest.mark.mlx
@pytest.mark.parametrize("case", _MODE_CASES)
def test_upstream_tensor_product_bilinear_right_equivariant_and_weight_linear(case) -> None:
    mx = mlx_backend._require()
    product = _make_mode_product(case)
    x1 = mx.random.normal(shape=(2, product.irreps_in1.dim))
    x2 = mx.random.normal(shape=(2, product.irreps_in1.dim))
    y1 = mx.random.normal(shape=(2, product.irreps_in2.dim))
    y2 = mx.random.normal(shape=(2, product.irreps_in2.dim))
    weight = mx.random.normal(shape=(product.weight_numel,)) if product.weight_numel else None

    def run(left, right, selected_weight=weight):
        return product(_array(product.irreps_in1, left), _array(product.irreps_in2, right), selected_weight).array

    actual = run(x1 + 1.7 * x2, y1 - y2)
    assert _max_abs(actual - run(x1, y1 - y2) - 1.7 * run(x2, y1 - y2)) < 3e-5
    assert _max_abs(actual - run(x1 + 1.7 * x2, y1) + run(x1 + 1.7 * x2, y2)) < 3e-5

    right_operator = product.right(_array(product.irreps_in2, y1), weight)
    assert _max_abs(run(x1, y1) - mx.einsum("...i,...ij->...j", x1, right_operator)) < 3e-5

    angles = o3.rand_angles()
    d1 = o3.irreps_wigner_d(product.irreps_in1, *angles)
    d2 = o3.irreps_wigner_d(product.irreps_in2, *angles)
    dout = o3.irreps_wigner_d(product.irreps_out, *angles)
    rotated = run(x1 @ mx.swapaxes(d1, -1, -2), y1 @ mx.swapaxes(d2, -1, -2))
    expected = run(x1, y1) @ mx.swapaxes(dout, -1, -2)
    assert _max_abs(rotated - expected) < 4e-4

    if product.weight_numel:
        weight1 = mx.random.normal(shape=(product.weight_numel,))
        weight2 = mx.random.normal(shape=(product.weight_numel,))
        assert _max_abs(run(x1, y1, weight1) + 1.5 * run(x1, y1, weight2) - run(x1, y1, weight1 + 1.5 * weight2)) < 3e-5


@pytest.mark.mlx
@pytest.mark.parametrize("path_normalization", ["element", "path"])
def test_upstream_fully_connected_tensor_product_is_statistically_normalized(path_normalization: str) -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct(
        "5x0e + 5x1o",
        "5x0e + 5x1e",
        "4x0e + 4x1o + 4x1e",
        path_normalization=path_normalization,
        internal_weights=False,
        shared_weights=False,
        compile_left_right=False,
    )
    samples = 4096
    left = mx.random.normal(shape=(samples, product.irreps_in1.dim))
    right = mx.random.normal(shape=(samples, product.irreps_in2.dim))
    weights = mx.random.normal(shape=(samples, product.weight_numel))
    output = product(_array(product.irreps_in1, left), _array(product.irreps_in2, right), weights).array
    variance = mx.mean(output * output, axis=0)
    assert 0.65 < float(mx.mean(variance)) < 1.35


@pytest.mark.mlx
def test_upstream_tensor_product_empty_outputs_and_inputs() -> None:
    mx = mlx_backend._require()
    disconnected = o3.TensorProduct(
        "1o + 2e",
        "0e + 1o + 2e",
        "1o",
        [],
        internal_weights=False,
        compile_left_right=False,
    )
    left = _array(disconnected.irreps_in1, mx.random.normal(shape=(4, disconnected.irreps_in1.dim)))
    right = _array(disconnected.irreps_in2, mx.random.normal(shape=(4, disconnected.irreps_in2.dim)))
    output = disconnected(left, right)
    assert output.shape == (4, disconnected.irreps_out.dim)
    assert _max_abs(output.array) == 0.0
    assert disconnected.right(right).shape == (4, disconnected.irreps_in1.dim, disconnected.irreps_out.dim)

    empty_input = o3.FullyConnectedTensorProduct("0e + 1e", "", "0e + 1e", compile_left_right=False)
    output = empty_input(
        _array(empty_input.irreps_in1, mx.random.normal(shape=(1, 2, 4))),
        _array("", mx.zeros((2, 1, 0))),
    )
    assert output.shape == (2, 2, 4)
    assert _max_abs(output.array) == 0.0

    empty_output = o3.FullTensorProduct("", "1o", compile_left_right=False)
    result = empty_output(_array("", mx.zeros((3, 0))), _array("1o", mx.random.normal(shape=(3, 3))))
    assert result.shape == (3, 0)


@pytest.mark.mlx
def test_upstream_tensor_product_empty_leading_dimension_broadcasts() -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct("0e + 1e", "0e + 1e", "0e + 1e", compile_left_right=False)
    left = _array(product.irreps_in1, mx.random.normal(shape=(2, 1, 0, 1, 4)))
    right = _array(product.irreps_in2, mx.random.normal(shape=(1, 2, 0, 3, 4)))
    assert product(left, right).shape == (2, 2, 0, 3, 4)
    assert product.right(right).shape == (1, 2, 0, 3, 4, 4)


@pytest.mark.mlx
def test_upstream_tensor_product_compiled_and_eager_paths_match() -> None:
    mx = mlx_backend._require()
    case = ("2x1o", "3x1e", "4x1o", "uvw", True)
    eager = _make_mode_product(case, compile_left_right=False)
    compiled = _make_mode_product(case, compile_left_right=True)
    left = _array(eager.irreps_in1, mx.random.normal(shape=(8, eager.irreps_in1.dim)))
    right = _array(eager.irreps_in2, mx.random.normal(shape=(8, eager.irreps_in2.dim)))
    weight = mx.random.normal(shape=(eager.weight_numel,))
    assert _max_abs(eager(left, right, weight).array - compiled(left, right, weight).array) < 2e-6
    assert _max_abs(eager.right(right, weight) - compiled.right(right, weight)) < 2e-6


@pytest.mark.mlx
def test_upstream_tensor_product_unshared_weight_broadcast_and_validation() -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct(
        "0e + 1e",
        "0e + 1e",
        "0e + 1e",
        shared_weights=False,
        compile_left_right=False,
    )
    assert not product.internal_weights
    with pytest.raises(RuntimeError, match="Weights must be provided"):
        product(
            _array(product.irreps_in1, mx.random.normal(shape=(2, 4))),
            _array(product.irreps_in2, mx.random.normal(shape=(2, 4))),
        )

    left_values = mx.random.normal(shape=(2, 1, 4, 4))
    right_values = mx.random.normal(shape=(2, 3, 1, 4))
    weights = mx.random.normal(shape=(3, 4, product.weight_numel))
    output = product(_array(product.irreps_in1, left_values), _array(product.irreps_in2, right_values), weights)
    assert output.shape == (2, 3, 4, 4)

    flat = product(
        _array(product.irreps_in1, mx.broadcast_to(left_values, (2, 3, 4, 4)).reshape(24, 4)),
        _array(product.irreps_in2, mx.broadcast_to(right_values, (2, 3, 4, 4)).reshape(24, 4)),
        mx.broadcast_to(weights[None], (2, 3, 4, product.weight_numel)).reshape(24, product.weight_numel),
    )
    assert _max_abs(output.array.reshape(24, 4) - flat.array) < 2e-6


@pytest.mark.mlx
@pytest.mark.parametrize("shared", [True, False])
def test_upstream_tensor_product_accepts_per_instruction_weight_lists(shared: bool) -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct(
        "1e + 2e",
        "1e + 2e",
        "0e + 1e + 2e",
        internal_weights=False,
        shared_weights=shared,
        compile_left_right=False,
    )
    batch = 3
    left = _array(product.irreps_in1, mx.random.normal(shape=(batch, product.irreps_in1.dim)))
    right = _array(product.irreps_in2, mx.random.normal(shape=(batch, product.irreps_in2.dim)))
    if shared:
        weight_list = [mx.random.normal(shape=instruction.path_shape) for instruction in product.instructions]
        flat = mx.concatenate([value.reshape(-1) for value in weight_list])
    else:
        weight_list = [mx.random.normal(shape=(batch, *instruction.path_shape)) for instruction in product.instructions]
        flat = mx.concatenate([value.reshape(batch, -1) for value in weight_list], axis=-1)
    assert _max_abs(product(left, right, weight_list).array - product(left, right, flat).array) < 2e-6
    with pytest.raises(ValueError, match="per-instruction weights"):
        product(left, right, weight_list[:-1])


@pytest.mark.mlx
def test_upstream_tensor_product_single_connected_output_with_extra_zeros() -> None:
    mx = mlx_backend._require()
    first = o3.TensorProduct("5x0e", "5x0e", "5x0e", [(0, 0, 0, "uvw", True)], compile_left_right=False)
    second = o3.TensorProduct("5x0e", "5x0e", "5x0e + 3x0o", [(0, 0, 0, "uvw", True)], compile_left_right=False)
    second.update({"weight": first.weight})
    left = _array("5x0e", mx.random.normal(shape=(3, 5)))
    right = _array("5x0e", mx.random.normal(shape=(3, 5)))
    output1, output2 = first(left, right), second(left, right)
    assert _max_abs(output1.array - output2.array[:, :5]) < 2e-6
    assert _max_abs(output2.array[:, 5:]) == 0.0


@pytest.mark.mlx
def test_upstream_tensor_product_weight_views_copy_and_save_load() -> None:
    mx = mlx_backend._require()
    product = o3.FullyConnectedTensorProduct("1e + 2e", "1e + 2e", "0e + 1e + 2e", compile_left_right=False)
    assert [view.shape for view in product.weight_views()] == [instruction.path_shape for instruction in product.instructions if instruction.has_weight]
    zeros = mx.zeros((product.weight_numel,))
    left = _array(product.irreps_in1, mx.random.normal(shape=(2, product.irreps_in1.dim)))
    right = _array(product.irreps_in2, mx.random.normal(shape=(2, product.irreps_in2.dim)))
    assert _max_abs(product(left, right, zeros).array) == 0.0

    duplicate = copy.deepcopy(product)
    assert _max_abs(product(left, right).array - duplicate(left, right).array) < 2e-6

    with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
        product.save_weights(handle.name)
        restored = o3.FullyConnectedTensorProduct("1e + 2e", "1e + 2e", "0e + 1e + 2e", compile_left_right=False)
        restored.load_weights(handle.name)
        assert _max_abs(product(left, right).array - restored(left, right).array) < 2e-6


@pytest.mark.mlx
def test_upstream_triangular_and_wrapper_tensor_products() -> None:
    mx = mlx_backend._require()
    triangular = o3.TensorProduct(
        "10x0e", "10x0e", "45x0e", [(0, 0, 0, "uvu<v", False)], compile_left_right=True
    )
    left = _array("10x0e", mx.arange(10, dtype=mx.float32)[None, :])
    right = _array("10x0e", (1 + mx.arange(10, dtype=mx.float32))[None, :])
    expected = [float(left.array[0, i] * right.array[0, j]) for i in range(10) for j in range(i + 1, 10)]
    assert triangular(left, right).array.tolist()[0] == pytest.approx(expected)

    for irreps1, irreps2 in [("15x0e", "5x0e + 5x1o + 5x1e"), ("2x0e + 1x1e", "2x0o + 1x1e")]:
        elementwise = o3.ElementwiseTensorProduct(irreps1, irreps2, compile_left_right=True)
        values1 = _array(elementwise.irreps_in1, mx.random.normal(shape=(5, elementwise.irreps_in1.dim)))
        values2 = _array(elementwise.irreps_in2, mx.random.normal(shape=(5, elementwise.irreps_in2.dim)))
        assert elementwise(values1, values2).shape == (5, elementwise.irreps_out.dim)

    for irreps1, irreps2 in [("0e", "2x0e"), ("0e + 1e", "2x0e + 3x1e")]:
        full = o3.FullTensorProduct(irreps1, irreps2, compile_left_right=True)
        values1 = _array(full.irreps_in1, mx.random.normal(shape=(5, full.irreps_in1.dim)))
        values2 = _array(full.irreps_in2, mx.random.normal(shape=(5, full.irreps_in2.dim)))
        assert full(values1, values2).shape == (5, full.irreps_out.dim)


@pytest.mark.mlx
def test_upstream_tensor_square_elasticity_output_irreps() -> None:
    first = o3.TensorSquare("1o", compile_left_right=False)
    second = o3.TensorSquare(first.irreps_out, compile_left_right=False)
    assert second.irreps_out.simplify() == o3.Irreps("2x0e + 2x2e + 4e")


@pytest.mark.mlx
def test_upstream_identity_module_and_fully_connected_split_normalization() -> None:
    mx = mlx_backend._require()
    identity = o3.Identity("1e + 2e + 3x3o", "1e + 2e + 3x3o")
    values = mx.random.normal(shape=(7, identity.irreps_in.dim))
    output = identity(_array(identity.irreps_in, values))
    assert output.array is values
    assert bool(mx.all(identity.output_mask))
    assert identity.parameters() == {}
    compiled = mx.compile(lambda raw: identity(_array(identity.irreps_in, raw)).array)
    assert _max_abs(compiled(values) - values) == 0.0
    with pytest.raises(ValueError, match="equal"):
        o3.Identity("0e", "0o")

    first = o3.FullyConnectedTensorProduct("10x0e", "10x0e", "0e", compile_left_right=False)
    split = o3.FullyConnectedTensorProduct("3x0e + 7x0e", "3x0e + 7x0e", "0e", compile_left_right=False)
    first.update({"weight": mx.ones(first.weight.shape)})
    split.update({"weight": mx.ones(split.weight.shape)})
    left = mx.random.normal(shape=(2, 3, 10))
    right = mx.random.normal(shape=(2, 3, 10))
    assert _max_abs(first(_array("10x0e", left), _array("10x0e", right)).array - split(_array(split.irreps_in1, left), _array(split.irreps_in2, right)).array) < 2e-6


@pytest.mark.mlx
@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_upstream_tensor_square_statistical_normalization(normalization: str) -> None:
    mx = mlx_backend._require()
    irreps = o3.Irreps("0e + 1e + 2e")
    square = o3.TensorSquare(irreps, irrep_normalization=normalization, compile_left_right=False)
    samples = 20_000
    chunks = []
    for part in irreps:
        chunk = mx.random.normal(shape=(samples, part.dim))
        if normalization == "norm":
            chunk = chunk / mx.sqrt(mx.sum(chunk * chunk, axis=-1, keepdims=True))
        chunks.append(chunk)
    inputs = _array(irreps, mx.concatenate(chunks, axis=-1))
    output = square(inputs)
    if normalization == "norm":
        squared_norms = o3.Norm(square.irreps_out, squared=True)(output).array
        means = mx.mean(squared_norms, axis=0)
    else:
        means = mx.mean(output.array**2, axis=0)
    assert float(mx.min(means)) > 0.8
    assert float(mx.max(means)) < 1.2
