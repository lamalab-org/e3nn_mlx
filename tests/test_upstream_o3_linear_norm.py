"""MLX adaptations of upstream e3nn Linear and Norm tests."""

from __future__ import annotations

import pytest

import e3nn_mlx as o3
from e3nn_mlx.backend import mlx_backend


def _array(irreps, values):
    return o3.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
def test_upstream_linear_equivariance_compile_and_normalization() -> None:
    mx = mlx_backend._require()
    module = o3.Linear("1e + 2e + 3x3o", "1e + 2e + 3x3o", bias=False, compile=True)
    values = mx.random.normal(shape=(32, module.irreps_in.dim))
    inputs = _array(module.irreps_in, values)
    angles = o3.rand_angles()
    d_in = o3.irreps_wigner_d(module.irreps_in, *angles)
    d_out = o3.irreps_wigner_d(module.irreps_out, *angles)
    actual = module(_array(module.irreps_in, values @ mx.swapaxes(d_in, -1, -2))).array
    expected = module(inputs).array @ mx.swapaxes(d_out, -1, -2)
    assert _max_abs(actual - expected) < 2e-4
    assert module(inputs).shape == (32, module.irreps_out.dim)
    variance = mx.mean(module(inputs).array ** 2)
    assert 0.2 < float(variance) < 2.5


@pytest.mark.mlx
def test_upstream_linear_per_output_biases() -> None:
    mx = mlx_backend._require()
    module = o3.Linear(
        "2x0e + 1e + 2x0e",
        "3x0e + 1e + 3x0e + 5x0e",
        biases=[True, False, False, True],
    )
    module.update({"bias": mx.ones((8,))})
    output = module(_array(module.irreps_in, mx.zeros((module.irreps_in.dim,)))).array
    assert output.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.mlx
def test_upstream_linear_single_output_and_output_mask() -> None:
    mx = mlx_backend._require()
    first = o3.Linear("5x0e", "5x0e", bias=False)
    second = o3.Linear("5x0e", "5x0e + 3x0o", bias=False)
    second.update({"weight": first.weight})
    inputs = _array("5x0e", mx.random.normal(shape=(3, 5)))
    output1, output2 = first(inputs), second(inputs)
    assert _max_abs(output1.array - output2.array[:, :5]) < 1e-6
    assert _max_abs(output2.array[:, 5:]) == 0.0

    disconnected = o3.Linear("1e + 2e", "3e + 5x2o", bias=False)
    assert not bool(mx.any(disconnected.output_mask))


@pytest.mark.mlx
def test_upstream_linear_matches_equivalent_tensor_product() -> None:
    mx = mlx_backend._require()
    irreps_in = o3.Irreps("2x0e + 3x1e + 2e")
    irreps_out = o3.Irreps("4x0e + 2x1e + 2e")
    linear = o3.Linear(irreps_in, irreps_out, bias=False)
    instructions = [
        (instruction.i_in, 0, instruction.i_out, "uvw", True, 1.0)
        for instruction in linear.instructions
    ]
    product = o3.TensorProduct(
        irreps_in,
        "0e",
        irreps_out,
        instructions,
        internal_weights=False,
        compile_left_right=False,
    )
    values = mx.random.normal(shape=(4, irreps_in.dim))
    scalar = mx.ones((4, 1))
    expected = product(_array(irreps_in, values), _array("0e", scalar), weight=linear.weight)
    assert _max_abs(linear(_array(irreps_in, values)).array - expected.array) < 2e-6


@pytest.mark.mlx
def test_upstream_linear_instruction_validation_and_empty_paths() -> None:
    mx = mlx_backend._require()
    disconnected = o3.Linear("4x0e + 3x4o", "1x2e + 4x0o", bias=False)
    assert disconnected.instructions == ()
    assert not bool(mx.any(disconnected.output_mask))

    with pytest.raises(ValueError, match="identical irreps"):
        o3.Linear("4x0e + 3x4o", "1x2e + 4x0e", instructions=[(0, 0)])
    with pytest.raises(IndexError, match="out of range"):
        o3.Linear("4x0e + 3x4o", "1x2e + 4x0e", instructions=[(4, 0)])

    empty = o3.Linear(o3.Irreps.spherical_harmonics(3), o3.Irreps.spherical_harmonics(3), instructions=[], bias=False)
    values = mx.random.normal(shape=(3, empty.irreps_in.dim))
    assert _max_abs(empty(_array(empty.irreps_in, values)).array) == 0.0

    selected = o3.Linear("4x0e + 3x1o + 2x0e", "2x1o + 8x0e", instructions=[(0, 1), (1, 0)], bias=False)
    assert {(instruction.i_in, instruction.i_out) for instruction in selected.instructions} == {(0, 1), (1, 0)}
    assert {instruction.path_shape for instruction in selected.instructions} == {(4, 8), (3, 2)}


@pytest.mark.mlx
def test_upstream_linear_weight_views_and_unshared_weights() -> None:
    mx = mlx_backend._require()
    shared = o3.Linear("4x0e + 3x1o + 2x0e", "2x1o + 8x0e", instructions=[(0, 1), (1, 0)], bias=False)
    assert shared.weight_view_for_instruction(0).shape == (4, 8)
    assert shared.weight_view_for_instruction(1).shape == (3, 2)
    assert [view.shape for view in shared.weight_views()] == [(4, 8), (3, 2)]
    yielded = list(shared.weight_views(yield_instruction=True))
    assert [index for index, _, _ in yielded] == [0, 1]

    unshared = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
        instructions=[(0, 1), (1, 0)],
        bias=False,
        internal_weights=False,
        shared_weights=False,
    )
    weights = mx.random.normal(shape=(7, unshared.weight_numel))
    assert unshared.weight_view_for_instruction(0, weights).shape == (7, 4, 8)
    assert unshared.weight_view_for_instruction(1, weights).shape == (7, 3, 2)
    inputs = _array(unshared.irreps_in, mx.random.normal(shape=(7, unshared.irreps_in.dim)))
    assert unshared(inputs, weights).shape == (7, unshared.irreps_out.dim)


@pytest.mark.mlx
def test_upstream_linear_feature_channels() -> None:
    mx = mlx_backend._require()
    module = o3.Linear("0e + 1e + 2e", "0e + 2x1e + 2e", f_in=44, f_out=25, bias=False, compile=True)
    assert module.weight_numel == 4
    assert module.weight.shape == (44, 25, 4)
    values = mx.random.normal(shape=(10, 44, module.irreps_in.dim))
    output = module(_array(module.irreps_in, values))
    assert output.shape == (10, 25, module.irreps_out.dim)
    assert 0.6 < float(mx.mean(output.array**2)) < 1.5

    angles = o3.rand_angles()
    d_in = o3.irreps_wigner_d(module.irreps_in, *angles)
    d_out = o3.irreps_wigner_d(module.irreps_out, *angles)
    rotated = module(_array(module.irreps_in, values @ mx.swapaxes(d_in, -1, -2))).array
    expected = output.array @ mx.swapaxes(d_out, -1, -2)
    assert _max_abs(rotated - expected) < 3e-4


@pytest.mark.mlx
def test_upstream_linear_defaults_to_no_bias_and_external_unshared_weights() -> None:
    mx = mlx_backend._require()
    default = o3.Linear("0e", "0e")
    assert default.bias is None
    assert set(default.parameters()) == {"weight"}

    external = o3.Linear("2x0e", "3x0e", shared_weights=False, bias=False)
    assert not external.internal_weights and external.parameters() == {}
    values = mx.random.normal(shape=(5, 2))
    weights = mx.random.normal(shape=(5, external.weight_numel))
    assert external(_array("2x0e", values), weights).shape == (5, 3)


@pytest.mark.mlx
@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"])
@pytest.mark.parametrize("squared", [True, False])
def test_upstream_norm_equivariance_compile_and_empty(irreps_in: str, squared: bool) -> None:
    mx = mlx_backend._require()
    module = o3.Norm(irreps_in, squared=squared)
    values = mx.random.normal(shape=(4, module.irreps_in.dim))
    inputs = _array(module.irreps_in, values)
    output = module(inputs)
    assert output.irreps == module.irreps_out
    assert output.shape == (4, module.irreps_in.num_irreps)
    compiled = mx.compile(lambda raw: module(_array(module.irreps_in, raw)).array)
    compiled_output = compiled(values)
    if output.array.size:
        assert _max_abs(compiled_output - output.array) < 2e-6
    else:
        assert compiled_output.shape == output.array.shape
    if module.irreps_in.dim:
        angles = o3.rand_angles()
        transform = o3.irreps_wigner_d(module.irreps_in, *angles)
        rotated = module(_array(module.irreps_in, values @ mx.swapaxes(transform, -1, -2)))
        assert _max_abs(rotated.array - output.array) < 2e-4


@pytest.mark.mlx
@pytest.mark.parametrize("squared", [True, False])
def test_upstream_norm_zero_gradient_and_vector_values(squared: bool) -> None:
    mx = mlx_backend._require()
    irreps = o3.Irreps("2x0e + 3x0o")
    module = o3.Norm(irreps, squared=squared)
    gradient = mx.grad(lambda value: mx.sum(module(_array(irreps, value)).array))(mx.zeros((irreps.dim,)))
    mx.eval(gradient)
    assert _max_abs(gradient) == 0.0

    vectors = mx.random.normal(shape=(3, 10, 3))
    vector_module = o3.Norm("10x1o", squared=squared)
    actual = vector_module(_array("10x1o", vectors.reshape(3, -1))).array
    expected = mx.sum(vectors * vectors, axis=-1)
    if not squared:
        expected = mx.sqrt(expected)
    assert _max_abs(actual - expected) < 2e-6
