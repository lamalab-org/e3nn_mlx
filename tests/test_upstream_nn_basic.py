"""MLX adaptations of upstream NN activation, dropout, and extraction tests."""

from __future__ import annotations

import copy

import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend


def _array(irreps, values):
    return e3nn.IrrepsArray(irreps, values)


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value)))


@pytest.mark.mlx
@pytest.mark.parametrize(
    "irreps_in,acts,expected",
    [
        ("256x0o", "abs", "256x0e"),
        ("37x0e", "tanh", "37x0e"),
        ("4x0e + 3x0o", "silu_abs", "4x0e + 3x0e"),
    ],
)
def test_upstream_activation_parity_compile_equivariance_and_normalization(irreps_in, acts, expected) -> None:
    mx = mlx_backend._require()
    functions = {
        "abs": [mx.abs],
        "tanh": [mx.tanh],
        "silu_abs": [lambda value: value * mx.sigmoid(value), mx.abs],
    }[acts]
    module = e3nn.Activation(irreps_in, functions)
    assert module.irreps_out == e3nn.Irreps(expected)
    values = mx.random.normal(shape=(20_000, module.irreps_in.dim))
    output = module(_array(module.irreps_in, values))
    assert output.shape == values.shape
    assert 0.94 < float(mx.mean(output.array**2)) < 1.06

    for chunk_in, chunk_out, activation in zip(
        _array(module.irreps_in, values).chunk_arrays(), output.chunk_arrays(), functions, strict=True
    ):
        ratio = chunk_out / activation(chunk_in)
        finite = mx.isfinite(ratio)
        selected = mx.where(finite, ratio, mx.zeros_like(ratio))
        reference = selected.reshape(-1)[mx.argmax(finite.reshape(-1))]
        assert _max_abs(mx.where(finite, selected - reference, mx.zeros_like(selected))) < 2e-5

    angles = e3nn.rand_angles()
    d_in = e3nn.irreps_wigner_d(module.irreps_in, *angles)
    d_out = e3nn.irreps_wigner_d(module.irreps_out, *angles)
    rotated = module(_array(module.irreps_in, values[:8] @ mx.swapaxes(d_in, -1, -2))).array
    expected_rotated = output.array[:8] @ mx.swapaxes(d_out, -1, -2)
    assert _max_abs(rotated - expected_rotated) < 3e-5
    compiled = mx.compile(lambda raw: module(_array(module.irreps_in, raw)).array)
    assert _max_abs(compiled(values[:8]) - output.array[:8]) < 2e-6
    assert module.parameters() == {}


@pytest.mark.mlx
def test_upstream_activation_validation() -> None:
    mx = mlx_backend._require()
    with pytest.raises(ValueError, match="counts differ"):
        e3nn.Activation("0e + 0o", [mx.tanh])
    with pytest.raises(ValueError, match="only be applied to scalar"):
        e3nn.Activation("1o", [mx.tanh])
    with pytest.raises(ValueError, match="even or odd"):
        e3nn.Activation("0o", [mx.sigmoid])


@pytest.mark.mlx
def test_upstream_equivariant_dropout_training_eval_compile_and_copy() -> None:
    mx = mlx_backend._require()
    module = e3nn.Dropout("10x1e + 10x0e", p=0.75)
    values = mx.random.normal(shape=(5, 2, module.irreps.dim))
    features = _array(module.irreps, values)

    module.eval()
    assert module(features) is features

    module.train()
    mx.random.seed(17)
    output = module(features)
    assert bool(mx.all((output.array == values / 0.25) | (output.array == 0)))
    for part, chunk in zip(module.irreps, output.chunk_arrays(), strict=True):
        block = chunk.reshape(5, 2, part.mul, part.ir.dim)
        if part.ir.dim > 1:
            zero_pattern = block == 0
            assert bool(mx.all(zero_pattern == zero_pattern[..., :1]))

    angles = e3nn.rand_angles()
    representation = e3nn.irreps_wigner_d(module.irreps, *angles)
    rotated_values = values @ mx.swapaxes(representation, -1, -2)
    mx.random.seed(123)
    actual = module(_array(module.irreps, rotated_values)).array
    mx.random.seed(123)
    expected = module(features).array @ mx.swapaxes(representation, -1, -2)
    assert _max_abs(actual - expected) < 3e-5
    assert isinstance(copy.deepcopy(module), e3nn.Dropout)


@pytest.mark.mlx
def test_upstream_dropout_boundary_probabilities() -> None:
    mx = mlx_backend._require()
    features = _array("0e + 1e", mx.ones((3, 4)))
    zero = e3nn.Dropout("0e + 1e", 0.0)
    one = e3nn.Dropout("0e + 1e", 1.0)
    zero.train()
    one.train()
    assert _max_abs(zero(features).array - features.array) == 0.0
    assert _max_abs(one(features).array) == 0.0


@pytest.mark.mlx
def test_upstream_extract_multiple_single_ir_and_compile_copy() -> None:
    mx = mlx_backend._require()
    values = mx.array([0.0, 0.0, 0.0, 1.0, 2.0])
    features = _array("1e + 0e + 0e", values)
    multiple = e3nn.Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])
    first, second = multiple(features)
    assert first.array.tolist() == [1.0]
    assert second.array.tolist() == [2.0]
    assert multiple.irreps_outs == (e3nn.Irreps("0e"), e3nn.Irreps("0e"))

    for squeeze in (True, False):
        single = e3nn.Extract("1e + 0e + 0e", ["0e"], [(1,)], squeeze_out=squeeze)
        output = single(features)
        selected = output if squeeze else output[0]
        assert selected.array.tolist() == [1.0]
        compiled = mx.compile(
            (lambda raw: single(_array(single.irreps_in, raw)).array)
            if squeeze
            else (lambda raw: single(_array(single.irreps_in, raw))[0].array)
        )
        compiled_output = compiled(values)
        assert compiled_output.tolist() == [1.0]

    extract_ir = e3nn.ExtractIr("1e + 0e + 0e", "0e")
    selected = extract_ir(features)
    assert selected.irreps == e3nn.Irreps("0e + 0e")
    assert selected.array.tolist() == [1.0, 2.0]
    assert isinstance(copy.deepcopy(multiple), e3nn.Extract)
    assert isinstance(copy.deepcopy(extract_ir), e3nn.ExtractIr)


@pytest.mark.mlx
def test_upstream_extract_validation() -> None:
    with pytest.raises(ValueError, match="one extraction instruction"):
        e3nn.Extract("0e", ["0e"], [])
    with pytest.raises(ValueError, match="does not match"):
        e3nn.Extract("0e + 1e", ["0e"], [(1,)])
