"""Compatibility tests for the e3nn-shaped high-level MLX API."""

from __future__ import annotations

import pytest

import e3nn_mlx
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray


def _max_abs(value) -> float:
    mx = mlx_backend._require()
    return float(mx.max(mx.abs(value))) if value.size else 0.0


@pytest.mark.mlx
def test_e3nn_style_namespaces_and_rotation_aliases() -> None:
    from e3nn_mlx import math, nn, o3
    from e3nn_mlx.nn.models.v2106 import Compose, SimpleNetwork

    assert e3nn_mlx.o3 is o3
    assert e3nn_mlx.nn is nn
    assert e3nn_mlx.math is math
    assert o3.wigner_D is o3.wigner_d
    assert e3nn_mlx.wigner_D is e3nn_mlx.wigner_d
    assert o3.Instruction is e3nn_mlx.Instruction
    assert SimpleNetwork.__module__ == "e3nn_mlx.nn.models.v2106.gate_points_networks"
    assert callable(Compose.forward)


@pytest.mark.mlx
def test_o3_linear_raw_and_typed_calls_are_exact_and_compile() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3

    module = o3.Linear("2x0e + 2x1o", "3x0e + 2x1o", bias=True)
    values = mx.random.normal(shape=(64, module.irreps_in.dim))
    raw = module(values)
    typed = module(IrrepsArray(module.irreps_in, values))
    assert isinstance(raw, mx.array)
    assert isinstance(typed, IrrepsArray)
    assert _max_abs(raw - typed.array) == 0.0
    assert _max_abs(module.forward(values) - raw) == 0.0

    compiled = mx.compile(module)
    assert _max_abs(compiled(values) - raw) < 2e-6
    gradient = mx.grad(lambda array: mx.mean(module(array) ** 2))(values)
    assert gradient.shape == values.shape
    assert bool(mx.all(mx.isfinite(gradient)))


@pytest.mark.mlx
@pytest.mark.parametrize(
    "factory,arity",
    [
        (lambda o3: o3.FullTensorProduct("1o", "1o"), 2),
        (lambda o3: o3.FullyConnectedTensorProduct("1o", "1o", "0e + 1e + 2e"), 2),
        (lambda o3: o3.ElementwiseTensorProduct("2x1o", "2x1o"), 2),
        (lambda o3: o3.TensorSquare("1o"), 1),
    ],
)
def test_o3_tensor_products_preserve_input_style(factory, arity) -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3

    module = factory(o3)
    if arity == 1:
        raw_inputs = [mx.random.normal(shape=(16, module.irreps_in.dim))]
        typed_inputs = [IrrepsArray(module.irreps_in, raw_inputs[0])]
    else:
        raw_inputs = [
            mx.random.normal(shape=(16, module.irreps_in1.dim)),
            mx.random.normal(shape=(16, module.irreps_in2.dim)),
        ]
        typed_inputs = [
            IrrepsArray(module.irreps_in1, raw_inputs[0]),
            IrrepsArray(module.irreps_in2, raw_inputs[1]),
        ]
    raw = module(*raw_inputs)
    typed = module(*typed_inputs)
    assert isinstance(raw, mx.array)
    assert isinstance(typed, IrrepsArray)
    assert _max_abs(raw - typed.array) < 2e-6
    assert _max_abs(module.forward(*raw_inputs) - raw) < 2e-6


@pytest.mark.mlx
def test_o3_norm_reduced_tensor_products_and_spherical_harmonics_raw_api() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import o3

    values = mx.random.normal(shape=(12, 4))
    norm = o3.Norm("0e + 1o")
    assert norm(values).shape == (12, 2)
    assert isinstance(norm(IrrepsArray("0e + 1o", values)), IrrepsArray)

    reduced = o3.ReducedTensorProducts("ij=ji", i="1o")
    vectors = mx.random.normal(shape=(12, 3))
    raw = reduced(vectors, vectors)
    typed = reduced(IrrepsArray("1o", vectors), IrrepsArray("1o", vectors))
    assert _max_abs(raw - typed.array) < 2e-6

    harmonics = o3.SphericalHarmonics([0, 1, 2], True, "component")
    output = harmonics(vectors)
    assert output.shape == (12, 9)
    assert _max_abs(harmonics.forward(vectors) - output) == 0.0


@pytest.mark.mlx
def test_nn_modules_accept_raw_arrays_without_changing_typed_behavior() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx import nn

    activation = nn.Activation("2x0e", [mx.tanh])
    values = mx.random.normal(shape=(32, 2))
    assert _max_abs(
        activation(values)
        - activation(IrrepsArray(activation.irreps_in, values)).array
    ) == 0.0

    gate = nn.Gate("0e", [mx.tanh], "0e", [mx.sigmoid], "1o")
    gate_values = mx.random.normal(shape=(32, gate.irreps_in.dim))
    assert _max_abs(
        gate(gate_values) - gate(IrrepsArray(gate.irreps_in, gate_values)).array
    ) < 2e-6

    batch_norm = nn.BatchNorm("0e + 1o")
    batch_norm.eval()
    normalized = batch_norm(mx.random.normal(shape=(8, 4)))
    assert normalized.shape == (8, 4)

    extraction = nn.ExtractIr("0e + 1o + 2x0e", "0e")
    extracted = extraction(mx.random.normal(shape=(8, 6)))
    assert extracted.shape == (8, 3)


@pytest.mark.mlx
def test_v2106_compatibility_convolution_matches_typed_core_exactly() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx.models.v2106 import Convolution as CoreConvolution
    from e3nn_mlx.nn.models.v2106.points_convolution import Convolution

    module = Convolution("0e + 1o", "0e", "0e + 1o", "0e + 1o", [3, 8], 2.0)
    core = CoreConvolution("0e + 1o", "0e", "0e + 1o", "0e + 1o", [3, 8], 2.0)
    core.update(module.parameters())
    edge_src = mx.array([0, 1, 2, 0], dtype=mx.int32)
    edge_dst = mx.array([1, 2, 0, 2], dtype=mx.int32)
    node_input = mx.random.normal(shape=(3, 4))
    node_attr = mx.ones((3, 1))
    edge_attr = mx.random.normal(shape=(4, 4))
    edge_scalars = mx.random.normal(shape=(4, 3))
    arguments = node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    raw = module(*arguments)
    expected = core.forward_arrays(*arguments)
    assert isinstance(raw, mx.array)
    assert _max_abs(raw - expected) == 0.0
    assert _max_abs(module.forward(*arguments) - expected) == 0.0
    compiled = mx.compile(module.forward)
    assert _max_abs(compiled(*arguments) - expected) < 3e-5


@pytest.mark.mlx
def test_v2106_network_compatibility_path_returns_raw_but_legacy_path_stays_typed() -> None:
    mx = mlx_backend._require()
    from e3nn_mlx.models.v2106 import SimpleNetwork as TypedNetwork
    from e3nn_mlx.nn.models.v2106 import SimpleNetwork

    compatibility = SimpleNetwork(
        "0e + 1o", "0e", 2.0, 3.0, 4.0, mul=2, layers=1, lmax=1
    )
    typed = TypedNetwork(
        "0e + 1o", "0e", 2.0, 3.0, 4.0, mul=2, layers=1, lmax=1
    )
    typed.update(compatibility.parameters())
    data = {
        "pos": mx.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.7]]
        ),
        "x": mx.random.normal(shape=(4, 4)),
    }
    raw = compatibility(data)
    legacy = typed(data)
    assert isinstance(raw, mx.array)
    assert isinstance(legacy, IrrepsArray)
    assert _max_abs(raw - legacy.array) < 2e-6

    typed_data = dict(data)
    typed_data["x"] = IrrepsArray(compatibility.irreps_in, data["x"])
    typed_result = compatibility(typed_data)
    assert isinstance(typed_result, IrrepsArray)
    assert _max_abs(typed_result.array - raw) < 2e-6
