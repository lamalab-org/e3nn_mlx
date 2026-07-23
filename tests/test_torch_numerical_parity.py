"""Numerical parity with deterministic fixtures from pinned Torch/e3nn."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import e3nn_mlx as e3nn
from e3nn_mlx.backend import mlx_backend
from e3nn_mlx.irreps_array import IrrepsArray


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference_data"
MANIFEST = json.loads((REFERENCE / "torch_parity_manifest.json").read_text())


@pytest.fixture(scope="module")
def fixture():
    with np.load(REFERENCE / "torch_parity_reference.npz") as arrays:
        yield arrays


def _mx(array):
    mx = mlx_backend._require()
    return mx.array(np.asarray(array), dtype=mx.float32)


def _assert_close(actual, expected, *, atol: float = 3e-5, rtol: float = 3e-5):
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        atol=atol,
        rtol=rtol,
    )


def _loss(output):
    mx = mlx_backend._require()
    coefficients = mx.linspace(0.3, 1.1, output.size).reshape(output.shape)
    return mx.sum(mx.sin(output) * coefficients) + 0.07 * mx.sum(output**2)


def _gradient_and_hessian(
    function: Callable[[Any], Any],
    value,
):
    mx = mlx_backend._require()
    loss = lambda argument: _loss(function(argument))
    gradient_function = mx.grad(loss)
    gradient = gradient_function(value)
    rows = [
        mx.grad(
            lambda argument, index=index: gradient_function(argument)[index]
        )(value)
        for index in range(value.size)
    ]
    return loss(value), gradient, mx.stack(rows)


def _tensor_product(case):
    from e3nn_mlx.ops_tp import TensorProduct

    return TensorProduct(
        case["irreps_in1"],
        case["irreps_in2"],
        case["irreps_out"],
        [tuple(instruction) for instruction in case["instructions"]],
        internal_weights=False,
        shared_weights=case.get("shared_weights", True),
        use_custom_kernel=False,
    )


def _tp_output(module, left, right, weight):
    if weight is None:
        return module.differentiable_arrays(left, right)
    return module.differentiable_arrays(left, right, weight)


def test_torch_parity_generator_and_versions_are_pinned() -> None:
    generator = REFERENCE.parent / "reference_generation" / "generate_torch_parity.py"
    ast.parse(generator.read_text(), filename=str(generator))
    assert MANIFEST["schema"] == 1
    assert MANIFEST["e3nn"] == "0.5.8"
    assert MANIFEST["torch"] == "2.7.1"
    assert MANIFEST["numpy"] == "2.3.1"
    assert MANIFEST["dtype"] == "float64"


@pytest.mark.mlx
@pytest.mark.parametrize(
    "case", MANIFEST["tensor_products"], ids=lambda case: case["name"]
)
def test_every_torch_tensor_product_instruction_mode_matches_forward(
    fixture, case
) -> None:
    module = _tensor_product(case)
    prefix = f"tp.{case['name']}"
    left = _mx(fixture[f"{prefix}.left"])
    right = _mx(fixture[f"{prefix}.right"])
    weight = _mx(fixture[f"{prefix}.weight"]) if f"{prefix}.weight" in fixture else None
    actual = _tp_output(module, left, right, weight)
    _assert_close(actual, fixture[f"{prefix}.output"])


@pytest.mark.mlx
@pytest.mark.parametrize(
    "case",
    [case for case in MANIFEST["tensor_products"] if "derivative_splits" in case],
    ids=lambda case: case["name"],
)
def test_every_torch_tensor_product_mode_matches_gradients_hessians_and_mixed_blocks(
    fixture, case
) -> None:
    module = _tensor_product(case)
    prefix = f"tp.{case['name']}"
    packed = _mx(fixture[f"{prefix}.derivative_input"])
    split1, split2 = case["derivative_splits"]
    dim1 = split1
    dim2 = split2 - split1

    def output(value):
        left = value[:split1].reshape(1, dim1)
        right = value[split1:split2].reshape(1, dim2)
        weight = value[split2:] if case["weight_numel"] else None
        if weight is not None and not case.get("shared_weights", True):
            weight = weight.reshape(1, case["weight_numel"])
        return _tp_output(module, left, right, weight)

    loss, gradient, hessian = _gradient_and_hessian(output, packed)
    _assert_close(loss, fixture[f"{prefix}.loss"], atol=8e-5, rtol=8e-5)
    _assert_close(
        gradient, fixture[f"{prefix}.gradient"], atol=2e-4, rtol=2e-4
    )
    _assert_close(
        hessian, fixture[f"{prefix}.hessian"], atol=5e-4, rtol=5e-4
    )
    # Explicitly retain the mixed position/parameter block in the contract.
    if case["weight_numel"]:
        _assert_close(
            hessian[:split2, split2:],
            fixture[f"{prefix}.hessian"][:split2, split2:],
            atol=5e-4,
            rtol=5e-4,
        )


@pytest.mark.mlx
@pytest.mark.parametrize(
    "case", MANIFEST["tensor_products"], ids=lambda case: case["name"]
)
def test_torch_and_mlx_tensor_product_equivariance_errors_match(
    fixture, case
) -> None:
    mx = mlx_backend._require()
    module = _tensor_product(case)
    prefix = f"tp.{case['name']}"
    left = _mx(fixture[f"{prefix}.left"])
    right = _mx(fixture[f"{prefix}.right"])
    weight = _mx(fixture[f"{prefix}.weight"]) if f"{prefix}.weight" in fixture else None
    rotation = _mx(fixture["models.rotation"])
    d1 = e3nn.o3.irreps_wigner_d_from_matrix(module.irreps_in1, rotation)
    d2 = e3nn.o3.irreps_wigner_d_from_matrix(module.irreps_in2, rotation)
    dout = e3nn.o3.irreps_wigner_d_from_matrix(module.irreps_out, rotation)
    baseline = _tp_output(module, left, right, weight)
    transformed = _tp_output(
        module,
        left @ mx.swapaxes(d1, -1, -2),
        right @ mx.swapaxes(d2, -1, -2),
        weight,
    )
    error = mx.max(mx.abs(transformed - baseline @ mx.swapaxes(dout, -1, -2)))
    torch_error = float(fixture[f"{prefix}.equivariance_error"])
    assert float(error) < 4e-5
    assert abs(float(error) - torch_error) < 4e-5


@pytest.mark.mlx
@pytest.mark.parametrize(
    "case", MANIFEST["tensor_product_wrappers"], ids=lambda case: case["name"]
)
def test_torch_tensor_product_wrappers_match(fixture, case) -> None:
    name = case["name"]
    prefix = f"wrapper.{name}"
    left = _mx(fixture[f"{prefix}.left"])
    right = _mx(fixture[f"{prefix}.right"]) if f"{prefix}.right" in fixture else None
    weight = _mx(fixture[f"{prefix}.weight"]) if f"{prefix}.weight" in fixture else None
    if name == "full":
        module = e3nn.FullTensorProduct(case["irreps_in1"], case["irreps_in2"])
        actual = module(
            IrrepsArray(module.irreps_in1, left),
            IrrepsArray(module.irreps_in2, right),
        ).array
    elif name == "fully_connected":
        module = e3nn.FullyConnectedTensorProduct(
            case["irreps_in1"],
            case["irreps_in2"],
            case["irreps_out"],
            internal_weights=False,
        )
        actual = module(
            IrrepsArray(module.irreps_in1, left),
            IrrepsArray(module.irreps_in2, right),
            weight,
        ).array
    elif name == "elementwise":
        module = e3nn.ElementwiseTensorProduct(
            case["irreps_in1"], case["irreps_in2"]
        )
        actual = module(
            IrrepsArray(module.irreps_in1, left),
            IrrepsArray(module.irreps_in2, right),
        ).array
    else:
        module = e3nn.TensorSquare(case["irreps_in1"])
        actual = module(IrrepsArray(module.irreps_in, left)).array
    assert module.irreps_out == e3nn.Irreps(case["irreps_out"]).simplify()
    _assert_close(actual, fixture[f"{prefix}.output"])


def _module_case(name: str) -> dict[str, Any]:
    return next(case for case in MANIFEST["modules"] if case["name"] == name)


def _parameterless_module(name: str, case: dict[str, Any]):
    mx = mlx_backend._require()
    if name == "norm":
        return e3nn.Norm(case["irreps_in"])
    if name == "activation":
        return e3nn.Activation(case["irreps_in"], [mx.tanh, mx.abs])
    if name == "gate":
        return e3nn.Gate(
            "2x0e", [mx.tanh], "2x0e", [mx.sigmoid], "2x1o"
        )
    if name == "norm_activation":
        return e3nn.NormActivation(
            case["irreps_in"], mx.tanh, normalize=True, epsilon=1e-6
        )
    if name == "identity":
        return e3nn.Identity(case["irreps_in"], case["irreps_out"])
    if name == "extract":
        return e3nn.Extract(
            case["irreps_in"], ["1o", "2x0e"], [[1], [2]], squeeze_out=False
        )
    if name == "extract_ir":
        return e3nn.ExtractIr(case["irreps_in"], "0e")
    module = e3nn.Dropout(case["irreps_in"], p=0.35)
    module.eval()
    return module


@pytest.mark.mlx
def test_torch_linear_forward_input_parameter_gradient_hessian_and_mixed_derivatives(
    fixture,
) -> None:
    mx = mlx_backend._require()
    case = _module_case("linear")
    module = e3nn.Linear(
        case["irreps_in"],
        case["irreps_out"],
        internal_weights=False,
        bias=True,
    )
    prefix = "module.linear"
    values = _mx(fixture[f"{prefix}.input"])
    weight = _mx(fixture[f"{prefix}.weight"])
    bias = _mx(fixture[f"{prefix}.bias"])
    actual = module(IrrepsArray(module.irreps_in, values), weight, bias=bias).array
    _assert_close(actual, fixture[f"{prefix}.output"])

    packed = _mx(fixture[f"{prefix}.derivative_input"])
    split_x, split_w = case["derivative_splits"]

    def output(value):
        return module(
            IrrepsArray(module.irreps_in, value[:split_x].reshape(1, split_x)),
            value[split_x:split_w],
            bias=value[split_w:],
        ).array

    loss, gradient, hessian = _gradient_and_hessian(output, packed)
    _assert_close(loss, fixture[f"{prefix}.loss"], atol=8e-5, rtol=8e-5)
    _assert_close(
        gradient, fixture[f"{prefix}.gradient"], atol=2e-4, rtol=2e-4
    )
    _assert_close(
        hessian, fixture[f"{prefix}.hessian"], atol=5e-4, rtol=5e-4
    )

    rotation = _mx(fixture["models.rotation"])
    din = e3nn.o3.irreps_wigner_d_from_matrix(module.irreps_in, rotation)
    dout = e3nn.o3.irreps_wigner_d_from_matrix(module.irreps_out, rotation)
    rotated = module(
        IrrepsArray(module.irreps_in, values @ mx.swapaxes(din, -1, -2)),
        weight,
        bias=bias,
    ).array
    error = mx.max(mx.abs(rotated - actual @ mx.swapaxes(dout, -1, -2)))
    assert float(error) < 4e-5
    assert abs(float(error) - float(fixture[f"{prefix}.equivariance_error"])) < 4e-5


@pytest.mark.mlx
@pytest.mark.parametrize(
    "name",
    [
        "norm",
        "activation",
        "gate",
        "norm_activation",
        "identity",
        "extract",
        "extract_ir",
        "dropout_eval",
    ],
)
def test_torch_parameterless_nn_modules_match(fixture, name: str) -> None:
    case = _module_case(name)
    values = _mx(fixture[f"module.{name}.input"])
    module = _parameterless_module(name, case)
    output = module(IrrepsArray(module.irreps_in, values))
    tolerance = 6e-4 if name in {"activation", "gate"} else 5e-5
    if isinstance(output, tuple):
        for index, item in enumerate(output):
            _assert_close(
                item.array,
                fixture[f"module.{name}.output.{index}"],
                atol=tolerance,
                rtol=tolerance,
            )
    else:
        _assert_close(
            output.array,
            fixture[f"module.{name}.output"],
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.mlx
@pytest.mark.parametrize("name", ["norm", "activation", "gate", "norm_activation"])
def test_torch_parameterless_nn_input_gradients_and_hessians_match(
    fixture, name: str
) -> None:
    case = _module_case(name)
    module = _parameterless_module(name, case)
    packed = _mx(fixture[f"module.{name}.derivative_input"])

    def output(value):
        result = module(
            IrrepsArray(module.irreps_in, value.reshape(1, value.size))
        )
        return result.array

    loss, gradient, hessian = _gradient_and_hessian(output, packed)
    prefix = f"module.{name}"
    tolerance = 8e-4 if name in {"activation", "gate"} else 5e-4
    _assert_close(loss, fixture[f"{prefix}.loss"], atol=tolerance, rtol=tolerance)
    _assert_close(
        gradient, fixture[f"{prefix}.gradient"], atol=tolerance, rtol=tolerance
    )
    _assert_close(
        hessian, fixture[f"{prefix}.hessian"], atol=tolerance, rtol=tolerance
    )


@pytest.mark.mlx
@pytest.mark.parametrize("name", ["s2_activation", "so3_activation"])
def test_torch_spherical_grid_activation_modules_match(fixture, name: str) -> None:
    mx = mlx_backend._require()
    case = _module_case(name)
    if name == "s2_activation":
        module = e3nn.S2Activation(
            case["irreps_in"],
            mx.tanh,
            case["resolution"],
            normalization="component",
            lmax_out=case["lmax_out"],
            random_rot=False,
        )
    else:
        module = e3nn.SO3Activation(
            case["lmax_in"],
            case["lmax_out"],
            mx.tanh,
            case["resolution"],
        )
    actual = module(_mx(fixture[f"module.{name}.input"]))
    _assert_close(
        actual,
        fixture[f"module.{name}.output"],
        atol=8e-4,
        rtol=8e-4,
    )


@pytest.mark.mlx
def test_torch_batch_norm_state_and_output_match(fixture) -> None:
    module = e3nn.BatchNorm("2x0e+2x1o")
    module.eval()
    module.weight = _mx(fixture["module.batch_norm.weight"])
    module.bias = _mx(fixture["module.batch_norm.bias"])
    module._running_mean = _mx(fixture["module.batch_norm.running_mean"])
    module._running_var = _mx(fixture["module.batch_norm.running_var"])
    values = _mx(fixture["module.batch_norm.input"])
    actual = module(IrrepsArray(module.irreps_in, values)).array
    _assert_close(actual, fixture["module.batch_norm.output"], atol=5e-5, rtol=5e-5)


def _load_parameter_tree(module, fixture, prefix: str, metadata) -> None:
    from mlx.utils import tree_unflatten

    parameters = [
        (entry["name"], _mx(fixture[f"{prefix}.parameter.{entry['name']}"]))
        for entry in metadata
    ]
    module.update(tree_unflatten(parameters))


@pytest.mark.mlx
def test_torch_fully_connected_net_parameters_and_output_match(fixture) -> None:
    mx = mlx_backend._require()
    case = _module_case("fully_connected_net")
    module = e3nn.FullyConnectedNet(case["widths"], mx.tanh, out_act=True)
    _load_parameter_tree(module, fixture, "module.fully_connected_net", case["parameters"])
    actual = module(_mx(fixture["module.fully_connected_net.input"]))
    _assert_close(actual, fixture["module.fully_connected_net.output"], atol=6e-5, rtol=6e-5)


def _model_case(name: str):
    return next(case for case in MANIFEST["models"] if case["name"] == name)


@pytest.mark.mlx
@pytest.mark.parametrize("name", ["gate_points", "v2106"])
def test_torch_and_mlx_models_match_identical_weights_outputs_and_equivariance_errors(
    fixture, name: str
) -> None:
    mx = mlx_backend._require()
    case = _model_case(name)
    if name == "gate_points":
        module = e3nn.GatePointsNetwork(
            "0e+1o",
            "2x0e+2x1o",
            "0e+1o",
            "0e",
            e3nn.Irreps.spherical_harmonics(1),
            layers=1,
            max_radius=2.5,
            number_of_basis=3,
            radial_layers=1,
            radial_neurons=4,
            num_neighbors=2.0,
            num_nodes=4.0,
            reduce_output=False,
        )
    else:
        module = e3nn.NetworkForAGraphWithAttributes(
            "0e+1o",
            "0e",
            "0e",
            "0e+1o",
            max_radius=2.5,
            num_neighbors=2.0,
            num_nodes=4.0,
            mul=2,
            layers=1,
            lmax=1,
            pool_nodes=False,
        )
    _load_parameter_tree(module, fixture, f"model.{name}", case["parameters"])
    prefix = f"model.{name}"
    positions = _mx(fixture[f"{prefix}.positions"])
    node_input = _mx(fixture[f"{prefix}.node_input"])
    node_attr = _mx(fixture[f"{prefix}.node_attr"])
    batch = mx.array(np.asarray(fixture[f"{prefix}.batch"]), dtype=mx.int32)
    edge_src = mx.array(np.asarray(fixture[f"{prefix}.edge_src"]), dtype=mx.int32)
    edge_dst = mx.array(np.asarray(fixture[f"{prefix}.edge_dst"]), dtype=mx.int32)

    if name == "gate_points":
        def forward(pos, features):
            return module.forward_with_edges(
                pos,
                features,
                node_attr,
                batch,
                edge_src,
                edge_dst,
                num_graphs=1,
            ).array
    else:
        edge_attr = _mx(fixture[f"{prefix}.edge_attr"])

        def forward(pos, features):
            return module.forward_with_edges(
                pos,
                features,
                node_attr,
                edge_attr,
                batch,
                edge_src,
                edge_dst,
                num_graphs=1,
            ).array

    output = forward(positions, node_input)
    _assert_close(output, fixture[f"{prefix}.output"], atol=2e-4, rtol=2e-4)
    rotation = _mx(fixture["models.rotation"])
    din = e3nn.o3.irreps_wigner_d_from_matrix(case["irreps_in"], rotation)
    dout = e3nn.o3.irreps_wigner_d_from_matrix(case["irreps_out"], rotation)
    rotated = forward(
        positions @ mx.swapaxes(rotation, -1, -2),
        node_input @ mx.swapaxes(din, -1, -2),
    )
    _assert_close(
        rotated, fixture[f"{prefix}.rotated_output"], atol=3e-4, rtol=3e-4
    )
    error = mx.max(mx.abs(rotated - output @ mx.swapaxes(dout, -1, -2)))
    torch_error = float(fixture[f"{prefix}.equivariance_error"])
    assert float(error) < 5e-4
    assert abs(float(error) - torch_error) < 5e-4
