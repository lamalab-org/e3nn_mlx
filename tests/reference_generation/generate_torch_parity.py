"""Generate deterministic Torch/e3nn parity fixtures for the MLX port.

Run this script only in the pinned reference environment.  The normal test
suite consumes the resulting NPZ/JSON files and does not import Torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import e3nn
from e3nn import nn, o3
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reference_data"
SEED = 20260723
DTYPE = torch.float64


TP_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "uvw",
        "irreps_in1": "1o",
        "irreps_in2": "1o",
        "irreps_out": "2x0e+2x1e+2x2e",
        "instructions": (
            (0, 0, 0, "uvw", True),
            (0, 0, 1, "uvw", True),
            (0, 0, 2, "uvw", True),
        ),
    },
    {
        "name": "uvu",
        "irreps_in1": "2x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "2x0e",
        "instructions": ((0, 0, 0, "uvu", True),),
    },
    {
        "name": "uvu_grouped",
        "irreps_in1": "2x1o",
        "irreps_in2": "3x1o",
        "irreps_out": "2x0e+2x1e+2x2e",
        "instructions": (
            (0, 0, 0, "uvu", True),
            (0, 0, 1, "uvu", True),
            (0, 0, 2, "uvu", True),
        ),
    },
    {
        "name": "uvv",
        "irreps_in1": "2x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "3x0e",
        "instructions": ((0, 0, 0, "uvv", True),),
    },
    {
        "name": "uuw",
        "irreps_in1": "2x0e",
        "irreps_in2": "2x0e",
        "irreps_out": "2x0e",
        "instructions": ((0, 0, 0, "uuw", True),),
    },
    {
        "name": "uuu",
        "irreps_in1": "2x0e",
        "irreps_in2": "2x0e",
        "irreps_out": "2x0e",
        "instructions": ((0, 0, 0, "uuu", True),),
    },
    {
        "name": "uvuv",
        "irreps_in1": "2x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "6x0e",
        "instructions": ((0, 0, 0, "uvuv", True),),
    },
    {
        "name": "uvu_lt_v",
        "irreps_in1": "3x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "3x0e",
        "instructions": ((0, 0, 0, "uvu<v", True),),
    },
    {
        "name": "u_lt_vw",
        "irreps_in1": "3x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "2x0e",
        "instructions": ((0, 0, 0, "u<vw", True),),
    },
)


def _array(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _deterministic(shape: tuple[int, ...], offset: int, *, scale: float = 0.2) -> torch.Tensor:
    size = int(np.prod(shape))
    values = torch.arange(offset, offset + size, dtype=DTYPE)
    return (scale * torch.sin(values + 0.37)).reshape(shape)


def _store_parameters(
    arrays: dict[str, np.ndarray],
    prefix: str,
    module: torch.nn.Module,
) -> list[dict[str, Any]]:
    metadata = []
    offset = 1
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = _deterministic(tuple(parameter.shape), offset)
            parameter.copy_(value)
            arrays[f"{prefix}.parameter.{name}"] = _array(value)
            metadata.append({"name": name, "shape": list(parameter.shape)})
            offset += parameter.numel() + 3
    return metadata


def _derivatives(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variable = value.detach().clone().requires_grad_(True)

    def loss(argument: torch.Tensor) -> torch.Tensor:
        output = function(argument)
        coefficients = torch.linspace(
            0.3, 1.1, output.numel(), dtype=output.dtype, device=output.device
        ).reshape(output.shape)
        return torch.sum(torch.sin(output) * coefficients) + 0.07 * torch.sum(output**2)

    loss_value = loss(variable)
    (gradient,) = torch.autograd.grad(loss_value, (variable,), create_graph=True)
    hessian = torch.autograd.functional.hessian(loss, variable)
    return _array(loss_value), _array(gradient), _array(hessian)


def _tp_fixtures(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    cases = []
    for case_index, raw in enumerate(TP_CASES):
        case = dict(raw)
        case["instructions"] = [list(value) for value in raw["instructions"]]
        prefix = f"tp.{case['name']}"
        module = o3.TensorProduct(
            case["irreps_in1"],
            case["irreps_in2"],
            case["irreps_out"],
            raw["instructions"],
            internal_weights=False,
            shared_weights=True,
            irrep_normalization="component",
            path_normalization="element",
            _specialized_code=False,
        )
        dim1 = module.irreps_in1.dim
        dim2 = module.irreps_in2.dim
        left = _deterministic((2, dim1), 11 + 29 * case_index, scale=0.7)
        right = _deterministic((2, dim2), 17 + 31 * case_index, scale=0.6)
        weight = (
            _deterministic((module.weight_numel,), 23 + 37 * case_index, scale=0.5)
            if module.weight_numel
            else None
        )
        output = module(left, right, weight) if weight is not None else module(left, right)
        arrays[f"{prefix}.left"] = _array(left)
        arrays[f"{prefix}.right"] = _array(right)
        if weight is not None:
            arrays[f"{prefix}.weight"] = _array(weight)
        arrays[f"{prefix}.output"] = _array(output)
        rotation = _fixed_rotation()
        d1 = module.irreps_in1.D_from_matrix(rotation)
        d2 = module.irreps_in2.D_from_matrix(rotation)
        dout = module.irreps_out.D_from_matrix(rotation)
        rotated_output = (
            module(left @ d1.T, right @ d2.T, weight)
            if weight is not None
            else module(left @ d1.T, right @ d2.T)
        )
        arrays[f"{prefix}.equivariance_error"] = _array(
            torch.max(torch.abs(rotated_output - output @ dout.T))
        )

        derivative_left = left[:1]
        derivative_right = right[:1]
        pieces = [derivative_left.reshape(-1), derivative_right.reshape(-1)]
        if weight is not None:
            pieces.append(weight)
        packed = torch.cat(pieces)
        split1 = derivative_left.numel()
        split2 = split1 + derivative_right.numel()

        def packed_output(value: torch.Tensor) -> torch.Tensor:
            local_left = value[:split1].reshape_as(derivative_left)
            local_right = value[split1:split2].reshape_as(derivative_right)
            local_weight = value[split2:] if weight is not None else None
            return (
                module(local_left, local_right, local_weight)
                if local_weight is not None
                else module(local_left, local_right)
            )

        loss, gradient, hessian = _derivatives(packed_output, packed)
        arrays[f"{prefix}.derivative_input"] = _array(packed)
        arrays[f"{prefix}.loss"] = loss
        arrays[f"{prefix}.gradient"] = gradient
        arrays[f"{prefix}.hessian"] = hessian
        case["weight_numel"] = module.weight_numel
        case["derivative_splits"] = [split1, split2]
        cases.append(case)

    # Explicitly exercise unshared per-sample weights.
    case = {
        "name": "uvu_unshared",
        "irreps_in1": "2x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "2x0e",
        "instructions": [(0, 0, 0, "uvu", True)],
        "shared_weights": False,
    }
    module = o3.TensorProduct(
        case["irreps_in1"],
        case["irreps_in2"],
        case["irreps_out"],
        case["instructions"],
        internal_weights=False,
        shared_weights=False,
        _specialized_code=False,
    )
    left = _deterministic((2, module.irreps_in1.dim), 401, scale=0.7)
    right = _deterministic((2, module.irreps_in2.dim), 421, scale=0.6)
    weight = _deterministic((2, module.weight_numel), 443, scale=0.5)
    prefix = "tp.uvu_unshared"
    arrays[f"{prefix}.left"] = _array(left)
    arrays[f"{prefix}.right"] = _array(right)
    arrays[f"{prefix}.weight"] = _array(weight)
    output = module(left, right, weight)
    arrays[f"{prefix}.output"] = _array(output)
    rotation = _fixed_rotation()
    d1 = module.irreps_in1.D_from_matrix(rotation)
    d2 = module.irreps_in2.D_from_matrix(rotation)
    dout = module.irreps_out.D_from_matrix(rotation)
    rotated_output = module(left @ d1.T, right @ d2.T, weight)
    arrays[f"{prefix}.equivariance_error"] = _array(
        torch.max(torch.abs(rotated_output - output @ dout.T))
    )

    derivative_left = left[:1]
    derivative_right = right[:1]
    derivative_weight = weight[:1]
    split1 = derivative_left.numel()
    split2 = split1 + derivative_right.numel()
    packed = torch.cat(
        [
            derivative_left.reshape(-1),
            derivative_right.reshape(-1),
            derivative_weight.reshape(-1),
        ]
    )

    def packed_output(value: torch.Tensor) -> torch.Tensor:
        return module(
            value[:split1].reshape_as(derivative_left),
            value[split1:split2].reshape_as(derivative_right),
            value[split2:].reshape_as(derivative_weight),
        )

    loss, gradient, hessian = _derivatives(packed_output, packed)
    arrays[f"{prefix}.derivative_input"] = _array(packed)
    arrays[f"{prefix}.loss"] = loss
    arrays[f"{prefix}.gradient"] = gradient
    arrays[f"{prefix}.hessian"] = hessian
    case["weight_numel"] = module.weight_numel
    case["derivative_splits"] = [split1, split2]
    cases.append(case)
    manifest["tensor_products"] = cases


def _wrapper_fixtures(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    cases = []

    def record(name: str, module, left, right=None, weight=None) -> None:
        prefix = f"wrapper.{name}"
        arrays[f"{prefix}.left"] = _array(left)
        if right is not None:
            arrays[f"{prefix}.right"] = _array(right)
        if weight is not None:
            arrays[f"{prefix}.weight"] = _array(weight)
        if right is None:
            output = module(left, weight) if weight is not None else module(left)
        else:
            output = module(left, right, weight) if weight is not None else module(left, right)
        arrays[f"{prefix}.output"] = _array(output)
        cases.append(
            {
                "name": name,
                "class": type(module).__name__,
                "irreps_in1": str(getattr(module, "irreps_in1", getattr(module, "irreps_in", ""))),
                "irreps_in2": str(getattr(module, "irreps_in2", "")),
                "irreps_out": str(module.irreps_out),
                "weight_numel": int(getattr(module, "weight_numel", 0)),
            }
        )

    full = o3.FullTensorProduct("2x1o", "3x1o")
    record(
        "full",
        full,
        _deterministic((2, full.irreps_in1.dim), 501),
        _deterministic((2, full.irreps_in2.dim), 541),
    )
    connected = o3.FullyConnectedTensorProduct(
        "2x0e+2x1o",
        "1x0e+1x1o",
        "3x0e+2x1e+2x2e",
        internal_weights=False,
        shared_weights=True,
    )
    record(
        "fully_connected",
        connected,
        _deterministic((2, connected.irreps_in1.dim), 601),
        _deterministic((2, connected.irreps_in2.dim), 641),
        _deterministic((connected.weight_numel,), 677),
    )
    elementwise = o3.ElementwiseTensorProduct("2x1o", "2x1o")
    record(
        "elementwise",
        elementwise,
        _deterministic((2, elementwise.irreps_in1.dim), 701),
        _deterministic((2, elementwise.irreps_in2.dim), 733),
    )
    square = o3.TensorSquare("2x1o")
    record("tensor_square", square, _deterministic((2, square.irreps_in.dim), 773))
    manifest["tensor_product_wrappers"] = cases


def _module_fixtures(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    cases: list[dict[str, Any]] = []

    # Linear includes both input/parameter first derivatives and all mixed
    # second-derivative blocks through one packed Hessian.
    linear = o3.Linear(
        "2x0e+2x1o",
        "3x0e+2x1o",
        internal_weights=False,
        shared_weights=True,
        biases=True,
    )
    x = _deterministic((3, linear.irreps_in.dim), 801)
    weight = _deterministic((linear.weight_numel,), 851)
    bias = _deterministic((linear.bias_numel,), 887)
    output = linear(x, weight, bias)
    prefix = "module.linear"
    arrays[f"{prefix}.input"] = _array(x)
    arrays[f"{prefix}.weight"] = _array(weight)
    arrays[f"{prefix}.bias"] = _array(bias)
    arrays[f"{prefix}.output"] = _array(output)
    rotation = _fixed_rotation()
    d_in = linear.irreps_in.D_from_matrix(rotation)
    d_out = linear.irreps_out.D_from_matrix(rotation)
    rotated_output = linear(x @ d_in.T, weight, bias)
    arrays[f"{prefix}.equivariance_error"] = _array(
        torch.max(torch.abs(rotated_output - output @ d_out.T))
    )
    packed = torch.cat([x[:1].reshape(-1), weight, bias])
    split_x = x.shape[1]
    split_w = split_x + weight.numel()

    def linear_output(value: torch.Tensor) -> torch.Tensor:
        return linear(
            value[:split_x].reshape(1, -1),
            value[split_x:split_w],
            value[split_w:],
        )

    loss, gradient, hessian = _derivatives(linear_output, packed)
    arrays[f"{prefix}.derivative_input"] = _array(packed)
    arrays[f"{prefix}.loss"] = loss
    arrays[f"{prefix}.gradient"] = gradient
    arrays[f"{prefix}.hessian"] = hessian
    cases.append(
        {
            "name": "linear",
            "irreps_in": str(linear.irreps_in),
            "irreps_out": str(linear.irreps_out),
            "weight_numel": linear.weight_numel,
            "bias_numel": linear.bias_numel,
            "derivative_splits": [split_x, split_w],
        }
    )

    def record_parameterless(
        name: str,
        module,
        value: torch.Tensor,
        *,
        derivatives: bool = False,
    ) -> None:
        prefix = f"module.{name}"
        arrays[f"{prefix}.input"] = _array(value)
        output = module(value)
        if isinstance(output, tuple):
            for index, item in enumerate(output):
                arrays[f"{prefix}.output.{index}"] = _array(item)
            output_count = len(output)
        else:
            arrays[f"{prefix}.output"] = _array(output)
            output_count = 1
        if derivatives:
            derivative_input = value[:1].reshape(-1)
            loss, gradient, hessian = _derivatives(
                lambda packed: module(packed.reshape(1, -1)),
                derivative_input,
            )
            arrays[f"{prefix}.derivative_input"] = _array(derivative_input)
            arrays[f"{prefix}.loss"] = loss
            arrays[f"{prefix}.gradient"] = gradient
            arrays[f"{prefix}.hessian"] = hessian
        irreps_in = getattr(module, "irreps_in", None)
        if irreps_in is None:
            irreps_in = module.irreps
        irreps_out = getattr(module, "irreps_out", None)
        if irreps_out is None:
            irreps_out = getattr(module, "irreps", "")
        cases.append(
            {
                "name": name,
                "irreps_in": str(irreps_in),
                "irreps_out": str(irreps_out),
                "output_count": output_count,
                "derivatives": derivatives,
            }
        )

    record_parameterless(
        "norm",
        o3.Norm("2x0e+2x1o"),
        _deterministic((4, 8), 901),
        derivatives=True,
    )
    record_parameterless(
        "activation",
        nn.Activation("2x0e+1x0o", [torch.tanh, torch.abs]),
        _deterministic((4, 3), 941),
        derivatives=True,
    )
    record_parameterless(
        "gate",
        nn.Gate("2x0e", [torch.tanh], "2x0e", [torch.sigmoid], "2x1o"),
        _deterministic((4, 10), 977),
        derivatives=True,
    )
    record_parameterless(
        "norm_activation",
        nn.NormActivation("2x0e+2x1o", torch.tanh, normalize=True, epsilon=1e-6),
        _deterministic((4, 8), 1011),
        derivatives=True,
    )
    record_parameterless(
        "identity",
        nn.Identity("2x0e+2x1o", "2x0e+2x1o"),
        _deterministic((4, 8), 1051),
    )
    record_parameterless(
        "extract",
        nn.Extract("0e+1o+2x0e", ["1o", "2x0e"], [[1], [2]], squeeze_out=False),
        _deterministic((4, 6), 1091),
    )
    record_parameterless(
        "extract_ir",
        nn.ExtractIr("0e+1o+2x0e", "0e"),
        _deterministic((4, 6), 1111),
    )
    dropout = nn.Dropout("2x0e+2x1o", p=0.35)
    dropout.eval()
    record_parameterless("dropout_eval", dropout, _deterministic((4, 8), 1131))

    batch_norm = nn.BatchNorm("2x0e+2x1o")
    batch_norm.eval()
    with torch.no_grad():
        batch_norm.weight.copy_(_deterministic(tuple(batch_norm.weight.shape), 1171, scale=0.4) + 1)
        batch_norm.bias.copy_(_deterministic(tuple(batch_norm.bias.shape), 1191, scale=0.2))
        batch_norm.running_mean.copy_(_deterministic(tuple(batch_norm.running_mean.shape), 1211))
        batch_norm.running_var.copy_(
            _deterministic(tuple(batch_norm.running_var.shape), 1231, scale=0.1).abs() + 0.8
        )
    value = _deterministic((4, 8), 1271)
    arrays["module.batch_norm.input"] = _array(value)
    arrays["module.batch_norm.output"] = _array(batch_norm(value))
    for name, tensor in (
        ("weight", batch_norm.weight),
        ("bias", batch_norm.bias),
        ("running_mean", batch_norm.running_mean),
        ("running_var", batch_norm.running_var),
    ):
        arrays[f"module.batch_norm.{name}"] = _array(tensor)
    cases.append({"name": "batch_norm", "irreps_in": "2x0e+2x1o"})

    fc = nn.FullyConnectedNet([3, 5, 2], torch.tanh, out_act=True)
    parameters = _store_parameters(arrays, "module.fully_connected_net", fc)
    value = _deterministic((4, 3), 1301)
    arrays["module.fully_connected_net.input"] = _array(value)
    arrays["module.fully_connected_net.output"] = _array(fc(value))
    cases.append(
        {
            "name": "fully_connected_net",
            "widths": [3, 5, 2],
            "parameters": parameters,
        }
    )

    s2 = nn.S2Activation(
        o3.Irreps.spherical_harmonics(2),
        torch.tanh,
        res=20,
        normalization="component",
        lmax_out=2,
        random_rot=False,
    )
    value = _deterministic((2, s2.irreps_in.dim), 1341)
    arrays["module.s2_activation.input"] = _array(value)
    arrays["module.s2_activation.output"] = _array(s2(value))
    cases.append(
        {
            "name": "s2_activation",
            "irreps_in": str(s2.irreps_in),
            "irreps_out": str(s2.irreps_out),
            "resolution": 20,
            "lmax_out": 2,
        }
    )

    so3 = nn.SO3Activation(1, 1, torch.tanh, resolution=6)
    value = _deterministic((2, 10), 1371)
    arrays["module.so3_activation.input"] = _array(value)
    arrays["module.so3_activation.output"] = _array(so3(value))
    cases.append(
        {
            "name": "so3_activation",
            "lmax_in": 1,
            "lmax_out": 1,
            "resolution": 6,
        }
    )
    manifest["modules"] = cases


def _fixed_rotation() -> torch.Tensor:
    angles = (
        torch.tensor(0.37, dtype=DTYPE),
        torch.tensor(-0.51, dtype=DTYPE),
        torch.tensor(0.83, dtype=DTYPE),
    )
    return o3.angles_to_matrix(*angles)


def _equivariance_error(
    baseline: torch.Tensor,
    transformed: torch.Tensor,
    irreps_out,
    rotation: torch.Tensor,
) -> torch.Tensor:
    d_out = o3.Irreps(irreps_out).D_from_matrix(rotation)
    expected = baseline @ d_out.T
    return torch.max(torch.abs(transformed - expected))


def _model_fixtures(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    from e3nn.nn.models import gate_points_2102 as gate_module
    from e3nn.nn.models.gate_points_2102 import Network as GateNetwork
    from e3nn.nn.models.v2106.gate_points_networks import (
        NetworkForAGraphWithAttributes,
    )

    rotation = _fixed_rotation()
    arrays["models.rotation"] = _array(rotation)
    edge_src = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 0, 2, 0, 3, 1], dtype=torch.long)
    edge_index = torch.stack([edge_src, edge_dst])
    positions = _deterministic((4, 3), 1401, scale=0.9)
    node_input = _deterministic((4, 4), 1441, scale=0.7)
    node_attr = _deterministic((4, 1), 1491, scale=0.5)
    batch = torch.zeros((4,), dtype=torch.long)
    d_input = o3.Irreps("0e+1o").D_from_matrix(rotation)
    rotated_positions = positions @ rotation.T
    rotated_input = node_input @ d_input.T

    gate = GateNetwork(
        "0e+1o",
        "2x0e+2x1o",
        "0e+1o",
        "0e",
        o3.Irreps.spherical_harmonics(1),
        layers=1,
        max_radius=2.5,
        number_of_basis=3,
        radial_layers=1,
        radial_neurons=4,
        num_neighbors=2.0,
        num_nodes=4.0,
        reduce_output=False,
    )
    gate_parameters = _store_parameters(arrays, "model.gate_points", gate)
    # Avoid an optional torch-cluster dependency while retaining the exact
    # upstream forward path and a fixed topology shared with MLX.
    original_radius_graph = gate_module.radius_graph
    gate_module.radius_graph = lambda *_args, **_kwargs: edge_index
    try:
        gate_data = {"pos": positions, "x": node_input, "z": node_attr, "batch": batch}
        gate_rotated_data = {
            "pos": rotated_positions,
            "x": rotated_input,
            "z": node_attr,
            "batch": batch,
        }
        gate_output = gate(gate_data)
        gate_rotated = gate(gate_rotated_data)
    finally:
        gate_module.radius_graph = original_radius_graph
    for name, value in (
        ("positions", positions),
        ("node_input", node_input),
        ("node_attr", node_attr),
        ("batch", batch),
        ("edge_src", edge_src),
        ("edge_dst", edge_dst),
        ("output", gate_output),
        ("rotated_output", gate_rotated),
    ):
        arrays[f"model.gate_points.{name}"] = _array(value)
    arrays["model.gate_points.equivariance_error"] = _array(
        _equivariance_error(gate_output, gate_rotated, gate.irreps_out, rotation)
    )

    v2106 = NetworkForAGraphWithAttributes(
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
    v2106_parameters = _store_parameters(arrays, "model.v2106", v2106)
    edge_attr = _deterministic((edge_src.numel(), 1), 1531, scale=0.4)
    v_data = {
        "pos": positions,
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "batch": batch,
    }
    v_rotated_data = dict(v_data)
    v_rotated_data["pos"] = rotated_positions
    v_rotated_data["node_input"] = rotated_input
    v_output = v2106(v_data)
    v_rotated = v2106(v_rotated_data)
    for name, value in (
        ("positions", positions),
        ("node_input", node_input),
        ("node_attr", node_attr),
        ("edge_attr", edge_attr),
        ("batch", batch),
        ("edge_src", edge_src),
        ("edge_dst", edge_dst),
        ("output", v_output),
        ("rotated_output", v_rotated),
    ):
        arrays[f"model.v2106.{name}"] = _array(value)
    arrays["model.v2106.equivariance_error"] = _array(
        _equivariance_error(v_output, v_rotated, v2106.irreps_node_output, rotation)
    )
    manifest["models"] = [
        {
            "name": "gate_points",
            "parameters": gate_parameters,
            "irreps_in": "0e+1o",
            "irreps_out": "0e+1o",
        },
        {
            "name": "v2106",
            "parameters": v2106_parameters,
            "irreps_in": "0e+1o",
            "irreps_out": "0e+1o",
        },
    ]


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "schema": 1,
        "seed": SEED,
        "e3nn": e3nn.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float64",
    }
    _tp_fixtures(arrays, manifest)
    _wrapper_fixtures(arrays, manifest)
    _module_fixtures(arrays, manifest)
    _model_fixtures(arrays, manifest)
    np.savez_compressed(OUTPUT / "torch_parity_reference.npz", **arrays)
    (OUTPUT / "torch_parity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
