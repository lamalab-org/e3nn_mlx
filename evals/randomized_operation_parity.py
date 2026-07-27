"""Randomized Torch/e3nn versus MLX parity for major numerical operations."""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

try:
    from evals.randomized_tensor_product_parity import (
        ROOT,
        _format_irreps,
        _irreps_dim,
        _metrics,
        _part,
        _random_values,
    )
except ModuleNotFoundError:  # direct ``python evals/...py`` execution
    from randomized_tensor_product_parity import (
        ROOT,
        _format_irreps,
        _irreps_dim,
        _metrics,
        _part,
        _random_values,
    )


KINDS = (
    "rotation_matrix",
    "wigner_d",
    "spherical_harmonics",
    "linear",
    "norm",
    "activation",
    "gate",
    "norm_activation",
    "batch_norm",
    "s2_activation",
    "so3_activation",
    "radial_basis",
    "soft_unit_step",
    "reduced_tensor_products",
    "scatter_sum",
    "full_tensor_product",
    "fully_connected_tensor_product",
    "elementwise_tensor_product",
    "tensor_square",
)


def _spec(rng: np.random.Generator, shape: tuple[int, ...]) -> dict[str, Any]:
    return {"shape": list(shape), "values": _random_values(rng, shape)}


def _numpy(specification: dict[str, Any]) -> np.ndarray:
    return np.asarray(specification["values"], dtype=np.float32).reshape(
        specification["shape"]
    )


def _random_irreps(
    rng: np.random.Generator,
    *,
    max_l: int,
    blocks: int | None = None,
    include_scalar: bool = False,
) -> tuple[str, list[dict[str, int]]]:
    count = int(rng.integers(1, 4)) if blocks is None else blocks
    parts = []
    if include_scalar:
        parts.append(_part(int(rng.integers(1, 4)), 0, 1))
    while len(parts) < count:
        parts.append(
            _part(
                int(rng.integers(1, 4)),
                int(rng.integers(0, max_l + 1)),
                1 if rng.integers(0, 2) else -1,
            )
        )
    return _format_irreps(parts), parts


def _batch_shape(rng: np.random.Generator, dimension: int) -> tuple[int, ...]:
    layout = int(rng.integers(0, 3))
    if layout == 0:
        return (dimension,)
    if layout == 1:
        return int(rng.integers(1, 6)), dimension
    return int(rng.integers(1, 4)), int(rng.integers(1, 4)), dimension


def _activation_name(rng: np.random.Generator, parity: int) -> str:
    if parity == -1:
        return str(rng.choice(("tanh", "abs")))
    return str(rng.choice(("tanh", "abs", "silu")))


def _make_case(
    rng: np.random.Generator,
    kind: str,
    *,
    seed: int,
    index: int,
    max_l: int,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": f"seed-{seed}-{kind}-{index:04d}",
        "seed": seed,
        "index": index,
        "kind": kind,
    }
    if kind in {"rotation_matrix", "wigner_d"}:
        shape = () if rng.random() < 0.35 else (int(rng.integers(1, 6)),)
        case["angles"] = [_spec(rng, shape) for _ in range(3)]
        if kind == "wigner_d":
            case["l"] = int(rng.integers(0, max_l + 1))
        return case

    if kind == "spherical_harmonics":
        lmax = int(rng.integers(0, max_l + 1))
        shape = _batch_shape(rng, 3)
        case.update(
            {
                "lmax": lmax,
                "normalize": bool(rng.integers(0, 2)),
                "normalization": str(
                    rng.choice(("component", "norm", "integral"))
                ),
                "input": _spec(rng, shape),
            }
        )
        return case

    if kind == "linear":
        block_count = int(rng.integers(1, 4))
        quantum_numbers = [
            (
                int(rng.integers(0, max_l + 1)),
                1 if rng.integers(0, 2) else -1,
            )
            for _ in range(block_count)
        ]
        input_parts = [
            _part(int(rng.integers(1, 4)), l, parity)
            for l, parity in quantum_numbers
        ]
        output_parts = [
            _part(int(rng.integers(1, 4)), l, parity)
            for l, parity in quantum_numbers
        ]
        rng.shuffle(output_parts)
        weight_numel = sum(
            left["mul"] * right["mul"]
            for left in input_parts
            for right in output_parts
            if (left["l"], left["parity"]) == (right["l"], right["parity"])
        )
        bias = bool(
            rng.integers(0, 2)
            and any(part["l"] == 0 and part["parity"] == 1 for part in output_parts)
        )
        bias_numel = (
            sum(
                part["mul"]
                for part in output_parts
                if part["l"] == 0 and part["parity"] == 1
            )
            if bias
            else 0
        )
        case.update(
            {
                "irreps_in": _format_irreps(input_parts),
                "irreps_out": _format_irreps(output_parts),
                "path_normalization": str(rng.choice(("element", "path"))),
                "bias": bias,
                "input": _spec(
                    rng, _batch_shape(rng, _irreps_dim(input_parts))
                ),
                "weight": _spec(rng, (weight_numel,)),
                "bias_value": (
                    _spec(rng, (bias_numel,)) if bias_numel else None
                ),
            }
        )
        return case

    if kind == "norm":
        irreps, parts = _random_irreps(rng, max_l=max_l)
        case.update(
            {
                "irreps": irreps,
                "squared": bool(rng.integers(0, 2)),
                "input": _spec(rng, _batch_shape(rng, _irreps_dim(parts))),
            }
        )
        return case

    if kind == "activation":
        parts = [
            _part(int(rng.integers(1, 4)), 0, 1),
            _part(int(rng.integers(1, 4)), 0, -1),
        ]
        case.update(
            {
                "irreps": _format_irreps(parts),
                "activations": [
                    _activation_name(rng, part["parity"]) for part in parts
                ],
                "input": _spec(rng, _batch_shape(rng, _irreps_dim(parts))),
            }
        )
        return case

    if kind == "gate":
        scalar_parts = [
            _part(int(rng.integers(1, 4)), 0, 1),
            _part(int(rng.integers(1, 3)), 0, -1),
        ]
        gate_mul = int(rng.integers(1, 4))
        gate_parts = [_part(gate_mul, 0, 1)]
        gated_parts = [
            _part(
                gate_mul,
                int(rng.integers(1, max_l + 1)),
                1 if rng.integers(0, 2) else -1,
            )
        ]
        input_dim = (
            _irreps_dim(scalar_parts)
            + _irreps_dim(gate_parts)
            + _irreps_dim(gated_parts)
        )
        case.update(
            {
                "irreps_scalars": _format_irreps(scalar_parts),
                "scalar_activations": ["tanh", "abs"],
                "irreps_gates": _format_irreps(gate_parts),
                "gate_activations": ["sigmoid"],
                "irreps_gated": _format_irreps(gated_parts),
                "input": _spec(rng, _batch_shape(rng, input_dim)),
            }
        )
        return case

    if kind == "norm_activation":
        irreps, parts = _random_irreps(rng, max_l=max_l)
        # e3nn 0.5.8 rejects normalize=False with epsilon=None because of an
        # upstream constructor bug. Keep randomized parity on the shared,
        # valid normalization contract.
        normalize = True
        case.update(
            {
                "irreps": irreps,
                "normalize": normalize,
                "epsilon": 1e-6 if normalize else None,
                "input": _spec(rng, _batch_shape(rng, _irreps_dim(parts))),
            }
        )
        return case

    if kind == "batch_norm":
        irreps, parts = _random_irreps(
            rng, max_l=max_l, blocks=3, include_scalar=True
        )
        scalar_numel = sum(
            part["mul"]
            for part in parts
            if part["l"] == 0 and part["parity"] == 1
        )
        feature_numel = sum(part["mul"] for part in parts)
        batch = int(rng.integers(2, 5))
        samples = int(rng.integers(1, 5))
        case.update(
            {
                "irreps": irreps,
                "normalization": str(rng.choice(("component", "norm"))),
                "input": _spec(
                    rng, (batch, samples, _irreps_dim(parts))
                ),
                "running_mean": _spec(rng, (scalar_numel,)),
                "running_var": {
                    "shape": [feature_numel],
                    "values": (
                        np.abs(
                            np.asarray(
                                _random_values(rng, (feature_numel,)),
                                dtype=np.float32,
                            )
                        )
                        + 0.7
                    ).tolist(),
                },
                "weight": _spec(rng, (feature_numel,)),
                "bias": _spec(rng, (scalar_numel,)),
            }
        )
        return case

    if kind == "s2_activation":
        lmax = int(rng.integers(1, min(max_l, 3) + 1))
        lmax_out = int(rng.integers(0, min(max_l, 3) + 1))
        parts = [_part(1, l, 1) for l in range(lmax + 1)]
        case.update(
            {
                "irreps": _format_irreps(parts),
                "lmax_out": lmax_out,
                "resolution": int(rng.choice((10, 12, 16))),
                "normalization": str(rng.choice(("component", "norm"))),
                "input": _spec(
                    rng, (int(rng.integers(1, 4)), _irreps_dim(parts))
                ),
            }
        )
        return case

    if kind == "so3_activation":
        lmax_in = int(rng.integers(0, min(max_l, 2) + 1))
        lmax_out = int(rng.integers(0, min(max_l, 2) + 1))
        input_dim = sum((2 * l + 1) ** 2 for l in range(lmax_in + 1))
        case.update(
            {
                "lmax_in": lmax_in,
                "lmax_out": lmax_out,
                "resolution": int(rng.choice((5, 6))),
                "normalization": "component",
                "input": _spec(rng, (int(rng.integers(1, 3)), input_dim)),
            }
        )
        return case

    if kind == "radial_basis":
        basis = str(
            rng.choice(("gaussian", "cosine", "smooth_finite", "fourier", "bessel"))
        )
        cutoff = bool(rng.integers(0, 2))
        number = int(rng.integers(2, 7))
        case.update(
            {
                "basis": basis,
                "cutoff": cutoff,
                "number": number,
                "start": -0.2,
                "end": 2.3,
                "input": {
                    "shape": [int(rng.integers(2, 9))],
                    "values": rng.uniform(-0.1, 2.2, size=int(rng.integers(2, 9))).astype(np.float32).tolist(),
                },
            }
        )
        # The shape and sampled value count must be identical.
        case["input"]["shape"] = [len(case["input"]["values"])]
        return case

    if kind == "soft_unit_step":
        size = int(rng.integers(2, 10))
        case["input"] = {
            "shape": [size],
            "values": rng.uniform(-1.0, 2.0, size=size).astype(np.float32).tolist(),
        }
        return case

    if kind == "reduced_tensor_products":
        l = int(rng.integers(1, min(max_l, 2) + 1))
        parity = 1 if rng.integers(0, 2) else -1
        parts = [_part(1, l, parity)]
        irreps = _format_irreps(parts)
        dimension = 2 * l + 1
        formula = str(rng.choice(("ij=ji", "ij=-ji")))
        case.update(
            {
                "formula": formula,
                "irreps": irreps,
                "inputs": [
                    _spec(rng, (int(rng.integers(1, 4)), dimension))
                    for _ in range(2)
                ],
            }
        )
        # Both inputs must use the same leading shape.
        case["inputs"][1] = _spec(
            rng, tuple(case["inputs"][0]["shape"])
        )
        return case

    if kind == "scatter_sum":
        items = int(rng.integers(1, 20))
        features = int(rng.integers(1, 9))
        dim_size = int(rng.integers(1, 8))
        case.update(
            {
                "dim_size": dim_size,
                "index": rng.integers(0, dim_size, size=items).tolist(),
                "input": _spec(rng, (items, features)),
            }
        )
        return case

    if kind in {
        "full_tensor_product",
        "fully_connected_tensor_product",
        "elementwise_tensor_product",
        "tensor_square",
    }:
        l1 = int(rng.integers(0, max_l + 1))
        l2 = int(rng.integers(0, max_l + 1))
        p1 = 1 if rng.integers(0, 2) else -1
        p2 = 1 if rng.integers(0, 2) else -1
        mul1 = int(rng.integers(1, 4))
        mul2 = mul1 if kind == "elementwise_tensor_product" else int(rng.integers(1, 4))
        left_parts = [_part(mul1, l1, p1)]
        right_parts = [_part(mul2, l2, p2)]
        batch = int(rng.integers(1, 5))
        case.update(
            {
                "irreps_in1": _format_irreps(left_parts),
                "irreps_in2": _format_irreps(right_parts),
                "left": _spec(rng, (batch, _irreps_dim(left_parts))),
                "right": _spec(rng, (batch, _irreps_dim(right_parts))),
            }
        )
        if kind == "fully_connected_tensor_product":
            lout = int(rng.integers(abs(l1 - l2), l1 + l2 + 1))
            output_parts = [
                _part(int(rng.integers(1, 4)), lout, p1 * p2)
            ]
            weight_numel = mul1 * mul2 * output_parts[0]["mul"]
            case["irreps_out"] = _format_irreps(output_parts)
            case["weight"] = _spec(rng, (weight_numel,))
        elif kind == "tensor_square":
            case["irreps_in"] = case.pop("irreps_in1")
            case["input"] = case.pop("left")
            case.pop("irreps_in2")
            case.pop("right")
        return case

    raise ValueError(f"unsupported randomized operation kind {kind!r}")


def generate_cases(
    *,
    seed: int,
    cases_per_kind: int,
    kinds: tuple[str, ...] = KINDS,
    max_l: int = 4,
) -> list[dict[str, Any]]:
    if cases_per_kind < 1:
        raise ValueError("cases_per_kind must be positive")
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown operation kinds: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    return [
        _make_case(
            rng,
            kind,
            seed=seed,
            index=index,
            max_l=max_l,
        )
        for kind in kinds
        for index in range(cases_per_kind)
    ]


def _torch_activation(name: str):
    import torch

    return {
        "tanh": torch.tanh,
        "abs": torch.abs,
        "silu": torch.nn.functional.silu,
        "sigmoid": torch.sigmoid,
    }[name]


def _mlx_activation(name: str):
    import mlx.core as mx

    return {
        "tanh": mx.tanh,
        "abs": mx.abs,
        "silu": lambda value: value * mx.sigmoid(value),
        "sigmoid": mx.sigmoid,
    }[name]


def _torch_operation(case: dict[str, Any]):
    import torch
    from e3nn import math as e3math
    from e3nn import nn, o3

    def variable(specification):
        return torch.from_numpy(_numpy(specification)).requires_grad_(True)

    kind = case["kind"]
    metadata: dict[str, Any] = {}
    if kind in {"rotation_matrix", "wigner_d"}:
        primals = tuple(variable(value) for value in case["angles"])
        if kind == "rotation_matrix":
            function = lambda a, b, c: o3.angles_to_matrix(a, b, c)
        else:
            function = lambda a, b, c: o3.wigner_D(case["l"], a, b, c)
    elif kind == "spherical_harmonics":
        primals = (variable(case["input"]),)
        function = lambda value: o3.spherical_harmonics(
            list(range(case["lmax"] + 1)),
            value,
            normalize=case["normalize"],
            normalization=case["normalization"],
        )
    elif kind == "linear":
        module = o3.Linear(
            case["irreps_in"],
            case["irreps_out"],
            internal_weights=False,
            biases=case["bias"],
            path_normalization=case["path_normalization"],
        )
        primals = [variable(case["input"]), variable(case["weight"])]
        if case["bias_value"] is not None:
            primals.append(variable(case["bias_value"]))
            function = lambda value, weight, bias: module(value, weight, bias)
        else:
            function = lambda value, weight: module(value, weight)
        primals = tuple(primals)
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "norm":
        module = o3.Norm(case["irreps"], squared=case["squared"])
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "activation":
        module = nn.Activation(
            case["irreps"],
            [_torch_activation(name) for name in case["activations"]],
        )
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "gate":
        module = nn.Gate(
            case["irreps_scalars"],
            [_torch_activation(name) for name in case["scalar_activations"]],
            case["irreps_gates"],
            [_torch_activation(name) for name in case["gate_activations"]],
            case["irreps_gated"],
        )
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "norm_activation":
        module = nn.NormActivation(
            case["irreps"],
            torch.tanh,
            normalize=case["normalize"],
            epsilon=case["epsilon"],
        )
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "batch_norm":
        module = nn.BatchNorm(
            case["irreps"],
            normalization=case["normalization"],
        )
        module.eval()
        with torch.no_grad():
            module.running_mean.copy_(
                torch.from_numpy(_numpy(case["running_mean"]))
            )
            module.running_var.copy_(
                torch.from_numpy(_numpy(case["running_var"]))
            )
            module.weight.copy_(torch.from_numpy(_numpy(case["weight"])))
            module.bias.copy_(torch.from_numpy(_numpy(case["bias"])))
        primals = (variable(case["input"]),)
        function = module
    elif kind == "s2_activation":
        module = nn.S2Activation(
            case["irreps"],
            torch.tanh,
            case["resolution"],
            normalization=case["normalization"],
            lmax_out=case["lmax_out"],
            random_rot=False,
        )
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "so3_activation":
        module = nn.SO3Activation(
            case["lmax_in"],
            case["lmax_out"],
            torch.tanh,
            case["resolution"],
            normalization=case["normalization"],
        )
        primals = (variable(case["input"]),)
        function = module
    elif kind == "radial_basis":
        primals = (variable(case["input"]),)
        function = lambda value: e3math.soft_one_hot_linspace(
            value,
            case["start"],
            case["end"],
            case["number"],
            basis=case["basis"],
            cutoff=case["cutoff"],
        )
    elif kind == "soft_unit_step":
        primals = (variable(case["input"]),)
        function = e3math.soft_unit_step
    elif kind == "reduced_tensor_products":
        module = o3.ReducedTensorProducts(
            case["formula"], i=case["irreps"]
        )
        primals = tuple(variable(value) for value in case["inputs"])
        function = lambda *values: torch.einsum(
            "kij,...k->...ij",
            module.change_of_basis,
            module(*values),
        )
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "scatter_sum":
        source = variable(case["input"])
        index = torch.tensor(case["index"], dtype=torch.long)
        primals = (source,)

        def function(value):
            output = torch.zeros(
                (case["dim_size"], *value.shape[1:]),
                dtype=value.dtype,
            )
            return output.index_add(0, index, value)
    elif kind in {
        "full_tensor_product",
        "fully_connected_tensor_product",
        "elementwise_tensor_product",
    }:
        left = variable(case["left"])
        right = variable(case["right"])
        if kind == "full_tensor_product":
            module = o3.FullTensorProduct(
                case["irreps_in1"], case["irreps_in2"]
            )
            primals = (left, right)
            function = module
        elif kind == "elementwise_tensor_product":
            module = o3.ElementwiseTensorProduct(
                case["irreps_in1"], case["irreps_in2"]
            )
            primals = (left, right)
            function = module
        else:
            module = o3.FullyConnectedTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                case["irreps_out"],
                internal_weights=False,
            )
            primals = (left, right, variable(case["weight"]))
            function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "tensor_square":
        module = o3.TensorSquare(case["irreps_in"])
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    else:
        raise ValueError(kind)
    return function, tuple(primals), metadata


def _mlx_operation(case: dict[str, Any], variant: str):
    import mlx.core as mx

    import e3nn_mlx as e3nn

    kind = case["kind"]
    primals: tuple[Any, ...]
    metadata: dict[str, Any] = {}
    variable = lambda specification: mx.array(_numpy(specification))
    if kind in {"rotation_matrix", "wigner_d"}:
        primals = tuple(variable(value) for value in case["angles"])
        if kind == "rotation_matrix":
            function = lambda a, b, c: e3nn.o3.angles_to_matrix(a, b, c)
        else:
            function = lambda a, b, c: e3nn.o3.wigner_d(case["l"], a, b, c)
    elif kind == "spherical_harmonics":
        primals = (variable(case["input"]),)
        function = lambda value: e3nn.spherical_harmonics(
            list(range(case["lmax"] + 1)),
            value,
            normalize=case["normalize"],
            normalization=case["normalization"],
            use_custom_kernel=variant == "custom-requested",
        )
    elif kind == "linear":
        module = e3nn.o3.Linear(
            case["irreps_in"],
            case["irreps_out"],
            internal_weights=False,
            bias=case["bias"],
            path_normalization=case["path_normalization"],
            compile=variant == "compiled",
        )
        values = [variable(case["input"]), variable(case["weight"])]
        if case["bias_value"] is not None:
            values.append(variable(case["bias_value"]))
            function = lambda value, weight, bias: module(
                value, weight, bias=bias
            )
        else:
            function = lambda value, weight: module(value, weight)
        primals = tuple(values)
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "norm":
        module = e3nn.o3.Norm(case["irreps"], squared=case["squared"])
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "activation":
        module = e3nn.Activation(
            case["irreps"],
            [_mlx_activation(name) for name in case["activations"]],
        )
        primals = (variable(case["input"]),)
        function = lambda value: module(e3nn.IrrepsArray(module.irreps_in, value)).array
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "gate":
        module = e3nn.Gate(
            case["irreps_scalars"],
            [_mlx_activation(name) for name in case["scalar_activations"]],
            case["irreps_gates"],
            [_mlx_activation(name) for name in case["gate_activations"]],
            case["irreps_gated"],
        )
        primals = (variable(case["input"]),)
        function = lambda value: module(e3nn.IrrepsArray(module.irreps_in, value)).array
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "norm_activation":
        module = e3nn.NormActivation(
            case["irreps"],
            mx.tanh,
            normalize=case["normalize"],
            epsilon=case["epsilon"],
        )
        primals = (variable(case["input"]),)
        function = lambda value: module(e3nn.IrrepsArray(module.irreps_in, value)).array
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "batch_norm":
        module = e3nn.BatchNorm(
            case["irreps"],
            normalization=case["normalization"],
        )
        module.eval()
        module._running_mean = variable(case["running_mean"])
        module._running_var = variable(case["running_var"])
        module.weight = variable(case["weight"])
        module.bias = variable(case["bias"])
        primals = (variable(case["input"]),)
        function = lambda value: module(e3nn.IrrepsArray(module.irreps, value)).array
    elif kind == "s2_activation":
        module = e3nn.S2Activation(
            case["irreps"],
            mx.tanh,
            case["resolution"],
            normalization=case["normalization"],
            lmax_out=case["lmax_out"],
            random_rot=False,
        )
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "so3_activation":
        module = e3nn.SO3Activation(
            case["lmax_in"],
            case["lmax_out"],
            mx.tanh,
            case["resolution"],
            normalization=case["normalization"],
        )
        primals = (variable(case["input"]),)
        function = module
    elif kind == "radial_basis":
        primals = (variable(case["input"]),)
        function = lambda value: e3nn.soft_one_hot_linspace(
            value,
            case["start"],
            case["end"],
            case["number"],
            basis=case["basis"],
            cutoff=case["cutoff"],
        )
    elif kind == "soft_unit_step":
        primals = (variable(case["input"]),)
        function = e3nn.soft_unit_step
    elif kind == "reduced_tensor_products":
        module = e3nn.o3.ReducedTensorProducts(
            case["formula"], i=case["irreps"]
        )
        primals = tuple(variable(value) for value in case["inputs"])
        function = lambda *values: mx.einsum(
            "kij,...k->...ij",
            module.change_of_basis,
            module(*values),
        )
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "scatter_sum":
        index = mx.array(case["index"], dtype=mx.int32)
        primals = (variable(case["input"]),)
        function = lambda value: e3nn.scatter_sum(
            value,
            index,
            case["dim_size"],
            use_custom_kernel=variant == "custom-requested",
        )
    elif kind in {
        "full_tensor_product",
        "fully_connected_tensor_product",
        "elementwise_tensor_product",
    }:
        left = variable(case["left"])
        right = variable(case["right"])
        custom = variant == "custom-requested"
        if kind == "full_tensor_product":
            module = e3nn.o3.FullTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                use_custom_kernel=custom,
            )
            primals = (left, right)
            function = module
        elif kind == "elementwise_tensor_product":
            module = e3nn.o3.ElementwiseTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                use_custom_kernel=custom,
            )
            primals = (left, right)
            function = module
        else:
            module = e3nn.o3.FullyConnectedTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                case["irreps_out"],
                internal_weights=False,
                use_custom_kernel=custom,
            )
            primals = (left, right, variable(case["weight"]))
            function = module
        metadata["irreps_out"] = str(module.irreps_out)
    elif kind == "tensor_square":
        module = e3nn.o3.TensorSquare(case["irreps_in"])
        primals = (variable(case["input"]),)
        function = module
        metadata["irreps_out"] = str(module.irreps_out)
    else:
        raise ValueError(kind)
    return function, primals, metadata


def _torch_evaluate(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    try:
        function, primals, metadata = _torch_operation(case)
        output = function(*primals)
        cotangent = torch.linspace(
            0.3, 1.1, output.numel(), dtype=output.dtype
        ).reshape(output.shape)
        if output.requires_grad:
            raw_gradients = torch.autograd.grad(
                output,
                primals,
                grad_outputs=cotangent,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(primal) if gradient is None else gradient
                for primal, gradient in zip(
                    primals, raw_gradients, strict=True
                )
            )
        else:
            gradients = tuple(torch.zeros_like(primal) for primal in primals)
        return {
            "status": "ok",
            "output": output.detach().cpu().numpy().tolist(),
            "shape": list(output.shape),
            "gradients": [
                {
                    "shape": list(gradient.shape),
                    "values": gradient.detach().cpu().numpy().tolist(),
                }
                for gradient in gradients
            ],
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _mlx_variants(kind: str) -> tuple[str, ...]:
    if kind in {
        "spherical_harmonics",
        "scatter_sum",
        "full_tensor_product",
        "fully_connected_tensor_product",
        "elementwise_tensor_product",
    }:
        return "general", "custom-requested"
    if kind == "linear":
        return "eager", "compiled"
    return ("mlx",)


def _mlx_evaluate(case: dict[str, Any], variant: str) -> dict[str, Any]:
    import mlx.core as mx

    try:
        function, primals, metadata = _mlx_operation(case, variant)
        output = function(*primals)
        cotangent = mx.linspace(0.3, 1.1, output.size).reshape(output.shape)
        _, gradients = mx.vjp(function, primals, (cotangent,))
        mx.eval(output, *gradients)
        return {
            "name": variant,
            "status": "ok",
            "output": np.asarray(output).tolist(),
            "shape": list(output.shape),
            "gradients": [
                {
                    "shape": list(gradient.shape),
                    "values": np.asarray(gradient).tolist(),
                }
                for gradient in gradients
            ],
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "name": variant,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _torch_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import e3nn

    return {
        "backend": "torch",
        "torch": torch.__version__,
        "e3nn": e3nn.__version__,
        "results": [
            {"id": case["id"], **_torch_evaluate(case)} for case in cases
        ],
    }


def _mlx_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend": "mlx",
        "mlx": version("mlx"),
        "results": [
            {
                "id": case["id"],
                "variants": [
                    _mlx_evaluate(case, variant)
                    for variant in _mlx_variants(case["kind"])
                ],
            }
            for case in cases
        ],
    }


def _comparison_tolerance(kind: str) -> tuple[float, float]:
    if kind == "so3_activation":
        # The two implementations use different SO(3) quadrature paths.
        # Absolute errors grow slightly with signal magnitude, while the
        # norm-scaled discrepancy remains below one part in a thousand.
        return 8e-3, 2e-3
    if kind == "s2_activation":
        return 3e-3, 3e-3
    if kind in {"activation", "gate"}:
        return 5e-3, 5e-3
    if kind == "radial_basis":
        return 8e-4, 8e-4
    return 1e-4, 1e-4


def _canonical_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    canonical = dict(metadata)
    if "irreps_out" in canonical:
        from e3nn_core import Irreps

        canonical["irreps_out"] = str(
            Irreps(canonical["irreps_out"])
            .remove_zero_multiplicities()
            .simplify()
        )
    return canonical


def compare_results(
    cases: list[dict[str, Any]],
    torch_result: dict[str, Any],
    mlx_result: dict[str, Any],
) -> dict[str, Any]:
    torch_rows = {row["id"]: row for row in torch_result["results"]}
    mlx_rows = {row["id"]: row for row in mlx_result["results"]}
    rows = []
    failures = []
    for case in cases:
        reference = torch_rows[case["id"]]
        candidate = mlx_rows[case["id"]]
        atol, norm_tolerance = _comparison_tolerance(case["kind"])
        comparisons = []
        failed = reference["status"] != "ok"
        for variant in candidate["variants"]:
            comparison = {"name": variant["name"], "status": variant["status"]}
            if reference["status"] == variant["status"] == "ok":
                output_reference = np.asarray(reference["output"], dtype=np.float32)
                output_actual = np.asarray(variant["output"], dtype=np.float32)
                output_metrics = _metrics(output_actual, output_reference)
                gradient_metrics = []
                shapes_match = output_actual.shape == output_reference.shape
                gradients_match = len(reference["gradients"]) == len(
                    variant["gradients"]
                )
                if gradients_match:
                    for actual, expected in zip(
                        variant["gradients"],
                        reference["gradients"],
                        strict=True,
                    ):
                        actual_value = np.asarray(actual["values"], dtype=np.float32)
                        expected_value = np.asarray(expected["values"], dtype=np.float32)
                        gradients_match = (
                            gradients_match
                            and actual_value.shape == expected_value.shape
                        )
                        if gradients_match:
                            gradient_metrics.append(
                                _metrics(actual_value, expected_value)
                            )
                metadata_match = _canonical_metadata(
                    variant.get("metadata")
                ) == _canonical_metadata(reference.get("metadata"))
                maximum_gradient_abs = max(
                    (metric["max_abs"] for metric in gradient_metrics),
                    default=0.0,
                )
                maximum_gradient_norm = max(
                    (metric["norm_scaled"] for metric in gradient_metrics),
                    default=0.0,
                )
                passed = bool(
                    shapes_match
                    and gradients_match
                    and metadata_match
                    and output_metrics["max_abs"] <= atol
                    and output_metrics["norm_scaled"] <= norm_tolerance
                    and maximum_gradient_abs <= atol * 4
                    and maximum_gradient_norm <= norm_tolerance * 4
                )
                comparison.update(
                    {
                        "passed": passed,
                        "output": output_metrics,
                        "gradient_max_abs": maximum_gradient_abs,
                        "gradient_norm_scaled": maximum_gradient_norm,
                        "metadata_match": metadata_match,
                    }
                )
            else:
                comparison["passed"] = False
                comparison["error"] = variant.get(
                    "error", reference.get("error", "worker error")
                )
            failed = failed or not comparison["passed"]
            comparisons.append(comparison)
        row = {
            "id": case["id"],
            "kind": case["kind"],
            "comparisons": comparisons,
            "passed": not failed,
        }
        rows.append(row)
        if failed:
            failures.append(
                {
                    "case": case,
                    "torch": reference,
                    "mlx": candidate,
                    "comparisons": comparisons,
                }
            )
    return {
        "schema": 1,
        "seed": cases[0]["seed"] if cases else None,
        "case_count": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "torch_version": torch_result.get("torch"),
        "e3nn_version": torch_result.get("e3nn"),
        "mlx_version": mlx_result.get("mlx"),
        "rows": rows,
        "failures": failures,
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Randomized operation parity",
        "",
        f"- Seed: `{report['seed']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Passed: `{report['passed']}`",
        f"- Failed: `{report['failed']}`",
        f"- Torch/e3nn: `{report['torch_version']}` / `{report['e3nn_version']}`",
        f"- MLX: `{report['mlx_version']}`",
        "",
        "| Case | Operation | Variant | Output max abs | VJP max abs | Result |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        for comparison in row["comparisons"]:
            lines.append(
                "| {case} | {kind} | {variant} | {output} | {gradient} | {result} |".format(
                    case=row["id"],
                    kind=row["kind"],
                    variant=comparison["name"],
                    output=(
                        f"{comparison['output']['max_abs']:.3e}"
                        if "output" in comparison
                        else "—"
                    ),
                    gradient=(
                        f"{comparison['gradient_max_abs']:.3e}"
                        if "gradient_max_abs" in comparison
                        else "—"
                    ),
                    result="pass" if comparison.get("passed") else "FAIL",
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_torch_python() -> Path:
    candidate = ROOT / "evals" / ".venv-torch" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--cases-per-kind", type=int, default=10)
    parser.add_argument("--kind", action="append", choices=KINDS)
    parser.add_argument("--max-l", type=int, default=4)
    parser.add_argument("--torch-python", type=Path, default=_default_torch_python())
    parser.add_argument("--mlx-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--worker-backend", choices=("torch", "mlx"), help=argparse.SUPPRESS)
    parser.add_argument("--cases-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _worker(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))["cases"]
    result = _torch_worker(cases) if args.worker_backend == "torch" else _mlx_worker(cases)
    args.worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_backend:
        if args.cases_file is None or args.worker_output is None:
            raise SystemExit("worker mode requires --cases-file and --worker-output")
        return _worker(args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_dir or (
        ROOT / "evals" / "results" / "operation-randomized" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replay:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))
        cases = [replay["case"] if "case" in replay else replay]
        args.seed = cases[0]["seed"]
    else:
        cases = generate_cases(
            seed=args.seed,
            cases_per_kind=args.cases_per_kind,
            kinds=tuple(args.kind) if args.kind else KINDS,
            max_l=args.max_l,
        )
    cases_file = output_dir / "cases.json"
    cases_file.write_text(
        json.dumps(
            {"schema": 1, "seed": args.seed, "cases": cases},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    results = {}
    for backend, interpreter in (
        ("torch", args.torch_python),
        ("mlx", args.mlx_python),
    ):
        destination = output_dir / f"{backend}.json"
        subprocess.run(
            [
                str(interpreter),
                str(Path(__file__).resolve()),
                "--worker-backend",
                backend,
                "--cases-file",
                str(cases_file),
                "--worker-output",
                str(destination),
            ],
            cwd=ROOT,
            check=True,
        )
        results[backend] = json.loads(destination.read_text(encoding="utf-8"))
    report = compare_results(cases, results["torch"], results["mlx"])
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(report, output_dir / "report.md")
    for failure in report["failures"]:
        failure_dir = output_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        (failure_dir / f"{failure['case']['id']}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"Randomized operation report: {output_dir / 'report.md'}")
    print(
        f"Cases: {report['case_count']}; passed: {report['passed']}; "
        f"failed: {report['failed']}"
    )
    return 0 if report["failed"] == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
