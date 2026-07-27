"""Randomized numerical parity for important e3nn Torch and MLX operations."""

from __future__ import annotations

import argparse
from collections import Counter
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
        _numpy_value,
        _part,
    )
except ModuleNotFoundError:
    from randomized_tensor_product_parity import (
        ROOT,
        _format_irreps,
        _irreps_dim,
        _metrics,
        _numpy_value,
        _part,
    )


FAMILIES = (
    "wigner_3j",
    "rotations",
    "spherical_harmonics",
    "linear",
    "norm",
    "norm_activation",
    "gate",
    "radial",
    "scatter",
    "tensor_product_wrappers",
    "s2_activation",
    "so3_activation",
)


def _random_irreps_parts(
    rng: np.random.Generator,
    *,
    max_l: int,
    count: int | None = None,
    max_mul: int = 3,
) -> list[dict[str, int]]:
    count = int(rng.integers(1, 4)) if count is None else count
    return [
        _part(
            int(rng.integers(1, max_mul + 1)),
            int(rng.integers(0, max_l + 1)),
            1 if rng.integers(0, 2) else -1,
        )
        for _ in range(count)
    ]


def _array_spec(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    scale: float = 0.7,
) -> dict[str, Any]:
    values = rng.normal(0.0, scale, size=shape).astype(np.float32)
    return {"shape": list(shape), "values": values.tolist()}


def _base_case(
    family: str,
    seed: int,
    index: int,
    family_index: int,
) -> dict[str, Any]:
    return {
        "id": f"seed-{seed}-{family}-{family_index:04d}",
        "seed": seed,
        "index": index,
        "family_index": family_index,
        "family": family,
    }


def _generate_wigner(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    l1 = int(rng.integers(0, max_l + 1))
    l2 = int(rng.integers(0, max_l + 1))
    l3 = int(rng.integers(abs(l1 - l2), l1 + l2 + 1))
    case.update({"l1": l1, "l2": l2, "l3": l3, "tolerance": 2e-5})


def _generate_rotations(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    parts = _random_irreps_parts(rng, max_l=max_l, max_mul=2)
    batch = int(rng.integers(1, 5))
    angles = rng.normal(0.0, 1.5, size=(batch, 3)).astype(np.float32)
    case.update(
        {
            "irreps": _format_irreps(parts),
            "angles": {"shape": [batch, 3], "values": angles.tolist()},
            "tolerance": 2e-4,
        }
    )


def _generate_spherical_harmonics(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    lmax = int(rng.integers(0, max_l + 1))
    degrees = sorted(
        set(
            int(value)
            for value in rng.choice(
                np.arange(lmax + 1),
                size=int(rng.integers(1, lmax + 2)),
                replace=True,
            )
        )
    )
    layout = str(rng.choice(("vector", "batch", "grid")))
    if layout == "vector":
        shape = (3,)
    elif layout == "batch":
        shape = (int(rng.integers(1, 7)), 3)
    else:
        shape = (int(rng.integers(1, 4)), int(rng.integers(1, 4)), 3)
    case.update(
        {
            "degrees": degrees,
            "vectors": _array_spec(rng, shape),
            "normalize": bool(rng.integers(0, 2)),
            "normalization": str(rng.choice(("component", "norm", "integral"))),
            "layout": layout,
            "tolerance": 8e-5,
        }
    )


def _generate_linear(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    input_parts = _random_irreps_parts(rng, max_l=max_l, max_mul=3)
    output_parts = []
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.8:
            source = input_parts[int(rng.integers(0, len(input_parts)))]
            output_parts.append(
                _part(int(rng.integers(1, 4)), source["l"], source["parity"])
            )
        else:
            output_parts.extend(
                _random_irreps_parts(rng, max_l=max_l, count=1, max_mul=2)
            )
    weight_numel = sum(
        source["mul"] * target["mul"]
        for source in input_parts
        for target in output_parts
        if source["l"] == target["l"] and source["parity"] == target["parity"]
    )
    instructions = [
        [input_index, output_index]
        for input_index, source in enumerate(input_parts)
        for output_index, target in enumerate(output_parts)
        if source["l"] == target["l"] and source["parity"] == target["parity"]
    ]
    bias_numel = sum(
        target["mul"]
        for target in output_parts
        if target["l"] == 0 and target["parity"] == 1
    )
    batch = int(rng.integers(1, 5))
    layout = str(rng.choice(("batch", "grid")))
    leading = (batch,) if layout == "batch" else (batch, int(rng.integers(1, 4)))
    shared_weights = bool(rng.integers(0, 2))
    weight_leading = () if shared_weights else (
        leading if rng.random() < 0.5 else (1,) * len(leading)
    )
    case.update(
        {
            "irreps_in": _format_irreps(input_parts),
            "irreps_out": _format_irreps(output_parts),
            "input": _array_spec(rng, (*leading, _irreps_dim(input_parts))),
            "weight": _array_spec(rng, (*weight_leading, weight_numel)),
            "instructions": instructions,
            "bias": (
                _array_spec(rng, (bias_numel,), scale=0.3)
                if bias_numel and layout == "batch" and shared_weights
                else None
            ),
            "shared_weights": shared_weights,
            "path_normalization": str(rng.choice(("element", "path"))),
            "layout": layout,
            "tolerance": 8e-5,
        }
    )


def _generate_norm(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    parts = _random_irreps_parts(rng, max_l=max_l, max_mul=3)
    leading = (int(rng.integers(1, 5)), int(rng.integers(1, 4)))
    case.update(
        {
            "irreps": _format_irreps(parts),
            "input": _array_spec(rng, (*leading, _irreps_dim(parts))),
            "squared": bool(rng.integers(0, 2)),
            "tolerance": 4e-5,
        }
    )


def _generate_norm_activation(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    parts = _random_irreps_parts(rng, max_l=max_l, max_mul=3)
    # e3nn 0.5.8 cannot construct normalize=False with epsilon=None.
    normalize = True
    case.update(
        {
            "irreps": _format_irreps(parts),
            "input": _array_spec(
                rng,
                (int(rng.integers(1, 5)), _irreps_dim(parts)),
            ),
            "normalize": normalize,
            "epsilon": 1e-6 if normalize else None,
            "tolerance": 8e-5,
        }
    )


def _generate_gate(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    scalar_parts = [
        _part(int(rng.integers(1, 4)), 0, 1),
        _part(int(rng.integers(1, 3)), 0, -1),
    ]
    gated_parts = [
        _part(
            int(rng.integers(1, 4)),
            int(rng.integers(1, max(2, max_l + 1))),
            1 if rng.integers(0, 2) else -1,
        )
        for _ in range(int(rng.integers(1, 3)))
    ]
    gate_count = sum(part["mul"] for part in gated_parts)
    gate_parts = [_part(gate_count, 0, 1)]
    # Upstream Gate sorts its combined input irreps.
    input_parts = sorted(
        [*scalar_parts, *gate_parts, *gated_parts],
        key=lambda part: (part["l"], part["parity"]),
    )
    case.update(
        {
            "irreps_scalars": _format_irreps(scalar_parts),
            "scalar_activations": ["tanh"] * len(scalar_parts),
            "irreps_gates": _format_irreps(gate_parts),
            "gate_activations": ["sigmoid"],
            "irreps_gated": _format_irreps(gated_parts),
            "input": _array_spec(
                rng,
                (int(rng.integers(1, 6)), _irreps_dim(input_parts)),
            ),
            # MLX's sigmoid approximation differs slightly from Torch's
            # implementation while preserving the same gate semantics.
            "tolerance": 2e-3,
        }
    )


def _generate_radial(
    rng: np.random.Generator,
    case: dict[str, Any],
    **_kwargs,
) -> None:
    start = float(rng.uniform(-1.5, -0.2))
    end = float(rng.uniform(0.5, 2.5))
    shape = (
        int(rng.integers(1, 7)),
        int(rng.integers(1, 4)),
    )
    values = rng.uniform(start - 0.4, end + 0.4, size=shape).astype(np.float32)
    case.update(
        {
            "input": {"shape": list(shape), "values": values.tolist()},
            "start": start,
            "end": end,
            "number": int(rng.integers(2, 9)),
            "basis": str(
                rng.choice(
                    ("fourier", "bessel", "gaussian", "cosine", "smooth_finite")
                )
            ),
            "cutoff": bool(rng.integers(0, 2)),
            "tolerance": 8e-5,
        }
    )


def _generate_scatter(
    rng: np.random.Generator,
    case: dict[str, Any],
    **_kwargs,
) -> None:
    dim_size = int(rng.integers(1, 9))
    edges = int(rng.integers(0, 20))
    width = int(rng.integers(1, 9))
    index = rng.integers(0, dim_size, size=(edges,), dtype=np.int32)
    case.update(
        {
            "source": _array_spec(rng, (edges, width)),
            "index": {"shape": [edges], "values": index.tolist()},
            "dim_size": dim_size,
            "tolerance": 2e-6,
        }
    )


def _couples(first: dict[str, int], second: dict[str, int], output: dict[str, int]) -> bool:
    return (
        first["parity"] * second["parity"] == output["parity"]
        and abs(first["l"] - second["l"]) <= output["l"] <= first["l"] + second["l"]
    )


def _unique_irreps_parts(
    rng: np.random.Generator,
    *,
    max_l: int,
    count: int,
    multiplicities: list[int] | None = None,
) -> list[dict[str, int]]:
    choices = [(l, parity) for l in range(max_l + 1) for parity in (-1, 1)]
    selected = rng.choice(len(choices), size=count, replace=False)
    return [
        _part(
            (
                multiplicities[index]
                if multiplicities is not None
                else int(rng.integers(1, 3))
            ),
            choices[int(choice)][0],
            choices[int(choice)][1],
        )
        for index, choice in enumerate(selected)
    ]


def _merge_duplicate_irreps(parts: list[dict[str, int]]) -> list[dict[str, int]]:
    merged: list[dict[str, int]] = []
    locations: dict[tuple[int, int], int] = {}
    for part in parts:
        key = (part["l"], part["parity"])
        if key in locations:
            merged[locations[key]]["mul"] += part["mul"]
        else:
            locations[key] = len(merged)
            merged.append(dict(part))
    return merged


def _generate_wrapper(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    kind = ("full", "fully_connected", "elementwise", "tensor_square")[
        case["family_index"] % 4
    ]
    block_count = int(rng.integers(1, min(4, 2 * (max_l + 1)) + 1))
    first = _unique_irreps_parts(
        rng, max_l=max_l, count=block_count
    )
    if kind == "elementwise":
        second = _unique_irreps_parts(
            rng,
            max_l=max_l,
            count=len(first),
            multiplicities=[part["mul"] for part in first],
        )
    else:
        second = _unique_irreps_parts(
            rng,
            max_l=max_l,
            count=int(rng.integers(1, min(4, 2 * (max_l + 1)) + 1)),
        )
    batch = int(rng.integers(1, 5))
    case.update(
        {
            "kind": kind,
            "irreps_in1": _format_irreps(first),
            "left": _array_spec(rng, (batch, _irreps_dim(first))),
            "tolerance": 1e-4,
        }
    )
    if kind != "tensor_square":
        case["irreps_in2"] = _format_irreps(second)
        case["right"] = _array_spec(rng, (batch, _irreps_dim(second)))
    if kind == "fully_connected":
        candidates = []
        for left_part in first:
            for right_part in second:
                lout = int(
                    rng.integers(
                        abs(left_part["l"] - right_part["l"]),
                        left_part["l"] + right_part["l"] + 1,
                    )
                )
                candidates.append(
                    _part(
                        int(rng.integers(1, 3)),
                        lout,
                        left_part["parity"] * right_part["parity"],
                    )
                )
        output = _merge_duplicate_irreps([
            candidates[int(rng.integers(0, len(candidates)))]
            for _ in range(int(rng.integers(1, min(4, len(candidates) + 1))))
        ])
        weight_numel = sum(
            left_part["mul"] * right_part["mul"] * out_part["mul"]
            for left_part in first
            for right_part in second
            for out_part in output
            if _couples(left_part, right_part, out_part)
        )
        case["irreps_out"] = _format_irreps(output)
        case["weight"] = _array_spec(rng, (weight_numel,))


def _generate_s2(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    lmax_in = int(rng.integers(0, min(3, max_l) + 1))
    lmax_out = int(rng.integers(0, min(3, max_l) + 1))
    parts = [_part(1, l, (-1) ** l) for l in range(lmax_in + 1)]
    case.update(
        {
            "irreps": _format_irreps(parts),
            "lmax_out": lmax_out,
            "resolution": int(rng.choice((12, 16, 20))),
            "normalization": str(rng.choice(("component", "norm"))),
            "input": _array_spec(
                rng,
                (int(rng.integers(1, 4)), (lmax_in + 1) ** 2),
            ),
            "tolerance": 1.2e-3,
        }
    )


def _so3_dim(lmax: int) -> int:
    return sum((2 * l + 1) ** 2 for l in range(lmax + 1))


def _generate_so3(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    max_l: int,
) -> None:
    lmax_in = int(rng.integers(0, min(2, max_l) + 1))
    lmax_out = int(rng.integers(0, min(2, max_l) + 1))
    case.update(
        {
            "lmax_in": lmax_in,
            "lmax_out": lmax_out,
            "resolution": int(rng.choice((4, 5, 6))),
            "normalization": "component",
            "input": _array_spec(
                rng,
                (int(rng.integers(1, 3)), _so3_dim(lmax_in)),
            ),
            "tolerance": 1.5e-3,
        }
    )


_GENERATORS = {
    "wigner_3j": _generate_wigner,
    "rotations": _generate_rotations,
    "spherical_harmonics": _generate_spherical_harmonics,
    "linear": _generate_linear,
    "norm": _generate_norm,
    "norm_activation": _generate_norm_activation,
    "gate": _generate_gate,
    "radial": _generate_radial,
    "scatter": _generate_scatter,
    "tensor_product_wrappers": _generate_wrapper,
    "s2_activation": _generate_s2,
    "so3_activation": _generate_so3,
}


def generate_core_cases(
    *,
    seed: int,
    cases_per_family: int,
    families: tuple[str, ...] = FAMILIES,
    max_l: int = 4,
) -> list[dict[str, Any]]:
    if cases_per_family < 1:
        raise ValueError("cases_per_family must be positive")
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown operation families: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    cases = []
    index = 0
    for family in families:
        for family_index in range(cases_per_family):
            case = _base_case(family, seed, index, family_index)
            _GENERATORS[family](rng, case, max_l=max_l)
            cases.append(case)
            index += 1
    return cases


def _activation(name: str, framework):
    return {"tanh": framework.tanh, "sigmoid": framework.sigmoid}[name]


def _torch_module_output(case: dict[str, Any]) -> np.ndarray:
    import torch
    from e3nn import nn, o3

    family = case["family"]
    if family == "wigner_3j":
        output = o3.wigner_3j(case["l1"], case["l2"], case["l3"])
    elif family == "rotations":
        angles = torch.from_numpy(_numpy_value(case["angles"]))
        matrix = o3.angles_to_matrix(angles[:, 0], angles[:, 1], angles[:, 2])
        representation = o3.Irreps(case["irreps"]).D_from_angles(
            angles[:, 0], angles[:, 1], angles[:, 2]
        )
        output = torch.cat(
            [matrix.reshape(matrix.shape[0], -1), representation.reshape(representation.shape[0], -1)],
            dim=-1,
        )
    elif family == "spherical_harmonics":
        output = o3.spherical_harmonics(
            case["degrees"],
            torch.from_numpy(_numpy_value(case["vectors"])),
            normalize=case["normalize"],
            normalization=case["normalization"],
        )
    elif family == "linear":
        module = o3.Linear(
            case["irreps_in"],
            case["irreps_out"],
            internal_weights=False,
            shared_weights=case["shared_weights"],
            biases=case["bias"] is not None,
            path_normalization=case["path_normalization"],
            instructions=[tuple(value) for value in case["instructions"]],
        )
        value = torch.from_numpy(_numpy_value(case["input"]))
        weight = torch.from_numpy(_numpy_value(case["weight"]))
        bias = (
            None
            if case["bias"] is None
            else torch.from_numpy(_numpy_value(case["bias"]))
        )
        output = module(value, weight, bias)
    elif family == "norm":
        output = o3.Norm(case["irreps"], squared=case["squared"])(
            torch.from_numpy(_numpy_value(case["input"]))
        )
    elif family == "norm_activation":
        output = nn.NormActivation(
            case["irreps"],
            torch.tanh,
            normalize=case["normalize"],
            epsilon=case["epsilon"],
        )(torch.from_numpy(_numpy_value(case["input"])))
    elif family == "gate":
        module = nn.Gate(
            case["irreps_scalars"],
            [_activation(name, torch) for name in case["scalar_activations"]],
            case["irreps_gates"],
            [_activation(name, torch) for name in case["gate_activations"]],
            case["irreps_gated"],
        )
        output = module(torch.from_numpy(_numpy_value(case["input"])))
    elif family == "radial":
        from e3nn.math import soft_one_hot_linspace

        output = soft_one_hot_linspace(
            torch.from_numpy(_numpy_value(case["input"])),
            case["start"],
            case["end"],
            case["number"],
            basis=case["basis"],
            cutoff=case["cutoff"],
        )
    elif family == "scatter":
        source = torch.from_numpy(_numpy_value(case["source"]))
        index = torch.as_tensor(case["index"]["values"], dtype=torch.long)
        output = torch.zeros(
            (case["dim_size"], source.shape[-1]), dtype=source.dtype
        ).index_add(0, index, source)
    elif family == "tensor_product_wrappers":
        kind = case["kind"]
        left = torch.from_numpy(_numpy_value(case["left"]))
        if kind == "full":
            module = o3.FullTensorProduct(case["irreps_in1"], case["irreps_in2"])
            output = module(left, torch.from_numpy(_numpy_value(case["right"])))
        elif kind == "fully_connected":
            module = o3.FullyConnectedTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                case["irreps_out"],
                internal_weights=False,
            )
            output = module(
                left,
                torch.from_numpy(_numpy_value(case["right"])),
                torch.from_numpy(_numpy_value(case["weight"])),
            )
        elif kind == "elementwise":
            module = o3.ElementwiseTensorProduct(
                case["irreps_in1"], case["irreps_in2"]
            )
            output = module(left, torch.from_numpy(_numpy_value(case["right"])))
        else:
            module = o3.TensorSquare(case["irreps_in1"])
            output = module(left)
    elif family == "s2_activation":
        output = nn.S2Activation(
            case["irreps"],
            torch.tanh,
            case["resolution"],
            normalization=case["normalization"],
            lmax_out=case["lmax_out"],
            random_rot=False,
        )(torch.from_numpy(_numpy_value(case["input"])))
    elif family == "so3_activation":
        output = nn.SO3Activation(
            case["lmax_in"],
            case["lmax_out"],
            torch.tanh,
            case["resolution"],
            normalization=case["normalization"],
        )(torch.from_numpy(_numpy_value(case["input"])))
    else:
        raise ValueError(f"unsupported family {family!r}")
    return output.detach().cpu().numpy()


def _torch_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import e3nn

    rows = []
    for case in cases:
        try:
            value = _torch_module_output(case)
            rows.append(
                {
                    "id": case["id"],
                    "status": "ok",
                    "shape": list(value.shape),
                    "output": value.tolist(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case["id"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "backend": "torch",
        "torch": torch.__version__,
        "e3nn": e3nn.__version__,
        "results": rows,
    }


def _mlx_variants(case: dict[str, Any]) -> tuple[tuple[str, bool, bool], ...]:
    if (
        case["family"] == "tensor_product_wrappers"
        and case["kind"] == "tensor_square"
    ):
        return (("default", True, True),)
    if case["family"] in {"spherical_harmonics", "scatter", "tensor_product_wrappers"}:
        return (
            ("generic-eager", False, False),
            ("generic-compiled", True, False),
            ("custom-requested", True, True),
        )
    if case["family"] == "linear":
        return (("generic-eager", False, False), ("generic-compiled", True, False))
    return (("general", False, False),)


def _mlx_module_output(
    case: dict[str, Any],
    *,
    compiled: bool,
    custom: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import mlx.core as mx
    import e3nn_mlx as o3

    family = case["family"]
    metadata: dict[str, Any] = {}
    if family == "wigner_3j":
        output = o3.wigner_3j(case["l1"], case["l2"], case["l3"])
    elif family == "rotations":
        angles = mx.array(_numpy_value(case["angles"]))
        matrix = o3.angles_to_matrix(angles[:, 0], angles[:, 1], angles[:, 2])
        representation = o3.irreps_wigner_d(
            case["irreps"], angles[:, 0], angles[:, 1], angles[:, 2]
        )
        output = mx.concatenate(
            [matrix.reshape(matrix.shape[0], -1), representation.reshape(representation.shape[0], -1)],
            axis=-1,
        )
    elif family == "spherical_harmonics":
        output = o3.spherical_harmonics(
            case["degrees"],
            mx.array(_numpy_value(case["vectors"])),
            normalize=case["normalize"],
            normalization=case["normalization"],
            use_custom_kernel=custom,
        )
        metadata["custom_eligible"] = bool(
            custom
            and len(case["vectors"]["shape"]) == 2
            and case["vectors"]["shape"][0] > 0
            and max(case["degrees"]) <= 4
        )
        metadata["kernel_kind"] = (
            "spherical_harmonics" if metadata["custom_eligible"] else None
        )
    elif family == "linear":
        module = o3.Linear(
            case["irreps_in"],
            case["irreps_out"],
            internal_weights=False,
            shared_weights=case["shared_weights"],
            biases=case["bias"] is not None,
            path_normalization=case["path_normalization"],
            instructions=[tuple(value) for value in case["instructions"]],
            compile=compiled,
        )
        bias = (
            None if case["bias"] is None else mx.array(_numpy_value(case["bias"]))
        )
        output = module(
            o3.IrrepsArray(module.irreps_in, mx.array(_numpy_value(case["input"]))),
            mx.array(_numpy_value(case["weight"])),
            bias=bias,
        ).array
    elif family == "norm":
        module = o3.Norm(case["irreps"], squared=case["squared"])
        output = module(
            o3.IrrepsArray(module.irreps_in, mx.array(_numpy_value(case["input"])))
        ).array
    elif family == "norm_activation":
        module = o3.NormActivation(
            case["irreps"],
            mx.tanh,
            normalize=case["normalize"],
            epsilon=case["epsilon"],
        )
        output = module(
            o3.IrrepsArray(module.irreps_in, mx.array(_numpy_value(case["input"])))
        ).array
    elif family == "gate":
        module = o3.Gate(
            case["irreps_scalars"],
            [_activation(name, mx) for name in case["scalar_activations"]],
            case["irreps_gates"],
            [_activation(name, mx) for name in case["gate_activations"]],
            case["irreps_gated"],
        )
        output = module(
            o3.IrrepsArray(module.irreps_in, mx.array(_numpy_value(case["input"])))
        ).array
    elif family == "radial":
        output = o3.soft_one_hot_linspace(
            mx.array(_numpy_value(case["input"])),
            case["start"],
            case["end"],
            case["number"],
            basis=case["basis"],
            cutoff=case["cutoff"],
        )
    elif family == "scatter":
        source = mx.array(_numpy_value(case["source"]))
        index = mx.array(case["index"]["values"], dtype=mx.int32)
        output = o3.scatter_sum(
            source,
            index,
            case["dim_size"],
            use_custom_kernel=custom,
        )
        metadata["custom_eligible"] = bool(custom and case["source"]["shape"][0] > 0)
        metadata["kernel_kind"] = (
            "scatter_sum" if metadata["custom_eligible"] else None
        )
    elif family == "tensor_product_wrappers":
        kind = case["kind"]
        options = {
            "compile_left_right": compiled,
            "use_custom_kernel": custom,
        }
        left_values = mx.array(_numpy_value(case["left"]))
        if kind == "full":
            module = o3.FullTensorProduct(
                case["irreps_in1"], case["irreps_in2"], **options
            )
            output = module(
                o3.IrrepsArray(module.irreps_in1, left_values),
                o3.IrrepsArray(
                    module.irreps_in2, mx.array(_numpy_value(case["right"]))
                ),
            ).array
        elif kind == "fully_connected":
            module = o3.FullyConnectedTensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                case["irreps_out"],
                internal_weights=False,
                **options,
            )
            output = module(
                o3.IrrepsArray(module.irreps_in1, left_values),
                o3.IrrepsArray(
                    module.irreps_in2, mx.array(_numpy_value(case["right"]))
                ),
                mx.array(_numpy_value(case["weight"])),
            ).array
        elif kind == "elementwise":
            module = o3.ElementwiseTensorProduct(
                case["irreps_in1"], case["irreps_in2"], **options
            )
            output = module(
                o3.IrrepsArray(module.irreps_in1, left_values),
                o3.IrrepsArray(
                    module.irreps_in2, mx.array(_numpy_value(case["right"]))
                ),
            ).array
        else:
            module = o3.TensorSquare(
                case["irreps_in1"],
                compile_left_right=compiled,
            )
            output = module(o3.IrrepsArray(module.irreps_in, left_values)).array
        metadata["custom_eligible"] = bool(
            custom
            and module._try_metal(
                mx.array(_numpy_value(case["left"])),
                (
                    mx.array(_numpy_value(case["right"]))
                    if case.get("right") is not None
                    else mx.array(_numpy_value(case["left"]))
                ),
                (
                    None
                    if case.get("weight") is None
                    else mx.array(_numpy_value(case["weight"]))
                ),
            )
            is not None
        )
        metadata["kernel_kind"] = getattr(module, "_metal_kernel_kind", None)
    elif family == "s2_activation":
        output = o3.S2Activation(
            case["irreps"],
            mx.tanh,
            case["resolution"],
            normalization=case["normalization"],
            lmax_out=case["lmax_out"],
            random_rot=False,
        )(mx.array(_numpy_value(case["input"])))
    elif family == "so3_activation":
        output = o3.SO3Activation(
            case["lmax_in"],
            case["lmax_out"],
            mx.tanh,
            case["resolution"],
            normalization=case["normalization"],
        )(mx.array(_numpy_value(case["input"])))
    else:
        raise ValueError(f"unsupported family {family!r}")
    mx.eval(output)
    return np.asarray(output), metadata


def _mlx_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        variants = []
        for name, compiled, custom in _mlx_variants(case):
            try:
                value, metadata = _mlx_module_output(
                    case, compiled=compiled, custom=custom
                )
                variants.append(
                    {
                        "name": name,
                        "status": "ok",
                        "shape": list(value.shape),
                        "output": value.tolist(),
                        **metadata,
                    }
                )
            except Exception as exc:
                variants.append(
                    {
                        "name": name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        rows.append({"id": case["id"], "variants": variants})
    return {"backend": "mlx", "mlx": version("mlx"), "results": rows}


def compare_core_results(
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
        comparisons = []
        failed = reference["status"] != "ok"
        for variant in candidate["variants"]:
            comparison = {
                "name": variant["name"],
                "status": variant["status"],
                "custom_eligible": variant.get("custom_eligible", False),
                "kernel_kind": variant.get("kernel_kind"),
            }
            if reference["status"] == "ok" and variant["status"] == "ok":
                expected = np.asarray(reference["output"], dtype=np.float32)
                actual = np.asarray(variant["output"], dtype=np.float32)
                if expected.shape == actual.shape:
                    comparison.update(_metrics(actual, expected))
                    tolerance = float(case["tolerance"])
                    comparison["passed"] = bool(
                        comparison["max_abs"]
                        <= tolerance
                        + tolerance * float(np.max(np.abs(expected), initial=0.0))
                        and comparison["norm_scaled"] <= tolerance
                    )
                else:
                    comparison["passed"] = False
                    comparison["shape_mismatch"] = {
                        "torch": list(expected.shape),
                        "mlx": list(actual.shape),
                    }
            else:
                comparison["passed"] = False
                comparison["error"] = variant.get("error", reference.get("error"))
            failed = failed or not comparison["passed"]
            comparisons.append(comparison)
        row = {
            "id": case["id"],
            "family": case["family"],
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
    counts = Counter(row["family"] for row in rows if row["passed"])
    return {
        "schema": 1,
        "seed": cases[0]["seed"] if cases else None,
        "case_count": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "passed_by_family": dict(sorted(counts.items())),
        "torch_version": torch_result.get("torch"),
        "e3nn_version": torch_result.get("e3nn"),
        "mlx_version": mlx_result.get("mlx"),
        "rows": rows,
        "failures": failures,
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Randomized core-operation parity",
        "",
        f"- Seed: `{report['seed']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Passed: `{report['passed']}`",
        f"- Failed: `{report['failed']}`",
        f"- Torch/e3nn: `{report['torch_version']}` / `{report['e3nn_version']}`",
        f"- MLX: `{report['mlx_version']}`",
        "",
        "| Case | Family | Variant | Kernel | Max abs | Norm-scaled | Result |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        for comparison in row["comparisons"]:
            kernel = (
                comparison.get("kernel_kind")
                if comparison.get("custom_eligible")
                else "fallback"
            )
            lines.append(
                f"| {row['id']} | {row['family']} | {comparison['name']} | "
                f"{kernel or 'general'} | "
                f"{comparison.get('max_abs', float('nan')):.3e} | "
                f"{comparison.get('norm_scaled', float('nan')):.3e} | "
                f"{'pass' if comparison.get('passed') else 'FAIL'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_torch_python() -> Path:
    candidate = ROOT / "evals" / ".venv-torch" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--cases-per-family", type=int, default=12)
    parser.add_argument("--family", action="append", choices=FAMILIES)
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


def _worker_main(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))["cases"]
    result = _torch_worker(cases) if args.worker_backend == "torch" else _mlx_worker(cases)
    args.worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_backend:
        if args.cases_file is None or args.worker_output is None:
            raise SystemExit("worker mode requires --cases-file and --worker-output")
        return _worker_main(args)

    output_dir = args.output_dir or (
        ROOT
        / "evals"
        / "results"
        / "core-randomized"
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replay:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))
        cases = [replay["case"] if "case" in replay else replay]
    else:
        cases = generate_core_cases(
            seed=args.seed,
            cases_per_family=args.cases_per_family,
            families=tuple(args.family or FAMILIES),
            max_l=args.max_l,
        )
    cases_path = output_dir / "cases.json"
    cases_path.write_text(
        json.dumps({"schema": 1, "cases": cases}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    outputs = {}
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
                str(cases_path),
                "--worker-output",
                str(destination),
            ],
            cwd=ROOT,
            check=True,
        )
        outputs[backend] = json.loads(destination.read_text(encoding="utf-8"))

    report = compare_core_results(cases, outputs["torch"], outputs["mlx"])
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(report, output_dir / "report.md")
    for failure in report["failures"]:
        failure_dir = output_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        (failure_dir / f"{failure['case']['id']}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"Randomized core parity report: {output_dir / 'report.md'}")
    print(
        f"Cases: {report['case_count']}; passed: {report['passed']}; "
        f"failed: {report['failed']}"
    )
    return 0 if report["failed"] == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
