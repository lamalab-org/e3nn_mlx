"""Seeded randomized TensorProduct parity qualification for Torch and MLX.

The parent process generates JSON cases and NumPy values once, then invokes
isolated Torch and MLX worker interpreters. This keeps Torch out of the normal
MLX environment while ensuring both implementations consume identical data.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = ("uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw")
MLX_VARIANTS = (
    ("generic-eager", False, False),
    ("generic-compiled", True, False),
    ("custom-requested", True, True),
)


def _irrep(l: int, parity: int) -> str:
    return f"{l}{'e' if parity == 1 else 'o'}"


def _part(mul: int, l: int, parity: int) -> dict[str, int]:
    return {"mul": int(mul), "l": int(l), "parity": int(parity)}


def _format_irreps(parts: list[dict[str, int]]) -> str:
    return " + ".join(
        f"{part['mul']}x{_irrep(part['l'], part['parity'])}"
        for part in parts
    )


def _irreps_dim(parts: list[dict[str, int]]) -> int:
    return sum(part["mul"] * (2 * part["l"] + 1) for part in parts)


def _path_shape(mode: str, mul1: int, mul2: int, mul_out: int) -> tuple[int, ...]:
    if mode == "uvw":
        return mul1, mul2, mul_out
    if mode in {"uvu", "uvv"}:
        return mul1, mul2
    if mode == "uuw":
        return mul1, mul_out
    if mode == "uuu":
        return (mul1,)
    if mode == "uvuv":
        return mul1, mul2
    pair_count = mul1 * (mul1 - 1) // 2
    if mode == "uvu<v":
        return (pair_count,)
    if mode == "u<vw":
        return pair_count, mul_out
    raise ValueError(f"unsupported mode {mode!r}")


def _mode_multiplicities(
    rng: np.random.Generator,
    mode: str,
    max_mul: int,
) -> tuple[int, int, int]:
    if mode in {"uvu<v", "u<vw"}:
        mul1 = int(rng.integers(2, max(3, max_mul + 1)))
        mul2 = mul1
        if mode == "uvu<v":
            mul_out = mul1 * (mul1 - 1) // 2
        else:
            mul_out = int(rng.integers(1, max_mul + 1))
        return mul1, mul2, mul_out
    mul1 = int(rng.integers(1, max_mul + 1))
    mul2 = int(rng.integers(1, max_mul + 1))
    if mode == "uvw":
        mul_out = int(rng.integers(1, max_mul + 1))
    elif mode == "uvu":
        mul_out = mul1
    elif mode == "uvv":
        mul_out = mul2
    elif mode == "uuw":
        mul2 = mul1
        mul_out = int(rng.integers(1, max_mul + 1))
    elif mode == "uuu":
        mul2 = mul1
        mul_out = mul1
    elif mode == "uvuv":
        # Keep the Cartesian multiplicity manageable in randomized runs.
        mul1 = min(mul1, 3)
        mul2 = min(mul2, 3)
        mul_out = mul1 * mul2
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    return mul1, mul2, mul_out


def _insert_noise_parts(
    rng: np.random.Generator,
    core_parts: list[dict[str, int]],
    *,
    max_l: int,
    include_zero: bool,
) -> tuple[list[dict[str, int]], list[int]]:
    tagged: list[tuple[dict[str, int], int | None]] = [
        (part, index) for index, part in enumerate(core_parts)
    ]
    for _ in range(int(rng.integers(0, 3))):
        tagged.append(
            (
                _part(
                    int(rng.integers(1, 4)),
                    int(rng.integers(0, max_l + 1)),
                    1 if rng.integers(0, 2) else -1,
                ),
                None,
            )
        )
    if include_zero:
        for _ in range(int(rng.integers(1, 3))):
            tagged.append(
                (
                    _part(
                        0,
                        int(rng.integers(0, max_l + 1)),
                        1 if rng.integers(0, 2) else -1,
                    ),
                    None,
                )
            )
    rng.shuffle(tagged)
    positions = [0] * len(core_parts)
    for position, (_, core_index) in enumerate(tagged):
        if core_index is not None:
            positions[core_index] = position
    return [part for part, _ in tagged], positions


def _random_values(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> list[Any]:
    return rng.normal(0.0, 0.7, size=shape).astype(np.float32).tolist()


def generate_cases(
    *,
    seed: int,
    count: int,
    max_l: int = 4,
    max_mul: int = 4,
) -> list[dict[str, Any]]:
    """Generate reproducible, valid, mode-aware TensorProduct cases."""

    if count < 1:
        raise ValueError("count must be positive")
    if max_l < 0:
        raise ValueError("max_l must be non-negative")
    if max_mul < 2:
        raise ValueError("max_mul must be at least two")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    for index in range(count):
        # Cycle before sampling so every sufficiently large run covers all modes.
        mode = MODES[index % len(MODES)]
        mul1, mul2, mul_out = _mode_multiplicities(rng, mode, max_mul)
        l1 = int(rng.integers(0, max_l + 1))
        l2 = int(rng.integers(0, max_l + 1))
        lout = int(rng.integers(abs(l1 - l2), l1 + l2 + 1))
        parity1 = 1 if rng.integers(0, 2) else -1
        parity2 = 1 if rng.integers(0, 2) else -1
        parity_out = parity1 * parity2

        left_core_count = int(rng.integers(1, 4))
        right_core_count = int(rng.integers(1, 4))
        output_core_count = int(rng.integers(1, 3))
        left_core = [_part(mul1, l1, parity1) for _ in range(left_core_count)]
        right_core = [_part(mul2, l2, parity2) for _ in range(right_core_count)]
        output_core = [
            _part(mul_out, lout, parity_out) for _ in range(output_core_count)
        ]
        include_zero = bool(rng.random() < 0.45)
        left_parts, left_positions = _insert_noise_parts(
            rng, left_core, max_l=max_l, include_zero=include_zero
        )
        right_parts, right_positions = _insert_noise_parts(
            rng, right_core, max_l=max_l, include_zero=include_zero
        )
        output_parts, output_positions = _insert_noise_parts(
            rng, output_core, max_l=max_l, include_zero=include_zero
        )

        instruction_count = int(rng.integers(1, 5))
        triples = [
            (
                int(rng.integers(0, left_core_count)),
                int(rng.integers(0, right_core_count)),
                int(rng.integers(0, output_core_count)),
            )
            for _ in range(instruction_count)
        ]
        # Force repeated accumulation regularly instead of hoping random
        # topology happens to produce it.
        if instruction_count >= 2 and index % 3 == 0:
            triples[1] = (triples[1][0], triples[1][1], triples[0][2])

        requires_weights = mode in {"uvw", "u<vw"} or (
            mode == "uuw" and mul_out != 1
        )
        weighted_flags = [
            True if requires_weights else bool(rng.integers(0, 2))
            for _ in triples
        ]
        if not requires_weights and instruction_count >= 2 and index % 4 == 0:
            weighted_flags[0] = False
            weighted_flags[1] = True
        weight_numel = 0
        path_size = math.prod(_path_shape(mode, mul1, mul2, mul_out))
        instructions = []
        for triple, weighted in zip(triples, weighted_flags, strict=True):
            path_weight = float(rng.choice((0.25, 1.0, 4.0)))
            instructions.append(
                [
                    left_positions[triple[0]],
                    right_positions[triple[1]],
                    output_positions[triple[2]],
                    mode,
                    weighted,
                    path_weight,
                ]
            )
            if weighted:
                weight_numel += path_size

        shared_weights = weight_numel == 0 or bool(rng.integers(0, 2))
        if shared_weights:
            layout = str(
                rng.choice(("scalar-batch", "shared-grid", "shared-vector"))
            )
        else:
            layout = str(
                rng.choice(("unshared-batch", "unshared-singleton", "unshared-grid"))
            )

        dim1 = _irreps_dim(left_parts)
        dim2 = _irreps_dim(right_parts)
        if layout == "shared-vector":
            left_shape = (dim1,)
            right_shape = (dim2,)
            weight_shape = (weight_numel,)
        elif layout in {"shared-grid", "unshared-grid"}:
            batch = int(rng.integers(1, 4))
            frames = int(rng.integers(1, 4))
            left_shape = (batch, 1, dim1)
            right_shape = (1, frames, dim2)
            if layout == "unshared-grid":
                weight_shape = (
                    (batch, frames, weight_numel)
                    if rng.random() < 0.5
                    else (1, frames, weight_numel)
                )
            else:
                weight_shape = (weight_numel,)
        else:
            batch = int(rng.integers(1, 6))
            left_shape = (batch, dim1)
            right_shape = (batch, dim2)
            if layout == "unshared-batch":
                weight_shape = (batch, weight_numel)
            elif layout == "unshared-singleton":
                weight_shape = (1, weight_numel)
            else:
                weight_shape = (weight_numel,)

        case = {
            "id": f"seed-{seed}-case-{index:04d}",
            "seed": seed,
            "index": index,
            "mode": mode,
            "irreps_in1": _format_irreps(left_parts),
            "irreps_in2": _format_irreps(right_parts),
            "irreps_out": _format_irreps(output_parts),
            "instructions": instructions,
            "irrep_normalization": str(rng.choice(("component", "norm"))),
            "path_normalization": str(rng.choice(("element", "path"))),
            "shared_weights": shared_weights,
            "weight_layout": layout,
            "weight_numel": weight_numel,
            "left": {"shape": list(left_shape), "values": _random_values(rng, left_shape)},
            "right": {
                "shape": list(right_shape),
                "values": _random_values(rng, right_shape),
            },
            "weight": (
                None
                if weight_numel == 0
                else {
                    "shape": list(weight_shape),
                    "values": _random_values(rng, weight_shape),
                }
            ),
        }
        cases.append(case)
    return cases


def _numpy_value(spec: dict[str, Any]) -> np.ndarray:
    return np.asarray(spec["values"], dtype=np.float32).reshape(spec["shape"])


def _torch_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from e3nn import o3

    results = []
    for case in cases:
        try:
            module = o3.TensorProduct(
                case["irreps_in1"],
                case["irreps_in2"],
                case["irreps_out"],
                [tuple(instruction) for instruction in case["instructions"]],
                internal_weights=False,
                shared_weights=case["shared_weights"],
                irrep_normalization=case["irrep_normalization"],
                path_normalization=case["path_normalization"],
                _specialized_code=False,
            )
            left = torch.from_numpy(_numpy_value(case["left"]))
            right = torch.from_numpy(_numpy_value(case["right"]))
            weight = (
                None
                if case["weight"] is None
                else torch.from_numpy(_numpy_value(case["weight"]))
            )
            output = module(left, right, weight) if weight is not None else module(left, right)
            value = output.detach().cpu().numpy()
            results.append(
                {
                    "id": case["id"],
                    "status": "ok",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "output": value.tolist(),
                }
            )
        except Exception as exc:  # qualification output must retain the case
            results.append(
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
        "results": results,
    }


def _mlx_worker(cases: list[dict[str, Any]]) -> dict[str, Any]:
    import mlx.core as mx

    import e3nn_mlx as o3

    results = []
    for case in cases:
        variants = []
        for name, compile_left_right, use_custom_kernel in MLX_VARIANTS:
            try:
                module = o3.TensorProduct(
                    case["irreps_in1"],
                    case["irreps_in2"],
                    case["irreps_out"],
                    [tuple(instruction) for instruction in case["instructions"]],
                    internal_weights=False,
                    shared_weights=case["shared_weights"],
                    irrep_normalization=case["irrep_normalization"],
                    path_normalization=case["path_normalization"],
                    compile_left_right=compile_left_right,
                    use_custom_kernel=use_custom_kernel,
                )
                left = mx.array(_numpy_value(case["left"]))
                right = mx.array(_numpy_value(case["right"]))
                weight = (
                    None
                    if case["weight"] is None
                    else mx.array(_numpy_value(case["weight"]))
                )
                eligible = False
                if use_custom_kernel:
                    eligible = module._try_metal(left, right, weight) is not None
                output = module(
                    o3.IrrepsArray(module.irreps_in1, left),
                    o3.IrrepsArray(module.irreps_in2, right),
                    weight,
                ).array
                mx.eval(output)
                value = np.asarray(output)
                variants.append(
                    {
                        "name": name,
                        "status": "ok",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "output": value.tolist(),
                        "custom_eligible": eligible,
                        "kernel_kind": getattr(module, "_metal_kernel_kind", None),
                    }
                )
            except Exception as exc:  # qualification output must retain the case
                variants.append(
                    {
                        "name": name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        results.append({"id": case["id"], "variants": variants})
    return {
        "backend": "mlx",
        "mlx": version("mlx"),
        "results": results,
    }


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = np.abs(actual - expected)
    absolute = float(np.max(difference)) if difference.size else 0.0
    relative = (
        float(np.max(difference / np.maximum(np.abs(expected), 1e-8)))
        if difference.size
        else 0.0
    )
    denominator = float(np.linalg.norm(expected.reshape(-1)))
    normalized = float(np.linalg.norm(difference.reshape(-1)) / (denominator + 1e-12))
    return {
        "max_abs": absolute,
        "max_rel": relative,
        "norm_scaled": normalized,
    }


def compare_results(
    cases: list[dict[str, Any]],
    torch_result: dict[str, Any],
    mlx_result: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    norm_tolerance: float,
) -> dict[str, Any]:
    """Compare workers and retain metrics and complete replayable failures."""

    torch_rows = {row["id"]: row for row in torch_result["results"]}
    mlx_rows = {row["id"]: row for row in mlx_result["results"]}
    rows = []
    failures = []
    for case in cases:
        case_id = case["id"]
        reference = torch_rows[case_id]
        candidate = mlx_rows[case_id]
        comparisons = []
        case_failed = reference["status"] != "ok"
        for variant in candidate["variants"]:
            comparison: dict[str, Any] = {
                "name": variant["name"],
                "status": variant["status"],
                "custom_eligible": variant.get("custom_eligible", False),
                "kernel_kind": variant.get("kernel_kind"),
            }
            if reference["status"] == "ok" and variant["status"] == "ok":
                expected = np.asarray(reference["output"], dtype=np.float32)
                actual = np.asarray(variant["output"], dtype=np.float32)
                if actual.shape != expected.shape:
                    comparison["passed"] = False
                    comparison["shape_mismatch"] = {
                        "torch": list(expected.shape),
                        "mlx": list(actual.shape),
                    }
                else:
                    values = _metrics(actual, expected)
                    comparison.update(values)
                    comparison["passed"] = bool(
                        values["max_abs"] <= atol + rtol * float(np.max(np.abs(expected)))
                        and values["norm_scaled"] <= norm_tolerance
                    )
            else:
                comparison["passed"] = False
                if variant["status"] == "error":
                    comparison["error"] = variant["error"]
            case_failed = case_failed or not comparison["passed"]
            comparisons.append(comparison)
        row = {
            "id": case_id,
            "mode": case["mode"],
            "weight_layout": case["weight_layout"],
            "shared_weights": case["shared_weights"],
            "torch_status": reference["status"],
            "comparisons": comparisons,
            "passed": not case_failed,
        }
        rows.append(row)
        if case_failed:
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
        "mlx_version": mlx_result.get("mlx"),
        "rows": rows,
        "failures": failures,
    }


def _write_markdown(report: dict[str, Any], destination: Path) -> None:
    lines = [
        "# Randomized TensorProduct parity",
        "",
        f"- Seed: `{report['seed']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Passed: `{report['passed']}`",
        f"- Failed: `{report['failed']}`",
        f"- Torch: `{report['torch_version']}`",
        f"- MLX: `{report['mlx_version']}`",
        "",
        "| Case | Mode | Layout | Variant | Kernel | Max abs | Norm-scaled | Result |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        for comparison in row["comparisons"]:
            lines.append(
                "| {case} | {mode} | {layout} | {variant} | {kernel} | "
                "{absolute} | {normalized} | {result} |".format(
                    case=row["id"],
                    mode=row["mode"],
                    layout=row["weight_layout"],
                    variant=comparison["name"],
                    kernel=(
                        comparison.get("kernel_kind")
                        if comparison.get("custom_eligible")
                        else "fallback"
                    )
                    or "general",
                    absolute=(
                        f"{comparison['max_abs']:.3e}"
                        if "max_abs" in comparison
                        else "—"
                    ),
                    normalized=(
                        f"{comparison['norm_scaled']:.3e}"
                        if "norm_scaled" in comparison
                        else "—"
                    ),
                    result="pass" if comparison.get("passed") else "FAIL",
                )
            )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker_main(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))["cases"]
    if args.worker_backend == "torch":
        result = _torch_worker(cases)
    else:
        result = _mlx_worker(cases)
    args.worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _default_torch_python() -> Path:
    candidate = ROOT / "evals" / ".venv-torch" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--max-l", type=int, default=4)
    parser.add_argument("--max-mul", type=int, default=4)
    parser.add_argument("--torch-python", type=Path, default=_default_torch_python())
    parser.add_argument("--mlx-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay one saved failure JSON instead of generating cases.",
    )
    parser.add_argument("--atol", type=float, default=6e-5)
    parser.add_argument("--rtol", type=float, default=6e-5)
    parser.add_argument("--norm-tolerance", type=float, default=6e-5)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--worker-backend", choices=("torch", "mlx"), help=argparse.SUPPRESS)
    parser.add_argument("--cases-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_backend:
        if args.cases_file is None or args.worker_output is None:
            raise SystemExit("worker mode requires --cases-file and --worker-output")
        return _worker_main(args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_dir or (
        ROOT / "evals" / "results" / "tensor-product-randomized" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replay is not None:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))
        cases = [replay["case"] if "case" in replay else replay]
        args.seed = int(cases[0]["seed"])
    else:
        cases = generate_cases(
            seed=args.seed,
            count=args.cases,
            max_l=args.max_l,
            max_mul=args.max_mul,
        )
    cases_file = output_dir / "cases.json"
    cases_file.write_text(
        json.dumps(
            {
                "schema": 1,
                "seed": args.seed,
                "max_l": args.max_l,
                "max_mul": args.max_mul,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve()
    worker_outputs = {}
    for backend, interpreter in (
        ("torch", args.torch_python),
        ("mlx", args.mlx_python),
    ):
        destination = output_dir / f"{backend}.json"
        command = [
            str(interpreter),
            str(script),
            "--worker-backend",
            backend,
            "--cases-file",
            str(cases_file),
            "--worker-output",
            str(destination),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        worker_outputs[backend] = json.loads(destination.read_text(encoding="utf-8"))

    report = compare_results(
        cases,
        worker_outputs["torch"],
        worker_outputs["mlx"],
        atol=args.atol,
        rtol=args.rtol,
        norm_tolerance=args.norm_tolerance,
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "report.md")
    failure_dir = output_dir / "failures"
    for failure in report["failures"]:
        failure_dir.mkdir(exist_ok=True)
        (failure_dir / f"{failure['case']['id']}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"Randomized parity report: {output_dir / 'report.md'}")
    print(
        f"Cases: {report['case_count']}; passed: {report['passed']}; "
        f"failed: {report['failed']}"
    )
    return 0 if report["failed"] == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
