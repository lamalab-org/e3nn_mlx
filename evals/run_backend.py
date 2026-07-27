#!/usr/bin/env python3
"""Run one isolated worker for the compact three-way benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import cold_start, measure, new_run_metadata, write_document
from evals.workloads import DEFAULT_TIMING, case_names, get_workloads


def _value(text: str):
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def apply_overrides(workloads: dict[str, dict[str, Any]], overrides: list[str]) -> None:
    for override in overrides:
        try:
            target, raw_value = override.split("=", 1)
            case, key = target.split(".", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid override {override!r}; expected CASE.KEY=VALUE"
            ) from exc
        if case not in workloads:
            raise ValueError(f"override refers to unselected case {case!r}")
        if key not in workloads[case]:
            raise ValueError(f"case {case!r} has no setting {key!r}")
        workloads[case][key] = _value(raw_value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("torch-cpu", "mlx", "mlx-kernel"),
        required=True,
    )
    parser.add_argument("--preset", choices=tuple(DEFAULT_TIMING), default="smoke")
    parser.add_argument("--case", action="append", choices=case_names(), dest="cases")
    parser.add_argument(
        "--phase",
        action="append",
        choices=("forward", "train"),
        dest="phases",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args(argv)


def _result_row(
    task,
    *,
    backend: str,
    execution: str,
    phase: str,
    timing: dict[str, Any] | None = None,
    compile_ms: float | None = None,
    peak_memory_bytes: int | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    row = {
        "backend": backend,
        "execution": execution,
        "phase": phase,
        "case": task.name,
        "description": task.description,
        "config": task.config,
        "item_count": int(task.item_count),
        "compile_ms": compile_ms,
        "peak_memory_bytes": peak_memory_bytes,
        "status": status,
        "error": error,
    }
    if timing is not None:
        row.update(timing)
        row["items_per_second"] = (
            task.item_count / (timing["median_ms"] / 1_000)
            if timing["median_ms"] > 0
            else None
        )
    return row


def _benchmark(
    task,
    *,
    backend: str,
    phase: str,
    function,
    compiler,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    execution = "eager" if compiler is None else "compiled"
    try:
        if task.reset_peak_memory is not None:
            task.reset_peak_memory()
        compile_ms = None
        measured = function
        if compiler is not None:
            measured, compile_ms = cold_start(compiler, task.synchronize)
        timing = measure(
            measured,
            task.synchronize,
            warmup=warmup,
            samples=samples,
            inner_repeats=1,
        )
        peak = task.peak_memory() if task.peak_memory is not None else None
        return _result_row(
            task,
            backend=backend,
            execution=execution,
            phase=phase,
            timing=timing,
            compile_ms=compile_ms,
            peak_memory_bytes=int(peak) if peak is not None else None,
        )
    except Exception as exc:
        return _result_row(
            task,
            backend=backend,
            execution=execution,
            phase=phase,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        print("\n".join(case_names()))
        return 0

    workloads = get_workloads(args.preset, args.cases)
    apply_overrides(workloads, args.overrides)
    defaults = DEFAULT_TIMING[args.preset]
    warmup = defaults["warmup"] if args.warmup is None else args.warmup
    samples = defaults["samples"] if args.samples is None else args.samples
    phases = args.phases or ["forward", "train"]

    if args.backend == "torch-cpu":
        from evals import torch_cases as implementation

        implementation.configure()
        device = "cpu"
        execution = "eager"
    else:
        import mlx.core as mx

        from evals import mlx_cases as implementation

        implementation.configure(use_custom_kernels=args.backend == "mlx-kernel")
        mx.random.seed(0)
        device = str(mx.default_device())
        execution = "compiled"

    metadata = new_run_metadata(args.preset, args.backend, device)
    metadata["backend_details"] = implementation.backend_metadata()
    metadata["timing"] = {"warmup": warmup, "samples": samples}
    metadata["overrides"] = args.overrides
    results = []

    for name, config in workloads.items():
        print(f"[{args.backend}] constructing {name}", flush=True)
        try:
            task = implementation.BUILDERS[name](config)
        except Exception as exc:
            results.append(
                {
                    "preset": args.preset,
                    "backend": args.backend,
                    "execution": "construction",
                    "phase": "setup",
                    "case": name,
                    "description": "",
                    "config": config,
                    "item_count": 0,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        modes = []
        if "forward" in phases:
            modes.append(
                (
                    "forward",
                    task.forward if execution == "eager" else None,
                    task.compile_forward if execution == "compiled" else None,
                )
            )
        if "train" in phases:
            modes.append(
                (
                    "train",
                    task.train if execution == "eager" else None,
                    task.compile_train if execution == "compiled" else None,
                )
            )
        for phase, function, compiler in modes:
            print(f"[{args.backend}] {name}: {phase}", flush=True)
            row = _benchmark(
                task,
                backend=args.backend,
                phase=phase,
                function=function,
                compiler=compiler,
                warmup=warmup,
                samples=samples,
            )
            row["preset"] = args.preset
            results.append(row)
            if row["status"] == "ok":
                print(f"  median {row['median_ms']:.3f} ms", flush=True)
            else:
                print(f"  ERROR {row['error']}", flush=True)

    write_document(args.output, metadata, results)
    failures = sum(row["status"] != "ok" for row in results)
    print(
        json.dumps(
            {"output": str(args.output), "rows": len(results), "failures": failures}
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
