#!/usr/bin/env python3
"""Run one benchmark backend inside its isolated Python interpreter."""

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


def result_row(
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
        "family": task.family,
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


def benchmark_mode(
    task,
    *,
    backend: str,
    execution: str,
    phase: str,
    function,
    compiler,
    warmup: int,
    samples: int,
    inner_repeats: int,
) -> dict[str, Any]:
    try:
        if task.reset_peak_memory is not None:
            task.reset_peak_memory()
        compile_ms = None
        measured = function
        if execution == "compiled":
            measured, compile_ms = cold_start(compiler, task.synchronize)
        timing = measure(
            measured,
            task.synchronize,
            warmup=warmup,
            samples=samples,
            inner_repeats=inner_repeats,
        )
        peak = task.peak_memory() if task.peak_memory is not None else None
        return result_row(
            task,
            backend=backend,
            execution=execution,
            phase=phase,
            timing=timing,
            compile_ms=compile_ms,
            peak_memory_bytes=int(peak) if peak is not None else None,
        )
    except Exception as exc:  # benchmark failures belong in the result artifact
        return result_row(
            task,
            backend=backend,
            execution=execution,
            phase=phase,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "torch"), required=True)
    parser.add_argument("--preset", choices=tuple(DEFAULT_TIMING), default="smoke")
    parser.add_argument("--case", action="append", choices=case_names(), dest="cases")
    parser.add_argument("--phase", action="append", choices=("forward", "train"), dest="phases")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--inner-repeats", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args(argv)


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
    repeats = (
        defaults["inner_repeats"]
        if args.inner_repeats is None
        else args.inner_repeats
    )
    phases = args.phases or ["forward", "train"]

    if args.backend == "mlx":
        import mlx.core as mx

        from evals import mlx_cases as implementation

        mx.random.seed(0)
        device = str(mx.default_device())
    else:
        from evals import torch_cases as implementation

        implementation.configure(
            device=args.device, enable_compile=args.torch_compile
        )
        device = implementation.backend_metadata()["device"]

    result_backend = f"torch-{device}" if args.backend == "torch" else args.backend
    metadata = new_run_metadata(args.preset, result_backend, device)
    metadata["backend_details"] = implementation.backend_metadata()
    metadata["timing"] = {
        "warmup": warmup,
        "samples": samples,
        "inner_repeats": repeats,
    }
    metadata["overrides"] = args.overrides
    results = []
    for name, config in workloads.items():
        print(f"[{result_backend}] constructing {name}", flush=True)
        try:
            task = implementation.BUILDERS[name](config)
        except Exception as exc:
            results.append(
                {
                    "preset": args.preset,
                    "backend": result_backend,
                    "execution": "construction",
                    "phase": "setup",
                    "case": name,
                    "family": name,
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
            modes.append(("eager", "forward", task.forward, None))
            if task.compile_forward is not None:
                modes.append(
                    ("compiled", "forward", None, task.compile_forward)
                )
        if "train" in phases and task.train is not None:
            modes.append(("eager", "train", task.train, None))
            if task.compile_train is not None:
                modes.append(("compiled", "train", None, task.compile_train))
        for execution, phase, function, compiler in modes:
            print(f"[{result_backend}] {name}: {execution} {phase}", flush=True)
            row = benchmark_mode(
                task,
                backend=result_backend,
                execution=execution,
                phase=phase,
                function=function,
                compiler=compiler,
                warmup=warmup,
                samples=samples,
                inner_repeats=repeats,
            )
            row["preset"] = args.preset
            results.append(row)
            if row["status"] == "ok":
                print(f"  median {row['median_ms']:.3f} ms", flush=True)
            else:
                print(f"  ERROR {row['error']}", flush=True)

    write_document(args.output, metadata, results)
    failures = sum(row["status"] != "ok" for row in results)
    print(json.dumps({"output": str(args.output), "rows": len(results), "failures": failures}))
    return 1 if args.fail_on_error and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
