#!/usr/bin/env python3
"""Orchestrate isolated MLX and PyTorch benchmark processes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import SCHEMA_VERSION, load_documents
from evals.workloads import DEFAULT_TIMING, case_names


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("both", "mlx", "torch"), default="both")
    parser.add_argument(
        "--backend-order",
        choices=("mlx-first", "torch-first"),
        default="mlx-first",
    )
    parser.add_argument("--preset", choices=tuple(DEFAULT_TIMING), default="smoke")
    parser.add_argument("--case", action="append", choices=case_names(), dest="cases")
    parser.add_argument("--phase", action="append", choices=("forward", "train"), dest="phases")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--inner-repeats", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--mlx-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--torch-python", type=Path, default=ROOT / "evals/.venv-torch/bin/python"
    )
    parser.add_argument(
        "--torch-device",
        choices=("auto", "mps", "cpu", "both"),
        default="auto",
        help="Torch device to benchmark; 'both' runs isolated MPS and CPU workers",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals/results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args(argv)


def worker_command(
    backend: str,
    interpreter: Path,
    output: Path,
    args,
    *,
    torch_device: str | None = None,
) -> list[str]:
    command = [
        str(interpreter),
        str(ROOT / "evals/run_backend.py"),
        "--backend",
        backend,
        "--preset",
        args.preset,
        "--output",
        str(output),
    ]
    for case in args.cases or []:
        command.extend(["--case", case])
    for phase in args.phases or []:
        command.extend(["--phase", phase])
    for name in ("warmup", "samples", "inner_repeats"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    for override in args.overrides:
        command.extend(["--set", override])
    if backend == "torch":
        command.extend(["--device", torch_device or args.torch_device])
        if args.torch_compile:
            command.append("--torch-compile")
    if args.fail_on_error:
        command.append("--fail-on-error")
    return command


def selected_workers(args) -> list[tuple[str, str | None]]:
    """Return backend/device workers in the requested thermal ordering."""
    torch_devices = (
        ["mps", "cpu"] if args.torch_device == "both" else [args.torch_device]
    )
    torch_workers = [("torch", device) for device in torch_devices]
    if args.backend == "mlx":
        return [("mlx", None)]
    if args.backend == "torch":
        return torch_workers
    mlx_worker = [("mlx", None)]
    return (
        mlx_worker + torch_workers
        if args.backend_order == "mlx-first"
        else torch_workers + mlx_worker
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    interpreters = {"mlx": args.mlx_python, "torch": args.torch_python}
    outputs = []
    failures = 0
    for backend, torch_device in selected_workers(args):
        interpreter = interpreters[backend]
        result_name = f"torch-{torch_device}" if backend == "torch" else backend
        output = run_dir / f"{result_name}.json"
        if not interpreter.exists():
            print(
                f"{backend} interpreter does not exist: {interpreter}\n"
                "See evals/README.md for environment setup.",
                file=sys.stderr,
            )
            failures += 1
            continue
        command = worker_command(
            backend,
            interpreter,
            output,
            args,
            torch_device=torch_device,
        )
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if output.exists():
            outputs.append(output)
        if completed.returncode != 0:
            failures += 1

    if not outputs:
        return 1
    runs, rows = load_documents(outputs)
    combined = run_dir / "combined.json"
    combined.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": SCHEMA_VERSION,
                    "backend": "combined",
                    "preset": args.preset,
                    "source_runs": runs,
                },
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Combined results: {combined}")
    if args.plot:
        plot_dir = run_dir / "plots"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "evals/plot_results.py"),
                str(combined),
                "--output-dir",
                str(plot_dir),
            ],
            cwd=ROOT,
            check=False,
        )
        failures += completed.returncode != 0
    return 1 if failures and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
