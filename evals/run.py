#!/usr/bin/env python3
"""Run Torch CPU, general MLX, and kernel-enabled MLX in isolation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import SCHEMA_VERSION, load_documents
from evals.workloads import DEFAULT_TIMING, case_names


ROOT = Path(__file__).resolve().parents[1]
WORKERS = ("torch-cpu", "mlx", "mlx-kernel")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--mlx-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=ROOT / "evals/.venv-torch/bin/python",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals/results")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def worker_command(
    backend: str,
    interpreter: Path,
    output: Path,
    args,
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
    for name in ("warmup", "samples"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name}", str(value)])
    for override in args.overrides:
        command.extend(["--set", override])
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    interpreters = {
        "torch-cpu": args.torch_python,
        "mlx": args.mlx_python,
        "mlx-kernel": args.mlx_python,
    }
    outputs = []
    failures = 0

    for backend in WORKERS:
        interpreter = interpreters[backend]
        output = run_dir / f"{backend}.json"
        if not interpreter.exists():
            print(
                f"{backend} interpreter does not exist: {interpreter}\n"
                "See evals/README.md for setup.",
                file=sys.stderr,
            )
            failures += 1
            continue
        command = worker_command(backend, interpreter, output, args)
        environment = os.environ.copy()
        if backend.startswith("mlx"):
            environment["E3NN_MLX_REQUIRE_RUNTIME"] = "1"
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            env=environment,
        )
        if output.exists():
            outputs.append(output)
        if completed.returncode != 0:
            failures += 1

    if len(outputs) != len(WORKERS):
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

    if not args.no_plot:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "evals/plot_results.py"),
                str(combined),
                "--output-dir",
                str(run_dir / "plots"),
            ],
            cwd=ROOT,
            check=False,
        )
        failures += completed.returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
