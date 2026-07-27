"""Tests for the compact three-way evaluation harness."""

from __future__ import annotations

import json

import pytest

from evals.common import SCHEMA_VERSION, load_documents, summarize, write_document
from evals.plot_results import main as plot_main, speedup_entries
from evals.run import WORKERS, parse_args, worker_command
from evals.run_backend import apply_overrides, parse_args as parse_worker_args
from evals.workloads import case_names, get_workloads, spherical_irreps


def _row(backend, median, *, phase="forward"):
    return {
        "preset": "smoke",
        "case": "linear",
        "phase": phase,
        "backend": backend,
        "execution": "eager" if backend == "torch-cpu" else "compiled",
        "dispatch": "torch-eager" if backend == "torch-cpu" else "general-mlx",
        "status": "ok",
        "median_ms": median,
        "p25_ms": median * 0.9,
        "p75_ms": median * 1.1,
        "items_per_second": 1000 / median,
        "compile_ms": None if backend == "torch-cpu" else 5.0,
        "peak_memory_bytes": None if backend == "torch-cpu" else 1_000_000,
        "item_count": 64,
        "config": {"items": 64, "mul": 2, "lmax": 1},
        "error": None,
    }


def test_statistics_and_workload_surface_are_small_and_fixed() -> None:
    summary = summarize([4.0, 1.0, 3.0, 2.0])
    assert summary["median_ms"] == 2.5
    assert summary["p25_ms"] == 1.75
    assert summary["p75_ms"] == 3.25
    assert set(case_names()) == {
        "spherical_harmonics",
        "full_tensor_product",
        "fully_connected_tensor_product",
        "weighted_tensor_product_uvu",
        "linear",
        "scatter_sum",
    }
    assert set(get_workloads("smoke")) == set(case_names())
    assert set(get_workloads("full")) == set(case_names())
    assert spherical_irreps(3, 2) == "3x0e + 3x1o + 3x2e"
    with pytest.raises(ValueError, match="unknown preset"):
        get_workloads("medium")


def test_overrides_are_typed_and_validated() -> None:
    workloads = get_workloads("smoke", ["linear"])
    apply_overrides(workloads, ["linear.items=123", "linear.mul=5"])
    assert workloads["linear"]["items"] == 123
    assert workloads["linear"]["mul"] == 5
    with pytest.raises(ValueError, match="unselected"):
        apply_overrides(workloads, ["spherical_harmonics.items=1"])
    with pytest.raises(ValueError, match="no setting"):
        apply_overrides(workloads, ["linear.missing=1"])


def test_runner_always_builds_exactly_three_worker_commands(tmp_path) -> None:
    args = parse_args(
        [
            "--preset",
            "full",
            "--case",
            "linear",
            "--phase",
            "forward",
            "--warmup",
            "2",
            "--samples",
            "4",
        ]
    )
    assert WORKERS == ("torch-cpu", "mlx", "mlx-kernel")
    for backend in WORKERS:
        interpreter = tmp_path / backend
        command = worker_command(
            backend,
            interpreter,
            tmp_path / f"{backend}.json",
            args,
        )
        assert command[command.index("--backend") + 1] == backend
        assert command[command.index("--preset") + 1] == "full"
        assert "--device" not in command
        assert "--mlx-kernels" not in command
        assert "--torch-compile" not in command


def test_worker_cli_has_only_the_three_public_backend_labels() -> None:
    for backend in WORKERS:
        args = parse_worker_args(
            ["--backend", backend, "--output", f"{backend}.json"]
        )
        assert args.backend == backend
    with pytest.raises(SystemExit):
        parse_worker_args(["--backend", "torch-mps", "--output", "bad.json"])


def test_result_round_trip_speedups_and_compact_plots(tmp_path) -> None:
    rows = [
        _row("torch-cpu", 10.0),
        _row("mlx", 4.0),
        _row("mlx-kernel", 2.0),
        _row("torch-cpu", 12.0, phase="train"),
        _row("mlx", 6.0, phase="train"),
        _row("mlx-kernel", 3.0, phase="train"),
    ]
    result_path = tmp_path / "results.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "backend": "combined",
        "preset": "smoke",
    }
    write_document(result_path, metadata, rows)
    loaded_metadata, loaded_rows = load_documents([result_path])
    assert loaded_metadata == [metadata]
    assert loaded_rows == rows
    assert [entry[1] for entry in speedup_entries(rows, "forward")] == [2.5, 5.0]

    output = tmp_path / "plots"
    assert plot_main([str(result_path), "--output-dir", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {
        "latency_forward.svg",
        "latency_train.svg",
        "speedup_forward.svg",
        "speedup_train.svg",
        "summary.csv",
        "report.html",
    }
    assert "5.00×" in (output / "speedup_forward.svg").read_text()
    latency = (output / "latency_forward.svg").read_text()
    assert "torch-cpu" in latency
    assert "mlx-kernel" in latency
    report = (output / "report.html").read_text()
    assert "Selected path" in report
    assert "general-mlx" in report
    assert json.loads(result_path.read_text())["metadata"]["schema_version"] == 1


def test_backend_builders_match_the_documented_case_set() -> None:
    from evals import mlx_cases, torch_cases

    assert tuple(mlx_cases.BUILDERS) == case_names()
    assert tuple(torch_cases.BUILDERS) == case_names()


@pytest.mark.mlx
def test_mlx_benchmark_reports_actual_tensor_product_dispatch() -> None:
    from evals import mlx_cases

    mlx_cases.configure(use_custom_kernels=True)
    small = mlx_cases.build_fully_connected_tensor_product(
        {"items": 16, "mul": 8, "lmax": 2}
    )
    dense = mlx_cases.build_fully_connected_tensor_product(
        {"items": 64, "mul": 8, "lmax": 2}
    )
    mlx_cases.configure(use_custom_kernels=False)
    general = mlx_cases.build_fully_connected_tensor_product(
        {"items": 16, "mul": 8, "lmax": 2}
    )

    assert small.dispatch == "metal-scalar-paths"
    assert dense.dispatch == "general-mlx (kernel fallback)"
    assert general.dispatch == "general-mlx"
