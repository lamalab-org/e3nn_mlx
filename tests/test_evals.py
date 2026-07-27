"""Tests for the cross-framework evaluation harness."""

from __future__ import annotations

import json

import pytest

from evals.common import SCHEMA_VERSION, load_documents, summarize, write_document
from evals.plot_results import main as plot_main, speedup_entries, write_scaling_plots
from evals.run import parse_args, selected_workers, worker_command
from evals.run_backend import apply_overrides
from evals.workloads import case_names, get_workloads, ring_edges, spherical_irreps


def _row(backend, execution, median, *, phase="forward"):
    return {
        "preset": "smoke",
        "case": "linear",
        "phase": phase,
        "backend": backend,
        "execution": execution,
        "status": "ok",
        "median_ms": median,
        "p25_ms": median * 0.9,
        "p75_ms": median * 1.1,
        "items_per_second": 1000 / median,
        "compile_ms": 5.0 if execution == "compiled" else None,
        "peak_memory_bytes": 1_000_000 if backend == "mlx" else None,
        "item_count": 64,
        "config": {"items": 64, "mul": 2, "lmax": 1},
        "error": None,
    }


def test_statistics_and_workload_helpers() -> None:
    summary = summarize([4.0, 1.0, 3.0, 2.0])
    assert summary["median_ms"] == 2.5
    assert summary["p25_ms"] == 1.75
    assert summary["p75_ms"] == 3.25
    assert set(get_workloads("smoke")) == set(case_names())
    assert spherical_irreps(3, 2) == "3x0e + 3x1o + 3x2e"
    source, destination = ring_edges(4, 2)
    assert len(source) == len(destination) == 8
    assert all(left != right for left, right in zip(source, destination, strict=True))


def test_overrides_are_typed_and_validated() -> None:
    workloads = get_workloads("smoke", ["linear"])
    apply_overrides(workloads, ["linear.items=123", "linear.mul=5"])
    assert workloads["linear"]["items"] == 123
    assert workloads["linear"]["mul"] == 5
    with pytest.raises(ValueError, match="unselected"):
        apply_overrides(workloads, ["spherical_harmonics.items=1"])
    with pytest.raises(ValueError, match="no setting"):
        apply_overrides(workloads, ["linear.missing=1"])


def test_result_round_trip_speedups_and_dependency_free_plots(tmp_path) -> None:
    rows = [
        _row("torch-mps", "eager", 4.0),
        _row("torch-cpu", "eager", 10.0),
        _row("mlx", "eager", 2.0),
        _row("mlx", "compiled", 1.0),
        _row("torch-mps", "eager", 8.0, phase="train"),
        _row("torch-cpu", "eager", 12.0, phase="train"),
        _row("mlx", "compiled", 2.0, phase="train"),
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
    forward_speedups = speedup_entries(rows, "forward")
    assert sorted(entry[1] for entry in forward_speedups) == [2.0, 4.0, 5.0, 10.0]

    output = tmp_path / "plots"
    assert plot_main([str(result_path), "--output-dir", str(output)]) == 0
    expected = {
        "latency_forward.svg",
        "latency_train.svg",
        "speedup_forward.svg",
        "speedup_train.svg",
        "kernel_speedup_forward.svg",
        "kernel_speedup_train.svg",
        "compile_cost.svg",
        "peak_memory.svg",
        "summary.csv",
        "report.html",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert "4.00×" in (output / "speedup_forward.svg").read_text()
    latency = (output / "latency_forward.svg").read_text()
    assert "torch-mps-eager" in latency
    assert "torch-cpu-eager" in latency
    assert json.loads(result_path.read_text())["metadata"]["schema_version"] == 1


def test_scaling_plot_uses_multiple_workload_sizes(tmp_path) -> None:
    small = _row("mlx", "compiled", 2.0)
    large = _row("mlx", "compiled", 3.0)
    large["item_count"] = 128
    large["items_per_second"] = 128_000 / 3
    large["config"] = {"items": 128, "mul": 2, "lmax": 1}
    generated = write_scaling_plots(tmp_path, [small, large])
    assert generated == ["scaling_linear.svg"]
    plot = (tmp_path / generated[0]).read_text()
    assert "throughput scaling" in plot
    assert "mlx-compiled" in plot


def test_torch_both_expands_to_isolated_mps_and_cpu_workers(tmp_path) -> None:
    args = parse_args(
        [
            "--backend",
            "both",
            "--torch-device",
            "both",
            "--mlx-kernels",
            "both",
            "--torch-python",
            str(tmp_path / "torch-python"),
        ]
    )
    assert selected_workers(args) == [
        ("mlx", None, "on"),
        ("mlx", None, "off"),
        ("torch", "mps", None),
        ("torch", "cpu", None),
    ]
    for mode in ("on", "off"):
        mlx_command = worker_command(
            "mlx",
            args.mlx_python,
            tmp_path / f"mlx-{mode}.json",
            args,
            mlx_kernels=mode,
        )
        assert mlx_command[mlx_command.index("--mlx-kernels") + 1] == mode
    command = worker_command(
        "torch",
        args.torch_python,
        tmp_path / "torch-cpu.json",
        args,
        torch_device="cpu",
    )
    assert command[command.index("--device") + 1] == "cpu"


@pytest.mark.mlx
@pytest.mark.parametrize("enabled", [False, True])
def test_v2106_mlx_builders_pass_selected_kernel_mode(enabled, monkeypatch) -> None:
    from evals import mlx_cases
    from e3nn_mlx.models.v2106 import (
        Convolution,
        MessagePassing,
        NetworkForAGraphWithAttributes,
    )

    seen = {"convolution": [], "message_passing": [], "network": []}

    def record_init(kind, original):
        def wrapped(instance, *args, **kwargs):
            seen[kind].append(kwargs.get("use_custom_kernel"))
            original(instance, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        Convolution,
        "__init__",
        record_init("convolution", Convolution.__init__),
    )
    monkeypatch.setattr(
        MessagePassing,
        "__init__",
        record_init("message_passing", MessagePassing.__init__),
    )
    monkeypatch.setattr(
        NetworkForAGraphWithAttributes,
        "__init__",
        record_init("network", NetworkForAGraphWithAttributes.__init__),
    )
    mlx_cases.configure(use_custom_kernels=enabled)
    try:
        workloads = get_workloads(
            "smoke",
            ["v2106_convolution", "v2106_message_passing", "v2106_network"],
        )

        mlx_cases.build_v2106_convolution(workloads["v2106_convolution"])
        assert seen["convolution"] == [enabled]

        seen["convolution"].clear()
        mlx_cases.build_v2106_message_passing(workloads["v2106_message_passing"])
        assert seen["message_passing"] == [enabled]
        assert seen["convolution"] and set(seen["convolution"]) == {enabled}

        seen["convolution"].clear()
        seen["message_passing"].clear()
        mlx_cases.build_v2106_network(workloads["v2106_network"])
        assert seen["network"] == [enabled]
        assert seen["message_passing"] == [enabled]
        assert seen["convolution"] and set(seen["convolution"]) == {enabled}
    finally:
        mlx_cases.configure(use_custom_kernels=True)
