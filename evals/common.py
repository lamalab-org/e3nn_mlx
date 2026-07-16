"""Framework-independent benchmark timing, statistics, and result I/O."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable
import uuid


SCHEMA_VERSION = 1


@dataclass(slots=True)
class Task:
    """A constructed benchmark task for one backend."""

    name: str
    family: str
    description: str
    config: dict[str, Any]
    item_count: int
    forward: Callable[[], Any]
    synchronize: Callable[[Any], None]
    compile_forward: Callable[[], Callable[[], Any]] | None = None
    train: Callable[[], Any] | None = None
    compile_train: Callable[[], Callable[[], Any]] | None = None
    reset_peak_memory: Callable[[], None] | None = None
    peak_memory: Callable[[], int | None] | None = None


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(samples_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(value) for value in samples_ms)
    if not values:
        raise ValueError("at least one timing sample is required")
    return {
        "min_ms": values[0],
        "p25_ms": percentile(values, 0.25),
        "median_ms": statistics.median(values),
        "p75_ms": percentile(values, 0.75),
        "max_ms": values[-1],
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.pstdev(values),
    }


def measure(
    function: Callable[[], Any],
    synchronize: Callable[[Any], None],
    *,
    warmup: int,
    samples: int,
    inner_repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or samples <= 0 or inner_repeats <= 0:
        raise ValueError("warmup must be non-negative; samples and repeats must be positive")
    for _ in range(warmup):
        synchronize(function())
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = None
        for _ in range(inner_repeats):
            result = function()
            synchronize(result)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000 / inner_repeats
        timings.append(elapsed_ms)
    return {
        **summarize(timings),
        "samples": samples,
        "inner_repeats": inner_repeats,
        "raw_samples_ms": timings,
    }


def cold_start(
    compiler: Callable[[], Callable[[], Any]],
    synchronize: Callable[[Any], None],
) -> tuple[Callable[[], Any], float]:
    start = time.perf_counter_ns()
    compiled = compiler()
    synchronize(compiled())
    return compiled, (time.perf_counter_ns() - start) / 1_000_000


def machine_metadata() -> dict[str, Any]:
    command = ["system_profiler", "SPHardwareDataType", "-json"]
    try:
        raw_hardware = json.loads(
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
        )
        overview = raw_hardware.get("SPHardwareDataType", [{}])[0]
        hardware = {
            key: overview[key]
            for key in (
                "chip_type",
                "machine_model",
                "machine_name",
                "number_processors",
                "physical_memory",
            )
            if key in overview
        }
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        hardware = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "hardware": hardware,
    }


def new_run_metadata(preset: str, backend: str, device: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "backend": backend,
        "device": device,
        "machine": machine_metadata(),
    }


def write_document(path: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, sort_keys=True)
        + "\n"
    )
    temporary.replace(path)


def load_documents(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = []
    rows = []
    for path in paths:
        document = json.loads(path.read_text())
        if document.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path} does not use result schema {SCHEMA_VERSION}")
        metadata.append(document["metadata"])
        rows.extend(document.get("results", []))
    return metadata, rows
