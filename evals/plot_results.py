#!/usr/bin/env python3
"""Generate dependency-free SVG plots and CSV summaries from benchmark JSON."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import load_documents


COLORS = {
    "mlx-compiled": "#5b5bd6",
    "mlx-eager": "#8b8be8",
    "torch-compiled": "#d95f42",
    "torch-eager": "#ed947f",
    "torch-mps-compiled": "#d95f42",
    "torch-mps-eager": "#ed947f",
    "torch-cpu-compiled": "#b36b00",
    "torch-cpu-eager": "#e0a32f",
    "speedup": "#2f9e73",
    "slowdown": "#d95f42",
    "compile": "#8d65b5",
    "memory": "#3386a8",
}


def _label(row) -> str:
    return f"{row['backend']}-{row['execution']}"


def _config_label(config) -> str:
    keys = ("items", "nodes", "neighbors", "mul", "lmax", "layers")
    return ", ".join(f"{key}={config[key]}" for key in keys if key in config)


def _context_label(row) -> str:
    preset = row.get("preset", "custom")
    config = _config_label(row.get("config", {}))
    return f"{preset}; {config}" if config else str(preset)


def _number(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value:.3g} ms"
    if unit == "x":
        return f"{value:.2f}×"
    if unit == "MB":
        return f"{value:.1f} MB"
    return f"{value:.3g}"


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("preset"),
            row["case"],
            row["phase"],
            row["backend"],
            row["execution"],
            json.dumps(row.get("config", {}), sort_keys=True),
        )
        groups.setdefault(key, []).append(row)
    aggregated = []
    for group in groups.values():
        output = dict(group[0])
        samples = [
            float(sample)
            for row in group
            for sample in row.get("raw_samples_ms", [row["median_ms"]])
        ]
        output["median_ms"] = statistics.median(samples)
        output["items_per_second"] = output["item_count"] / (
            output["median_ms"] / 1_000
        )
        compile_times = [
            float(row["compile_ms"])
            for row in group
            if row.get("compile_ms") is not None
        ]
        output["compile_ms"] = (
            statistics.median(compile_times) if compile_times else None
        )
        memory = [
            int(row["peak_memory_bytes"])
            for row in group
            if row.get("peak_memory_bytes") is not None
        ]
        output["peak_memory_bytes"] = max(memory) if memory else None
        aggregated.append(output)
    return aggregated


def horizontal_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    entries: list[tuple[str, float, str]],
    unit: str,
    reference: float | None = None,
) -> None:
    width = 1280
    left = 530
    right = 150
    top = 105
    row_height = 32
    height = max(220, top + row_height * max(1, len(entries)) + 80)
    plot_width = width - left - right
    maximum = max((value for _, value, _ in entries), default=1.0)
    if reference is not None:
        maximum = max(maximum, reference)
    maximum = maximum * 1.08 if maximum > 0 else 1.0
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        f'<text x="40" y="42" font-family="system-ui" font-size="25" '
        f'font-weight="700" fill="#202124">{escape(title)}</text>',
        f'<text x="40" y="70" font-family="system-ui" font-size="14" '
        f'fill="#5f6368">{escape(subtitle)}</text>',
    ]
    if reference is not None:
        x = left + plot_width * reference / maximum
        elements.append(
            f'<line x1="{x:.2f}" y1="{top - 12}" x2="{x:.2f}" '
            f'y2="{height - 55}" stroke="#6f7378" stroke-dasharray="4 4"/>'
        )
        elements.append(
            f'<text x="{x + 5:.2f}" y="{top - 17}" font-family="system-ui" '
            'font-size="12" fill="#5f6368">parity</text>'
        )
    for index, (label, value, color_key) in enumerate(entries):
        y = top + index * row_height
        bar_width = plot_width * value / maximum
        elements.extend(
            [
                f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" '
                f'font-family="ui-monospace, monospace" font-size="12" '
                f'fill="#303238">{escape(label)}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_width:.2f}" height="20" '
                f'rx="3" fill="{COLORS.get(color_key, "#777")}"/>',
                f'<text x="{left + bar_width + 7:.2f}" y="{y + 19}" '
                f'font-family="system-ui" font-size="12" fill="#303238">'
                f'{escape(_number(value, unit))}</text>',
            ]
        )
    if not entries:
        elements.append(
            '<text x="40" y="135" font-family="system-ui" font-size="16" '
            'fill="#8a3b2d">No matching successful rows were found.</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def latency_entries(rows, phase: str):
    selected = [
        row
        for row in aggregate_rows(rows)
        if row.get("phase") == phase
    ]
    selected.sort(key=lambda row: (row["case"], _label(row)))
    return [
        (
            f"{row['case']} [{_context_label(row)}] · {_label(row)}",
            float(row["median_ms"]),
            _label(row),
        )
        for row in selected
    ]


def speedup_entries(rows, phase: str):
    successful = [
        row
        for row in aggregate_rows(rows)
        if row.get("phase") == phase
    ]
    groups = {}
    for row in successful:
        config = json.dumps(row.get("config", {}), sort_keys=True)
        groups.setdefault((row.get("preset"), row["case"], config), {})[
            _label(row)
        ] = row
    entries = []
    for (preset, case, config_json), group in sorted(groups.items()):
        torch_labels = [
            label
            for label in ("torch-eager", "torch-mps-eager", "torch-cpu-eager")
            if label in group
        ]
        for torch_label in torch_labels:
            torch_row = group[torch_label]
            for mlx_label in ("mlx-eager", "mlx-compiled"):
                mlx_row = group.get(mlx_label)
                if mlx_row is None:
                    continue
                ratio = float(torch_row["median_ms"]) / float(mlx_row["median_ms"])
                entries.append(
                    (
                        f"{case} [{preset}; {_config_label(json.loads(config_json))}] "
                        f"· {mlx_label} vs {torch_label}",
                        ratio,
                        "speedup" if ratio >= 1.0 else "slowdown",
                    )
                )
    return entries


def compile_entries(rows):
    selected = [
        row
        for row in aggregate_rows(rows)
        if row.get("execution") == "compiled"
        and row.get("compile_ms") is not None
    ]
    selected.sort(key=lambda row: (row["phase"], row["case"], row["backend"]))
    return [
        (
            f"{row['phase']} · {row['case']} "
            f"[{_context_label(row)}] · {row['backend']}",
            float(row["compile_ms"]),
            "compile",
        )
        for row in selected
    ]


def memory_entries(rows):
    selected = [
        row
        for row in aggregate_rows(rows)
        if row.get("peak_memory_bytes") is not None
    ]
    selected.sort(key=lambda row: (row["phase"], row["case"], _label(row)))
    return [
        (
            f"{row['phase']} · {row['case']} "
            f"[{_context_label(row)}] · {_label(row)}",
            float(row["peak_memory_bytes"]) / 1_000_000,
            "memory",
        )
        for row in selected
    ]


def write_scaling_plots(output_dir: Path, rows) -> list[str]:
    successful = aggregate_rows(rows)
    generated = []
    cases = sorted({row["case"] for row in successful})
    for case in cases:
        case_rows = [row for row in successful if row["case"] == case]
        x_values = sorted({int(row["item_count"]) for row in case_rows})
        if len(x_values) < 2:
            continue
        y_values = [float(row["items_per_second"]) for row in case_rows]
        if min(x_values) <= 0 or min(y_values) <= 0:
            continue
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_low, x_high = math.log10(x_min), math.log10(x_max)
        y_low, y_high = math.log10(y_min), math.log10(y_max)
        if x_low == x_high or y_low == y_high:
            continue
        width, height = 1280, 650
        left, right, top, bottom = 110, 50, 115, 80
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_position(value):
            return left + (math.log10(value) - x_low) / (x_high - x_low) * plot_width

        def y_position(value):
            return top + (y_high - math.log10(value)) / (y_high - y_low) * plot_height

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fbfbfd"/>',
            f'<text x="40" y="42" font-family="system-ui" font-size="25" '
            f'font-weight="700" fill="#202124">{escape(case)} throughput scaling</text>',
            '<text x="40" y="70" font-family="system-ui" font-size="14" '
            'fill="#5f6368">Log-log axes; higher throughput is better.</text>',
        ]
        for index in range(5):
            fraction = index / 4
            x_value = 10 ** (x_low + fraction * (x_high - x_low))
            y_value = 10 ** (y_low + fraction * (y_high - y_low))
            x = left + fraction * plot_width
            y = top + (1 - fraction) * plot_height
            elements.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                    f'y2="{top + plot_height}" stroke="#d9dce1"/>',
                    f'<text x="{x:.1f}" y="{top + plot_height + 23}" '
                    f'text-anchor="middle" font-family="system-ui" font-size="11" '
                    f'fill="#5f6368">{x_value:.2g}</text>',
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                    f'y2="{y:.1f}" stroke="#d9dce1"/>',
                    f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                    f'font-family="system-ui" font-size="11" fill="#5f6368">'
                    f'{y_value:.2g}</text>',
                ]
            )
        series = {}
        for row in case_rows:
            series.setdefault(f"{row['phase']} · {_label(row)}", []).append(row)
        for series_index, (label, points) in enumerate(sorted(series.items())):
            points.sort(key=lambda row: row["item_count"])
            backend_label = _label(points[0])
            color = COLORS.get(backend_label, "#777")
            dash = ' stroke-dasharray="7 5"' if points[0]["phase"] == "train" else ""
            coordinates = " ".join(
                f"{x_position(row['item_count']):.1f},{y_position(row['items_per_second']):.1f}"
                for row in points
            )
            elements.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="2"{dash}/>'
            )
            for row in points:
                elements.append(
                    f'<circle cx="{x_position(row["item_count"]):.1f}" '
                    f'cy="{y_position(row["items_per_second"]):.1f}" r="4" '
                    f'fill="{color}"/>'
                )
            legend_x = left + (series_index % 4) * 270
            legend_y = 92 + (series_index // 4) * 18
            elements.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y - 4}" '
                    f'x2="{legend_x + 20}" y2="{legend_y - 4}" stroke="{color}" '
                    f'stroke-width="2"{dash}/>',
                    f'<text x="{legend_x + 26}" y="{legend_y}" '
                    f'font-family="system-ui" font-size="11" fill="#303238">'
                    f'{escape(label)}</text>',
                ]
            )
        elements.extend(
            [
                f'<text x="{left + plot_width / 2}" y="{height - 25}" '
                'text-anchor="middle" font-family="system-ui" font-size="13" '
                'fill="#303238">work items per invocation</text>',
                f'<text x="25" y="{top + plot_height / 2}" '
                'text-anchor="middle" font-family="system-ui" font-size="13" '
                'fill="#303238" transform="rotate(-90 25 '
                f'{top + plot_height / 2})">items per second</text>',
                "</svg>",
            ]
        )
        name = f"scaling_{case}.svg"
        (output_dir / name).write_text("\n".join(elements) + "\n")
        generated.append(name)
    return generated


def write_csv(path: Path, rows) -> None:
    fields = [
        "preset",
        "case",
        "phase",
        "backend",
        "execution",
        "status",
        "median_ms",
        "p25_ms",
        "p75_ms",
        "items_per_second",
        "compile_ms",
        "peak_memory_bytes",
        "item_count",
        "config",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["config"] = json.dumps(row.get("config", {}), sort_keys=True)
            writer.writerow(output)


def write_report(path: Path, files: list[str], metadata) -> None:
    images = "\n".join(
        f'<figure><img src="{escape(name)}" alt="{escape(name)}"></figure>'
        for name in files
    )
    path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>e3nn benchmark report</title>"
        "<style>body{font:16px system-ui;margin:2rem;color:#202124}"
        "img{max-width:100%;border:1px solid #ddd;margin:.5rem 0 2rem}"
        "code,pre{background:#f3f4f6}pre{padding:1rem;overflow:auto}</style>"
        "<h1>e3nn-mlx / PyTorch benchmark report</h1>"
        "<p>A ratio above 1× means MLX is faster. Latencies are synchronized medians.</p>"
        f"{images}<h2>Run metadata</h2><pre>{escape(json.dumps(metadata, indent=2))}</pre>"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metadata, rows = load_documents(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for phase in ("forward", "train"):
        latency_name = f"latency_{phase}.svg"
        horizontal_bars(
            args.output_dir / latency_name,
            title=f"Synchronized {phase} latency",
            subtitle="Lower is better; bars show median wall time per invocation.",
            entries=latency_entries(rows, phase),
            unit="ms",
        )
        generated.append(latency_name)
        speedup_name = f"speedup_{phase}.svg"
        horizontal_bars(
            args.output_dir / speedup_name,
            title=f"MLX {phase} speed relative to PyTorch eager",
            subtitle=(
                "Separate MPS and CPU ratios = PyTorch median / MLX median; "
                "above 1× favors MLX."
            ),
            entries=speedup_entries(rows, phase),
            unit="x",
            reference=1.0,
        )
        generated.append(speedup_name)
    horizontal_bars(
        args.output_dir / "compile_cost.svg",
        title="Cold compilation cost",
        subtitle="Compiler creation plus the first synchronized invocation.",
        entries=compile_entries(rows),
        unit="ms",
    )
    generated.append("compile_cost.svg")
    horizontal_bars(
        args.output_dir / "peak_memory.svg",
        title="Peak framework memory",
        subtitle="Backend-reported peak; MLX includes active allocations for the measured mode.",
        entries=memory_entries(rows),
        unit="MB",
    )
    generated.append("peak_memory.svg")
    generated.extend(write_scaling_plots(args.output_dir, rows))
    write_csv(args.output_dir / "summary.csv", rows)
    write_report(args.output_dir / "report.html", generated, metadata)
    print(f"Generated {len(generated)} SVG plots and report.html in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
