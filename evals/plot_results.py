#!/usr/bin/env python3
"""Create compact SVG, CSV, and HTML reports for the three-way benchmark."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics
import sys
from typing import NamedTuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import load_documents, summarize


BACKENDS = ("torch-cpu", "mlx", "mlx-kernel")
COLORS = {
    "torch-cpu": "#d97706",
    "mlx": "#2563eb",
    "mlx-kernel": "#059669",
    "speedup": "#059669",
    "slowdown": "#dc2626",
}


class BarEntry(NamedTuple):
    label: str
    value: float
    color_key: str
    lower: float
    upper: float
    repeat_count: int


def successful_rows(rows, phase: str):
    selected = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("phase") == phase
    ]
    selected.sort(
        key=lambda row: (
            row["case"],
            BACKENDS.index(row["backend"]),
        )
    )
    return selected


def _run_summary(values) -> dict[str, float]:
    return summarize(float(value) for value in values)


def latency_entries(rows, phase: str):
    grouped = {}
    for row in successful_rows(rows, phase):
        grouped.setdefault((row["case"], row["backend"]), []).append(
            float(row["median_ms"])
        )
    entries = []
    for (case, backend), medians in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], BACKENDS.index(item[0][1])),
    ):
        summary = _run_summary(medians)
        entries.append(
            BarEntry(
                f"{case} · {backend}",
                summary["median_ms"],
                backend,
                summary["p25_ms"],
                summary["p75_ms"],
                len(medians),
            )
        )
    return entries


def speedup_entries(rows, phase: str):
    grouped: dict[str, dict[str, dict[int, dict]]] = {}
    occurrence: dict[tuple[str, str], int] = {}
    for row in successful_rows(rows, phase):
        key = (row["case"], row["backend"])
        fallback = occurrence.get(key, 0)
        repeat_index = int(row.get("repeat_index", fallback))
        occurrence[key] = fallback + 1
        grouped.setdefault(row["case"], {}).setdefault(row["backend"], {})[
            repeat_index
        ] = row
    entries = []
    for case, group in sorted(grouped.items()):
        references = group.get("torch-cpu")
        if not references:
            continue
        for backend in ("mlx", "mlx-kernel"):
            candidates = group.get(backend)
            if not candidates:
                continue
            ratios = [
                float(references[index]["median_ms"])
                / float(candidates[index]["median_ms"])
                for index in sorted(references.keys() & candidates.keys())
            ]
            if not ratios:
                continue
            summary = _run_summary(ratios)
            entries.append(
                BarEntry(
                    f"{case} · {backend} vs torch-cpu",
                    summary["median_ms"],
                    (
                        "speedup"
                        if summary["median_ms"] >= 1.0
                        else "slowdown"
                    ),
                    summary["p25_ms"],
                    summary["p75_ms"],
                    len(ratios),
                )
            )
    return entries


def horizontal_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    entries: list[BarEntry],
    unit: str,
    reference: float | None = None,
) -> None:
    width = 1280
    left = 510
    right = 300
    top = 100
    row_height = 31
    height = max(220, top + row_height * max(1, len(entries)) + 70)
    plot_width = width - left - right
    maximum = max((entry.upper for entry in entries), default=1.0)
    if reference is not None:
        maximum = max(maximum, reference)
    maximum = maximum * 1.08 if maximum > 0 else 1.0
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        f'<text x="35" y="38" font-family="system-ui" font-size="24" '
        f'font-weight="700" fill="#202124">{escape(title)}</text>',
        f'<text x="35" y="66" font-family="system-ui" font-size="14" '
        f'fill="#5f6368">{escape(subtitle)}</text>',
    ]
    if reference is not None:
        x = left + plot_width * reference / maximum
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top - 12}" x2="{x:.2f}" '
                f'y2="{height - 48}" stroke="#6b7280" stroke-dasharray="4 4"/>',
                f'<text x="{x + 5:.2f}" y="{top - 17}" font-family="system-ui" '
                'font-size="12" fill="#5f6368">parity</text>',
            ]
        )
    for index, entry in enumerate(entries):
        label, value, color_key = entry[:3]
        y = top + index * row_height
        bar_width = plot_width * value / maximum
        lower_x = left + plot_width * entry.lower / maximum
        upper_x = left + plot_width * entry.upper / maximum
        formatted = (
            f"{value:.3f} {unit}"
            if unit == "ms"
            else f"{value:.2f}×"
        )
        if entry.repeat_count > 1:
            formatted += f" (IQR {entry.lower:.3f}–{entry.upper:.3f}, n={entry.repeat_count})"
        elements.extend(
            [
                f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" '
                f'font-family="ui-monospace, monospace" font-size="11" '
                f'fill="#303238">{escape(label)}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_width:.2f}" '
                f'height="20" rx="3" fill="{COLORS[color_key]}"/>',
                f'<line class="error-bar" x1="{lower_x:.2f}" y1="{y + 14}" '
                f'x2="{upper_x:.2f}" y2="{y + 14}" stroke="#111827" '
                'stroke-width="2"/>',
                f'<line class="error-bar" x1="{lower_x:.2f}" y1="{y + 9}" '
                f'x2="{lower_x:.2f}" y2="{y + 19}" stroke="#111827"/>',
                f'<line class="error-bar" x1="{upper_x:.2f}" y1="{y + 9}" '
                f'x2="{upper_x:.2f}" y2="{y + 19}" stroke="#111827"/>',
                f'<text x="{left + bar_width + 7:.2f}" y="{y + 19}" '
                f'font-family="system-ui" font-size="12" fill="#303238">'
                f'{formatted}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def aggregate_rows(rows, *, expected_repeats: int | None = None):
    grouped = {}
    for row in rows:
        key = (row.get("phase"), row.get("case"), row.get("backend"))
        grouped.setdefault(key, []).append(row)
    aggregated = []
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        successful = [
            row
            for row in group
            if row.get("status") == "ok" and row.get("median_ms") is not None
        ]
        output = dict(successful[0] if successful else group[0])
        repeat_count = max(len(group), expected_repeats or 0)
        output["repeat_count"] = repeat_count
        output["successful_repeats"] = len(successful)
        if successful:
            timing = _run_summary(row["median_ms"] for row in successful)
            output.update(timing)
            compile_values = [
                float(row["compile_ms"])
                for row in successful
                if row.get("compile_ms") is not None
            ]
            output["compile_ms"] = (
                statistics.median(compile_values) if compile_values else None
            )
            output["status"] = (
                "ok" if len(successful) == repeat_count else "partial"
            )
            output["error"] = None
            output["items_per_second"] = (
                float(output["item_count"])
                / (float(output["median_ms"]) / 1_000)
                if output.get("item_count") is not None
                and float(output["median_ms"]) > 0
                else None
            )
            memory_values = [
                int(row["peak_memory_bytes"])
                for row in successful
                if row.get("peak_memory_bytes") is not None
            ]
            output["peak_memory_bytes"] = (
                int(statistics.median(memory_values))
                if memory_values
                else None
            )
        else:
            output["status"] = "error"
            output["median_ms"] = None
            output["p25_ms"] = None
            output["p75_ms"] = None
            output["error"] = "; ".join(
                str(row.get("error"))
                for row in group
                if row.get("error")
            )
        aggregated.append(output)
    return aggregated


def write_csv(path: Path, rows) -> None:
    fields = [
        "case",
        "phase",
        "backend",
        "execution",
        "dispatch",
        "status",
        "median_ms",
        "p25_ms",
        "p75_ms",
        "repeat_count",
        "successful_repeats",
        "compile_ms",
        "items_per_second",
        "peak_memory_bytes",
        "item_count",
        "config",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item.get("phase", ""),
                item.get("case", ""),
                item.get("backend", ""),
            ),
        ):
            output = dict(row)
            output["config"] = json.dumps(row.get("config", {}), sort_keys=True)
            writer.writerow(output)


def write_html(path: Path, rows, generated: list[str]) -> None:
    table_rows = []
    for row in sorted(
        rows,
        key=lambda item: (
            item.get("phase", ""),
            item.get("case", ""),
            item.get("backend", ""),
        ),
    ):
        median = (
            f"{float(row['median_ms']):.3f}"
            if row.get("median_ms") is not None
            else "—"
        )
        compile_ms = (
            f"{float(row['compile_ms']):.3f}"
            if row.get("compile_ms") is not None
            else "—"
        )
        interval = (
            f"{float(row['p25_ms']):.3f}–{float(row['p75_ms']):.3f}"
            if row.get("p25_ms") is not None
            and row.get("p75_ms") is not None
            else "—"
        )
        repetitions = (
            f"{row.get('successful_repeats', 0)}/{row.get('repeat_count', 1)}"
        )
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('phase', '')))}</td>"
            f"<td>{escape(str(row.get('case', '')))}</td>"
            f"<td>{escape(str(row.get('backend', '')))}</td>"
            f"<td>{escape(str(row.get('execution', '')))}</td>"
            f"<td>{escape(str(row.get('dispatch', '')))}</td>"
            f"<td>{median}</td><td>{interval}</td><td>{repetitions}</td>"
            f"<td>{compile_ms}</td>"
            f"<td>{escape(str(row.get('status', '')))}</td>"
            "</tr>"
        )
    images = "\n".join(
        f'<section><img src="{escape(name)}" alt="{escape(name)}"></section>'
        for name in generated
    )
    path.write_text(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>e3nn benchmark</title>
<style>
body{font-family:system-ui;margin:2rem;color:#202124}
img{max-width:100%;border:1px solid #e5e7eb;margin-bottom:1rem}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #d1d5db;padding:.45rem;text-align:left}
th{background:#f3f4f6}
</style></head><body>
<h1>Three-way e3nn performance comparison</h1>
<p>Torch CPU uses all available CPU threads. Both MLX workers use compiled
execution; only <code>mlx-kernel</code> enables generated kernels.</p>
"""
        + images
        + """
<h2>Measurements</h2>
<table><thead><tr><th>Phase</th><th>Case</th><th>Backend</th>
<th>Execution</th><th>Selected path</th><th>Run median (ms)</th>
<th>Between-run IQR (ms)</th><th>Successful runs</th><th>Compile (ms)</th>
<th>Status</th></tr></thead><tbody>
"""
        + "\n".join(table_rows)
        + "</tbody></table></body></html>\n"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metadata, rows = load_documents(args.results)
    requested_repeats = max(
        (
            int(document.get("repeat_count_requested", 0))
            for document in metadata
        ),
        default=0,
    )
    aggregated = aggregate_rows(
        rows,
        expected_repeats=requested_repeats or None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for phase in ("forward", "train"):
        latency_name = f"latency_{phase}.svg"
        horizontal_bars(
            args.output_dir / latency_name,
            title=f"{phase.title()} steady-state latency",
            subtitle=(
                "Lower is better; bars are medians of independent runs "
                "and whiskers show the between-run IQR."
            ),
            entries=latency_entries(rows, phase),
            unit="ms",
        )
        generated.append(latency_name)
        speedup_name = f"speedup_{phase}.svg"
        horizontal_bars(
            args.output_dir / speedup_name,
            title=f"{phase.title()} speedup over Torch CPU",
            subtitle=(
                "Paired by repeat; bars are median speedups and whiskers "
                "show the between-run IQR."
            ),
            entries=speedup_entries(rows, phase),
            unit="x",
            reference=1.0,
        )
        generated.append(speedup_name)
    write_csv(args.output_dir / "summary.csv", aggregated)
    write_html(args.output_dir / "report.html", aggregated, generated)
    print(f"Report: {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
