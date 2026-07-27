#!/usr/bin/env python3
"""Create compact SVG, CSV, and HTML reports for the three-way benchmark."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import load_documents


BACKENDS = ("torch-cpu", "mlx", "mlx-kernel")
COLORS = {
    "torch-cpu": "#d97706",
    "mlx": "#2563eb",
    "mlx-kernel": "#059669",
    "speedup": "#059669",
    "slowdown": "#dc2626",
}


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


def latency_entries(rows, phase: str):
    return [
        (
            f"{row['case']} · {row['backend']}",
            float(row["median_ms"]),
            row["backend"],
        )
        for row in successful_rows(rows, phase)
    ]


def speedup_entries(rows, phase: str):
    grouped = {}
    for row in successful_rows(rows, phase):
        grouped.setdefault(row["case"], {})[row["backend"]] = row
    entries = []
    for case, group in sorted(grouped.items()):
        reference = group.get("torch-cpu")
        if reference is None:
            continue
        for backend in ("mlx", "mlx-kernel"):
            candidate = group.get(backend)
            if candidate is None:
                continue
            ratio = float(reference["median_ms"]) / float(candidate["median_ms"])
            entries.append(
                (
                    f"{case} · {backend} vs torch-cpu",
                    ratio,
                    "speedup" if ratio >= 1.0 else "slowdown",
                )
            )
    return entries


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
    left = 510
    right = 150
    top = 100
    row_height = 31
    height = max(220, top + row_height * max(1, len(entries)) + 70)
    plot_width = width - left - right
    maximum = max((value for _, value, _ in entries), default=1.0)
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
    for index, (label, value, color_key) in enumerate(entries):
        y = top + index * row_height
        bar_width = plot_width * value / maximum
        formatted = f"{value:.3f} {unit}" if unit == "ms" else f"{value:.2f}×"
        elements.extend(
            [
                f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" '
                f'font-family="ui-monospace, monospace" font-size="11" '
                f'fill="#303238">{escape(label)}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_width:.2f}" '
                f'height="20" rx="3" fill="{COLORS[color_key]}"/>',
                f'<text x="{left + bar_width + 7:.2f}" y="{y + 19}" '
                f'font-family="system-ui" font-size="12" fill="#303238">'
                f'{formatted}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def write_csv(path: Path, rows) -> None:
    fields = [
        "case",
        "phase",
        "backend",
        "execution",
        "status",
        "median_ms",
        "p25_ms",
        "p75_ms",
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
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('phase', '')))}</td>"
            f"<td>{escape(str(row.get('case', '')))}</td>"
            f"<td>{escape(str(row.get('backend', '')))}</td>"
            f"<td>{escape(str(row.get('execution', '')))}</td>"
            f"<td>{median}</td><td>{compile_ms}</td>"
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
<th>Execution</th><th>Steady median (ms)</th><th>Compile (ms)</th>
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
    _, rows = load_documents(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for phase in ("forward", "train"):
        latency_name = f"latency_{phase}.svg"
        horizontal_bars(
            args.output_dir / latency_name,
            title=f"{phase.title()} steady-state latency",
            subtitle="Lower is better; synchronized median per invocation.",
            entries=latency_entries(rows, phase),
            unit="ms",
        )
        generated.append(latency_name)
        speedup_name = f"speedup_{phase}.svg"
        horizontal_bars(
            args.output_dir / speedup_name,
            title=f"{phase.title()} speedup over Torch CPU",
            subtitle="Above 1× favors MLX; below 1× favors Torch CPU.",
            entries=speedup_entries(rows, phase),
            unit="x",
            reference=1.0,
        )
        generated.append(speedup_name)
    write_csv(args.output_dir / "summary.csv", rows)
    write_html(args.output_dir / "report.html", rows, generated)
    print(f"Report: {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
