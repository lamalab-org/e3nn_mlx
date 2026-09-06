#!/usr/bin/env python3
"""Create compact SVG, CSV, and HTML reports for the three-way benchmark."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys
from typing import NamedTuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import load_documents, summarize


BACKENDS = ("torch-cpu", "mlx", "mlx-kernel")
# Categorical slots 1-3 of the validated reference palette, in fixed backend
# order (never cycled). All-pairs CVD dE 9.2 (deutan) on the light surface; the
# amber/green pairing this replaces sat at 7.9, inside the floor band.
COLORS = {
    "torch-cpu": "#2a78d6",
    "mlx": "#eb6834",
    "mlx-kernel": "#1baf7a",
}
INK = "#1c1c1a"
INK_MUTED = "#6b6a65"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


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
                    # Colour carries identity (which backend); faster/slower is
                    # already encoded by position relative to the parity rule.
                    backend,
                    summary["p25_ms"],
                    summary["p75_ms"],
                    len(ratios),
                )
            )
    return entries


def group_by_case(entries: list[BarEntry]) -> list[tuple[str, list[BarEntry]]]:
    """Collapse "case . series" rows into one row per case."""

    grouped: dict[str, list[BarEntry]] = {}
    for entry in entries:
        case = entry.label.split(" \u00b7 ")[0]
        grouped.setdefault(case, []).append(entry)
    return sorted(grouped.items())


def _log_ticks(low: float, high: float) -> list[float]:
    """Decade ticks, refined with 2/5 steps when the span is narrow."""

    ticks = []
    decade = math.floor(math.log10(low))
    while 10.0**decade <= high * 1.0000001:
        for step in (1.0, 2.0, 5.0):
            value = step * 10.0**decade
            if low <= value <= high:
                ticks.append(value)
        decade += 1
    return ticks or [low, high]


def _fmt(value: float, unit: str) -> str:
    if unit == "x":
        return f"{value:.2f}\u00d7"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def dot_plot(
    path: Path,
    *,
    title: str,
    subtitle: str,
    groups: list[tuple[str, list[BarEntry]]],
    unit: str,
    series: list[tuple[str, str]],
    reference: float | None = None,
    label_dots: bool = False,
) -> None:
    """One row per case, one dot per backend, on a log value axis.

    A dot plot rather than bars: these values span three orders of magnitude,
    and a bar has to start at zero to be honest, which a log axis cannot do.
    Position encoding carries the comparison instead, and each row becomes a
    single readable range rather than three separate bars.
    """

    left, right, top = 250, 190, 116
    row_height = 34
    width = 1180
    height = top + row_height * max(1, len(groups)) + 74
    plot_width = width - left - right

    values = [e.value for _, entries in groups for e in entries]
    values += [e.lower for _, entries in groups for e in entries if e.lower > 0]
    values += [e.upper for _, entries in groups for e in entries if e.upper > 0]
    values = [v for v in values if v > 0] or [1.0]
    low, high = min(values), max(values)
    if reference is not None:
        # Keep parity on the axis, but do not force symmetry about it: these
        # ratios sit almost entirely on one side, and a symmetric span would
        # spend half the width on an empty region.
        low, high = min(low, reference), max(high, reference)
    low, high = low / 1.45, high * 1.45
    log_low, log_high = math.log10(low), math.log10(high)

    def x_of(value: float) -> float:
        value = max(value, low)
        return left + plot_width * (math.log10(value) - log_low) / (log_high - log_low)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
        f'<text x="34" y="40" font-size="21" font-weight="600" fill="{INK}">{escape(title)}</text>',
        f'<text x="34" y="63" font-size="13" fill="{INK_MUTED}">{escape(subtitle)}</text>',
    ]

    # Legend: identity is never colour-alone, so every series is named here.
    # Only categories actually present are listed -- a legend entry with no
    # members on the chart is noise, and on the speedup plot it would advertise
    # a "slower" class that no case falls into.
    present = {entry.color_key for _, entries in groups for entry in entries}
    lx = 34
    for key, name in [item for item in series if item[0] in present]:
        out.append(f'<circle cx="{lx + 5}" cy="84" r="5" fill="{COLORS[key]}"/>')
        out.append(
            f'<text x="{lx + 16}" y="88" font-size="12" fill="{INK_MUTED}">{escape(name)}</text>'
        )
        lx += 30 + 7.6 * len(name)

    plot_bottom = top + row_height * len(groups)

    # Hairline grid, one shade off the surface, solid (never dashed).
    for tick in _log_ticks(low, high):
        x = x_of(tick)
        out.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{plot_bottom + 6}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 24}" text-anchor="middle" font-size="11" '
            f'fill="{INK_MUTED}" font-variant-numeric="tabular-nums">{_fmt(tick, unit)}</text>'
        )

    if reference is not None:
        x = x_of(reference)
        out.append(
            f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{plot_bottom + 6}" '
            f'stroke="{INK_MUTED}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{top - 20}" text-anchor="middle" font-size="11" '
            f'fill="{INK_MUTED}">parity</text>'
        )

    for index, (case, entries) in enumerate(groups):
        cy = top + index * row_height + row_height / 2
        if index % 2 == 1:
            out.append(
                f'<rect x="{left}" y="{cy - row_height / 2:.1f}" width="{plot_width}" '
                f'height="{row_height}" fill="#00000006"/>'
            )
        out.append(
            f'<text x="{left - 14}" y="{cy + 4:.1f}" text-anchor="end" font-size="12" '
            f'fill="{INK}">{escape(case)}</text>'
        )
        positioned = sorted(entries, key=lambda e: e.value)
        if len(positioned) > 1:
            out.append(
                f'<line x1="{x_of(positioned[0].value):.1f}" y1="{cy:.1f}" '
                f'x2="{x_of(positioned[-1].value):.1f}" y2="{cy:.1f}" '
                f'stroke="{GRID}" stroke-width="3" stroke-linecap="round"/>'
            )
        for entry in positioned:
            x = x_of(entry.value)
            if entry.repeat_count > 1 and entry.upper > entry.lower:
                out.append(
                    f'<line class="error-bar" x1="{x_of(entry.lower):.1f}" y1="{cy:.1f}" '
                    f'x2="{x_of(entry.upper):.1f}" y2="{cy:.1f}" '
                    f'stroke="{COLORS[entry.color_key]}" stroke-width="1" opacity="0.55"/>'
                )
            # 2px surface ring so overlapping dots stay separable.
            out.append(
                f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="6.5" fill="{SURFACE}"/>'
                f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="5" fill="{COLORS[entry.color_key]}"/>'
            )
        if label_dots:
            # Only the row's extremes are labelled, and only on the outside, so
            # the numbers never collide with a dot or with each other.
            lo, hi = positioned[0], positioned[-1]
            out.append(
                f'<text x="{x_of(lo.value) - 11:.1f}" y="{cy + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="{INK_MUTED}" font-variant-numeric="tabular-nums">'
                f'{_fmt(lo.value, unit)}</text>'
            )
            if hi is not lo:
                out.append(
                    f'<text x="{x_of(hi.value) + 11:.1f}" y="{cy + 4:.1f}" font-size="11" '
                    f'fill="{INK}" font-variant-numeric="tabular-nums">'
                    f'{_fmt(hi.value, unit)}</text>'
                )

    repeats = max(
        (entry.repeat_count for _, entries in groups for entry in entries), default=1
    )
    runs = f"n={repeats} independent runs" if repeats > 1 else "a single run"
    out.append(
        f'<text x="34" y="{height - 16}" font-size="11" fill="{INK_MUTED}">'
        f'{escape(f"Log scale. Dots are medians of {runs}; the faint bar through a dot is the between-run IQR.")}</text>'
    )
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


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
        f'<figure><img src="{escape(name)}" alt="{escape(name)}"></figure>'
        for name in generated
    )
    path.write_text(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>e3nn benchmark</title>
<style>
:root{color-scheme:light}
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:2.5rem 1.5rem;
  color:#1c1c1a;background:#fcfcfb;line-height:1.5}
main{max-width:1180px;margin:0 auto}
h1{font-size:1.6rem;font-weight:600;margin:0 0 .4rem}
h2{font-size:1.05rem;font-weight:600;margin:2.5rem 0 .75rem}
p{color:#6b6a65;max-width:64ch;margin:0 0 1.5rem}
figure{margin:0 0 1.75rem}
img{max-width:100%;display:block;border:1px solid #e6e5e1;border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:.82rem;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #e6e5e1;padding:.4rem .6rem;text-align:left}
th{font-weight:600;color:#6b6a65;border-bottom-width:2px;white-space:nowrap}
tbody tr:hover{background:#00000005}
code{font-family:ui-monospace,monospace;font-size:.9em}
</style></head><body>
<main>
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
        + "</tbody></table></main></body></html>\n"
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
        dot_plot(
            args.output_dir / latency_name,
            title=f"{phase.title()} steady-state latency",
            subtitle="Milliseconds per call, lower is better. One row per case.",
            groups=group_by_case(latency_entries(rows, phase)),
            unit="ms",
            series=[
                ("torch-cpu", "Torch CPU"),
                ("mlx", "MLX"),
                ("mlx-kernel", "MLX + kernels"),
            ],
        )
        generated.append(latency_name)
        speedup_name = f"speedup_{phase}.svg"
        dot_plot(
            args.output_dir / speedup_name,
            title=f"{phase.title()} speedup over Torch CPU",
            subtitle=(
                "Torch CPU median divided by MLX median, paired by repeat. "
                "Right of parity is faster."
            ),
            groups=group_by_case(speedup_entries(rows, phase)),
            unit="x",
            series=[
                ("mlx", "MLX"),
                ("mlx-kernel", "MLX + kernels"),
            ],
            reference=1.0,
            label_dots=True,
        )
        generated.append(speedup_name)
    write_csv(args.output_dir / "summary.csv", aggregated)
    write_html(args.output_dir / "report.html", aggregated, generated)
    print(f"Report: {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
