"""Reviewer-facing outputs: a figure, a JSONL record, and a self-contained page.

The figure puts kappa and the failure counts side by side on purpose. A reader
who sees only the left panel will conclude the arms are close; the right panel
is where the arms differ, and putting them on separate pages would be a choice
about what the reader notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsi_ablation.types import AblationCell

ACCENT = "#3c5a80"
WARM = "#b4593a"
MUTED = "#8a8f98"


def write_records(path: Path, cells: list[AblationCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for cell in cells:
            handle.write(json.dumps(cell.to_json(with_predictions=True), sort_keys=True) + "\n")


def summarise(cells: list[AblationCell]) -> dict[str, Any]:
    return {
        "n_cells": len(cells),
        "cells": [cell.to_json() for cell in cells],
        "best_kappa": max((c.metrics.kappa_w for c in cells), default=0.0),
        "worst_kappa": min((c.metrics.kappa_w for c in cells), default=0.0),
        "total_slides_lost": {
            arm: max(c.metrics.slides_lost for c in cells if c.tissue == arm)
            for arm in {c.tissue for c in cells}
        },
    }


def plot_overview(path: Path, cells: list[AblationCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{c.tissue}\n{c.colour} / {c.encoder}" for c in cells]
    kappa = [c.metrics.kappa_w for c in cells]
    lo = [c.metrics.kappa_w - c.metrics.kappa_w_lo for c in cells]
    hi = [c.metrics.kappa_w_hi - c.metrics.kappa_w for c in cells]
    lost = [c.metrics.slides_lost for c in cells]
    crossings = [c.metrics.threshold_crossings for c in cells]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), width_ratios=[1.35, 1.0])
    positions = range(len(cells))

    axes[0].bar(positions, kappa, color=ACCENT, width=0.68)
    axes[0].errorbar(
        list(positions), kappa, yerr=[lo, hi], fmt="none", ecolor="#2b3a4d", capsize=3, lw=1.1
    )
    axes[0].set_ylabel("quadratic weighted kappa")
    axes[0].set_title("Agreement on the held-out split", loc="left", fontsize=11)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", color="#e3e6ea", lw=0.8)
    axes[0].set_axisbelow(True)

    width = 0.4
    axes[1].bar([p - width / 2 for p in positions], lost, width=width, color=WARM, label="slides lost")
    axes[1].bar(
        [p + width / 2 for p in positions],
        crossings,
        width=width,
        color=MUTED,
        label="grade errors crossing the treatment line",
    )
    axes[1].set_title("What agreement does not show", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", color="#e3e6ea", lw=0.8)
    axes[1].set_axisbelow(True)

    for axis in axes:
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink:#171a1f; --muted:#5d646e; --rule:#e2e6ea; --accent:#3c5a80; --bg:#ffffff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e8eaed; --muted:#9aa2ad; --rule:#2c3138; --accent:#8fb0d8; --bg:#14171b; }}
}}
body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; color:var(--ink);
       background:var(--bg); margin:0 auto; max-width:1080px; padding:40px 24px 72px; }}
h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:15px; margin:34px 0 10px; padding-bottom:5px; border-bottom:1px solid var(--rule); }}
.sub {{ color:var(--muted); margin:0 0 6px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th,td {{ border-bottom:1px solid var(--rule); padding:6px 8px; text-align:left; }}
th {{ font-weight:600; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.04em; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ background:color-mix(in srgb, var(--rule) 55%, transparent); padding:1px 5px; border-radius:3px; font-size:12px; }}
img {{ max-width:100%; height:auto; border:1px solid var(--rule); border-radius:6px; }}
.note {{ border-left:3px solid var(--accent); padding:8px 0 8px 14px; color:var(--muted); margin:16px 0; }}
.wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="note">{note}</div>
<h2>Provenance</h2>
<div class="wrap"><table>
<tr><th>code version</th><td>{code_version}</td><th>git</th><td>{git_sha}</td></tr>
<tr><th>config hash</th><td colspan="3"><code>{config_hash}</code></td></tr>
<tr><th>data hash</th><td colspan="3"><code>{data_hash}</code></td></tr>
<tr><th>content digest</th><td><code>{content_digest}</code></td><th>python</th><td>{python}</td></tr>
</table></div>
<h2>Ablation grid</h2>
<img src="ablation_overview.png" alt="Agreement and failure counts for every cell of the ablation">
<div class="wrap"><table>
<tr><th>tissue</th><th>colour</th><th>encoder</th><th>kappa_w</th><th>95% CI</th>
<th>accuracy</th><th>slides lost</th><th>threshold crossings</th><th>graded</th><th>tiles/slide</th></tr>
{rows}
</table></div>
<h2>How to read this</h2>
<p class="sub">{closing}</p>
</body></html>
"""


def write_page(
    path: Path,
    cells: list[AblationCell],
    manifest: dict[str, Any],
    title: str,
    subtitle: str,
    note: str,
    closing: str,
) -> None:
    rows = "\n".join(
        f"<tr><td>{cell.tissue}</td><td>{cell.colour}</td><td>{cell.encoder}</td>"
        f"<td class='num'>{cell.metrics.kappa_w:.3f}</td><td class='num'>{cell.metrics.kappa_w_lo:.2f} to {cell.metrics.kappa_w_hi:.2f}</td>"
        f"<td class='num'>{cell.metrics.accuracy:.3f}</td><td class='num'>{cell.metrics.slides_lost}</td><td class='num'>{cell.metrics.threshold_crossings}</td>"
        f"<td class='num'>{cell.metrics.n_evaluated}</td><td class='num'>{cell.metrics.mean_tiles_per_slide:.1f}</td></tr>"
        for cell in cells
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _PAGE.format(
            title=title,
            subtitle=subtitle,
            note=note,
            closing=closing,
            rows=rows,
            **manifest,
        )
    )
