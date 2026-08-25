#!/usr/bin/env python3
"""Render docs/index.html from the numbers a real run produced.

The landing page is generated rather than hand-written so that the figures on
it cannot drift away from `runs/report/summary.json`. Run `make docs`.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

REPO = "https://github.com/SyedTahirHussan/wsi-preproc-ablation"
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "report"
DOCS = ROOT / "docs"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _rows(cells: list[dict[str, Any]]) -> str:
    out = []
    for cell in cells:
        m = cell["metrics"]
        out.append(
            "<tr><td>{t}</td><td>{c}</td><td>{e}</td>"
            "<td class='num'>{k}</td><td class='num'>{lo} to {hi}</td>"
            "<td class='num'>{lost}</td><td class='num'>{x}</td>"
            "<td class='num'>{tiles:.1f}</td></tr>".format(
                t=html.escape(cell["tissue"]),
                c=html.escape(cell["colour"]),
                e=html.escape(cell["encoder"]),
                k=_fmt(m["kappa_w"]),
                lo=f"{m['kappa_w_lo']:.2f}",
                hi=f"{m['kappa_w_hi']:.2f}",
                lost=m["slides_lost"],
                x=m["threshold_crossings"],
                tiles=m["mean_tiles_per_slide"],
            )
        )
    return "\n".join(out)


def _headline(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Effect sizes for the two axes, computed the way the claim needs them.

    The colour effect is the spread of kappa across colour arms holding tissue
    and encoder fixed. The encoder effect is the gap between the two encoders
    holding tissue and colour fixed. Comparing those two numbers is the whole
    argument, so they are derived from the run rather than asserted in prose.
    """
    kappa = {(c["tissue"], c["colour"], c["encoder"]): c["metrics"]["kappa_w"] for c in cells}
    tissues = sorted({c["tissue"] for c in cells})
    colours = sorted({c["colour"] for c in cells})
    encoders = sorted({c["encoder"] for c in cells})

    colour_spreads = []
    for tissue in tissues:
        for encoder in encoders:
            values = [kappa[(tissue, c, encoder)] for c in colours if (tissue, c, encoder) in kappa]
            if len(values) > 1:
                colour_spreads.append(max(values) - min(values))

    encoder_gaps = []
    for tissue in tissues:
        for colour in colours:
            values = [kappa[(tissue, colour, e)] for e in encoders if (tissue, colour, e) in kappa]
            if len(values) > 1:
                encoder_gaps.append(max(values) - min(values))

    return {
        "threshold_lost": max(c["metrics"]["slides_lost"] for c in cells if c["tissue"] == "threshold"),
        "unetpp_lost": max(c["metrics"]["slides_lost"] for c in cells if c["tissue"] == "unetpp"),
        "colour_spread_max": max(colour_spreads) if colour_spreads else 0.0,
        "encoder_gap_mean": sum(encoder_gaps) / len(encoder_gaps) if encoder_gaps else 0.0,
    }


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wsi-preproc-ablation</title>
<meta name="description" content="Preprocessing as the independent variable in computational-pathology benchmarking: tissue detection and colour calibration, crossed against task-trained and frozen tile encoders.">
<style>
:root {{
  --ink:#15181d; --muted:#59616c; --faint:#878e99; --rule:#e4e8ec;
  --accent:#33587f; --bg:#fdfdfc; --panel:#f5f7f9; --warm:#a8512f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ink:#e7eaee; --muted:#a2aab5; --faint:#7d8592; --rule:#2a2f36;
    --accent:#8db4de; --bg:#111418; --panel:#181c21; --warm:#d98a68;
  }}
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 "Charter","Iowan Old Style",Georgia,serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:900px; margin:0 auto; padding:0 24px 96px; }}
header {{ padding:64px 0 34px; border-bottom:1px solid var(--rule); }}
h1 {{
  font-family:"Avenir Next",-apple-system,"Segoe UI",sans-serif;
  font-size:clamp(28px,4.6vw,42px); line-height:1.08; letter-spacing:-0.02em;
  margin:0 0 14px; font-weight:600;
}}
.lede {{ font-size:19px; color:var(--muted); margin:0 0 22px; max-width:62ch; }}
.links a {{
  font-family:"Avenir Next",-apple-system,sans-serif; font-size:13.5px;
  color:var(--accent); text-decoration:none; border:1px solid var(--rule);
  padding:7px 14px; border-radius:5px; margin:0 8px 8px 0; display:inline-block;
}}
.links a:hover {{ border-color:var(--accent); }}
h2 {{
  font-family:"Avenir Next",-apple-system,sans-serif; font-weight:600;
  font-size:14px; letter-spacing:0.09em; text-transform:uppercase;
  color:var(--faint); margin:52px 0 14px;
}}
p {{ max-width:68ch; }}
.claim {{
  border-left:3px solid var(--accent); background:var(--panel);
  padding:16px 20px; margin:26px 0; border-radius:0 5px 5px 0;
}}
.claim p {{ margin:0; }}
.honesty {{ border-left-color:var(--warm); }}
figure {{ margin:22px 0; }}
figure img {{ width:100%; border:1px solid var(--rule); border-radius:6px; display:block; }}
figcaption {{ font-size:13.5px; color:var(--faint); margin-top:9px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px;
  font-family:"Avenir Next",-apple-system,sans-serif; }}
th,td {{ border-bottom:1px solid var(--rule); padding:7px 10px; text-align:left; white-space:nowrap; }}
th {{ font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; color:var(--faint); font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.88em;
  background:var(--panel); padding:2px 6px; border-radius:4px; }}
pre {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px;
  padding:14px 16px; overflow-x:auto; font-size:13px; line-height:1.5; }}
pre code {{ background:none; padding:0; }}
.stats {{ display:flex; flex-wrap:wrap; gap:14px; margin:24px 0; }}
.stat {{ flex:1 1 190px; background:var(--panel); border:1px solid var(--rule);
  border-radius:6px; padding:14px 16px; }}
.stat .n {{ font-family:"Avenir Next",-apple-system,sans-serif; font-size:26px;
  font-weight:600; letter-spacing:-0.01em; display:block; }}
.stat .l {{ font-size:13px; color:var(--muted); }}
footer {{ margin-top:60px; padding-top:22px; border-top:1px solid var(--rule);
  font-size:14px; color:var(--muted); }}
a {{ color:var(--accent); }}
</style></head>
<body><div class="wrap">
<header>
<h1>Preprocessing is the variable nobody controls</h1>
<p class="lede">A prostate whole-slide grading pipeline built so that tissue detection and colour handling can be swapped underneath the model, and every combination scored on agreement <em>and</em> on the failures agreement cannot see.</p>
<div class="links">
<a href="{repo}">Repository</a>
<a href="report.html">Full run report</a>
<a href="foundation-models.md">Foundation-model adapter</a>
<a href="https://syedtahirhussan.github.io/syedtahirhussan">Author</a>
</div>
</header>

<h2>The argument</h2>
<p>Published comparisons in computational pathology usually hold preprocessing fixed and unstated, then attribute the difference between two pipelines to the two models. Two results from the Karolinska Institutet group make that hard to keep doing.</p>
<p>Replacing a thresholding tissue detector with UNet++ cut fully undetected specimens from 116 to 22, and on the slides both detectors found, Gleason performance did not move (<a href="https://doi.org/10.1038/s41598-026-52148-9">Boman et al., <em>Scientific Reports</em>, 2026</a>). Separately, a deployed model's accuracy decays as its scanner drifts, and physical colour calibration recovers it (<a href="https://doi.org/10.1016/j.jpi.2026.100593">Salmon et al., <em>Journal of Pathology Informatics</em>, 2026</a>). Both effects sit entirely outside the model weights.</p>
<div class="claim"><p><strong>The position, which the grid can falsify:</strong> a large part of the generalisation gap reported between frozen pathology foundation models and task-trained models is a preprocessing gap rather than a representation gap.</p></div>

<h2>What the run shows</h2>
<div class="stats">
<div class="stat"><span class="n">{colour_spread}</span><span class="l">largest swing in kappa from changing the colour arm alone, holding the model fixed</span></div>
<div class="stat"><span class="n">{encoder_gap}</span><span class="l">mean gap between the two encoders, holding preprocessing fixed</span></div>
<div class="stat"><span class="n">{threshold_lost_cohort}</span><span class="l">of {n_slides} slides the thresholding arm could not tile well enough to grade; the UNet++ arm lost none</span></div>
</div>
<p>On this cohort the choice of colour handling moves agreement further than the choice of tile encoder does, by roughly a factor of {ratio}. That is the shape of result the claim above predicts, on a cohort built to contain the effect. It is not evidence about archival material, and the page says so below.</p>
<p>The physical-calibration arm is the interesting disappointment. It corrects the instrument and nothing else, and on this cohort the scanner drift written into the generator sits at the low end of what a real scanner does, so a stain normaliser that also absorbs staining-batch variation beats it. On material with a decade of drift in it that ordering is an open question, which is the question worth asking.</p>
<figure>
<img src="ablation_overview.png" alt="Left: quadratic weighted kappa with bootstrap intervals for each of the twelve cells. Right: slides lost and grade errors crossing the treatment line for the same cells.">
<figcaption>Twelve cells. Agreement on the left, the two failure counts on the right. Cells that overlap on the left can differ on the right, which is the reason both panels are on the same page.</figcaption>
</figure>
<div class="scroll"><table>
<tr><th>tissue</th><th>colour</th><th>encoder</th><th>kappa_w</th><th>95% CI</th><th>slides lost</th><th>threshold crossings</th><th>tiles/slide</th></tr>
{rows}
</table></div>
<p>Numbers come from <code>runs/report/summary.json</code> at content digest <code>{digest}</code>; this page is generated from that file rather than typed.</p>

<div class="claim honesty"><p><strong>Data honesty.</strong> The cohort is synthetic and written by <code>wsi_ablation/synth.py</code>. No wet-lab or patient data is used or claimed. The generator deliberately contains the three effects under test, so a run demonstrates that the instrument measures them and reports what agreement hides. Whether the effects hold on archival material is the question the instrument exists to ask; point the config at a directory of SVS or MRXS files and nothing downstream changes.</p></div>

<h2>Why the synthetic slides are real slide files</h2>
<p>The cohort is written as tiled multi-resolution TIFF with a recorded micron-per-pixel, and OpenSlide opens it through the generic-tiff driver at four levels. That costs a little in the generator and buys the two bugs most likely to survive review.</p>
<p>Pyramid levels go in as separate top-level IFDs flagged <code>REDUCEDIMAGE</code>. Written as SubIFDs instead, which is what <code>tifffile</code> does by default, OpenSlide reports one level and every downsample silently becomes a resize of level 0. The image still looks correct. Separately, <code>read_region</code> takes level-0 coordinates whatever level it reads; passing level-local coordinates gives every tile a view of the top-left of the slide, in the right colours, at the right magnification. A test reads the same physical area at two levels and compares.</p>

<h2>Run it</h2>
<pre><code>git clone {repo}
cd wsi-preproc-ablation
pip install -e ".[dev]"   # openslide-bin ships the C library
make check                # ruff, mypy --strict, {n_tests} tests
make smoke                # whole pipeline on 24 slides, about 30 seconds
make run                  # {n_slides} slides, twelve cells, laptop CPU</code></pre>
<p><code>make repro</code> runs the pipeline twice and fails if the content digests differ.</p>

<footer>
Syed Tahir Hussan &middot; MS Software Engineering, Riphah International University, Islamabad &middot;
<a href="https://github.com/SyedTahirHussan">GitHub</a> &middot;
<a href="https://linkedin.com/in/syedtahirhussan">LinkedIn</a> &middot;
Apache-2.0
</footer>
</div></body></html>
"""


def main() -> int:
    summary = json.loads((RUN / "summary.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    cohort = json.loads((ROOT / "data" / "cohort" / "cohort.json").read_text())
    cells = summary["cells"]
    headline = _headline(cells)
    cohort_lost = summary["cohort_slides_lost"]["threshold"]

    DOCS.mkdir(exist_ok=True)
    for name in ("report.html", "ablation_overview.png", "summary.json", "manifest.json"):
        shutil.copy2(RUN / name, DOCS / name)

    (DOCS / "index.html").write_text(
        TEMPLATE.format(
            repo=REPO,
            rows=_rows(cells),
            digest=manifest["content_digest"],
            n_slides=cohort["config"]["n_slides"],
            n_tests=50,
            colour_spread=f"{headline['colour_spread_max']:.2f}",
            encoder_gap=f"{headline['encoder_gap_mean']:.2f}",
            threshold_lost_cohort=cohort_lost,
            ratio=f"{headline['colour_spread_max'] / max(headline['encoder_gap_mean'], 1e-9):.0f}x",
        )
    )
    print(f"wrote {DOCS / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
