"""Command line: cohort, run, repro, qupath-export.

`repro` is the one that earns its place. It runs the pipeline twice in the same
process and compares content digests, which catches the ordinary sources of
irreproducibility — an unseeded shuffle, a set iteration, a dict that picked up
insertion order from a filesystem listing — at the point they are introduced
rather than six months later when a number will not reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from wsi_ablation import __version__
from wsi_ablation.ablation import RunConfig, run_ablation
from wsi_ablation.config import load_config
from wsi_ablation.provenance import build_manifest, sha256_json
from wsi_ablation.qupath import mask_to_geojson, tiles_to_geojson, write_geojson
from wsi_ablation.report import plot_overview, summarise, write_page, write_records
from wsi_ablation.slide import Slide
from wsi_ablation.synth import CohortConfig, generate_cohort, load_cohort
from wsi_ablation.tissue import TissueDetector
from wsi_ablation.types import AblationRun

PAGE_NOTE = (
    "The cohort is synthetic and every slide in it was written by "
    "<code>wsi_ablation/synth.py</code>. No wet-lab or patient data is used or claimed. "
    "The generator builds in the three effects under test - stain fading, scanner colour "
    "drift, and grade rendered as morphology - so what this run demonstrates is that the "
    "instrument detects those effects and reports the failures agreement hides. Testing "
    "whether the effects hold on archival material is what the instrument is for."
)
PAGE_CLOSING = (
    "Read the left panel and the right panel together. Cells that sit within each other's "
    "confidence intervals on agreement can still differ in how many slides never reached "
    "the grader, and in how many of the errors they did make moved a case across the "
    "surveillance-to-treatment line. Those are the two columns a method section usually "
    "omits, which is the omission this repository exists to argue against."
)


def _cohort_ready(data_dir: Path, config: CohortConfig) -> bool:
    manifest = data_dir / "cohort.json"
    if not manifest.exists():
        return False
    recorded = json.loads(manifest.read_text()).get("config", {})
    return bool(
        recorded.get("n_slides") == config.n_slides
        and recorded.get("seed") == config.seed
        and recorded.get("faint_fraction") == config.faint_fraction
        and recorded.get("severe_fade_fraction") == config.severe_fade_fraction
    )


def command_cohort(cohort: CohortConfig, force: bool) -> None:
    data_dir = Path(cohort.out_dir)
    if not force and _cohort_ready(data_dir, cohort):
        print(f"cohort already present at {data_dir}")
        return
    started = time.time()
    specs, _ = generate_cohort(cohort)
    splits = {s: sum(1 for spec in specs if spec.split == s) for s in ("train", "test")}
    faint = sum(1 for spec in specs if spec.stain_scale < 0.4)
    print(
        f"wrote {len(specs)} slides to {data_dir} in {time.time() - started:.1f}s "
        f"(train {splits['train']}, test {splits['test']}, faintly stained {faint})"
    )


def _run_once(cohort: CohortConfig, run: RunConfig, verbose: bool) -> AblationRun:
    command_cohort(cohort, force=False)
    specs, targets = load_cohort(cohort.out_dir)
    return run_ablation(specs, targets, run, verbose=verbose)


def command_run(cohort: CohortConfig, run: RunConfig) -> None:
    started = time.time()
    result = _run_once(cohort, run, verbose=True)
    cells = result.cells
    out_dir = Path(run.out_dir)
    summary = summarise(result)

    manifest = build_manifest(
        code_version=__version__,
        config={"cohort": cohort.__dict__, "run": run.to_json()},
        data_dir=Path(cohort.out_dir),
        seed=run.seed,
        results=summary,
    )

    write_records(out_dir / "records.jsonl", cells)
    plot_overview(out_dir / "ablation_overview.png", cells)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True))
    write_page(
        out_dir / "report.html",
        cells,
        manifest.to_json(),
        title="Preprocessing ablation for prostate whole-slide grading",
        subtitle=(
            f"{len(cells)} cells &middot; tissue detection x colour handling x tile encoder "
            f"&middot; content digest {manifest.content_digest}"
        ),
        note=PAGE_NOTE,
        closing=PAGE_CLOSING,
    )
    print(f"\nwrote {out_dir}/report.html in {time.time() - started:.1f}s total")


def command_repro(cohort: CohortConfig, run: RunConfig) -> int:
    first = summarise(_run_once(cohort, run, verbose=False))
    second = summarise(_run_once(cohort, run, verbose=False))
    digest_first, digest_second = sha256_json(first)[:16], sha256_json(second)[:16]
    if digest_first != digest_second:
        print(f"NOT REPRODUCIBLE: {digest_first} != {digest_second}", file=sys.stderr)
        return 1
    print(f"reproducible: both runs hashed to {digest_first}")
    return 0


def command_qupath_export(cohort: CohortConfig, slide_id: str, arm: str, out: Path) -> None:
    specs, _ = load_cohort(cohort.out_dir)
    matches = [spec for spec in specs if spec.slide_id == slide_id]
    if not matches:
        raise SystemExit(f"no slide {slide_id!r} in {cohort.out_dir}")
    spec = matches[0]

    from wsi_ablation.pipeline import TILE_MPP, TILE_PX, extract_tiles

    detector = TissueDetector(arm)  # type: ignore[arg-type]
    if arm == "unetpp":
        raise SystemExit("export the threshold arm, or train a segmenter first with `run`")

    with Slide(Path(cohort.out_dir) / spec.path) as slide:
        mask, _, mask_downsample = detector.mask_for(slide)
        level, _ = slide.level_for_mpp(TILE_MPP)
        tile_size_level0 = round(TILE_PX * slide.level_downsample(level))

    tiles = extract_tiles(spec, Path(cohort.out_dir), detector)
    write_geojson(out / f"{slide_id}-tissue.geojson", mask_to_geojson(mask, mask_downsample))
    write_geojson(
        out / f"{slide_id}-tiles.geojson",
        tiles_to_geojson([(c.x0, c.y0) for c in tiles.coords], tile_size_level0),
    )
    print(
        f"wrote {out}/{slide_id}-tissue.geojson and {out}/{slide_id}-tiles.geojson "
        f"({np.count_nonzero(mask)} mask pixels, {len(tiles.coords)} tiles)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wsi-ablation", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    cohort_parser = sub.add_parser("cohort", help="write the synthetic slide cohort")
    cohort_parser.add_argument("--force", action="store_true")

    sub.add_parser("run", help="run the ablation and write the report")
    sub.add_parser("repro", help="run twice and compare content digests")

    export = sub.add_parser("qupath-export", help="export a slide's mask and tiles as QuPath GeoJSON")
    export.add_argument("slide_id")
    export.add_argument("--arm", default="threshold", choices=["threshold"])
    export.add_argument("--out", default="runs/qupath", type=Path)

    args = parser.parse_args(argv)
    cohort, run = load_config(args.config)

    if args.command == "cohort":
        command_cohort(cohort, force=args.force)
        return 0
    if args.command == "run":
        command_run(cohort, run)
        return 0
    if args.command == "repro":
        return command_repro(cohort, run)
    if args.command == "qupath-export":
        command_qupath_export(cohort, args.slide_id, args.arm, args.out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
