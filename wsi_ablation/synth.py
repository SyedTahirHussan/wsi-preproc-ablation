"""A deterministic synthetic cohort written as real pyramidal whole-slide files.

No wet-lab data is claimed anywhere in this repository. What is claimed is that
the files on disk are genuine multi-resolution tiled TIFF pyramids with a
recorded micron-per-pixel, that OpenSlide opens them through its generic-tiff
driver, and that everything downstream reads them the way it would read an SVS
or an MRXS. Point the config at a directory of real slides and nothing after
this module changes.

Three properties are built into the cohort on purpose, because they are the
three the ablation is designed to measure:

1. Gleason pattern is rendered as morphology, not as a colour tag. Pattern 3 is
   discrete glands with open lumina, pattern 4 fuses them and closes the lumina,
   pattern 5 is a sheet. A grader has to look at structure to recover the grade.
2. A subset of slides is faintly stained, standing in for archival material that
   has lost stain over two decades in a biobank. These are the slides a
   saturation threshold drops entirely.
3. Each scanner has a colour profile that drifts with the acquisition date, and
   each scanner also images a colour target. The drift is recoverable from the
   target alone, which is what makes the physical-calibration arm a fair test
   rather than an oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray

from wsi_ablation.colour import COLOUR_TARGET
from wsi_ablation.isup import isup_grade_group
from wsi_ablation.stain import concentrations_to_rgb
from wsi_ablation.types import ScannerProfile, SlideSpec, Split

ByteArr = NDArray[np.uint8]
FloatArr = NDArray[np.float64]

SCANNERS: tuple[ScannerProfile, ...] = (
    ScannerProfile("SC-A", "S1", (1.000, 1.000, 1.000), (0.0, 0.0, 0.0), (0.0022, -0.0009, -0.0030)),
    ScannerProfile("SC-B", "S2", (0.980, 1.010, 1.030), (2.0, -1.0, 3.0), (-0.0013, 0.0018, 0.0027)),
    ScannerProfile("SC-C", "S3", (1.035, 0.972, 0.941), (-3.0, 2.0, 5.0), (0.0030, -0.0024, -0.0036)),
)

# Gleason pattern -> (gland radius px, lumen ratio, glands per mm^2, nuclear rim)
_PATTERN_MORPHOLOGY: dict[int, tuple[float, float, float, float]] = {
    0: (170.0, 0.66, 14.0, 0.26),
    3: (100.0, 0.48, 42.0, 0.38),
    4: (55.0, 0.15, 150.0, 0.56),
    5: (30.0, 0.00, 420.0, 0.74),
}

_PATTERN_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 0), (3, 3), (3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (4, 5), (5, 5),
)

BASE_WIDTH = 4096
BASE_HEIGHT = 2048
BASE_MPP = 0.5
PYRAMID_LEVELS = 4
TILE_SIZE_TIFF = 256
ARCHIVE_START_YEAR = 2005
ARCHIVE_END_YEAR = 2020
REFERENCE_YEAR = ARCHIVE_START_YEAR


@dataclass(frozen=True)
class CohortConfig:
    n_slides: int = 140
    seed: int = 20260825
    faint_fraction: float = 0.20
    severe_fade_fraction: float = 0.04
    out_dir: str = "data/cohort"


def _scanner_transform(rgb: ByteArr, profile: ScannerProfile, age_years: float) -> ByteArr:
    gain = np.asarray(profile.gain_at(age_years), dtype=np.float64)
    offset = np.asarray(profile.offset, dtype=np.float64)
    out = np.asarray(rgb, dtype=np.float64) * gain + offset
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _stamp_disc(
    canvas: FloatArr, cy: float, cx: float, radius: float, value: float, feather: float = 1.5
) -> None:
    """Add a soft-edged disc in place, touching only its bounding box."""
    h, w = canvas.shape
    r = int(np.ceil(radius + feather)) + 1
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    dist = np.sqrt(yy * yy + xx * xx)
    weight = np.clip((radius - dist) / feather + 0.5, 0.0, 1.0)
    canvas[y0:y1, x0:x1] += value * weight


def _render_core(
    rng: np.random.Generator,
    height: int,
    width: int,
    y_centre: float,
    thickness: float,
    primary: int,
    secondary: int,
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Return (haematoxylin, eosin, tissue mask) for one biopsy core phantom."""
    haem = np.zeros((height, width), dtype=np.float64)
    eos = np.zeros((height, width), dtype=np.float64)
    mask = np.zeros((height, width), dtype=np.float64)

    # The core itself: a gently undulating strip of eosinophilic stroma.
    wave = y_centre + 40.0 * np.sin(np.linspace(0.0, 2.4 * np.pi, width) + rng.uniform(0, 6.28))
    rows = np.arange(height, dtype=np.float64)[:, None]
    half = thickness / 2.0
    body = np.clip((half - np.abs(rows - wave[None, :])) / 14.0 + 0.5, 0.0, 1.0)
    mask += body
    eos += body * rng.uniform(0.17, 0.25)
    haem += body * rng.uniform(0.035, 0.065)

    # Glands, drawn with the primary pattern over the left 70% of the core and
    # the secondary over the rest, so a tile can carry either.
    for pattern, x_lo, x_hi in ((primary, 0.0, 0.7), (secondary, 0.7, 1.0)):
        radius, lumen_ratio, per_mm2, rim = _PATTERN_MORPHOLOGY[pattern]
        span_px = (x_hi - x_lo) * width
        area_mm2 = (span_px * BASE_MPP / 1000.0) * (thickness * BASE_MPP / 1000.0)
        n_glands = max(1, round(per_mm2 * area_mm2))
        for _ in range(n_glands):
            cx = rng.uniform(x_lo * width, x_hi * width)
            cy = float(np.interp(cx, np.arange(width), wave)) + rng.normal(0.0, thickness * 0.22)
            if not (0.0 <= cy < height):
                continue
            r = radius * rng.uniform(0.75, 1.3)
            _stamp_disc(haem, cy, cx, r, rim * rng.uniform(0.8, 1.2))
            _stamp_disc(eos, cy, cx, r, -0.12)
            if lumen_ratio > 0.0:
                _stamp_disc(haem, cy, cx, r * lumen_ratio, -rim * 1.6)
                _stamp_disc(eos, cy, cx, r * lumen_ratio, -0.32)

    # Scattered stromal nuclei so that empty stroma is not perfectly flat.
    for _ in range(int(width * thickness / 900.0)):
        cx = rng.uniform(0, width)
        cy = float(np.interp(cx, np.arange(width), wave)) + rng.normal(0.0, thickness * 0.28)
        if 0.0 <= cy < height:
            _stamp_disc(haem, cy, cx, rng.uniform(3.0, 6.0), 0.45)

    np.clip(haem, 0.0, 2.5, out=haem)
    np.clip(eos, 0.0, 2.5, out=eos)
    np.clip(mask, 0.0, 1.0, out=mask)
    return haem, eos, mask


def _illumination_field(rng: np.random.Generator, height: int, width: int) -> FloatArr:
    """A smooth low-amplitude shading field, the way a real scanner lights a slide.

    Without it the background is numerically perfect and any saturation cutoff
    at all separates tissue from glass, which would make the detection arms look
    interchangeable for the wrong reason.
    """
    coarse = rng.normal(0.0, 1.0, size=(6, 10))
    ys = np.linspace(0, coarse.shape[0] - 1, height)
    xs = np.linspace(0, coarse.shape[1] - 1, width)
    rows = np.stack([np.interp(xs, np.arange(coarse.shape[1]), row) for row in coarse])
    field = np.stack([np.interp(ys, np.arange(coarse.shape[0]), rows[:, i]) for i in range(width)], axis=1)
    return np.asarray(field, dtype=np.float64)


def _build_pyramid(base: ByteArr) -> list[ByteArr]:
    """Area-average 2x decimation, the reduction a scanner writes into its file."""
    levels = [base]
    for _ in range(PYRAMID_LEVELS - 1):
        prev = levels[-1]
        h, w = prev.shape[0] // 2 * 2, prev.shape[1] // 2 * 2
        block = prev[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).astype(np.uint16)
        levels.append(np.ascontiguousarray(block.mean(axis=(1, 3)).astype(np.uint8)))
    return levels


def write_pyramid(path: Path, base: ByteArr, mpp: float = BASE_MPP) -> None:
    """Write a tiled multi-resolution TIFF that OpenSlide reads as a slide.

    Levels go in as separate top-level IFDs flagged REDUCEDIMAGE. OpenSlide's
    generic-tiff driver ignores SubIFDs, so a pyramid written that way opens as
    a single-level image and every downstream downsample silently becomes a
    resize of level 0.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(path), bigtiff=False) as writer:
        for index, level in enumerate(_build_pyramid(base)):
            level_mpp = mpp * (2**index)
            writer.write(
                level,
                tile=(TILE_SIZE_TIFF, TILE_SIZE_TIFF),
                photometric="rgb",
                compression="deflate",
                resolution=(10000.0 / level_mpp, 10000.0 / level_mpp),
                resolutionunit="CENTIMETER",
                subfiletype=0 if index == 0 else 1,
            )


def _assign_split(site_id: str, year: int) -> Split:
    """Train on two sites in the first half of the archive; test everywhere else.

    The held-out set therefore contains an unseen scanner and a later period on
    seen scanners, which are the two shifts the group reports in practice.
    """
    if site_id == "S3":
        return "test"
    return "train" if year <= 2016 else "test"


def generate_cohort(config: CohortConfig) -> tuple[list[SlideSpec], dict[str, list[list[float]]]]:
    """Write the cohort to disk and return its ground truth and colour targets."""
    rng = np.random.default_rng(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[SlideSpec] = []
    for index in range(config.n_slides):
        profile = SCANNERS[index % len(SCANNERS)]
        year = int(rng.integers(ARCHIVE_START_YEAR, ARCHIVE_END_YEAR + 1))
        age_years = float(year - REFERENCE_YEAR)
        primary, secondary = _PATTERN_PAIRS[int(rng.integers(0, len(_PATTERN_PAIRS)))]
        draw = float(rng.random())
        if draw < config.severe_fade_fraction:
            # Severely faded archival material. A slide like this is why the
            # count of specimens that never reached the grader belongs in the
            # results table rather than in a footnote.
            stain_scale = float(rng.uniform(0.04, 0.09))
        elif draw < config.severe_fade_fraction + config.faint_fraction:
            stain_scale = float(rng.uniform(0.13, 0.30))
        else:
            stain_scale = float(rng.uniform(0.85, 1.15))
        n_cores = int(rng.integers(1, 3))

        haem = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.float64)
        eos = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.float64)
        mask = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.float64)
        for core in range(n_cores):
            y_centre = BASE_HEIGHT * (core + 0.5) / n_cores
            h_c, e_c, m_c = _render_core(
                rng, BASE_HEIGHT, BASE_WIDTH, y_centre, 1500.0 / n_cores - 60.0,
                primary, secondary,
            )
            haem += h_c
            eos += e_c
            mask = np.maximum(mask, m_c)

        conc = np.stack([haem, eos], axis=-1) * stain_scale
        conc += rng.normal(0.0, 0.010, size=conc.shape)
        conc += _illumination_field(rng, BASE_HEIGHT, BASE_WIDTH)[..., None] * 0.018
        rgb = concentrations_to_rgb(np.clip(conc, 0.0, None))
        rgb = _scanner_transform(rgb, profile, age_years)

        slide_id = f"WSI-{index:04d}"
        rel_path = f"{slide_id}.tif"
        write_pyramid(out_dir / rel_path, rgb)
        np.save(out_dir / f"{slide_id}.mask.npy", (mask > 0.5))

        specs.append(
            SlideSpec(
                slide_id=slide_id,
                path=rel_path,
                site_id=profile.site_id,
                scanner_id=profile.scanner_id,
                acquisition_year=year,
                age_years=age_years,
                gleason_primary=primary,
                gleason_secondary=secondary,
                isup=isup_grade_group(primary, secondary),
                stain_scale=stain_scale,
                n_cores=n_cores,
                split=_assign_split(profile.site_id, year),
                mpp=BASE_MPP,
                width=BASE_WIDTH,
                height=BASE_HEIGHT,
            )
        )

    targets = _render_colour_targets(specs)
    manifest = {
        "config": {
            "n_slides": config.n_slides,
            "seed": config.seed,
            "faint_fraction": config.faint_fraction,
            "severe_fade_fraction": config.severe_fade_fraction,
        },
        "slides": [spec.to_json() for spec in specs],
        "colour_targets": targets,
    }
    (out_dir / "cohort.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return specs, targets


def _render_colour_targets(specs: list[SlideSpec]) -> dict[str, list[list[float]]]:
    """Image the reference chart on each scanner, at that scanner's mean age.

    A real deployment images the chart per session. Averaging the age over the
    slides a scanner contributed is the cheaper approximation, and it is the
    same approximation a site makes when it calibrates monthly rather than per
    slide, so the residual it leaves is informative rather than an artefact.
    """
    targets: dict[str, list[list[float]]] = {}
    for profile in SCANNERS:
        ages = [s.age_years for s in specs if s.scanner_id == profile.scanner_id]
        mean_age = float(np.mean(ages)) if ages else 0.0
        chart = COLOUR_TARGET.astype(np.uint8).reshape(-1, 1, 3)
        observed = _scanner_transform(chart, profile, mean_age).reshape(-1, 3)
        targets[profile.scanner_id] = observed.astype(float).tolist()
    return targets


def load_cohort(out_dir: str) -> tuple[list[SlideSpec], dict[str, list[list[float]]]]:
    manifest = json.loads((Path(out_dir) / "cohort.json").read_text())
    specs = [SlideSpec(**row) for row in manifest["slides"]]
    return specs, manifest["colour_targets"]
