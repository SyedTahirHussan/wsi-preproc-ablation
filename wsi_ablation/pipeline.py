"""Tiling, colour correction, and bag construction — one cell of the grid.

This is where the independent variable actually acts. A cell of the ablation is
a (tissue, colour, encoder) triple, and everything that differs between cells
happens in `extract_bags`: which tiles exist, and what colour they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wsi_ablation.colour import Calibration, apply_colour_arm
from wsi_ablation.encoders import FixedFeatureBank
from wsi_ablation.mil import Bag
from wsi_ablation.slide import Slide, TileCoord
from wsi_ablation.tissue import TissueDetector, is_detected, tissue_fraction
from wsi_ablation.types import ColourArm, EncoderArm, SlideSpec, TissueResult

ByteArr = NDArray[np.uint8]
BoolArr = NDArray[np.bool_]

TILE_MPP = 1.0
TILE_PX = 128
MAX_TILES = 24
MIN_TILE_COVERAGE = 0.35
MIN_TILES_FOR_GRADE = 4


@dataclass(frozen=True)
class SlideTiles:
    """Raw tiles for one slide under one tissue arm, before any colour work."""

    spec: SlideSpec
    tiles: ByteArr
    coords: list[TileCoord]
    result: TissueResult


def _coverage(mask: BoolArr, coord: TileCoord, level_downsample: float, mask_downsample: float) -> float:
    """Fraction of a tile covered by the mask, both mapped to level-0 pixels."""
    scale = mask_downsample
    y0 = int(coord.y0 / scale)
    x0 = int(coord.x0 / scale)
    span = max(1, round(coord.size * level_downsample / scale))
    window = mask[y0 : y0 + span, x0 : x0 + span]
    return float(window.mean()) if window.size else 0.0


def extract_tiles(
    spec: SlideSpec,
    root: Path,
    detector: TissueDetector,
    max_tiles: int = MAX_TILES,
) -> SlideTiles:
    """Tile one slide under one tissue arm, keeping the most covered tiles.

    Selection is deterministic: coverage descending, then reading order, so the
    same slide gives the same bag on every run and any difference between cells
    is attributable to the arms rather than to tie-breaking.
    """
    with Slide(root / spec.path) as slide:
        mask, mask_level, mask_downsample = detector.mask_for(slide)
        detected = is_detected(mask)
        grid = slide.tile_grid(TILE_MPP, TILE_PX)
        level_downsample = slide.level_downsample(grid[0].level) if grid else 1.0

        scored = [
            (coord, _coverage(mask, coord, level_downsample, mask_downsample))
            for coord in grid
        ]
        covered = [pair for pair in scored if pair[1] >= MIN_TILE_COVERAGE]
        covered.sort(key=lambda pair: (-pair[1], pair[0].row, pair[0].col))
        kept = [coord for coord, _ in covered[:max_tiles]]

        tiles = np.zeros((len(kept), TILE_PX, TILE_PX, 3), dtype=np.uint8)
        for index, coord in enumerate(kept):
            tiles[index] = slide.read_region_rgb(coord.x0, coord.y0, coord.level, coord.size)

    result = TissueResult(
        slide_id=spec.slide_id,
        arm=detector.arm,
        mask_level=mask_level,
        mask_downsample=mask_downsample,
        tissue_fraction=tissue_fraction(mask),
        detected=detected and len(kept) >= MIN_TILES_FOR_GRADE,
        n_tiles=len(kept),
    )
    return SlideTiles(spec=spec, tiles=tiles, coords=kept, result=result)


def apply_colour(tiles: ByteArr, arm: ColourArm, calibration: Calibration | None) -> ByteArr:
    if arm == "none":
        return tiles
    out = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        out[index] = apply_colour_arm(tile, arm, calibration)
    return out


def build_bag(
    slide_tiles: SlideTiles,
    colour: ColourArm,
    encoder: EncoderArm,
    calibration: Calibration | None,
    bank: FixedFeatureBank,
) -> Bag | None:
    """Bag for one slide in one cell, or None where the slide was lost."""
    if not slide_tiles.result.detected:
        return None
    tiles = apply_colour(slide_tiles.tiles, colour, calibration)
    if encoder == "task-trained":
        return Bag(slide_tiles.spec.slide_id, tiles=tiles, features=None, label=slide_tiles.spec.isup)
    return Bag(
        slide_tiles.spec.slide_id,
        tiles=None,
        features=bank.encode(tiles),
        label=slide_tiles.spec.isup,
    )
