"""OpenSlide contract tests. These are the ones that catch silent corruption."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide
import pytest

from wsi_ablation.slide import Slide
from wsi_ablation.types import SlideSpec


def test_written_files_are_real_openslide_pyramids(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    specs, root = tiny_cohort
    path = root / specs[0].path
    assert openslide.OpenSlide.detect_format(str(path)) == "generic-tiff"
    with Slide(path) as slide:
        assert slide.level_count == 4
        assert slide.level_dimensions(0) == (specs[0].width, specs[0].height)
        assert slide.level_dimensions(3) == (specs[0].width // 8, specs[0].height // 8)


def test_micron_per_pixel_is_declared_in_the_file(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    """A pyramid without a recorded resolution cannot be tiled at a fixed scale."""
    specs, root = tiny_cohort
    with Slide(root / specs[0].path) as slide:
        assert slide.mpp_is_declared
        assert slide.mpp == pytest.approx(specs[0].mpp, abs=1e-3)


def test_level_for_mpp_picks_the_pyramid_level_not_a_resize(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    specs, root = tiny_cohort
    with Slide(root / specs[0].path) as slide:
        level, residual = slide.level_for_mpp(2.0)
        assert level == 2
        assert residual == pytest.approx(1.0)


def test_read_region_coordinates_are_level_zero(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    """The same physical area read at two levels must show the same picture.

    If read_region were being passed level-local coordinates, the level-1 read
    would land at a quarter of the intended position and this would fail.
    """
    specs, root = tiny_cohort
    with Slide(root / specs[0].path) as slide:
        at_level_0 = slide.read_region_rgb(512, 256, 0, 128)
        at_level_1 = slide.read_region_rgb(512, 256, 1, 64)
        coarse = at_level_0[::2, ::2].astype(float)
        assert np.abs(coarse - at_level_1.astype(float)).mean() < 12.0


def test_tile_grid_covers_the_slide_without_overlap(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    specs, root = tiny_cohort
    with Slide(root / specs[0].path) as slide:
        grid = slide.tile_grid(target_mpp=1.0, tile_px=128)
        level_width, level_height = slide.level_dimensions(grid[0].level)
        assert len(grid) == (level_width // 128) * (level_height // 128)
        origins = {(coord.x0, coord.y0) for coord in grid}
        assert len(origins) == len(grid)
