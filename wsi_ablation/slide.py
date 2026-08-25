"""OpenSlide access: level selection by micron-per-pixel, and tile grids.

The reason this is its own module rather than three lines inline: the two
mistakes that quietly ruin a cross-scanner comparison both live here. The first
is tiling at a fixed level index instead of a fixed micron-per-pixel, which
compares 0.25 um/px tiles from one scanner against 0.5 um/px tiles from another
and calls the difference a model effect. The second is `read_region` coordinates,
which OpenSlide specifies in level-0 pixels regardless of the level being read;
passing level-local coordinates gives a picture of the top-left corner of the
slide for every tile, and it looks plausible enough to survive review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openslide
from numpy.typing import NDArray

ByteArr = NDArray[np.uint8]

DEFAULT_MPP = 0.5


@dataclass(frozen=True)
class TileCoord:
    """A tile's position, in level-0 pixels, as OpenSlide wants it."""

    x0: int
    y0: int
    level: int
    size: int
    row: int
    col: int


class Slide:
    """A thin, closable wrapper that keeps micron-per-pixel honest."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._os = openslide.OpenSlide(str(path))

    def close(self) -> None:
        self._os.close()

    def __enter__(self) -> Slide:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._os.dimensions

    @property
    def level_count(self) -> int:
        return self._os.level_count

    @property
    def mpp(self) -> float:
        """Micron per pixel at level 0, from the file's own metadata.

        Falls back to the standard 20x value with no warning suppressed: a slide
        that does not record its resolution is a data-management problem, and
        the caller is told through the return of `mpp_is_declared`.
        """
        value = self._os.properties.get(openslide.PROPERTY_NAME_MPP_X)
        return float(value) if value else DEFAULT_MPP

    @property
    def mpp_is_declared(self) -> bool:
        return self._os.properties.get(openslide.PROPERTY_NAME_MPP_X) is not None

    @property
    def vendor(self) -> str:
        return str(self._os.properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown"))

    def level_for_mpp(self, target_mpp: float) -> tuple[int, float]:
        """Best level for a target resolution, and the residual rescale factor.

        The residual is rarely 1.0 on real slides, because scanner pyramids are
        powers of two from a base of 0.2425 or 0.5 and the target is not. It is
        returned rather than swallowed so the caller can resample deliberately.
        """
        downsample = target_mpp / self.mpp
        level = int(self._os.get_best_level_for_downsample(max(downsample, 1.0)))
        return level, downsample / self._os.level_downsamples[level]

    def read_region_rgb(self, x0: int, y0: int, level: int, size: int) -> ByteArr:
        """RGB tile; alpha is composited onto white, the way a scanner shows it."""
        region = self._os.read_region((x0, y0), level, (size, size))
        rgba = np.asarray(region, dtype=np.float64)
        alpha = rgba[..., 3:4] / 255.0
        composited = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
        return np.clip(np.rint(composited), 0, 255).astype(np.uint8)

    def thumbnail(self, level: int) -> ByteArr:
        width, height = self._os.level_dimensions[level]
        return self.read_region_rgb(0, 0, level, max(width, height))[:height, :width]

    def level_dimensions(self, level: int) -> tuple[int, int]:
        return self._os.level_dimensions[level]

    def level_downsample(self, level: int) -> float:
        return float(self._os.level_downsamples[level])

    def tile_grid(self, target_mpp: float, tile_px: int) -> list[TileCoord]:
        """Every tile position on a non-overlapping grid at the target resolution."""
        level, _ = self.level_for_mpp(target_mpp)
        downsample = self._os.level_downsamples[level]
        width, height = self._os.level_dimensions[level]
        coords: list[TileCoord] = []
        for row, y in enumerate(range(0, height - tile_px + 1, tile_px)):
            for col, x in enumerate(range(0, width - tile_px + 1, tile_px)):
                coords.append(
                    TileCoord(
                        x0=round(x * downsample),
                        y0=round(y * downsample),
                        level=level,
                        size=tile_px,
                        row=row,
                        col=col,
                    )
                )
        return coords
