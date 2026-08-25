"""QuPath interoperability, in both directions.

Tissue masks and tile grids leave this pipeline as QuPath GeoJSON, so a
pathologist can open the slide, see exactly which tissue the detector kept and
which tiles the grader read, and correct either. Corrections come back the same
way and rasterise into a mask the pipeline treats like any other.

Coordinates are level-0 pixels throughout, which is what QuPath's image
coordinate space uses. Writing overview-level coordinates into a QuPath file is
the mistake that produces annotations sitting in the top-left eighth of the
slide, and it is why `mask_to_geojson` takes the downsample explicitly instead
of inferring one.

Groovy scripts for the QuPath side are in `qupath/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon as raster_polygon
from skimage.measure import approximate_polygon, find_contours

BoolArr = NDArray[np.bool_]

TISSUE_CLASS = {"name": "Tissue", "color": [0, 170, 90]}
TILE_CLASS = {"name": "Tile", "color": [70, 110, 200]}


def _feature(coords: list[list[float]], classification: dict[str, Any], name: str) -> dict[str, Any]:
    ring = [*coords, coords[0]] if coords[0] != coords[-1] else coords
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "objectType": "annotation",
            "name": name,
            "classification": classification,
            "isLocked": True,
        },
    }


def mask_to_geojson(
    mask: BoolArr,
    downsample: float,
    tolerance: float = 2.0,
    min_area_px: int = 24,
) -> dict[str, Any]:
    """Contour a binary mask into QuPath annotations in level-0 coordinates.

    Contours are simplified with Douglas-Peucker before scaling, because an
    unsimplified mask boundary at overview resolution becomes tens of thousands
    of vertices once scaled up, and QuPath will open it but nobody will enjoy it.
    """
    features: list[dict[str, Any]] = []
    padded = np.pad(mask.astype(float), 1, mode="constant")
    for index, contour in enumerate(find_contours(padded, 0.5)):
        simplified = approximate_polygon(contour - 1, tolerance=tolerance)
        if len(simplified) < 4:
            continue
        area = 0.5 * abs(
            np.dot(simplified[:, 1], np.roll(simplified[:, 0], 1))
            - np.dot(simplified[:, 0], np.roll(simplified[:, 1], 1))
        )
        if area < min_area_px:
            continue
        coords = [[float(x * downsample), float(y * downsample)] for y, x in simplified]
        features.append(_feature(coords, TISSUE_CLASS, f"tissue-{index}"))
    return {"type": "FeatureCollection", "features": features}


def tiles_to_geojson(
    coords: list[tuple[int, int]],
    tile_size_level0: int,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Tile squares as QuPath annotations, given their level-0 origins."""
    features = []
    for index, (x0, y0) in enumerate(coords):
        square = [
            [float(x0), float(y0)],
            [float(x0 + tile_size_level0), float(y0)],
            [float(x0 + tile_size_level0), float(y0 + tile_size_level0)],
            [float(x0), float(y0 + tile_size_level0)],
        ]
        name = names[index] if names else f"tile-{index:04d}"
        features.append(_feature(square, TILE_CLASS, name))
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: Path, collection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, indent=1))


def geojson_to_mask(
    collection: dict[str, Any],
    shape: tuple[int, int],
    downsample: float,
    classification: str | None = "Tissue",
) -> BoolArr:
    """Rasterise QuPath annotations back onto a mask of the given shape.

    Polygons only. QuPath will happily export points, lines and ellipses, and a
    silent skip would leave a reviewer wondering why their correction did
    nothing, so anything else raises.
    """
    mask = np.zeros(shape, dtype=bool)
    for feature in collection.get("features", []):
        properties = feature.get("properties", {})
        if classification is not None:
            found = (properties.get("classification") or {}).get("name")
            if found != classification:
                continue
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            raise ValueError(
                f"only Polygon annotations can be rasterised, got {geometry.get('type')!r}"
            )
        ring = np.asarray(geometry["coordinates"][0], dtype=np.float64) / downsample
        rows, cols = raster_polygon(ring[:, 1], ring[:, 0], shape=shape)
        mask[rows, cols] = True
    return mask
