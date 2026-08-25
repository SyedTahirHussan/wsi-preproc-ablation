from __future__ import annotations

import numpy as np
import pytest

from wsi_ablation.qupath import geojson_to_mask, mask_to_geojson, tiles_to_geojson


def test_mask_round_trips_through_qupath_geojson() -> None:
    mask = np.zeros((80, 160), dtype=bool)
    mask[20:60, 30:120] = True
    collection = mask_to_geojson(mask, downsample=8.0)
    recovered = geojson_to_mask(collection, mask.shape, downsample=8.0)
    intersection = np.logical_and(mask, recovered).sum()
    union = np.logical_or(mask, recovered).sum()
    assert intersection / union > 0.97


def test_annotations_carry_the_properties_qupath_expects() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    feature = mask_to_geojson(mask, downsample=1.0)["features"][0]
    assert feature["properties"]["objectType"] == "annotation"
    assert feature["properties"]["classification"]["name"] == "Tissue"
    assert feature["geometry"]["type"] == "Polygon"
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]


def test_coordinates_are_scaled_to_level_zero() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    at_one = mask_to_geojson(mask, downsample=1.0)["features"][0]["geometry"]["coordinates"][0]
    at_eight = mask_to_geojson(mask, downsample=8.0)["features"][0]["geometry"]["coordinates"][0]
    assert max(x for x, _ in at_eight) == pytest.approx(8 * max(x for x, _ in at_one))


def test_tile_squares_are_written_at_the_requested_size() -> None:
    collection = tiles_to_geojson([(0, 0), (256, 512)], tile_size_level0=256)
    ring = collection["features"][1]["geometry"]["coordinates"][0]
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    assert max(xs) - min(xs) == 256
    assert max(ys) - min(ys) == 256


def test_non_polygon_annotations_raise_rather_than_being_skipped() -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "properties": {"classification": {"name": "Tissue"}},
            }
        ],
    }
    with pytest.raises(ValueError, match="Polygon"):
        geojson_to_mask(collection, (10, 10), downsample=1.0)
