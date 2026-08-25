from __future__ import annotations

import numpy as np

from wsi_ablation.colour import (
    COLOUR_TARGET,
    apply_colour_arm,
    delta_to_reference,
    fit_calibration,
    identity_calibration,
)
from wsi_ablation.stain import (
    concentrations_to_rgb,
    macenko_normalise,
    rgb_to_concentrations,
    rgb_to_od,
)


def _drift(target: np.ndarray, gain: tuple[float, float, float], offset: tuple[float, float, float]) -> np.ndarray:
    return np.clip(target * np.asarray(gain) + np.asarray(offset), 0, 255)


def test_calibration_recovers_a_known_scanner_drift() -> None:
    """The mechanism the physical arm claims: fit on the chart, not on tissue."""
    observed = _drift(COLOUR_TARGET, (1.06, 0.94, 0.88), (-5.0, 4.0, 9.0))
    before = delta_to_reference(observed)
    calibration = fit_calibration("SC-X", observed)
    corrected = calibration.apply(observed.astype(np.uint8).reshape(-1, 1, 3)).reshape(-1, 3)
    assert before > 5.0
    assert delta_to_reference(corrected.astype(float)) < before / 4


def test_identity_calibration_leaves_pixels_untouched() -> None:
    tile = (np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5).astype(np.uint8)
    assert np.array_equal(identity_calibration().apply(tile), tile)


def test_stain_round_trip_recovers_concentrations() -> None:
    concentrations = np.zeros((8, 8, 2))
    concentrations[..., 0] = 0.9
    concentrations[..., 1] = 0.4
    rgb = concentrations_to_rgb(concentrations)
    recovered = rgb_to_concentrations(rgb)[..., :2]
    assert np.abs(recovered - concentrations).max() < 0.05


def test_optical_density_is_zero_for_white_and_positive_for_ink() -> None:
    white = rgb_to_od(np.full((2, 2, 3), 255, dtype=np.uint8))
    ink = rgb_to_od(np.full((2, 2, 3), 40, dtype=np.uint8))
    assert np.allclose(white, 0.0)
    assert (ink > 0.5).all()


def test_macenko_leaves_a_blank_tile_alone(rgb_tile: np.ndarray) -> None:
    """Normalising an unstained tile must not manufacture stain out of noise."""
    blank = np.full_like(rgb_tile, 252)
    assert np.array_equal(macenko_normalise(blank), blank)


def test_colour_arms_dispatch(rgb_tile: np.ndarray) -> None:
    assert np.array_equal(apply_colour_arm(rgb_tile, "none"), rgb_tile)
    assert apply_colour_arm(rgb_tile, "macenko").shape == rgb_tile.shape
    assert apply_colour_arm(rgb_tile, "physical", identity_calibration()).shape == rgb_tile.shape
