"""Colour arms of the ablation: none, Macenko, physical calibration.

The physical arm is the interesting one. It does not look at tissue at all. Each
scanner in the cohort images a colour target — a small chart of known patches —
and the calibration fits the affine map from that scanner's rendering of the
chart to the reference rendering. The fit is per scanner and per acquisition
period, so it tracks drift as the scanner ages, which is the correction Ji et
al. (Modern Pathology, 2025) apply physically and Salmon et al. (J Pathol Inform,
2026) show rescues a model that had started to age.

The distinction the ablation needs: Macenko normalises what the tissue looks
like after the fact and needs enough stain in the tile to estimate anything.
Calibration corrects the instrument and works on a blank slide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wsi_ablation.stain import macenko_normalise
from wsi_ablation.types import ColourArm

ByteArr = NDArray[np.uint8]
FloatArr = NDArray[np.float64]

# Sixteen neutral-to-chromatic patches spanning the range an H&E slide occupies.
# Values are the reference (uncorrupted) sRGB of each patch.
COLOUR_TARGET: FloatArr = np.array(
    [
        [255, 255, 255], [235, 235, 235], [200, 200, 200], [160, 160, 160],
        [120, 120, 120], [ 80,  80,  80], [ 40,  40,  40], [ 10,  10,  10],
        [200, 140, 190], [170,  95, 165], [130,  60, 140], [ 95,  35, 110],
        [245, 190, 205], [235, 150, 175], [215, 110, 145], [190,  75, 120],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Calibration:
    """Affine sRGB correction fitted from one scanner's colour target."""

    scanner_id: str
    matrix: FloatArr  # 3x3
    offset: FloatArr  # 3

    def apply(self, rgb: ByteArr) -> ByteArr:
        arr = np.asarray(rgb, dtype=np.float64)
        corrected = arr @ self.matrix.T + self.offset
        return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def fit_calibration(scanner_id: str, observed_target: FloatArr) -> Calibration:
    """Least-squares fit of observed target patches back to the reference chart.

    An intercept column is appended so the fit absorbs a black-level offset as
    well as channel gain; a gain-only fit leaves a visible cast behind.
    """
    if observed_target.shape != COLOUR_TARGET.shape:
        raise ValueError(
            f"colour target must be {COLOUR_TARGET.shape}, got {observed_target.shape}"
        )
    design = np.hstack([observed_target, np.ones((observed_target.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(design, COLOUR_TARGET, rcond=None)
    return Calibration(
        scanner_id=scanner_id,
        matrix=np.ascontiguousarray(solution[:3].T),
        offset=np.ascontiguousarray(solution[3]),
    )


def identity_calibration(scanner_id: str = "identity") -> Calibration:
    return Calibration(scanner_id, np.eye(3), np.zeros(3))


def apply_colour_arm(
    rgb: ByteArr,
    arm: ColourArm,
    calibration: Calibration | None = None,
) -> ByteArr:
    """Dispatch one tile through the colour arm under test."""
    if arm == "none":
        return np.asarray(rgb, dtype=np.uint8)
    if arm == "macenko":
        return macenko_normalise(rgb)
    if arm == "physical":
        if calibration is None:
            raise ValueError("the physical arm needs a fitted calibration for the scanner")
        return calibration.apply(rgb)
    raise ValueError(f"unknown colour arm: {arm}")


def delta_to_reference(observed_target: FloatArr) -> float:
    """Mean per-channel deviation of a scanner's chart from the reference.

    Reported before and after calibration as the honest check that the fit did
    something: a residual near zero means the instrument, not the tissue, was
    the thing that moved.
    """
    return float(np.abs(observed_target - COLOUR_TARGET).mean())
