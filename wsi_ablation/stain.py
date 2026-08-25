"""Beer-Lambert stain optics, shared by the renderer and the normalisers.

Tissue is rendered from haematoxylin and eosin concentrations rather than from
RGB directly, so that a colour arm which claims to work in stain space is
actually operating on the quantity it says it is. The stain vectors are
Ruifrok and Johnston's H&E basis, the same ones scikit-image and every Macenko
implementation start from.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
ByteArr = NDArray[np.uint8]

# Ruifrok & Johnston (2001), rows are haematoxylin, eosin, residual (DAB slot).
STAIN_MATRIX: FloatArr = np.array(
    [
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776],
    ],
    dtype=np.float64,
)

_EPS = 1e-6
_MAX_OD = 3.0


def rgb_to_od(rgb: ByteArr | FloatArr, background: float = 255.0) -> FloatArr:
    """Optical density, clipped so that a pure black pixel stays finite."""
    arr = np.asarray(rgb, dtype=np.float64)
    return np.asarray(np.clip(-np.log10(np.clip(arr, _EPS, background) / background), 0.0, _MAX_OD))


def od_to_rgb(od: FloatArr, background: float = 255.0) -> ByteArr:
    out = background * np.power(10.0, -np.clip(od, 0.0, _MAX_OD))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def concentrations_to_rgb(conc: FloatArr, stains: FloatArr | None = None) -> ByteArr:
    """Render (..., n_stains) concentrations through the stain basis to 8-bit RGB."""
    basis = STAIN_MATRIX[: conc.shape[-1]] if stains is None else stains
    od = conc @ basis
    return od_to_rgb(od)


def rgb_to_concentrations(rgb: ByteArr, stains: FloatArr | None = None) -> FloatArr:
    """Unmix RGB into stain concentrations by least squares on optical density."""
    basis = STAIN_MATRIX if stains is None else stains
    od = rgb_to_od(rgb)
    flat = od.reshape(-1, od.shape[-1])
    conc, *_ = np.linalg.lstsq(basis.T, flat.T, rcond=None)
    return np.ascontiguousarray(conc.T.reshape(*od.shape[:-1], basis.shape[0]))


def macenko_stain_vectors(
    rgb: ByteArr,
    od_threshold: float = 0.15,
    percentile: float = 1.0,
) -> FloatArr:
    """Estimate the two dominant stain vectors, Macenko et al. (ISBI 2009).

    Falls back to the reference H&E basis when a tile carries too little stain
    to estimate anything, which is exactly what happens on the faintly stained
    archival slides this repository is built to keep track of.
    """
    od = rgb_to_od(rgb).reshape(-1, 3)
    tissue = od[od.sum(axis=1) > od_threshold]
    if tissue.shape[0] < 32:
        return STAIN_MATRIX[:2].copy()

    cov = np.cov(tissue, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    plane = eigvecs[:, [2, 1]]
    if plane[0, 0] < 0:
        plane[:, 0] *= -1
    if plane[0, 1] < 0:
        plane[:, 1] *= -1

    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    lo = np.percentile(angles, percentile)
    hi = np.percentile(angles, 100.0 - percentile)

    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
    vectors = np.stack([v_lo, v_hi]) if v_lo[0] > v_hi[0] else np.stack([v_hi, v_lo])
    vectors = np.abs(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.maximum(norms, _EPS))


def macenko_normalise(
    rgb: ByteArr,
    target_vectors: FloatArr | None = None,
    target_concentrations: tuple[float, float] = (1.9, 1.2),
    od_threshold: float = 0.15,
) -> ByteArr:
    """Map a tile onto a reference stain appearance.

    Returns the input unchanged when the tile has no estimable stain content,
    rather than amplifying sensor noise into something that looks like tissue.
    """
    source = macenko_stain_vectors(rgb, od_threshold=od_threshold)
    target = STAIN_MATRIX[:2].copy() if target_vectors is None else target_vectors

    od = rgb_to_od(rgb)
    flat = od.reshape(-1, 3)
    conc, *_ = np.linalg.lstsq(source.T, flat.T, rcond=None)

    scale = np.percentile(conc, 99.0, axis=1)
    if not np.all(np.isfinite(scale)) or np.any(scale < 1e-3):
        return np.asarray(rgb, dtype=np.uint8)

    conc = conc / scale[:, None] * np.asarray(target_concentrations)[:, None]
    return od_to_rgb((conc.T @ target).reshape(od.shape))
