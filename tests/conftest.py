from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wsi_ablation.synth import CohortConfig, generate_cohort
from wsi_ablation.types import SlideSpec


@pytest.fixture(scope="session")
def tiny_cohort(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[SlideSpec], Path]:
    """Six slides, written once for the whole session."""
    out = tmp_path_factory.mktemp("cohort")
    specs, _ = generate_cohort(CohortConfig(n_slides=6, seed=7, out_dir=str(out)))
    return specs, out


@pytest.fixture
def rgb_tile() -> np.ndarray:
    rng = np.random.default_rng(3)
    base = np.full((64, 64, 3), 235, dtype=np.float64)
    base[16:48, 16:48] = (168, 96, 176)
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
