from __future__ import annotations

from pathlib import Path

import numpy as np

from wsi_ablation.colour import identity_calibration
from wsi_ablation.encoders import FIXED_BANK_DIM, FixedFeatureBank, tiles_to_cnn_input
from wsi_ablation.mil import Bag, MILGrader, ordinal_targets, predict, train_grader
from wsi_ablation.pipeline import MIN_TILES_FOR_GRADE, build_bag, extract_tiles
from wsi_ablation.tissue import TissueDetector, threshold_mask, train_unetpp, unetpp_mask
from wsi_ablation.types import SlideSpec


def test_tiles_are_selected_deterministically(tiny_cohort: tuple[list[SlideSpec], Path]) -> None:
    specs, root = tiny_cohort
    detector = TissueDetector("threshold")
    first = extract_tiles(specs[0], root, detector)
    second = extract_tiles(specs[0], root, detector)
    assert [c.__dict__ for c in first.coords] == [c.__dict__ for c in second.coords]
    assert np.array_equal(first.tiles, second.tiles)


def test_a_slide_with_too_few_tiles_is_reported_as_lost(
    tiny_cohort: tuple[list[SlideSpec], Path],
) -> None:
    """Losing a slide must be visible, not absorbed into a smaller denominator."""
    specs, root = tiny_cohort
    tiles = extract_tiles(specs[0], root, TissueDetector("threshold"), max_tiles=MIN_TILES_FOR_GRADE - 1)
    assert not tiles.result.detected
    bag = build_bag(tiles, "none", "fixed-bank", identity_calibration(), FixedFeatureBank())
    assert bag is None


def test_threshold_arm_refuses_to_mask_an_unstained_slide() -> None:
    blank = np.full((64, 128, 3), 250, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noisy = np.clip(blank + rng.normal(0, 2, blank.shape), 0, 255).astype(np.uint8)
    assert not threshold_mask(noisy).any()


def test_unetpp_forward_pass_returns_a_mask_of_the_input_shape() -> None:
    rng = np.random.default_rng(1)
    overview = (rng.random((30, 46, 3)) * 255).astype(np.uint8)
    mask = np.zeros((30, 46), dtype=bool)
    mask[8:20, 10:36] = True
    model = train_unetpp([overview], [mask], epochs=1, seed=0)
    assert unetpp_mask(model, overview).shape == (30, 46)


def test_fixed_bank_is_frozen_and_shaped() -> None:
    rng = np.random.default_rng(2)
    tiles = (rng.random((5, 64, 64, 3)) * 255).astype(np.uint8)
    bank = FixedFeatureBank()
    assert bank.dim == FIXED_BANK_DIM
    assert np.array_equal(bank.encode(tiles), bank.encode(tiles))


def test_cnn_input_is_decimated_not_interpolated() -> None:
    rng = np.random.default_rng(4)
    tiles = (rng.random((2, 128, 128, 3)) * 255).astype(np.uint8)
    tensor = tiles_to_cnn_input(tiles)
    assert tensor.shape == (2, 3, 48, 48)
    assert np.isclose(float(tensor[0, 0, 0, 0]) * 255, float(tiles[0, 0, 0, 0]))


def test_ordinal_targets_are_a_distribution_over_neighbours() -> None:
    for label in range(6):
        target = ordinal_targets(label).numpy()
        assert np.isclose(target.sum(), 1.0)
        assert target.argmax() == label
        nonzero = np.flatnonzero(target)
        assert nonzero.min() >= label - 1 and nonzero.max() <= label + 1


def test_grader_learns_a_separable_toy_task() -> None:
    """A head that cannot fit two obviously different bags cannot fit anything."""
    rng = np.random.default_rng(5)
    bags = []
    for label, offset in ((1, 0.0), (4, 3.0)):
        for _ in range(6):
            features = (rng.normal(offset, 0.15, size=(8, FIXED_BANK_DIM))).astype(np.float32)
            bags.append(Bag(f"toy-{label}", tiles=None, features=features, label=label))
    model = train_grader(bags, in_dim=FIXED_BANK_DIM, trainable_encoder=False, epochs=30, seed=0)
    assert isinstance(model, MILGrader)
    assert predict(model, bags[0])[0] == 1
    assert predict(model, bags[-1])[0] == 4
