from __future__ import annotations

import numpy as np

from wsi_ablation.metrics import (
    accuracy,
    bootstrap_kappa_ci,
    quadratic_weighted_kappa,
    threshold_crossings,
)


def test_perfect_agreement_is_one() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5])
    assert quadratic_weighted_kappa(labels, labels) == 1.0


def test_adjacent_errors_cost_less_than_distant_ones() -> None:
    truth = np.array([0, 1, 2, 3, 4, 5])
    adjacent = np.array([1, 2, 3, 4, 5, 4])
    distant = np.array([5, 4, 3, 2, 1, 0])
    assert quadratic_weighted_kappa(truth, adjacent) > quadratic_weighted_kappa(truth, distant)


def test_single_class_returns_zero_rather_than_one() -> None:
    """A cohort that collapsed to one grade has no agreement to measure."""
    constant = np.array([2, 2, 2, 2])
    assert quadratic_weighted_kappa(constant, constant) == 0.0


def test_kappa_matches_a_hand_computed_case() -> None:
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 1])
    assert quadratic_weighted_kappa(truth, predicted, n_classes=2) == 0.0


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 6, size=80)
    predicted = np.clip(truth + rng.integers(-1, 2, size=80), 0, 5)
    point = quadratic_weighted_kappa(truth, predicted)
    lo, hi = bootstrap_kappa_ci(truth, predicted, n_boot=200)
    assert lo <= point <= hi


def test_crossings_are_counted_not_averaged() -> None:
    truth = np.array([1, 1, 3])
    predicted = np.array([2, 1, 4])
    assert threshold_crossings(truth, predicted) == 1
    assert accuracy(truth, predicted) == 1 / 3
