"""Agreement, and the failures agreement hides.

Quadratic weighted kappa is the number this field reports, so it is reported
here. On its own it is not enough for a preprocessing comparison, for two
reasons this module makes explicit.

A slide that was never detected contributes nothing to kappa. Drop the hardest
tenth of a cohort and kappa improves. `slides_lost` is therefore reported beside
it, always, and never folded into it.

Kappa is also indifferent to where an error lands. Confusing Grade Group 4 with
Grade Group 5 and confusing Grade Group 1 with Grade Group 2 cost the same
quadratic penalty at different points on the scale, but only the second changes
what a patient is offered. `threshold_crossings` counts the second kind.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wsi_ablation.isup import crosses_treatment_threshold, label_grid

IntArr = NDArray[np.int_]


def confusion(truth: IntArr, predicted: IntArr, n_classes: int | None = None) -> NDArray[np.int_]:
    k = n_classes if n_classes is not None else len(label_grid())
    matrix = np.zeros((k, k), dtype=np.int_)
    for t, p in zip(truth, predicted, strict=True):
        matrix[int(t), int(p)] += 1
    return matrix


def quadratic_weighted_kappa(truth: IntArr, predicted: IntArr, n_classes: int | None = None) -> float:
    """Cohen's kappa with quadratic disagreement weights.

    Returns 0.0 for a degenerate case — one class present in both vectors —
    rather than a division by zero or a misleading 1.0. A cohort that collapsed
    to a single grade has no agreement to measure.
    """
    k = n_classes if n_classes is not None else len(label_grid())
    observed = confusion(truth, predicted, k).astype(np.float64)
    total = observed.sum()
    if total == 0:
        return 0.0

    indices = np.arange(k, dtype=np.float64)
    weights = (indices[:, None] - indices[None, :]) ** 2 / max((k - 1) ** 2, 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total

    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 0.0
    return float(1.0 - (weights * observed).sum() / denominator)


def bootstrap_kappa_ci(
    truth: IntArr,
    predicted: IntArr,
    n_boot: int = 1000,
    seed: int = 20260825,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap over slides, which is the resampling unit here."""
    n = len(truth)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        index = rng.integers(0, n, size=n)
        draws[i] = quadratic_weighted_kappa(truth[index], predicted[index])
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def threshold_crossings(truth: IntArr, predicted: IntArr) -> int:
    return int(sum(crosses_treatment_threshold(int(t), int(p)) for t, p in zip(truth, predicted, strict=True)))


def accuracy(truth: IntArr, predicted: IntArr) -> float:
    return float(np.mean(truth == predicted)) if len(truth) else 0.0
