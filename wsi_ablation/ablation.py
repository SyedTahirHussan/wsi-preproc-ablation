"""The experiment: preprocessing as the independent variable.

Twelve cells, from two tissue-detection arms, three colour arms, and two encoder
arms. Every cell trains its own attention head on the training split and is
scored on the held-out split, which contains an unseen scanner and a later
acquisition period on the seen scanners.

What is deliberately not done here: no cell is allowed to see the test split
during tissue-detector training, colour fitting, or head fitting. The colour
calibration is fitted from colour targets, which are instrument measurements
rather than patient data, and are available for a test scanner in exactly the
way a calibration slide is available at a new site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wsi_ablation.colour import Calibration, fit_calibration, identity_calibration
from wsi_ablation.encoders import FIXED_BANK_DIM, FixedFeatureBank
from wsi_ablation.metrics import (
    accuracy,
    bootstrap_kappa_ci,
    quadratic_weighted_kappa,
    threshold_crossings,
)
from wsi_ablation.mil import Bag, predict, train_grader
from wsi_ablation.pipeline import SlideTiles, extract_tiles
from wsi_ablation.slide import Slide
from wsi_ablation.tissue import TissueDetector, train_unetpp
from wsi_ablation.types import (
    AblationCell,
    AblationRun,
    CellMetrics,
    ColourArm,
    EncoderArm,
    GradePrediction,
    SlideSpec,
    TissueArm,
)

TISSUE_ARMS: tuple[TissueArm, ...] = ("threshold", "unetpp")
COLOUR_ARMS: tuple[ColourArm, ...] = ("none", "macenko", "physical")
ENCODER_ARMS: tuple[EncoderArm, ...] = ("task-trained", "fixed-bank")


@dataclass
class RunConfig:
    data_dir: str = "data/cohort"
    out_dir: str = "runs/report"
    seed: int = 20260825
    epochs: int = 40
    segmenter_epochs: int = 20
    segmenter_train_slides: int = 24
    bootstrap: int = 500
    tissue_arms: tuple[TissueArm, ...] = TISSUE_ARMS
    colour_arms: tuple[ColourArm, ...] = COLOUR_ARMS
    encoder_arms: tuple[EncoderArm, ...] = ENCODER_ARMS

    def to_json(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "segmenter_epochs": self.segmenter_epochs,
            "segmenter_train_slides": self.segmenter_train_slides,
            "bootstrap": self.bootstrap,
            "tissue_arms": list(self.tissue_arms),
            "colour_arms": list(self.colour_arms),
            "encoder_arms": list(self.encoder_arms),
        }


def fit_calibrations(colour_targets: dict[str, list[list[float]]]) -> dict[str, Calibration]:
    return {
        scanner: fit_calibration(scanner, np.asarray(target, dtype=np.float64))
        for scanner, target in colour_targets.items()
    }


def train_segmenter(
    specs: list[SlideSpec], root: Path, epochs: int, seed: int, max_slides: int = 24
) -> object:
    """Fit the UNet++ arm on training-split overviews and their reference masks.

    A subsample of the training split is enough: tissue against glass is a low
    variance target, and spending the compute budget on the segmenter rather
    than on the graders would trade an easy gain for the measurement itself.
    """
    overviews = []
    masks = []
    chosen = [spec for spec in specs if spec.split == "train"][:max_slides]
    for spec in chosen:
        with Slide(root / spec.path) as slide:
            level = min(3, slide.level_count - 1)
            overview = slide.thumbnail(level)
            downsample = round(slide.level_downsample(level))
        truth = np.load(root / f"{spec.slide_id}.mask.npy")
        reduced = truth[::downsample, ::downsample][: overview.shape[0], : overview.shape[1]]
        overviews.append(overview)
        masks.append(np.asarray(reduced, dtype=bool))
    return train_unetpp(overviews, masks, epochs=epochs, seed=seed)


def _score_cell(
    tissue: TissueArm,
    colour: ColourArm,
    encoder: EncoderArm,
    tiles_by_slide: dict[str, SlideTiles],
    specs: list[SlideSpec],
    calibrations: dict[str, Calibration],
    config: RunConfig,
) -> AblationCell:
    from wsi_ablation.pipeline import build_bag

    bank = FixedFeatureBank()
    train_bags: list[Bag] = []
    test_entries: list[tuple[SlideSpec, Bag | None]] = []

    for spec in specs:
        slide_tiles = tiles_by_slide[spec.slide_id]
        calibration = calibrations.get(spec.scanner_id, identity_calibration())
        bag = build_bag(slide_tiles, colour, encoder, calibration, bank)
        if spec.split == "train":
            if bag is not None:
                train_bags.append(bag)
        else:
            test_entries.append((spec, bag))

    model = train_grader(
        train_bags,
        in_dim=FIXED_BANK_DIM,
        trainable_encoder=encoder == "task-trained",
        epochs=config.epochs,
        seed=config.seed,
    )

    predictions: list[GradePrediction] = []
    for spec, bag in test_entries:
        if bag is None:
            predictions.append(
                GradePrediction(spec.slide_id, "test", spec.isup, -1, False, float("nan"))
            )
            continue
        grade, entropy = predict(model, bag)
        predictions.append(GradePrediction(spec.slide_id, "test", spec.isup, grade, True, entropy))

    graded = [p for p in predictions if p.detected]
    truth = np.array([p.true_isup for p in graded], dtype=np.int_)
    predicted = np.array([p.pred_isup for p in graded], dtype=np.int_)
    lo, hi = bootstrap_kappa_ci(truth, predicted, n_boot=config.bootstrap, seed=config.seed)

    tiles_seen = [tiles_by_slide[p.slide_id].result.n_tiles for p in graded]
    metrics = CellMetrics(
        kappa_w=quadratic_weighted_kappa(truth, predicted),
        kappa_w_lo=lo,
        kappa_w_hi=hi,
        accuracy=accuracy(truth, predicted),
        slides_lost=len(predictions) - len(graded),
        threshold_crossings=threshold_crossings(truth, predicted),
        n_evaluated=len(graded),
        mean_tiles_per_slide=float(np.mean(tiles_seen)) if tiles_seen else 0.0,
    )
    return AblationCell(tissue, colour, encoder, metrics, predictions)


def run_ablation(
    specs: list[SlideSpec],
    colour_targets: dict[str, list[list[float]]],
    config: RunConfig,
    verbose: bool = True,
) -> AblationRun:
    root = Path(config.data_dir)
    calibrations = fit_calibrations(colour_targets)
    segmenter = None
    cells: list[AblationCell] = []
    cohort_losses: dict[str, int] = {}

    for tissue in config.tissue_arms:
        if tissue == "unetpp" and segmenter is None:
            started = time.time()
            segmenter = train_segmenter(
                specs, root, config.segmenter_epochs, config.seed, config.segmenter_train_slides
            )
            if verbose:
                print(f"  trained UNet++ tissue detector in {time.time() - started:.1f}s")

        detector = TissueDetector(tissue, model=segmenter if tissue == "unetpp" else None)  # type: ignore[arg-type]
        tiles_by_slide = {
            spec.slide_id: extract_tiles(spec, root, detector) for spec in specs
        }
        lost = sum(1 for t in tiles_by_slide.values() if not t.result.detected)
        cohort_losses[tissue] = lost
        if verbose:
            print(f"  {tissue}: {lost} of {len(specs)} slides produced too little tissue to grade")

        for colour in config.colour_arms:
            for encoder in config.encoder_arms:
                started = time.time()
                cell = _score_cell(
                    tissue, colour, encoder, tiles_by_slide, specs, calibrations, config
                )
                cells.append(cell)
                if verbose:
                    m = cell.metrics
                    print(
                        f"  {cell.name:38s} kappa_w={m.kappa_w:.3f} "
                        f"lost={m.slides_lost} crossings={m.threshold_crossings} "
                        f"({time.time() - started:.1f}s)"
                    )
        del tiles_by_slide
    return AblationRun(cells=cells, cohort_losses=cohort_losses)
