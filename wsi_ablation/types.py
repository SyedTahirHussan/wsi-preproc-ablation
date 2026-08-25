"""Typed records that cross module boundaries.

Everything the pipeline hands from one stage to the next is one of these. They
are frozen so that a record cannot be mutated after the stage that produced it,
and JSON-serialisable so a run can be replayed from disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TissueArm = Literal["threshold", "unetpp"]
ColourArm = Literal["none", "macenko", "physical"]
EncoderArm = Literal["task-trained", "fixed-bank"]
Split = Literal["train", "test"]


@dataclass(frozen=True)
class ScannerProfile:
    """Per-scanner colour behaviour, plus the drift it accumulates over time.

    `gain` and `offset` act per RGB channel on the reference image before it is
    written to the slide file. `drift_per_year` is added to `gain` in proportion
    to the slide's acquisition age, which is what a calibration arm has to undo.
    """

    scanner_id: str
    site_id: str
    gain: tuple[float, float, float]
    offset: tuple[float, float, float]
    drift_per_year: tuple[float, float, float]

    def gain_at(self, age_years: float) -> tuple[float, float, float]:
        return (
            self.gain[0] + self.drift_per_year[0] * age_years,
            self.gain[1] + self.drift_per_year[1] * age_years,
            self.gain[2] + self.drift_per_year[2] * age_years,
        )


@dataclass(frozen=True)
class SlideSpec:
    """Ground truth for one synthetic slide, written alongside the pyramid."""

    slide_id: str
    path: str
    site_id: str
    scanner_id: str
    acquisition_year: int
    age_years: float
    gleason_primary: int
    gleason_secondary: int
    isup: int
    stain_scale: float
    n_cores: int
    split: Split
    mpp: float
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TissueResult:
    """What a tissue detector concluded about one slide."""

    slide_id: str
    arm: TissueArm
    mask_level: int
    mask_downsample: float
    tissue_fraction: float
    detected: bool
    n_tiles: int


@dataclass(frozen=True)
class GradePrediction:
    slide_id: str
    split: Split
    true_isup: int
    pred_isup: int
    detected: bool
    attention_entropy: float


@dataclass(frozen=True)
class CellMetrics:
    """Scores for one cell of the ablation grid."""

    kappa_w: float
    kappa_w_lo: float
    kappa_w_hi: float
    accuracy: float
    slides_lost: int
    threshold_crossings: int
    n_evaluated: int
    mean_tiles_per_slide: float


@dataclass(frozen=True)
class AblationCell:
    tissue: TissueArm
    colour: ColourArm
    encoder: EncoderArm
    metrics: CellMetrics
    predictions: list[GradePrediction] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.tissue}/{self.colour}/{self.encoder}"

    def to_json(self, with_predictions: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tissue": self.tissue,
            "colour": self.colour,
            "encoder": self.encoder,
            "metrics": asdict(self.metrics),
        }
        if with_predictions:
            out["predictions"] = [asdict(p) for p in self.predictions]
        return out
