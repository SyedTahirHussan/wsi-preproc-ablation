"""The two tissue-detection arms.

`threshold` is the classical arm: Otsu on the saturation channel of a low
resolution overview, with morphological cleanup. It is what most published
pipelines use and what most method sections do not mention.

`unetpp` is a small UNet++ (Zhou et al., DLMIA 2018) with the nested dense skip
pathway that distinguishes UNet++ from UNet, trained on the overview images of
the training split only. It never sees a test-split slide before inference.

A detector reports two things. The mask, and whether it found any tissue at all.
The second number is the one this repository exists to keep: a slide with no
detected tissue does not produce a wrong grade, it produces no grade, and it
leaves the agreement statistic untouched on its way out of the cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import threshold_otsu

from wsi_ablation.slide import Slide
from wsi_ablation.types import TissueArm

ByteArr = NDArray[np.uint8]
BoolArr = NDArray[np.bool_]
FloatArr = NDArray[np.float32]

OVERVIEW_LEVEL = 3
MIN_TISSUE_FRACTION = 0.004
MIN_OBJECT_PX = 64

# Absolute saturation below which a mask is a picture of sensor noise. Otsu will
# happily return a cutoff under this on a slide with no strong tissue mode, and
# the mask that comes back looks like a plausible speckled foreground. The floor
# is roughly five times the background variation these files carry.
SATURATION_FLOOR = 0.14


def _saturation(rgb: ByteArr) -> FloatArr:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    largest = arr.max(axis=-1)
    smallest = arr.min(axis=-1)
    return np.asarray((largest - smallest) / np.maximum(largest, 1e-6), dtype=np.float32)


def _cleanup(mask: BoolArr) -> BoolArr:
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((5, 5))))
    labels, count = ndimage.label(filled)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum_labels(filled, labels, index=np.arange(1, count + 1))
    keep = {i + 1 for i, size in enumerate(sizes) if size >= MIN_OBJECT_PX}
    return np.isin(labels, list(keep)) if keep else np.zeros_like(mask, dtype=bool)


def threshold_mask(overview: ByteArr) -> BoolArr:
    """Otsu on saturation, floored at the noise level.

    Otsu assumes two modes. A faintly stained archival slide is close to one
    mode, so the split lands inside the background distribution and the mask
    that comes back is speckle. Flooring the cutoff stops that, at the cost of
    dropping slides whose tissue never reaches the floor. Which cost a pipeline
    pays is a choice, and it is made here rather than by accident: this arm
    would rather lose a slide than grade noise.
    """
    saturation = _saturation(overview)
    cutoff = max(float(threshold_otsu(saturation)), SATURATION_FLOOR)
    return _cleanup(saturation > cutoff)


class _ConvBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.body(x))


class UNetPP(torch.nn.Module):
    """UNet++ with two down-samplings and the full nested skip grid.

    Kept small on purpose: tissue against background is an easy target, and the
    point of the arm is that a learned detector holds on faint stain, not that
    a large network was used.
    """

    def __init__(self, channels: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        c0, c1, c2 = channels
        self.pool = torch.nn.MaxPool2d(2)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.x00 = _ConvBlock(3, c0)
        self.x10 = _ConvBlock(c0, c1)
        self.x20 = _ConvBlock(c1, c2)
        self.x01 = _ConvBlock(c0 + c1, c0)
        self.x11 = _ConvBlock(c1 + c2, c1)
        self.x02 = _ConvBlock(c0 * 2 + c1, c0)
        self.head = torch.nn.Conv2d(c0, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x01 = self.x01(torch.cat([x00, self.up(x10)], dim=1))
        x20 = self.x20(self.pool(x10))
        x11 = self.x11(torch.cat([x10, self.up(x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], dim=1))
        return torch.as_tensor(self.head(x02))


def _pad_to_multiple(arr: FloatArr, multiple: int = 4) -> tuple[FloatArr, tuple[int, int]]:
    h, w = arr.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)], mode="edge")
    return arr, (h, w)


def train_unetpp(
    overviews: list[ByteArr],
    masks: list[BoolArr],
    epochs: int = 40,
    seed: int = 20260825,
    lr: float = 3e-3,
) -> UNetPP:
    """Train the detector on training-split overviews and their reference masks.

    Slides are fed one at a time because they differ in size after padding, and
    batching them would need either resizing (which changes the stain statistics
    the detector has to survive) or cropping (which removes the faint edges that
    are the hard case).
    """
    torch.manual_seed(seed)
    model = UNetPP()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    order = np.random.default_rng(seed).permutation(len(overviews))

    model.train()
    for _ in range(epochs):
        for index in order:
            scaled = (np.asarray(overviews[index], dtype=np.float32).transpose(2, 0, 1) / 255.0)
            image, _ = _pad_to_multiple(np.asarray(scaled, dtype=np.float32))
            target, _ = _pad_to_multiple(np.asarray(masks[index], dtype=np.float32)[None])
            logits = model(torch.from_numpy(image)[None])
            loss = loss_fn(logits, torch.from_numpy(target)[None])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model


@torch.no_grad()
def unetpp_mask(model: UNetPP, overview: ByteArr, cutoff: float = 0.5) -> BoolArr:
    image = np.asarray(overview, dtype=np.float32).transpose(2, 0, 1).astype(np.float32) / 255.0
    padded, (h, w) = _pad_to_multiple(np.asarray(image, dtype=np.float32))
    probability = torch.sigmoid(model(torch.from_numpy(padded)[None]))[0, 0].numpy()
    return _cleanup(np.asarray(probability[:h, :w] > cutoff))


@dataclass
class TissueDetector:
    """Dispatch for the two arms, so callers do not branch on strings."""

    arm: TissueArm
    model: UNetPP | None = None

    def mask_for(self, slide: Slide) -> tuple[BoolArr, int, float]:
        level = min(OVERVIEW_LEVEL, slide.level_count - 1)
        overview = slide.thumbnail(level)
        if self.arm == "threshold":
            mask = threshold_mask(overview)
        elif self.arm == "unetpp":
            if self.model is None:
                raise ValueError("the unetpp arm needs a trained model")
            mask = unetpp_mask(self.model, overview)
        else:
            raise ValueError(f"unknown tissue arm: {self.arm}")
        return mask, level, slide.level_downsample(level)


def tissue_fraction(mask: BoolArr) -> float:
    return float(mask.mean()) if mask.size else 0.0


def is_detected(mask: BoolArr) -> bool:
    """A slide counts as detected when enough of it survived to be tiled."""
    return tissue_fraction(mask) >= MIN_TISSUE_FRACTION
