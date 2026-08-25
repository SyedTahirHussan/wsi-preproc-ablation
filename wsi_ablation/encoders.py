"""The two encoder arms: one that adapts to the pipeline, one that does not.

`task-trained` is a small convolutional encoder optimised end to end with the
attention head, on whatever tiles the tissue and colour arms hand it. If the
colour arm shifts the input distribution, training absorbs part of the shift.

`fixed-bank` is a frozen, hand-specified feature bank: colour moments, stain
concentrations after unmixing, gradient structure, and a lumen proxy. Nothing
about it is fitted, so a colour shift at test time passes straight through into
the features, and only the linear attention head on top can compensate.

The frozen arm stands in for a public pathology foundation model used the way
those models are usually used in benchmarks: weights frozen, a light head
trained on top. It is a stand-in and this repository never calls it anything
else. `FrozenEncoder` is the protocol a real one satisfies, and
`docs/foundation-models.md` gives the adapter for UNI (Chen et al., Nature
Medicine 2024) and Virchow (Vorontsov et al., Nature Medicine 2024). Swapping
one in changes the numbers and does not change a line of the experiment.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from wsi_ablation.stain import rgb_to_concentrations

ByteArr = NDArray[np.uint8]
FloatArr = NDArray[np.float32]

CNN_INPUT_PX = 64
FIXED_BANK_DIM = 22
EMBED_DIM = 32


class FrozenEncoder(Protocol):
    """What the ablation needs from any frozen tile encoder, learned or not."""

    @property
    def dim(self) -> int: ...

    def encode(self, tiles: ByteArr) -> FloatArr:
        """Map (n, h, w, 3) uint8 tiles to (n, dim) float32 embeddings."""


def _gradient_histogram(grey: FloatArr, bins: int = 8) -> FloatArr:
    gy, gx = np.gradient(grey, axis=(0, 1))
    magnitude = np.sqrt(gy * gy + gx * gx).ravel()
    edges = np.linspace(0.0, 0.25, bins + 1)
    counts, _ = np.histogram(magnitude, bins=edges)
    total = counts.sum()
    return np.asarray(counts / total if total else counts, dtype=np.float32)


class FixedFeatureBank:
    """Frozen descriptor bank, deliberately sensitive to what the scanner did.

    The lumen proxy is the one feature that carries grade rather than colour:
    the fraction of a tile that is bright and unstructured is high in a pattern-3
    gland with an open lumen and near zero in a pattern-5 sheet.
    """

    @property
    def dim(self) -> int:
        return FIXED_BANK_DIM

    def encode(self, tiles: ByteArr) -> FloatArr:
        out = np.zeros((tiles.shape[0], FIXED_BANK_DIM), dtype=np.float32)
        for index, tile in enumerate(tiles):
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            grey = np.asarray(arr.mean(axis=-1), dtype=np.float32)
            concentrations = rgb_to_concentrations(tile)[..., :2]
            lumen = float((grey > 0.93).mean())
            dark = float((grey < 0.55).mean())
            features = np.concatenate(
                [
                    arr.reshape(-1, 3).mean(axis=0),
                    arr.reshape(-1, 3).std(axis=0),
                    concentrations.reshape(-1, 2).mean(axis=0),
                    concentrations.reshape(-1, 2).std(axis=0),
                    _gradient_histogram(grey),
                    np.array([lumen, dark, float(grey.mean()), float(grey.std())]),
                ]
            )
            out[index] = features.astype(np.float32)
        return out


class TileCNN(torch.nn.Module):
    """Small convolutional tile encoder, trained with the attention head.

    No batch normalisation, deliberately. A bag is one slide and a batch is that
    slide's tiles, so batch statistics are slide statistics: normalising by them
    subtracts exactly the quantity the grade lives in. Mean brightness and the
    fraction of a tile that is open lumen separate a pattern-3 gland from a
    pattern-5 sheet, and a BatchNorm in front of the first convolution removes
    both before the network sees them. That version of this encoder scored a
    kappa indistinguishable from zero on every cell of the grid.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 48, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(48, embed_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.body(tiles))


def tiles_to_cnn_input(tiles: ByteArr) -> torch.Tensor:
    """Resize by strided decimation to the CNN input size, then scale to [0, 1].

    Decimation rather than interpolation, so that the operation introduces no
    colour of its own into an experiment whose independent variable is colour.
    """
    _, h, w, _ = tiles.shape
    step_y = max(1, h // CNN_INPUT_PX)
    step_x = max(1, w // CNN_INPUT_PX)
    small = tiles[:, ::step_y, ::step_x][:, :CNN_INPUT_PX, :CNN_INPUT_PX]
    arr = np.ascontiguousarray(small.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    assert arr.shape[1:] == (3, CNN_INPUT_PX, CNN_INPUT_PX), arr.shape
    return torch.from_numpy(arr)
