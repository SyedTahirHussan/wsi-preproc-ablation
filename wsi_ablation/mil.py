"""Attention-based multiple-instance learning over tiles, to a Grade Group.

The bag is a slide, the instances are its tiles, and the label is the slide's
ISUP Grade Group. Gated attention pooling follows Ilse, Tomczak and Welling
(ICML 2018), which is the pooling the group's own end-to-end prostate model
uses; the point of reusing it here is that the head is not the variable under
test and should therefore not be novel.

Two details that matter for a preprocessing ablation:

The head is trained per cell of the grid. Training one head on the best
preprocessing and reusing it everywhere would measure how well a fixed head
tolerates a shifted input, which is a different and easier question.

The loss is ordinal-aware. Grade Groups are ordered, and a cross-entropy that
treats Grade Group 1 and Grade Group 5 as equally wrong optimises for a metric
nobody reports. Soft targets spread a little mass onto neighbouring grades,
which is both closer to how pathologists disagree and closer to what quadratic
weighted kappa rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from wsi_ablation.encoders import EMBED_DIM, TileCNN, tiles_to_cnn_input
from wsi_ablation.isup import MAX_GRADE_GROUP

ByteArr = NDArray[np.uint8]
FloatArr = NDArray[np.float32]

N_CLASSES = MAX_GRADE_GROUP + 1
LABEL_SMOOTHING_NEIGHBOUR = 0.12


class GatedAttentionPool(torch.nn.Module):
    """Gated attention pooling; returns the bag embedding and the tile weights."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.value = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.Tanh())
        self.gate = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.Sigmoid())
        self.score = torch.nn.Linear(hidden, 1)

    def forward(self, instances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(self.value(instances) * self.gate(instances)), dim=0)
        return (weights * instances).sum(dim=0), weights.squeeze(-1)


class MILGrader(torch.nn.Module):
    """Tile encoder (optional) plus gated attention plus a Grade Group head."""

    def __init__(self, in_dim: int, trainable_encoder: bool) -> None:
        super().__init__()
        self.encoder = TileCNN(EMBED_DIM) if trainable_encoder else None
        bag_dim = EMBED_DIM if trainable_encoder else in_dim
        self.norm = torch.nn.LayerNorm(bag_dim)
        self.pool = GatedAttentionPool(bag_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(bag_dim, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, N_CLASSES),
        )

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        instances = self.encoder(bag) if self.encoder is not None else bag
        pooled, weights = self.pool(self.norm(instances))
        return torch.as_tensor(self.head(pooled)), weights


def ordinal_targets(label: int, smoothing: float = LABEL_SMOOTHING_NEIGHBOUR) -> torch.Tensor:
    """A soft target that leaks mass onto the adjacent Grade Groups only."""
    target = torch.zeros(N_CLASSES)
    target[label] = 1.0 - smoothing
    neighbours = [n for n in (label - 1, label + 1) if 0 <= n < N_CLASSES]
    for neighbour in neighbours:
        target[neighbour] = smoothing / len(neighbours)
    return target


@dataclass
class Bag:
    """One slide's tiles, already through the tissue and colour arms."""

    slide_id: str
    tiles: ByteArr | None
    features: FloatArr | None
    label: int

    def to_tensor(self) -> torch.Tensor:
        if self.tiles is not None:
            return tiles_to_cnn_input(self.tiles)
        assert self.features is not None
        return torch.from_numpy(np.ascontiguousarray(self.features, dtype=np.float32))


def train_grader(
    bags: list[Bag],
    in_dim: int,
    trainable_encoder: bool,
    epochs: int = 24,
    lr: float = 1.5e-3,
    seed: int = 20260825,
) -> MILGrader:
    torch.manual_seed(seed)
    model = MILGrader(in_dim=in_dim, trainable_encoder=trainable_encoder)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    tensors = [(bag.to_tensor(), ordinal_targets(bag.label)) for bag in bags]

    model.train()
    for _ in range(epochs):
        for index in rng.permutation(len(tensors)):
            instances, target = tensors[index]
            logits, _ = model(instances)
            loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: MILGrader, bag: Bag) -> tuple[int, float]:
    """Predicted Grade Group and the entropy of the attention distribution.

    Attention entropy is reported because a bag that spreads attention evenly
    over every tile has not localised anything, and a grade produced that way
    deserves less trust than the same grade from a peaked distribution. It is a
    diagnostic, not a score: nothing in the ablation selects on it.
    """
    logits, weights = model(bag.to_tensor())
    probabilities = torch.softmax(logits, dim=-1)
    entropy = float(-(weights * torch.log(weights.clamp_min(1e-9))).sum())
    return int(torch.argmax(probabilities).item()), entropy
