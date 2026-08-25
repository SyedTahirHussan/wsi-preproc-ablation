# Dropping a real foundation model into the frozen arm

The frozen encoder shipped in this repository is a hand-specified feature bank. It stands in for a public pathology foundation model and the code never calls it anything else. This page is the adapter for the real ones.

Weights are not vendored here. UNI and Virchow are both gated on HuggingFace behind an access agreement, and a public repository that bundled them would be redistributing something it has no right to redistribute.

## The protocol

`wsi_ablation/encoders.py` defines what the ablation needs from any tile encoder:

```python
class FrozenEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, tiles: ByteArr) -> FloatArr:
        """Map (n, h, w, 3) uint8 tiles to (n, dim) float32 embeddings."""
```

`FixedFeatureBank` satisfies it. So does this.

## UNI

[Chen et al., *Nature Medicine* 30, 850–862 (2024)](https://doi.org/10.1038/s41591-024-02857-3). ViT-L/16 trained with DINOv2 on about 100 million tiles from 100,426 slides. Released as `MahmoodLab/UNI` on HuggingFace.

```python
import numpy as np
import timm
import torch
from PIL import Image


class UNIEncoder:
    """UNI as a frozen tile encoder. Requires `timm` and HuggingFace access."""

    def __init__(self, device: str = "cpu") -> None:
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        ).eval().to(device)
        config = timm.data.resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = timm.data.create_transform(**config, is_training=False)
        self.device = device

    @property
    def dim(self) -> int:
        return 1024

    @torch.inference_mode()
    def encode(self, tiles: np.ndarray) -> np.ndarray:
        batch = torch.stack([self.transform(Image.fromarray(t)) for t in tiles]).to(self.device)
        return self.model(batch).float().cpu().numpy().astype(np.float32)
```

## Virchow

[Vorontsov et al., *Nature Medicine* 30, 2924–2935 (2024)](https://doi.org/10.1038/s41591-024-03141-0). ViT-H/14 trained on 1.5 million slides. Released as `paige-ai/Virchow`. Virchow's published recipe concatenates the class token with the mean of the patch tokens, which is why the embedding is 2560-dimensional rather than 1280.

```python
class VirchowEncoder(UNIEncoder):
    def __init__(self, device: str = "cpu") -> None:
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval().to(device)
        config = timm.data.resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = timm.data.create_transform(**config, is_training=False)
        self.device = device

    @property
    def dim(self) -> int:
        return 2560

    @torch.inference_mode()
    def encode(self, tiles: np.ndarray) -> np.ndarray:
        batch = torch.stack([self.transform(Image.fromarray(t)) for t in tiles]).to(self.device)
        output = self.model(batch)
        pooled = torch.cat([output[:, 0], output[:, 1:].mean(dim=1)], dim=-1)
        return pooled.float().cpu().numpy().astype(np.float32)
```

## Wiring it in

`_score_cell` in `wsi_ablation/ablation.py` builds a `FixedFeatureBank()` and passes it to `build_bag`. Pass one of these instead, and set `in_dim` on `train_grader` to the encoder's `dim`.

Three things to check before believing the numbers that come out.

**Tile size and magnification.** Both encoders were trained on 224-pixel tiles at 20x, roughly 0.5 µm/px. This pipeline cuts 128-pixel tiles at 1.0 µm/px, which is a different field of view at a different scale. Change `TILE_MPP` and `TILE_PX` in `wsi_ablation/pipeline.py` to match the encoder, or the comparison against the task-trained arm is unfair to the encoder rather than informative about it.

**Normalisation order.** `timm`'s transform applies its own resize and ImageNet-style normalisation. The colour arm has already run by that point, which is the intended order: the colour arm is the independent variable and the encoder's own preprocessing is part of the encoder.

**Runtime.** ViT-H/14 on CPU is roughly two orders of magnitude slower per tile than the feature bank. `hpc/slurm_ablation.sh` is the shape that survives it; add `--gres=gpu:1` and set `device="cuda"`.

## What the comparison would then be worth

The falsifiable claim in the top-level README is that a large part of the reported generalisation gap between frozen foundation encoders and task-trained models is a preprocessing gap. With a real encoder in this slot and a real cohort under it, the physical-calibration row against the untouched row is a direct measurement of how much of that gap the instrument accounts for. The synthetic cohort cannot answer it, because the synthetic cohort is where the drift was put in by hand.
