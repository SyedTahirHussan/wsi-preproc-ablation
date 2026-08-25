# wsi-preproc-ablation

[![ci](https://github.com/SyedTahirHussan/wsi-preproc-ablation/actions/workflows/ci.yml/badge.svg)](https://github.com/SyedTahirHussan/wsi-preproc-ablation/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![report](https://img.shields.io/badge/live-report-orange)](https://syedtahirhussan.github.io/wsi-preproc-ablation/)

A whole-slide grading pipeline built so that **preprocessing is the independent variable**. Tissue detection and colour handling are swapped underneath two tile encoders, and every cell of the grid is scored three ways: quadratic weighted kappa on ISUP Grade Group, the number of slides that never reached the grader, and the number of grade errors that moved a case across the surveillance-to-treatment line.

Most published comparisons hold preprocessing fixed and unstated, then attribute the difference between models to the models. Two results from the Karolinska Institutet group make that hard to keep doing. Replacing a thresholding tissue detector with UNet++ cut fully undetected specimens from 116 (0.43%) to 22 (0.08%), and on slides both detectors found, Gleason performance did not move ([Boman et al., *Scientific Reports*, 2026](https://doi.org/10.1038/s41598-026-52148-9)). Separately, a deployed model's accuracy decays as its scanner drifts, and physical colour calibration recovers it ([Salmon et al., *Journal of Pathology Informatics*, 2026](https://doi.org/10.1016/j.jpi.2026.100593); [Ji et al., *Modern Pathology*, 2025](https://doi.org/10.1016/j.modpat.2025.100715)). Both effects live entirely outside the model weights.

**The position this repository takes, and can be wrong about:** a large part of the generalisation gap reported between frozen pathology foundation models and task-trained models is a preprocessing gap rather than a representation gap, and the grid below is the shape of experiment that would settle it.

> **Data honesty.** The cohort is synthetic. `wsi_ablation/synth.py` writes it, and no wet-lab or patient data is used or claimed anywhere in this repository. The generator deliberately contains the three effects under test — stain fading in archival material, scanner colour drift over an acquisition period, and Gleason pattern rendered as gland morphology rather than as colour — so a run demonstrates that the instrument measures those effects and surfaces the failures kappa hides. Whether the effects hold on real archival material is the question the instrument exists to ask. Point `cohort.out_dir` at a directory of SVS or MRXS files and nothing downstream of `synth.py` changes.

## The grid

| Axis | Arms |
|---|---|
| Tissue detection | Otsu-on-saturation with a noise floor · UNet++ trained on training-split overviews |
| Colour | untouched · Macenko stain normalisation · physical calibration fitted from a per-scanner colour target |
| Tile encoder | small CNN trained end to end with the attention head · frozen hand-specified feature bank |

Twelve cells. Each trains its own gated-attention MIL head ([Ilse et al., ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html)) on the training split and is scored on a held-out split containing an unseen scanner and a later acquisition period on the seen scanners. Reusing one head across cells would measure how well a fixed head tolerates a shifted input, which is an easier question than the one asked here.

The frozen arm is a stand-in for a public pathology foundation model used the way benchmarks use them: weights frozen, a light head trained on top. It is a stand-in and the code never calls it anything else. `FrozenEncoder` in `wsi_ablation/encoders.py` is the protocol a real encoder satisfies; [docs/foundation-models.md](docs/foundation-models.md) gives the twenty-line adapter for [UNI](https://doi.org/10.1038/s41591-024-02857-3) and [Virchow](https://doi.org/10.1038/s41591-024-03141-0). Swapping one in changes the numbers and changes no part of the experiment.

## Quickstart

```bash
pip install -e ".[dev]"      # openslide-bin ships the C library; no brew or apt needed
make check                   # ruff, mypy --strict, pytest
make smoke                   # whole pipeline on 24 slides, about 30 seconds
make run                     # 140 slides, twelve cells, about fifteen minutes on a laptop CPU
```

Outputs land in `runs/report/`: `report.html` for a reviewer, `records.jsonl` with a per-slide prediction and attention entropy for every cell, `summary.json`, `ablation_overview.png`, and `manifest.json` carrying the config hash, the data hash, the git commit and a content digest.

`make repro` runs the pipeline twice and fails if the digests differ. That gate catches the ordinary sources of irreproducibility at the commit that introduced them: an unseeded shuffle, a set iteration, a dict that inherited its order from a filesystem listing.

## Why the slide files are real slide files

The synthetic cohort is written as tiled multi-resolution TIFF with a recorded micron-per-pixel, and OpenSlide opens it through the generic-tiff driver at four levels. That costs a little in the generator and buys the two bugs most likely to survive review.

Pyramid levels go in as separate top-level IFDs flagged `REDUCEDIMAGE`. Written as SubIFDs instead, which is what `tifffile` does by default and what a reasonable person would try first, OpenSlide reports one level, and every downsample downstream silently becomes a resize of level 0. The image still looks correct.

`read_region` takes level-0 coordinates whatever level it reads. Passing level-local coordinates gives every tile a view of the top-left of the slide, in the right colours, at the right magnification. `tests/test_slide.py` reads the same physical area at two levels and compares.

Tiles are cut at a fixed micron-per-pixel rather than a fixed level index, because a fixed index compares 0.25 µm/px tiles from one scanner against 0.5 µm/px tiles from another and calls the difference a model effect.

## The two columns that are usually missing

Kappa cannot see a slide that was never detected, so dropping the hardest tenth of a cohort improves it. `slides_lost` is reported beside it and never folded into it.

Kappa is also indifferent to where an error lands. Grade Group 1 against Grade Group 2 is the difference between active surveillance and a treatment conversation; Grade Group 4 against Grade Group 5 usually is not. `threshold_crossings` counts the first kind. The grading table itself is the 2014 ISUP consensus, pinned in `tests/test_isup.py` rather than derived, because 3+4 and 4+3 are both Gleason 7 and different Grade Groups and that is the sort of thing a refactor quietly breaks.

## QuPath, in both directions

`wsi-ablation qupath-export WSI-0007` writes the tissue mask and the tile grid as QuPath GeoJSON in level-0 coordinates. `qupath/import_pipeline_annotations.groovy` loads them onto the open slide, so a pathologist can see exactly which tissue the detector kept and which tiles the grader read. Corrections come back through `qupath/export_reviewed_tissue.groovy` and rasterise into a mask that substitutes for either detector arm.

## Running it on a cluster

`hpc/slurm_ablation.sh` is a twelve-task array job, one task per cell, with the cohort written once before the array rather than by 140 tasks racing for the same paths. Thread counts are pinned, since torch will otherwise claim every core on the node and twelve tasks doing that run slower than one.

## Layout

```
wsi_ablation/
  synth.py        deterministic cohort, written as real OpenSlide pyramids
  slide.py        OpenSlide access, level selection by micron-per-pixel, tile grids
  tissue.py       the two detection arms, including a real nested-skip UNet++
  stain.py        Beer-Lambert optics, Ruifrok H&E basis, Macenko estimation
  colour.py       the three colour arms and the colour-target calibration fit
  encoders.py     task-trained CNN, frozen feature bank, the FrozenEncoder protocol
  mil.py          gated attention pooling and an ordinal-aware Grade Group head
  isup.py         the 2014 ISUP consensus table and the treatment threshold
  metrics.py      kappa with a bootstrap interval, losses, threshold crossings
  pipeline.py     tiling and bag construction, where the arms actually differ
  ablation.py     the twelve-cell experiment
  provenance.py   content-addressed manifests
  report.py       figure, JSONL records, self-contained HTML page
qupath/           Groovy scripts for the pathologist's side
hpc/              SLURM array job
configs/          default.yaml and the smaller smoke.yaml that CI runs
```

## What is not here

No real slides, no patient data, and no claim about either. No foundation-model weights: the adapter is written and tested against the protocol, and running it needs `timm` and a HuggingFace access agreement that a public repository should not carry. No GPU path, since every arm here fits on a laptop CPU and adding one would have made the results harder to reproduce rather than easier.

## Author

Syed Tahir Hussan — MS Software Engineering, Riphah International University, Islamabad. [Portfolio](https://syedtahirhussan.github.io/syedtahirhussan) · [LinkedIn](https://linkedin.com/in/syedtahirhussan) · tahirsherazi786@gmail.com

Apache-2.0. Citation metadata in `CITATION.cff`.
