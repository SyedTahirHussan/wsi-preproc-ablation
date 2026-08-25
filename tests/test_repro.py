"""Determinism, checked at the level a reader would check it: the digest."""

from __future__ import annotations

import json
from pathlib import Path

from wsi_ablation.ablation import RunConfig, run_ablation
from wsi_ablation.cli import main
from wsi_ablation.provenance import sha256_dir, sha256_json
from wsi_ablation.report import summarise
from wsi_ablation.synth import CohortConfig, generate_cohort, load_cohort
from wsi_ablation.types import SlideSpec


def test_same_seed_writes_byte_identical_slides(tmp_path: Path) -> None:
    config_a = CohortConfig(n_slides=3, seed=11, out_dir=str(tmp_path / "a"))
    config_b = CohortConfig(n_slides=3, seed=11, out_dir=str(tmp_path / "b"))
    generate_cohort(config_a)
    generate_cohort(config_b)
    assert sha256_dir(tmp_path / "a") == sha256_dir(tmp_path / "b")


def test_different_seed_writes_different_slides(tmp_path: Path) -> None:
    generate_cohort(CohortConfig(n_slides=3, seed=11, out_dir=str(tmp_path / "a")))
    generate_cohort(CohortConfig(n_slides=3, seed=12, out_dir=str(tmp_path / "b")))
    assert sha256_dir(tmp_path / "a") != sha256_dir(tmp_path / "b")


def test_cohort_manifest_round_trips(tiny_cohort: tuple[list[SlideSpec], Path]) -> None:
    specs, root = tiny_cohort
    loaded, targets = load_cohort(str(root))
    assert [s.to_json() for s in loaded] == [s.to_json() for s in specs]
    assert set(targets) == {"SC-A", "SC-B", "SC-C"}


def test_ablation_is_reproducible(tiny_cohort: tuple[list[SlideSpec], Path]) -> None:
    specs, root = tiny_cohort
    _, targets = load_cohort(str(root))
    config = RunConfig(
        data_dir=str(root),
        epochs=2,
        segmenter_epochs=1,
        segmenter_train_slides=2,
        bootstrap=20,
        tissue_arms=("threshold",),
        colour_arms=("none",),
        encoder_arms=("fixed-bank",),
    )
    first = run_ablation(specs, targets, config, verbose=False)
    second = run_ablation(specs, targets, config, verbose=False)
    assert first.cohort_losses == second.cohort_losses
    assert sha256_json(summarise(first)) == sha256_json(summarise(second))


def test_cli_run_writes_every_artefact(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "cohort:\n"
        "  n_slides: 4\n"
        "  seed: 13\n"
        f"  out_dir: {tmp_path / 'data'}\n"
        "run:\n"
        f"  out_dir: {tmp_path / 'out'}\n"
        "  epochs: 1\n"
        "  segmenter_epochs: 1\n"
        "  segmenter_train_slides: 2\n"
        "  bootstrap: 10\n"
        '  tissue_arms: ["threshold"]\n'
        '  colour_arms: ["none"]\n'
        '  encoder_arms: ["fixed-bank"]\n'
    )
    assert main(["--config", str(config), "run"]) == 0
    out = tmp_path / "out"
    for name in ("report.html", "records.jsonl", "summary.json", "manifest.json"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["content_digest"]) == 16
    assert "{" not in (out / "report.html").read_text().split("<body>")[1][:200]
