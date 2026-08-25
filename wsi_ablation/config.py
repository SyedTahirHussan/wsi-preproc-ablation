"""YAML configuration, loaded into the two dataclasses the pipeline uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wsi_ablation.ablation import RunConfig
from wsi_ablation.synth import CohortConfig


def load_config(path: str | Path) -> tuple[CohortConfig, RunConfig]:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    cohort_raw = dict(raw.get("cohort", {}))
    run_raw = dict(raw.get("run", {}))

    for key in ("tissue_arms", "colour_arms", "encoder_arms"):
        if key in run_raw:
            run_raw[key] = tuple(run_raw[key])

    cohort = CohortConfig(**cohort_raw)
    run = RunConfig(data_dir=cohort.out_dir, **run_raw)
    return cohort, run
