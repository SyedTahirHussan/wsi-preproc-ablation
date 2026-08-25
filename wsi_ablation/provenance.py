"""Content-addressed run records.

A preprocessing ablation is only worth reading if a reader can tell that two
numbers came from two different pipelines rather than from two different
afternoons. Every run writes a manifest with the config hash, the data hash, the
code version, the git commit where one is available, and the interpreter
version; `wsi-ablation repro` re-runs and compares content digests.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_dir(path: Path, patterns: tuple[str, ...] = ("*.tif", "cohort.json")) -> str:
    """Hash of a data directory, over file names and contents in sorted order."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    for file in sorted(set(files)):
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def git_sha(short: bool = True) -> str:
    """The commit this ran at, or `unversioned` outside a working tree."""
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unversioned"
    return completed.stdout.strip() or "unversioned"


@dataclass(frozen=True)
class Manifest:
    code_version: str
    git_sha: str
    python: str
    config_hash: str
    data_hash: str
    seed: int
    content_digest: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(
    code_version: str,
    config: dict[str, Any],
    data_dir: Path,
    seed: int,
    results: Any,
) -> Manifest:
    return Manifest(
        code_version=code_version,
        git_sha=git_sha(),
        python=platform.python_version(),
        config_hash=sha256_json(config),
        data_hash=sha256_dir(data_dir),
        seed=seed,
        content_digest=sha256_json(results)[:16],
    )
