"""Preprocessing: filesystem."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def ensure_project_dirs(project_root: Path) -> None:
    for rel_path in [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/splits",
        "reports/logs",
        "configs",
        "src/ingestion",
        "src/preprocessing",
        "src/utils",
    ]:
        (project_root / rel_path).mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
