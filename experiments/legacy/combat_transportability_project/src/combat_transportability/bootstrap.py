"""Transportability experiment: bootstrap."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_environment() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    local_pkgs = project_root / "local_py_pkgs"
    if local_pkgs.exists():
        sys.path.insert(0, str(local_pkgs))

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import numpy as np

    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    return project_root
