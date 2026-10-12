"""Transportability experiment: run dashi uci."""

#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_matplotlib_cache(project_root: Path) -> None:
    cache_root = project_root / ".cache"
    mplconfig = project_root / ".mplconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig.resolve()))


def main() -> int:
    project_root = Path(__file__).resolve().parent
    _configure_matplotlib_cache(project_root)

    from src.dashi_uci_pipeline import main as pipeline_main

    return pipeline_main(["--project-root", str(project_root), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
