"""Resolve research paths independently of machine and working directory."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def configured_path(key: str) -> str:
    """Return a configured path, with optional data and results root overrides.

    THESIS_PATHS_FILE selects a complete JSON registry. Relative registry values
    are repository-relative. THESIS_DATA_ROOT and THESIS_RESULTS_ROOT replace
    those two directory prefixes; the JSON registry never contains user paths.
    """
    if key == "python_executable":
        return sys.executable
    registry_file = Path(
        os.environ.get("THESIS_PATHS_FILE", REPOSITORY_ROOT / "configs" / "paths.json")
    ).expanduser()
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    value = registry[key]
    path = Path(value).expanduser()
    if not path.is_absolute():
        for prefix, variable in (
            ("data", "THESIS_DATA_ROOT"),
            ("results", "THESIS_RESULTS_ROOT"),
        ):
            if path.parts and path.parts[0] == prefix and variable in os.environ:
                path = Path(os.environ[variable]).expanduser() / Path(*path.parts[1:])
                break
        else:
            path = REPOSITORY_ROOT / path
    return str(path.resolve())


def resolve_config_paths(payload: dict) -> dict:
    """Resolve path-valued config fields while retaining scientific settings."""
    path_keys = {
        "project_root",
        "output_root",
        "desktop_output_root",
        "input_root",
        "phenotypic_file",
        "qap_dir",
        "master_table_path",
        "composite_phenotypic_path",
        "long_phenotypic_path",
        "input_samples_metadata",
        "input_expression_matrix",
        "expression_matrix",
        "clinical_dir",
        "names_file_path",
        "requested_sync_output_root",
    }
    resolved = payload.copy()
    for key, value in payload.items():
        if key in path_keys and value is not None:
            resolved[key] = resolve_path_value(value)
        elif key == "input_paths":
            resolved[key] = [resolve_path_value(item) for item in value]
    return resolved


def resolve_path_value(value: str) -> str:
    """Resolve an individual template path using the same root overrides."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        for prefix, variable in (
            ("data", "THESIS_DATA_ROOT"),
            ("results", "THESIS_RESULTS_ROOT"),
        ):
            if path.parts and path.parts[0] == prefix and variable in os.environ:
                path = Path(os.environ[variable]).expanduser() / Path(*path.parts[1:])
                break
        else:
            path = REPOSITORY_ROOT / path
    return str(path.resolve())
