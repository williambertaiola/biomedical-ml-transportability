"""Transportability experiment: common pipeline."""

from __future__ import annotations

# Locate the repository without depending on the current working directory.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys

_REPOSITORY_ROOT = next(
    parent
    for parent in _RepositoryPath(__file__).resolve().parents
    if (parent / ".thesis-root").is_file()
)
_repository_sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
from thesis_tools.paths import configured_path as _configured_path


import inspect
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    Path(_configured_path("results_root")) / "legacy" / "dashi_improved"
)


def resolve_existing_path(candidates: Iterable[str | Path], *, label: str) -> Path:
    checked: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        checked.append(path)
        if path.exists():
            return path
    checked_repr = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        f"Nessun path valido trovato per {label}.\nCandidati controllati:\n{checked_repr}"
    )


def get_output_root() -> Path:
    env_value = os.environ.get("DASHI_OUTPUT_ROOT")
    root = Path(env_value).expanduser() if env_value else DEFAULT_OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_output_dir(dataset_name: str, analysis_name: str) -> Path:
    out_dir = get_output_root() / dataset_name / analysis_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def configure_local_dashi_submodule(namespace: str, local_path: Path) -> None:
    import dashi

    installed_path = Path(dashi.__file__).resolve().parent / namespace
    if not local_path.exists():
        raise FileNotFoundError(f"Cartella non trovata: {local_path}")

    full_namespace = f"dashi.{namespace}"
    for module_name in list(sys.modules):
        if module_name == full_namespace or module_name.startswith(
            f"{full_namespace}."
        ):
            del sys.modules[module_name]

    local_module = types.ModuleType(full_namespace)
    local_module.__path__ = [str(local_path), str(installed_path)]
    local_module.__package__ = full_namespace
    sys.modules[full_namespace] = local_module
    setattr(dashi, namespace, local_module)


def assert_local_module(fn, local_path: Path) -> None:
    module_path = Path(inspect.getfile(fn)).resolve()
    if not str(module_path).startswith(str(local_path.resolve())):
        raise RuntimeError(
            f"Le funzioni non sono state caricate dalla cartella locale richiesta: {module_path}"
        )


def save_plotly_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def save_plotly_figure_list(figures, prefix_path: Path) -> None:
    for index, figure in enumerate(figures, start=1):
        output_path = prefix_path.with_name(f"{prefix_path.name}_{index}.html")
        save_plotly_figure(figure, output_path)


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        frame.to_csv(output_path, sep="\t", index=False)


def write_text(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
