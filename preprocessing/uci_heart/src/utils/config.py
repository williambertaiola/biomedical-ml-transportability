"""Preprocessing: config."""

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
from thesis_tools.paths import resolve_config_paths


import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    project_root: Path
    random_seed: int
    timezone: str
    min_valid_raw_fraction: float
    min_domain_samples_warning: int
    high_missingness_warning: float
    input_paths: list[Path]
    names_file_path: Path | None = None
    requested_sync_output_root: Path | None = None
    expected_rows_by_domain: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        payload = resolve_config_paths(json.loads(path.read_text(encoding="utf-8")))
        config_dir = path.resolve().parent

        def _resolve_maybe_relative(raw_path: str | None) -> Path | None:
            if raw_path is None:
                return None
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = (config_dir / candidate).resolve()
            return candidate.resolve()

        return cls(
            project_root=_resolve_maybe_relative(payload["project_root"]),
            random_seed=int(payload["random_seed"]),
            timezone=str(payload.get("timezone", "UTC")),
            min_valid_raw_fraction=float(payload.get("min_valid_raw_fraction", 0.95)),
            min_domain_samples_warning=int(
                payload.get("min_domain_samples_warning", 30)
            ),
            high_missingness_warning=float(
                payload.get("high_missingness_warning", 0.4)
            ),
            input_paths=[
                Path(item).expanduser().resolve() for item in payload["input_paths"]
            ],
            names_file_path=_resolve_maybe_relative(payload.get("names_file_path")),
            requested_sync_output_root=_resolve_maybe_relative(
                payload.get("requested_sync_output_root")
            ),
            expected_rows_by_domain={
                str(key): int(value)
                for key, value in payload.get("expected_rows_by_domain", {}).items()
            },
        )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.reports_dir / "logs"

    @property
    def config_dir(self) -> Path:
        return self.project_root / "configs"

    def resolved_paths(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "data_raw_dir": str(self.raw_dir),
            "data_interim_dir": str(self.interim_dir),
            "data_processed_dir": str(self.processed_dir),
            "data_splits_dir": str(self.splits_dir),
            "reports_dir": str(self.reports_dir),
            "logs_dir": str(self.logs_dir),
            "names_file_path": str(self.names_file_path)
            if self.names_file_path
            else "",
            "requested_sync_output_root": (
                str(self.requested_sync_output_root)
                if self.requested_sync_output_root
                else ""
            ),
        }

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "random_seed": self.random_seed,
            "timezone": self.timezone,
            "min_valid_raw_fraction": self.min_valid_raw_fraction,
            "min_domain_samples_warning": self.min_domain_samples_warning,
            "high_missingness_warning": self.high_missingness_warning,
            "input_paths": [str(path) for path in self.input_paths],
            "names_file_path": str(self.names_file_path)
            if self.names_file_path
            else None,
            "requested_sync_output_root": (
                str(self.requested_sync_output_root)
                if self.requested_sync_output_root
                else None
            ),
            "expected_rows_by_domain": self.expected_rows_by_domain,
            "resolved_paths": self.resolved_paths(),
        }
