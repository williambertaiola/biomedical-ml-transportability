"""Transportability experiment: config."""

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


from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_INPUT_DIR = Path(_configured_path("legacy_preprocessed_tcga_tcga"))
DEFAULT_OUTPUT_DIR = Path(_configured_path("legacy_cartella_senza_nome"))

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro",
    "pr_auc_macro",
    "brier_score",
    "ece",
]

PRIMARY_PLOT_METRICS = ["roc_auc_macro", "f1_macro", "balanced_accuracy"]
PSEUDO_THRESHOLD_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]


@dataclass
class ExperimentConfig:
    """Central configuration for the TCGA transductive learning experiment."""

    input_path: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    random_state: int = 42
    n_jobs: int = 1
    ece_bins: int = 10
    coral_regularization: float = 1e-3
    variance_threshold: float = 0.0
    top_variable_features: int = 2000
    pca_components: int = 128
    use_pca: bool = True
    pseudo_labeling_enabled: bool = True
    pseudo_threshold: float = 0.98
    pseudo_min_samples: int = 10
    pseudo_max_fraction: float = 0.15
    pseudo_class_cap_ratio: float = 0.35
    pseudo_sample_weight: float = 0.25
    max_points_per_site_plot: int = 250
    feature_shift_plot_top_n: int = 20
    feature_shift_save_top_n: int = 150
    save_models: bool = True
    primary_summary_metric: str = "balanced_accuracy"
    artificial_domain_count: int = 5
    confidence_threshold_grid: list[float] = field(
        default_factory=lambda: list(PSEUDO_THRESHOLD_GRID)
    )

    @property
    def variants(self) -> list[str]:
        variants = ["coral_only"]
        if self.pseudo_labeling_enabled:
            variants.append("coral_plus_pseudolabel")
        return variants

    def to_serializable_dict(self) -> dict[str, Any]:
        return _convert_for_json(asdict(self))


def _convert_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _convert_for_json(sub_value) for key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [_convert_for_json(item) for item in value]
    return value
