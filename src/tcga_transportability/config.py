"""Transportability pipeline module: config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "HistGradientBoosting": "Histogram Gradient Boosting",
}

MODEL_ORDER = ("LogisticRegression", "RandomForest", "HistGradientBoosting")
METHOD_ORDER = ("M0", "M1", "M2", "M3", "M4")


@dataclass
class ExperimentConfig:
    input_path: Path
    output_dir: Path
    dataset_name: str = "TCGA-BRCA Stage Transportability"
    enabled_methods: tuple[str, ...] = ("M0", "M1")
    target_sites: tuple[str, ...] = ()
    target_column: str = "target"
    target_label_column: str = "target_label"
    domain_column: str = "site"
    id_column: str = "subject_id"
    gene_prefix: str = "ENSG"
    target_test_size: float = 0.5
    random_seed: int = 42
    ece_bins: int = 10
    n_jobs: int = 1
    threshold: float = 0.5
    min_site_samples: int = 30
    min_site_class_count: int = 2
    min_variance_threshold: float = 1e-8
    combat_top_k_variance_genes: int = 500
    top_k_variance_genes: int = 2000
    top_k_model_genes: int = 500
    combat_label_covariate: str = "target"
    m2_weight_temperature: float = 0.12
    m2_min_domain_weight: float = 0.25
    m3_target_blend_weight: float = 0.5

    def to_snapshot(self) -> dict[str, object]:
        snapshot = asdict(self)
        snapshot["input_path"] = str(self.input_path)
        snapshot["output_dir"] = str(self.output_dir)
        snapshot["enabled_methods"] = list(self.enabled_methods)
        snapshot["target_sites"] = list(self.target_sites)
        snapshot["model_order"] = list(MODEL_ORDER)
        snapshot["method_order"] = list(METHOD_ORDER)
        return snapshot


def pretty_model_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(str(model_name), str(model_name))


def pretty_target_site(site: str) -> str:
    return str(site)
