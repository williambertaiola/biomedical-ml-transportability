"""Transportability pipeline module: config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "HistGradientBoosting": "Histogram Gradient Boosting",
}

TARGET_SITE_ORDER: tuple[str, ...] = ()
MODEL_ORDER = ("LogisticRegression", "RandomForest", "HistGradientBoosting")
METHOD_ORDER = ("M0", "M1", "M2", "M3", "M4")


@dataclass
class ExperimentConfig:
    input_path: Path
    schema_path: Path
    output_dir: Path
    dataset_name: str = "ABIDE II Core Multisite (strict phenotype)"
    enabled_methods: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4")
    target_column: str = "target"
    domain_column: str = "site"
    id_column: str = "subject_id"
    target_test_size: float = 0.5
    random_seed: int = 42
    ece_bins: int = 10
    n_jobs: int = 1
    threshold: float = 0.5
    target_sites: tuple[str, ...] = ()
    min_target_class_count: int = 2
    combat_continuous_covariates: tuple[str, ...] = ("age_at_scan",)
    combat_categorical_covariates: tuple[str, ...] = ("sex_code",)
    combat_low_cardinality_threshold: int = 10
    m2_weight_temperature: float = 0.12
    m2_min_domain_weight: float = 0.25
    m3_target_blend_weight: float = 0.5

    def to_snapshot(self) -> dict[str, object]:
        snapshot = asdict(self)
        snapshot["input_path"] = str(self.input_path)
        snapshot["schema_path"] = str(self.schema_path)
        snapshot["output_dir"] = str(self.output_dir)
        snapshot["enabled_methods"] = list(self.enabled_methods)
        snapshot["target_sites"] = list(self.target_sites)
        snapshot["combat_continuous_covariates"] = list(
            self.combat_continuous_covariates
        )
        snapshot["combat_categorical_covariates"] = list(
            self.combat_categorical_covariates
        )
        snapshot["target_site_order"] = list(TARGET_SITE_ORDER)
        snapshot["model_order"] = list(MODEL_ORDER)
        snapshot["method_order"] = list(METHOD_ORDER)
        return snapshot


def pretty_target_site(site: str) -> str:
    label = str(site).replace("ABIDEII-", "").replace("_", " ")
    return " ".join(label.split())


def pretty_model_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(str(model_name), str(model_name))
