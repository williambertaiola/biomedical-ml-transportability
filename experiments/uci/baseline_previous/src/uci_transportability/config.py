"""Transportability experiment: config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TARGET_SITE_DISPLAY = {
    "cleveland": "Cleveland",
    "hungarian": "Hungarian",
    "long_beach_va": "Long Beach VA",
    "switzerland": "Switzerland",
}

MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "HistGradientBoosting": "Histogram Gradient Boosting",
}

TARGET_SITE_ORDER = ("cleveland", "hungarian", "long_beach_va", "switzerland")
MODEL_ORDER = ("LogisticRegression", "RandomForest", "HistGradientBoosting")
METHOD_ORDER = ("M0", "M1", "M2", "M3", "M4")


@dataclass
class ExperimentConfig:
    input_path: Path
    schema_path: Path
    output_dir: Path
    dataset_name: str = "UCI Heart Disease"
    method_name: str = "M0"
    target_column: str = "target"
    domain_column: str = "domain_id"
    id_column: str = "subject_id"
    target_test_size: float = 0.5
    random_seed: int = 42
    ece_bins: int = 10
    n_jobs: int = 1
    threshold: float = 0.5

    def to_snapshot(self) -> dict[str, object]:
        snapshot = asdict(self)
        snapshot["input_path"] = str(self.input_path)
        snapshot["schema_path"] = str(self.schema_path)
        snapshot["output_dir"] = str(self.output_dir)
        snapshot["target_site_order"] = list(TARGET_SITE_ORDER)
        snapshot["model_order"] = list(MODEL_ORDER)
        snapshot["method_order"] = list(METHOD_ORDER)
        return snapshot


def pretty_target_site(site: str) -> str:
    return TARGET_SITE_DISPLAY.get(str(site), str(site))


def pretty_model_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(str(model_name), str(model_name))
