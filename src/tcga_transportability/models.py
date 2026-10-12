"""Transportability pipeline module: models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcga_transportability.config import ExperimentConfig, MODEL_ORDER
from tcga_transportability.feature_selection import TCGATrainOnlyFeatureSelector


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    scale_numeric: bool


def build_model_specs(config: ExperimentConfig) -> dict[str, ModelSpec]:
    """Construct the original estimators with their dataset-specific hyperparameters."""
    specs = {
        "LogisticRegression": ModelSpec(
            name="LogisticRegression",
            estimator=LogisticRegression(
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
                random_state=config.random_seed,
            ),
            scale_numeric=True,
        ),
        "RandomForest": ModelSpec(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=config.random_seed,
                n_jobs=config.n_jobs,
                class_weight="balanced_subsample",
            ),
            scale_numeric=False,
        ),
        "HistGradientBoosting": ModelSpec(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=300,
                random_state=config.random_seed,
            ),
            scale_numeric=False,
        ),
    }
    return {
        model_name: specs[model_name]
        for model_name in MODEL_ORDER
        if model_name in specs
    }


def build_pipeline(
    schema: dict[str, Any], model_spec: ModelSpec, config: ExperimentConfig
) -> Pipeline:
    """Assemble preprocessing and model fitting in the original order."""
    return Pipeline(
        steps=[
            (
                "selector",
                TCGATrainOnlyFeatureSelector(
                    gene_columns=schema["gene_features"],
                    min_variance_threshold=config.min_variance_threshold,
                    top_k_variance_genes=config.top_k_variance_genes,
                    top_k_model_genes=config.top_k_model_genes,
                ),
            ),
            ("scaler", StandardScaler() if model_spec.scale_numeric else "passthrough"),
            ("model", clone(model_spec.estimator)),
        ]
    )


def safe_predict_proba(pipeline: Pipeline, X) -> Any:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = pipeline.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not expose predict_proba or decision_function")
