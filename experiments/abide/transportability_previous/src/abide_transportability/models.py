"""Transportability experiment: models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from abide_transportability.config import ExperimentConfig, MODEL_ORDER


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    scale_numeric: bool


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    schema: dict[str, Any], *, scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema["numeric_features"]),
            ("categorical", categorical_pipeline, schema["categorical_features"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_specs(config: ExperimentConfig) -> dict[str, ModelSpec]:
    specs = {
        "LogisticRegression": ModelSpec(
            name="LogisticRegression",
            estimator=LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=config.random_seed,
            ),
            scale_numeric=True,
        ),
        "RandomForest": ModelSpec(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=500,
                random_state=config.random_seed,
                n_jobs=config.n_jobs,
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


def build_pipeline(schema: dict[str, Any], model_spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(schema, scale_numeric=model_spec.scale_numeric),
            ),
            ("variance_filter", VarianceThreshold()),
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
