"""Preprocessing: train only."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_train_only_preprocessor(
    schema: dict[str, Any],
    drop_zero_variance: bool = True,
) -> Pipeline:
    """Build a preprocessing pipeline that must be fit only on training data.

    This function intentionally does not inspect any dataset values. The caller is
    responsible for fitting the returned sklearn Pipeline inside each training fold.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema["numeric_features"]),
            ("categorical", categorical_pipeline, schema["categorical_features"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps: list[tuple[str, Any]] = [("column_transformer", column_transformer)]
    if drop_zero_variance:
        steps.append(("variance_filter", VarianceThreshold()))

    return Pipeline(steps=steps)
