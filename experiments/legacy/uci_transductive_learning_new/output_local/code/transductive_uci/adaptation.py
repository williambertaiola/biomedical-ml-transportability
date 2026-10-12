"""Transportability experiment: adaptation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentConfig
from .data_utils import InferredColumns, ensure_subject_ids


@dataclass
class PreparedFoldData:
    target_site: str
    source_ids: pd.Series
    target_ids: pd.Series
    y_source: np.ndarray
    y_target: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    variance_filter: VarianceThreshold
    coral_aligner: "CORALAligner"
    X_source_pre: np.ndarray
    X_target_pre: np.ndarray
    X_source_aligned: np.ndarray
    X_target_eval: np.ndarray


def make_dense_one_hot_encoder() -> OneHotEncoder:
    params: dict[str, object] = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in signature.parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_dense_one_hot_encoder()),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Cannot build a preprocessor with zero feature columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_fold_data(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inferred: InferredColumns,
    target_site: str,
    config: ExperimentConfig,
) -> PreparedFoldData:
    preprocessor = build_preprocessor(
        numeric_features=inferred.numeric_features,
        categorical_features=inferred.categorical_features,
    )

    X_source_df = source_df.loc[:, inferred.feature_columns].copy()
    X_target_df = target_df.loc[:, inferred.feature_columns].copy()

    X_source_pre = np.asarray(preprocessor.fit_transform(X_source_df), dtype=float)
    X_target_pre = np.asarray(preprocessor.transform(X_target_df), dtype=float)

    raw_feature_names = _extract_feature_names(preprocessor, inferred.feature_columns)
    variance_filter = VarianceThreshold(threshold=0.0)
    X_source_pre = variance_filter.fit_transform(X_source_pre)
    X_target_pre = variance_filter.transform(X_target_pre)

    support = variance_filter.get_support()
    filtered_feature_names = [
        name for name, keep in zip(raw_feature_names, support) if keep
    ]

    coral_aligner = CORALAligner(regularization=config.coral_regularization)
    coral_aligner.fit(X_source_pre, X_target_pre)
    X_source_aligned = coral_aligner.transform_source(X_source_pre)
    X_target_eval = coral_aligner.transform_target(X_target_pre)

    return PreparedFoldData(
        target_site=target_site,
        source_ids=ensure_subject_ids(source_df, inferred.subject_id_column),
        target_ids=ensure_subject_ids(target_df, inferred.subject_id_column),
        y_source=source_df[inferred.target_column].to_numpy(dtype=int),
        y_target=target_df[inferred.target_column].to_numpy(dtype=int),
        feature_names=filtered_feature_names,
        preprocessor=preprocessor,
        variance_filter=variance_filter,
        coral_aligner=coral_aligner,
        X_source_pre=np.asarray(X_source_pre, dtype=float),
        X_target_pre=np.asarray(X_target_pre, dtype=float),
        X_source_aligned=np.asarray(X_source_aligned, dtype=float),
        X_target_eval=np.asarray(X_target_eval, dtype=float),
    )


def coral_align(
    X_source: np.ndarray,
    X_target: np.ndarray,
    regularization: float = 1e-3,
) -> tuple["CORALAligner", np.ndarray, np.ndarray]:
    aligner = CORALAligner(regularization=regularization)
    aligner.fit(X_source, X_target)
    return (
        aligner,
        aligner.transform_source(X_source),
        aligner.transform_target(X_target),
    )


class CORALAligner:
    """
    CORAL aligns the source covariance to the unlabeled target covariance.

    The transform is fitted using:
    - source features after source-only preprocessing
    - target unlabeled features after source-only preprocessing

    During evaluation, the target representation is left in the target feature
    space while the source training representation is recolored toward the
    target covariance and target mean.
    """

    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization = float(regularization)
        self.source_mean_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.alignment_matrix_: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "CORALAligner":
        X_source = np.asarray(X_source, dtype=float)
        X_target = np.asarray(X_target, dtype=float)
        if X_source.ndim != 2 or X_target.ndim != 2:
            raise ValueError("CORALAligner expects 2D arrays.")
        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError("Source and target must have the same number of features.")

        self.source_mean_ = X_source.mean(axis=0)
        self.target_mean_ = X_target.mean(axis=0)

        source_centered = X_source - self.source_mean_
        target_centered = X_target - self.target_mean_

        cov_source = _regularized_covariance(source_centered, self.regularization)
        cov_target = _regularized_covariance(target_centered, self.regularization)

        cov_source_inv_sqrt = _symmetric_matrix_power(cov_source, power=-0.5)
        cov_target_sqrt = _symmetric_matrix_power(cov_target, power=0.5)
        self.alignment_matrix_ = cov_source_inv_sqrt @ cov_target_sqrt
        return self

    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_source = np.asarray(X_source, dtype=float)
        centered = X_source - self.source_mean_
        aligned = centered @ self.alignment_matrix_
        return aligned + self.target_mean_

    def transform_target(self, X_target: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(X_target, dtype=float)

    def _check_is_fitted(self) -> None:
        if (
            self.source_mean_ is None
            or self.target_mean_ is None
            or self.alignment_matrix_ is None
        ):
            raise RuntimeError("CORALAligner has not been fitted yet.")


def _extract_feature_names(
    preprocessor: ColumnTransformer, fallback_columns: list[str]
) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    return list(fallback_columns)


def _regularized_covariance(
    X_centered: np.ndarray, regularization: float
) -> np.ndarray:
    n_samples, n_features = X_centered.shape
    if n_samples <= 1:
        cov = np.eye(n_features, dtype=float)
    else:
        cov = np.cov(X_centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov += regularization * np.eye(cov.shape[0], dtype=float)
    return cov


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    clipped = np.clip(eigenvalues, a_min=1e-12, a_max=None)
    powered = np.power(clipped, power)
    return eigenvectors @ np.diag(powered) @ eigenvectors.T
