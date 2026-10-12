"""Transportability experiment: preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data_utils import InferredColumns, encode_target, ensure_subject_ids


@dataclass
class SourceFittedPreprocessor:
    imputer: SimpleImputer
    variance_filter: VarianceThreshold
    scaler: StandardScaler
    pca: PCA | None
    selected_feature_names: list[str]
    pca_feature_names: list[str]


@dataclass
class CORALAligner:
    regularization: float
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    alignment_matrix_: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "CORALAligner":
        X_source = np.asarray(X_source, dtype=np.float32)
        X_target = np.asarray(X_target, dtype=np.float32)
        self.source_mean_ = X_source.mean(axis=0)
        self.target_mean_ = X_target.mean(axis=0)

        source_centered = X_source - self.source_mean_
        target_centered = X_target - self.target_mean_

        cov_source = regularized_covariance(source_centered, self.regularization)
        cov_target = regularized_covariance(target_centered, self.regularization)
        cov_source_inv_sqrt = symmetric_matrix_power(cov_source, power=-0.5)
        cov_target_sqrt = symmetric_matrix_power(cov_target, power=0.5)
        self.alignment_matrix_ = cov_source_inv_sqrt @ cov_target_sqrt
        return self

    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        centered = np.asarray(X_source, dtype=np.float32) - self.source_mean_
        return (centered @ self.alignment_matrix_) + self.target_mean_

    def transform_target(self, X_target: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(X_target, dtype=np.float32)

    def _check_is_fitted(self) -> None:
        if (
            self.source_mean_ is None
            or self.target_mean_ is None
            or self.alignment_matrix_ is None
        ):
            raise RuntimeError("CORALAligner has not been fitted.")


@dataclass
class PreparedFoldData:
    target_domain: str
    source_ids: pd.Series
    target_ids: pd.Series
    y_source: np.ndarray
    y_target: np.ndarray
    source_preprocessor: SourceFittedPreprocessor
    coral_aligner: CORALAligner
    X_source_pre: np.ndarray
    X_target_pre: np.ndarray
    X_source_aligned: np.ndarray
    X_target_eval: np.ndarray
    selected_feature_names: list[str]
    pca_feature_names: list[str]


def preprocess_source(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inferred: InferredColumns,
    config: ExperimentConfig,
) -> tuple[SourceFittedPreprocessor, np.ndarray, np.ndarray]:
    X_source_raw = source_df.loc[:, inferred.feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    X_target_raw = target_df.loc[:, inferred.feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )

    imputer = SimpleImputer(strategy="median")
    X_source = imputer.fit_transform(X_source_raw).astype(np.float32, copy=False)
    X_target = imputer.transform(X_target_raw).astype(np.float32, copy=False)

    variance_filter = VarianceThreshold(threshold=config.variance_threshold)
    X_source = variance_filter.fit_transform(X_source).astype(np.float32, copy=False)
    X_target = variance_filter.transform(X_target).astype(np.float32, copy=False)

    selected_feature_names = [
        feature
        for feature, keep in zip(
            inferred.feature_columns, variance_filter.get_support()
        )
        if keep
    ]

    if X_source.shape[1] == 0:
        raise ValueError("No features survived the source-only variance threshold.")

    if X_source.shape[1] > config.top_variable_features:
        top_indices = np.argsort(np.var(X_source, axis=0))[
            -config.top_variable_features :
        ]
        top_indices = np.sort(top_indices)
        X_source = X_source[:, top_indices]
        X_target = X_target[:, top_indices]
        selected_feature_names = [selected_feature_names[idx] for idx in top_indices]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_source = scaler.fit_transform(X_source).astype(np.float32, copy=False)
    X_target = scaler.transform(X_target).astype(np.float32, copy=False)

    pca = None
    pca_feature_names = [str(name) for name in selected_feature_names]
    if config.use_pca:
        n_components = min(
            config.pca_components, X_source.shape[0] - 1, X_source.shape[1]
        )
        if n_components >= 2:
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=config.random_state,
            )
            X_source = pca.fit_transform(X_source).astype(np.float32, copy=False)
            X_target = pca.transform(X_target).astype(np.float32, copy=False)
            pca_feature_names = [f"PC{idx + 1}" for idx in range(X_source.shape[1])]

    fitted = SourceFittedPreprocessor(
        imputer=imputer,
        variance_filter=variance_filter,
        scaler=scaler,
        pca=pca,
        selected_feature_names=selected_feature_names,
        pca_feature_names=pca_feature_names,
    )
    return fitted, X_source, X_target


def prepare_fold_data(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inferred: InferredColumns,
    target_domain: str,
    config: ExperimentConfig,
) -> PreparedFoldData:
    fitted_preprocessor, X_source_pre, X_target_pre = preprocess_source(
        source_df=source_df,
        target_df=target_df,
        inferred=inferred,
        config=config,
    )
    coral_aligner = CORALAligner(regularization=config.coral_regularization)
    coral_aligner.fit(X_source_pre, X_target_pre)
    X_source_aligned = coral_aligner.transform_source(X_source_pre).astype(
        np.float32, copy=False
    )
    X_target_eval = coral_aligner.transform_target(X_target_pre).astype(
        np.float32, copy=False
    )

    return PreparedFoldData(
        target_domain=target_domain,
        source_ids=ensure_subject_ids(source_df, inferred),
        target_ids=ensure_subject_ids(target_df, inferred),
        y_source=encode_target(
            source_df[inferred.target_column], inferred.class_mapping
        ),
        y_target=encode_target(
            target_df[inferred.target_column], inferred.class_mapping
        ),
        source_preprocessor=fitted_preprocessor,
        coral_aligner=coral_aligner,
        X_source_pre=X_source_pre,
        X_target_pre=X_target_pre,
        X_source_aligned=X_source_aligned,
        X_target_eval=X_target_eval,
        selected_feature_names=fitted_preprocessor.selected_feature_names,
        pca_feature_names=fitted_preprocessor.pca_feature_names,
    )


def regularized_covariance(X_centered: np.ndarray, regularization: float) -> np.ndarray:
    n_samples, n_features = X_centered.shape
    if n_samples <= 1:
        cov = np.eye(n_features, dtype=np.float32)
    else:
        cov = np.cov(X_centered, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float32)
    cov = np.asarray(cov, dtype=np.float32)
    cov += regularization * np.eye(cov.shape[0], dtype=np.float32)
    return cov


def symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float32))
    eigenvalues = np.clip(eigenvalues, a_min=1e-8, a_max=None)
    powered = np.power(eigenvalues, power).astype(np.float32)
    return (eigenvectors @ np.diag(powered) @ eigenvectors.T).astype(np.float32)
