"""Transportability pipeline module: feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


@dataclass(frozen=True)
class FeatureSelectionSummary:
    n_input_genes: int
    n_after_variance_filter: int
    n_selected_genes: int
    selected_gene_preview: str


class TCGATrainOnlyFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        gene_columns: list[str],
        min_variance_threshold: float,
        top_k_variance_genes: int,
        top_k_model_genes: int,
    ) -> None:
        self.gene_columns = list(gene_columns)
        self.min_variance_threshold = float(min_variance_threshold)
        self.top_k_variance_genes = int(top_k_variance_genes)
        self.top_k_model_genes = int(top_k_model_genes)

    def fit(self, X: pd.DataFrame, y) -> "TCGATrainOnlyFeatureSelector":
        frame = (
            X[self.gene_columns]
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=self.gene_columns)
        )
        variances = frame.var(axis=0, ddof=0)
        variance_kept = variances.loc[
            variances > self.min_variance_threshold
        ].sort_values(ascending=False)
        if variance_kept.empty:
            variance_kept = variances.sort_values(ascending=False).head(1)

        if (
            self.top_k_variance_genes > 0
            and len(variance_kept) > self.top_k_variance_genes
        ):
            variance_kept = variance_kept.head(self.top_k_variance_genes)

        candidate_columns = variance_kept.index.tolist()
        if len(candidate_columns) == 1:
            selected_columns = candidate_columns
        else:
            y_array = np.asarray(y, dtype=int)
            score_values, _ = f_classif(
                frame[candidate_columns].to_numpy(dtype=np.float32), y_array
            )
            score_series = pd.Series(
                score_values, index=candidate_columns, dtype="float64"
            ).replace({np.nan: -np.inf})
            if (
                self.top_k_model_genes > 0
                and len(score_series) > self.top_k_model_genes
            ):
                score_series = score_series.sort_values(ascending=False).head(
                    self.top_k_model_genes
                )
            else:
                score_series = score_series.sort_values(ascending=False)
            selected_columns = score_series.index.tolist()

        self.variance_filtered_genes_ = candidate_columns
        self.selected_genes_ = selected_columns
        preview = "|".join(selected_columns[:10])
        self.summary_ = FeatureSelectionSummary(
            n_input_genes=int(len(self.gene_columns)),
            n_after_variance_filter=int(len(candidate_columns)),
            n_selected_genes=int(len(selected_columns)),
            selected_gene_preview=preview,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = (
            X[self.gene_columns]
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=self.gene_columns)
        )
        if not hasattr(self, "selected_genes_"):
            raise AttributeError(
                "TCGATrainOnlyFeatureSelector must be fit before transform."
            )
        return frame[self.selected_genes_].to_numpy(dtype=np.float32, copy=False)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        del input_features
        return np.asarray(getattr(self, "selected_genes_", []), dtype=object)

    def get_summary(self) -> FeatureSelectionSummary:
        return getattr(
            self,
            "summary_",
            FeatureSelectionSummary(
                n_input_genes=len(self.gene_columns),
                n_after_variance_filter=0,
                n_selected_genes=0,
                selected_gene_preview="",
            ),
        )
