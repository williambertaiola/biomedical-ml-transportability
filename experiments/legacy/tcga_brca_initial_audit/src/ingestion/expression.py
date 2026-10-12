"""Transportability experiment: expression."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import detect_delimiter

SAMPLE_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z](?:-.+)?$")
GENE_PATTERN = re.compile(r"^(?:ENSG\d+(?:\.\d+)?|[A-Za-z0-9_.-]{2,})$")


@dataclass
class ExpressionLoadResult:
    standardized_df: pd.DataFrame
    delimiter: str
    original_shape: tuple[int, int]
    original_orientation: str
    final_shape: tuple[int, int]
    transpose_applied: bool
    value_profile: dict[str, object]


def _sample_like_ratio(values: list[str]) -> float:
    if not values:
        return 0.0
    matches = sum(bool(SAMPLE_PATTERN.match(str(value))) for value in values)
    return matches / len(values)


def _gene_like_ratio(values: list[str]) -> float:
    if not values:
        return 0.0
    matches = sum(
        bool(GENE_PATTERN.match(str(value))) and not SAMPLE_PATTERN.match(str(value))
        for value in values
    )
    return matches / len(values)


def detect_orientation(preview: pd.DataFrame) -> str:
    if preview.empty or preview.shape[1] < 2:
        return "unknown"

    header_values = [
        str(value) for value in preview.columns[1 : min(preview.shape[1], 25)]
    ]
    first_column_values = preview.iloc[: min(len(preview), 25), 0].astype(str).tolist()

    header_sample_ratio = _sample_like_ratio(header_values)
    header_gene_ratio = _gene_like_ratio(header_values)
    first_col_sample_ratio = _sample_like_ratio(first_column_values)
    first_col_gene_ratio = _gene_like_ratio(first_column_values)

    if header_sample_ratio >= 0.6 and first_col_gene_ratio >= 0.6:
        return "genes_as_rows_samples_as_columns"
    if first_col_sample_ratio >= 0.6 and header_gene_ratio >= 0.6:
        return "samples_as_rows_genes_as_columns"
    if preview.shape[0] > preview.shape[1]:
        return "genes_as_rows_samples_as_columns"
    return "samples_as_rows_genes_as_columns"


def profile_matrix_values(df: pd.DataFrame, seed: int = 42) -> dict[str, object]:
    n_rows, n_cols = df.shape
    row_count = min(n_rows, 100)
    col_count = min(n_cols, 500)

    rng = np.random.default_rng(seed)
    row_idx = np.sort(rng.choice(n_rows, size=row_count, replace=False))
    col_idx = np.sort(rng.choice(n_cols, size=col_count, replace=False))
    sample = df.to_numpy(copy=False)[np.ix_(row_idx, col_idx)]

    finite_mask = np.isfinite(sample)
    finite_values = sample[finite_mask]
    if finite_values.size == 0:
        return {
            "sampled_cells": 0,
            "fractional_value_rate": np.nan,
            "min_value": np.nan,
            "max_value": np.nan,
            "median_value": np.nan,
            "value_type_guess": "unknown",
        }

    fractional_rate = float(
        np.mean(~np.isclose(finite_values, np.rint(finite_values), atol=1e-6))
    )
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    median_value = float(np.median(finite_values))

    if fractional_rate < 0.01 and min_value >= 0:
        value_type_guess = "likely_raw_counts"
    elif max_value <= 50 and fractional_rate >= 0.25:
        value_type_guess = "likely_transformed_or_normalized_continuous_expression"
    else:
        value_type_guess = "ambiguous_numeric_expression_matrix"

    return {
        "sampled_cells": int(finite_values.size),
        "fractional_value_rate": fractional_rate,
        "min_value": min_value,
        "max_value": max_value,
        "median_value": median_value,
        "value_type_guess": value_type_guess,
    }


def load_and_standardize_expression(
    expression_path: Path,
    logger: logging.Logger,
    *,
    seed: int = 42,
) -> ExpressionLoadResult:
    delimiter = detect_delimiter(expression_path)
    logger.info("Detected delimiter '%s' for %s", delimiter, expression_path)

    preview = pd.read_csv(expression_path, sep=delimiter, nrows=5)
    original_orientation = detect_orientation(preview)
    logger.info("Detected matrix orientation: %s", original_orientation)

    dtype_map = {column: np.float32 for column in preview.columns[1:]}
    raw_df = pd.read_csv(
        expression_path, sep=delimiter, index_col=0, dtype=dtype_map, low_memory=False
    )
    original_shape = raw_df.shape

    if original_orientation == "genes_as_rows_samples_as_columns":
        standardized_df = raw_df.transpose()
        transpose_applied = True
    else:
        standardized_df = raw_df.copy()
        transpose_applied = False

    standardized_df.index = standardized_df.index.astype(str)
    standardized_df.columns = standardized_df.columns.astype(str)
    standardized_df.index.name = "sample_barcode"
    standardized_df.columns.name = "gene_id"

    value_profile = profile_matrix_values(standardized_df, seed=seed)
    logger.info("Matrix value type guess: %s", value_profile["value_type_guess"])

    return ExpressionLoadResult(
        standardized_df=standardized_df,
        delimiter=delimiter,
        original_shape=original_shape,
        original_orientation=original_orientation,
        final_shape=standardized_df.shape,
        transpose_applied=transpose_applied,
        value_profile=value_profile,
    )
