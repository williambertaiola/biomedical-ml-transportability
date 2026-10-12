"""Transportability experiment: data utils."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TARGET_EXACT = [
    "target",
    "label",
    "outcome",
    "class",
    "diagnosis",
    "y",
    "num",
]
TARGET_PARTIAL = [
    "target",
    "label",
    "outcome",
    "diagnos",
    "class",
    "disease",
    "num",
]
DOMAIN_EXACT = [
    "domain_id",
    "site",
    "site_id",
    "domain",
    "hospital",
    "center",
    "centre",
    "source",
    "cohort",
    "dataset",
]
DOMAIN_PARTIAL = [
    "domain",
    "site",
    "hospital",
    "center",
    "centre",
    "source",
    "cohort",
    "batch",
]
ID_EXACT = [
    "subject_id",
    "sample_id",
    "patient_id",
    "case_id",
    "record_id",
    "id",
]
ID_PARTIAL = [
    "subject",
    "sample",
    "patient",
    "record",
    "case",
    "barcode",
    "submitter",
    "identifier",
]


@dataclass
class InferredColumns:
    target_column: str
    domain_column: str
    subject_id_column: str | None
    auxiliary_target_columns: list[str]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def load_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_csv}")
    return df


def infer_columns(df: pd.DataFrame) -> InferredColumns:
    target_column = _infer_target_column(df)
    domain_column = _infer_domain_column(df, excluded={target_column})
    subject_id_column = _infer_subject_id_column(
        df, excluded={target_column, domain_column}
    )
    auxiliary_target_columns = _find_auxiliary_target_columns(df, target_column)
    feature_columns = _infer_feature_columns(
        df,
        excluded={target_column, domain_column, *(auxiliary_target_columns)},
        subject_id_column=subject_id_column,
    )
    numeric_features, categorical_features = infer_feature_types(df, feature_columns)

    if not feature_columns:
        raise ValueError(
            "No predictive features remain after automatic column inference."
        )

    return InferredColumns(
        target_column=target_column,
        domain_column=domain_column,
        subject_id_column=subject_id_column,
        auxiliary_target_columns=auxiliary_target_columns,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def infer_feature_types(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_unique_threshold: int = 10,
) -> tuple[list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in feature_columns:
        series = df[column]
        if _is_categorical_feature(
            series, categorical_unique_threshold=categorical_unique_threshold
        ):
            categorical_features.append(column)
        else:
            numeric_features.append(column)

    return numeric_features, categorical_features


def get_feature_matrix(df: pd.DataFrame, inferred: InferredColumns) -> pd.DataFrame:
    return df.loc[:, inferred.feature_columns].copy()


def split_loso(
    df: pd.DataFrame, domain_column: str
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    for site in sorted(df[domain_column].dropna().astype(str).unique().tolist()):
        target_mask = df[domain_column].astype(str) == site
        target_df = df.loc[target_mask].copy()
        source_df = df.loc[~target_mask].copy()
        if source_df.empty or target_df.empty:
            continue
        yield site, source_df, target_df


def build_dataset_tables(
    df: pd.DataFrame, inferred: InferredColumns
) -> dict[str, pd.DataFrame]:
    overview = pd.DataFrame(
        [
            {"field": "n_rows", "value": int(df.shape[0])},
            {"field": "n_columns", "value": int(df.shape[1])},
            {"field": "target_column", "value": inferred.target_column},
            {"field": "domain_column", "value": inferred.domain_column},
            {"field": "subject_id_column", "value": inferred.subject_id_column or ""},
            {
                "field": "n_sites",
                "value": int(df[inferred.domain_column].nunique(dropna=True)),
            },
            {"field": "n_features", "value": int(len(inferred.feature_columns))},
            {
                "field": "n_numeric_features",
                "value": int(len(inferred.numeric_features)),
            },
            {
                "field": "n_categorical_features",
                "value": int(len(inferred.categorical_features)),
            },
        ]
    )

    column_roles = []
    for column in df.columns:
        role = "feature"
        if column == inferred.target_column:
            role = "target"
        elif column == inferred.domain_column:
            role = "domain"
        elif inferred.subject_id_column and column == inferred.subject_id_column:
            role = "subject_id"
        elif column in inferred.auxiliary_target_columns:
            role = "auxiliary_target_excluded"
        column_roles.append(
            {
                "column": column,
                "role": role,
                "dtype": str(df[column].dtype),
                "n_missing": int(df[column].isna().sum()),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
        )

    site_distribution = (
        df.groupby(inferred.domain_column, dropna=False)[inferred.target_column]
        .agg(site_n="size", target_positive="sum", target_rate="mean")
        .reset_index()
        .sort_values("site_n", ascending=False)
        .rename(columns={inferred.domain_column: "site"})
    )

    target_distribution = (
        df[inferred.target_column]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("target")
        .reset_index(name="n_samples")
    )
    target_distribution["fraction"] = target_distribution["n_samples"] / max(len(df), 1)

    return {
        "dataset_overview": overview,
        "column_roles": pd.DataFrame(column_roles),
        "site_distribution": site_distribution,
        "target_distribution": target_distribution,
    }


def format_initial_inspection(df: pd.DataFrame, inferred: InferredColumns) -> list[str]:
    lines = [
        f"[DATA] shape={df.shape}",
        f"[DATA] columns={list(df.columns)}",
        f"[DATA] inferred_target={inferred.target_column}",
        f"[DATA] inferred_domain={inferred.domain_column}",
        f"[DATA] inferred_subject_id={inferred.subject_id_column}",
        f"[DATA] auxiliary_target_columns_excluded={inferred.auxiliary_target_columns}",
    ]

    target_dist = (
        df[inferred.target_column].value_counts(dropna=False).sort_index().to_dict()
    )
    lines.append(f"[DATA] target_distribution={target_dist}")

    site_dist = (
        df[inferred.domain_column]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .to_dict()
    )
    lines.append(f"[DATA] site_distribution={site_dist}")
    return lines


def ensure_subject_ids(df: pd.DataFrame, subject_id_column: str | None) -> pd.Series:
    if subject_id_column and subject_id_column in df.columns:
        return df[subject_id_column].astype(str)
    return pd.Series(
        [f"row_{index:05d}" for index in range(len(df))],
        index=df.index,
        name="synthetic_subject_id",
        dtype="string",
    )


def _infer_target_column(df: pd.DataFrame) -> str:
    exact = _find_exact(df.columns, TARGET_EXACT)
    if exact is not None:
        return exact

    partial = _find_partial(df.columns, TARGET_PARTIAL)
    if partial is not None:
        return partial

    scored = []
    for column in df.columns:
        series = df[column]
        unique_count = series.nunique(dropna=True)
        if unique_count < 2 or unique_count > 20:
            continue
        score = 0
        lower = _normalize(column)
        if unique_count == 2:
            score += 4
        if any(token in lower for token in TARGET_PARTIAL):
            score += 5
        if pd.api.types.is_numeric_dtype(series):
            score += 1
        if series.nunique(dropna=True) >= len(series) * 0.9:
            score -= 5
        scored.append((score, column))

    if not scored:
        raise ValueError("Unable to infer the target column automatically.")

    scored.sort(reverse=True)
    return scored[0][1]


def _infer_domain_column(df: pd.DataFrame, excluded: set[str]) -> str:
    columns = [col for col in df.columns if col not in excluded]
    exact = _find_exact(columns, DOMAIN_EXACT)
    if exact is not None:
        return exact

    partial = _find_partial(columns, DOMAIN_PARTIAL)
    if partial is not None:
        return partial

    scored = []
    for column in columns:
        series = df[column]
        unique_count = series.nunique(dropna=True)
        if unique_count < 2 or unique_count > max(50, len(df) // 2):
            continue
        score = 0
        lower = _normalize(column)
        if any(token in lower for token in DOMAIN_PARTIAL):
            score += 5
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            score += 3
        if unique_count <= 20:
            score += 2
        if series.nunique(dropna=True) >= len(series) * 0.9:
            score -= 5
        scored.append((score, column))

    if not scored:
        raise ValueError("Unable to infer the site/domain column automatically.")

    scored.sort(reverse=True)
    return scored[0][1]


def _infer_subject_id_column(df: pd.DataFrame, excluded: set[str]) -> str | None:
    columns = [col for col in df.columns if col not in excluded]
    exact = _find_exact(columns, ID_EXACT)
    if exact is not None:
        return exact

    partial = _find_partial(columns, ID_PARTIAL)
    if partial is not None:
        return partial

    scored = []
    for column in columns:
        series = df[column]
        unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
        if unique_ratio < 0.95:
            continue
        score = 0
        lower = _normalize(column)
        if any(token in lower for token in ID_PARTIAL):
            score += 5
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            score += 2
        if unique_ratio >= 0.99:
            score += 1
        scored.append((score, column))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][1]


def _find_auxiliary_target_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    auxiliary: list[str] = []
    for column in df.columns:
        if column == target_column:
            continue
        lower = _normalize(column)
        if any(token in lower for token in TARGET_PARTIAL):
            auxiliary.append(column)
    return auxiliary


def _infer_feature_columns(
    df: pd.DataFrame,
    excluded: set[str],
    subject_id_column: str | None,
) -> list[str]:
    feature_columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if subject_id_column and column == subject_id_column:
            continue
        lower = _normalize(column)
        if (
            any(token in lower for token in ID_PARTIAL)
            and df[column].nunique(dropna=False) / max(len(df), 1) > 0.95
        ):
            continue
        feature_columns.append(column)
    return feature_columns


def _is_categorical_feature(
    series: pd.Series, categorical_unique_threshold: int = 10
) -> bool:
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return True

    if not pd.api.types.is_numeric_dtype(series):
        return True

    non_missing = pd.to_numeric(series.dropna(), errors="coerce")
    if non_missing.empty:
        return False

    unique_values = pd.unique(non_missing)
    is_integer_like = np.all(
        np.isclose(unique_values, np.round(unique_values), atol=1e-8)
    )
    return is_integer_like and len(unique_values) <= categorical_unique_threshold


def _find_exact(columns: list[str] | pd.Index, candidates: list[str]) -> str | None:
    lookup = {_normalize(column): str(column) for column in columns}
    for candidate in candidates:
        matched = lookup.get(_normalize(candidate))
        if matched is not None:
            return matched
    return None


def _find_partial(columns: list[str] | pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        for column in columns:
            if normalized_candidate in _normalize(str(column)):
                return str(column)
    return None


def _normalize(value: str) -> str:
    return value.strip().lower()
