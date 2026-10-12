"""Transportability experiment: data utils."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TARGET_EXACT = [
    "sample_type_label",
    "disease_level",
    "disease_status",
    "target",
    "label",
    "outcome",
    "class",
    "subtype",
    "diagnosis",
    "tumor_level",
]
TARGET_PARTIAL = [
    "disease",
    "status",
    "sample_type",
    "target",
    "label",
    "outcome",
    "class",
    "subtype",
    "diagnos",
    "tumor",
]
DOMAIN_EXACT = [
    "domain_primary",
    "domain_id",
    "site",
    "batch",
    "center",
    "centre",
    "tss_code",
    "tss",
]
DOMAIN_PARTIAL = [
    "domain",
    "site",
    "scanner",
    "acquisition",
    "batch",
    "center",
    "centre",
    "tss",
    "plate",
    "platform",
]
ID_EXACT = [
    "sample_barcode",
    "subject_id",
    "sample_id",
    "patient_id",
    "case_id",
    "submitter_id",
    "id",
]
ID_PARTIAL = [
    "barcode",
    "subject",
    "sample",
    "patient",
    "submitter",
    "case",
    "id",
]


@dataclass
class InferredColumns:
    target_column: str
    domain_column: str
    subject_id_column: str | None
    sample_id_column: str | None
    auxiliary_target_columns: list[str]
    auxiliary_domain_columns: list[str]
    feature_columns: list[str]
    class_labels: list[str]
    class_mapping: dict[str, int]
    used_artificial_domains: bool


def find_primary_input_file(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    candidates = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV/TSV file found inside {input_path}")

    candidates.sort(
        key=lambda path: (
            path.suffix.lower() != ".csv",
            -path.stat().st_size,
            path.name,
        )
    )
    return candidates[0]


def load_data(input_path: Path) -> tuple[pd.DataFrame, Path]:
    dataset_file = find_primary_input_file(input_path)
    header = read_header(dataset_file)
    dtype_map = build_dtype_map(header)

    read_kwargs = {"low_memory": False, "dtype": dtype_map}
    if dataset_file.suffix.lower() == ".tsv":
        read_kwargs["sep"] = "\t"

    df = pd.read_csv(dataset_file, **read_kwargs)
    return df, dataset_file


def read_header(dataset_file: Path) -> list[str]:
    delimiter = "\t" if dataset_file.suffix.lower() == ".tsv" else ","
    with dataset_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader)


def build_dtype_map(columns: list[str]) -> dict[str, object]:
    dtype_map: dict[str, object] = {}
    for column in columns:
        lower = normalize(column)
        if any(token == lower for token in ID_EXACT) or any(
            token in lower
            for token in [
                "barcode",
                "subject",
                "sample",
                "patient",
                "submitter",
                "case",
            ]
        ):
            dtype_map[column] = "string"
        elif lower in {
            "site",
            "domain_primary",
            "domain_source",
            "batch",
            "platform",
            "center",
            "centre",
            "tss",
            "tss_code",
        }:
            dtype_map[column] = "string"
        elif any(
            token in lower
            for token in [
                "label",
                "sample_type",
                "subtype",
                "diagnos",
                "phenotype",
                "disease",
                "status",
                "test_type",
            ]
        ):
            dtype_map[column] = "string"
        elif any(token in lower for token in ["tumor_level"]):
            dtype_map[column] = "Int64"
        else:
            dtype_map[column] = np.float32
    return dtype_map


def infer_columns(
    df: pd.DataFrame, random_state: int = 42, artificial_domain_count: int = 5
) -> tuple[pd.DataFrame, InferredColumns]:
    target_column = infer_target_column(df)
    domain_column = infer_domain_column(df, excluded={target_column})
    used_artificial_domains = False

    if domain_column is None:
        df, artificial_domain_column = create_artificial_domains(
            df=df,
            target_column=target_column,
            random_state=random_state,
            n_domains=artificial_domain_count,
        )
        domain_column = artificial_domain_column
        used_artificial_domains = True

    subject_id_column = infer_id_column(
        df, excluded={target_column, domain_column}, preferred="subject"
    )
    sample_id_column = infer_id_column(
        df, excluded={target_column, domain_column}, preferred="sample"
    )
    auxiliary_target_columns = find_auxiliary_target_columns(df, target_column)
    auxiliary_domain_columns = find_auxiliary_domain_columns(df, domain_column)
    feature_columns = infer_feature_columns(
        df,
        target_column=target_column,
        domain_column=domain_column,
        subject_id_column=subject_id_column,
        sample_id_column=sample_id_column,
        auxiliary_target_columns=auxiliary_target_columns,
        auxiliary_domain_columns=auxiliary_domain_columns,
    )
    if not feature_columns:
        raise ValueError(
            "No predictive feature columns detected after excluding metadata and label proxies."
        )

    class_labels, class_mapping = encode_class_mapping(df[target_column])
    inferred = InferredColumns(
        target_column=target_column,
        domain_column=domain_column,
        subject_id_column=subject_id_column,
        sample_id_column=sample_id_column,
        auxiliary_target_columns=auxiliary_target_columns,
        auxiliary_domain_columns=auxiliary_domain_columns,
        feature_columns=feature_columns,
        class_labels=class_labels,
        class_mapping=class_mapping,
        used_artificial_domains=used_artificial_domains,
    )
    return df, inferred


def build_dataset_tables(
    df: pd.DataFrame, inferred: InferredColumns
) -> dict[str, pd.DataFrame]:
    overview = pd.DataFrame(
        [
            {"field": "n_rows", "value": int(df.shape[0])},
            {"field": "n_columns", "value": int(df.shape[1])},
            {"field": "n_features", "value": int(len(inferred.feature_columns))},
            {"field": "target_column", "value": inferred.target_column},
            {"field": "domain_column", "value": inferred.domain_column},
            {"field": "subject_id_column", "value": inferred.subject_id_column or ""},
            {"field": "sample_id_column", "value": inferred.sample_id_column or ""},
            {
                "field": "n_domains",
                "value": int(df[inferred.domain_column].nunique(dropna=True)),
            },
            {
                "field": "used_artificial_domains",
                "value": str(bool(inferred.used_artificial_domains)),
            },
        ]
    )

    column_roles_rows = []
    for column in df.columns:
        role = "feature"
        if column == inferred.target_column:
            role = "target"
        elif column == inferred.domain_column:
            role = "domain"
        elif inferred.subject_id_column and column == inferred.subject_id_column:
            role = "subject_id"
        elif inferred.sample_id_column and column == inferred.sample_id_column:
            role = "sample_id"
        elif column in inferred.auxiliary_target_columns:
            role = "auxiliary_target_excluded"
        elif column in inferred.auxiliary_domain_columns:
            role = "auxiliary_domain_excluded"

        column_roles_rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(df[column].dtype),
                "n_missing": int(df[column].isna().sum()),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
        )

    target_distribution = (
        df[inferred.target_column]
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis("target_label")
        .reset_index(name="n_samples")
    )
    target_distribution["fraction"] = target_distribution["n_samples"] / max(len(df), 1)

    domain_distribution = (
        df.groupby(inferred.domain_column, dropna=False)[inferred.target_column]
        .agg(domain_n="size")
        .reset_index()
        .rename(columns={inferred.domain_column: "domain"})
        .sort_values("domain_n", ascending=False)
    )

    domain_target_distribution = (
        df.groupby([inferred.domain_column, inferred.target_column], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .rename(
            columns={
                inferred.domain_column: "domain",
                inferred.target_column: "target_label",
            }
        )
        .sort_values(["domain", "n_samples"], ascending=[True, False])
    )

    return {
        "dataset_overview": overview,
        "column_roles": pd.DataFrame(column_roles_rows),
        "target_distribution": target_distribution,
        "domain_distribution": domain_distribution,
        "domain_target_distribution": domain_target_distribution,
    }


def format_initial_inspection(df: pd.DataFrame, inferred: InferredColumns) -> list[str]:
    target_distribution = (
        df[inferred.target_column].astype("string").value_counts(dropna=False).to_dict()
    )
    domain_distribution = (
        df[inferred.domain_column]
        .astype("string")
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .to_dict()
    )
    preview_columns = list(df.columns[:80])
    lines = [
        f"[DATA] shape={df.shape}",
        f"[DATA] n_columns={df.shape[1]} | first_columns={preview_columns}",
        f"[DATA] inferred_target={inferred.target_column}",
        f"[DATA] inferred_domain={inferred.domain_column}",
        f"[DATA] inferred_subject_id={inferred.subject_id_column}",
        f"[DATA] inferred_sample_id={inferred.sample_id_column}",
        f"[DATA] auxiliary_target_columns_excluded={inferred.auxiliary_target_columns}",
        f"[DATA] auxiliary_domain_columns_excluded={inferred.auxiliary_domain_columns}",
        f"[DATA] target_distribution={target_distribution}",
        f"[DATA] domain_distribution={domain_distribution}",
    ]
    if inferred.used_artificial_domains:
        lines.append(
            "[DATA] domain fallback used: artificial domains generated by clustering."
        )
    return lines


def split_loso(
    df: pd.DataFrame, domain_column: str
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    for domain in sorted(df[domain_column].astype("string").dropna().unique().tolist()):
        target_mask = df[domain_column].astype("string") == domain
        target_df = df.loc[target_mask].copy()
        source_df = df.loc[~target_mask].copy()
        if source_df.empty or target_df.empty:
            continue
        yield str(domain), source_df, target_df


def ensure_subject_ids(df: pd.DataFrame, inferred: InferredColumns) -> pd.Series:
    for column in [inferred.sample_id_column, inferred.subject_id_column]:
        if column and column in df.columns:
            return df[column].astype("string")
    return pd.Series(
        [f"row_{index:06d}" for index in range(len(df))],
        index=df.index,
        name="synthetic_id",
        dtype="string",
    )


def encode_target(series: pd.Series, class_mapping: dict[str, int]) -> np.ndarray:
    encoded = series.astype("string").map(class_mapping)
    if encoded.isna().any():
        missing_labels = sorted(
            series.astype("string")[encoded.isna()].unique().tolist()
        )
        raise ValueError(f"Unmapped target labels detected: {missing_labels}")
    return encoded.to_numpy(dtype=int)


def create_artificial_domains(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
    n_domains: int,
) -> tuple[pd.DataFrame, str]:
    numeric_cols = [
        column
        for column in df.columns
        if column != target_column and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_cols:
        raise ValueError(
            "Cannot create artificial domains without numeric feature columns."
        )

    sample_cols = numeric_cols[: min(2000, len(numeric_cols))]
    matrix = df.loc[:, sample_cols].to_numpy(dtype=np.float32, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = StandardScaler(with_mean=True, with_std=True).fit_transform(matrix)
    n_components = min(10, matrix.shape[0] - 1, matrix.shape[1])
    if n_components >= 2:
        matrix = PCA(
            n_components=n_components, random_state=random_state
        ).fit_transform(matrix)

    n_clusters = max(3, min(n_domains, int(math.sqrt(len(df) / 20)) + 2))
    cluster_labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(matrix)
    artificial_domain_column = "__artificial_domain__"
    df = df.copy()
    df[artificial_domain_column] = pd.Series(
        [f"artificial_{label}" for label in cluster_labels], dtype="string"
    )
    return df, artificial_domain_column


def infer_target_column(df: pd.DataFrame) -> str:
    exact = find_exact(df.columns, TARGET_EXACT)
    if exact is not None:
        return exact

    partial = find_partial(df.columns, TARGET_PARTIAL)
    if partial is not None:
        return partial

    scored: list[tuple[int, str]] = []
    for column in df.columns:
        series = df[column]
        unique_count = series.nunique(dropna=True)
        if unique_count < 2 or unique_count > 20:
            continue
        score = 0
        lower = normalize(column)
        if any(token in lower for token in TARGET_PARTIAL):
            score += 5
        if pd.api.types.is_string_dtype(series):
            score += 2
        if pd.api.types.is_numeric_dtype(series):
            score += 1
        if unique_count <= 5:
            score += 1
        scored.append((score, str(column)))

    if not scored:
        raise ValueError("Unable to infer the target column automatically.")

    scored.sort(reverse=True)
    return scored[0][1]


def infer_domain_column(df: pd.DataFrame, excluded: set[str]) -> str | None:
    candidates = [column for column in df.columns if column not in excluded]
    exact = find_exact(candidates, DOMAIN_EXACT)
    if exact is not None:
        return exact

    partial = find_partial(candidates, DOMAIN_PARTIAL)
    if partial is not None:
        return partial

    scored: list[tuple[int, str]] = []
    for column in candidates:
        series = df[column]
        unique_count = series.nunique(dropna=True)
        if unique_count < 2 or unique_count > min(100, len(df) // 2):
            continue
        score = 0
        lower = normalize(str(column))
        if any(token in lower for token in DOMAIN_PARTIAL):
            score += 5
        if pd.api.types.is_string_dtype(series):
            score += 2
        if unique_count <= 30:
            score += 1
        scored.append((score, str(column)))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][1]


def infer_id_column(df: pd.DataFrame, excluded: set[str], preferred: str) -> str | None:
    candidates = [column for column in df.columns if column not in excluded]
    preferred_tokens = ["subject"] if preferred == "subject" else ["sample", "barcode"]
    for column in candidates:
        lower = normalize(str(column))
        if any(token in lower for token in preferred_tokens) and _is_high_uniqueness(
            df[column]
        ):
            return str(column)

    exact = find_exact(candidates, ID_EXACT)
    if exact is not None and _is_high_uniqueness(df[exact]):
        return exact

    partial = find_partial(candidates, ID_PARTIAL)
    if partial is not None and _is_high_uniqueness(df[partial]):
        return partial

    return None


def find_auxiliary_target_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    cols: list[str] = []
    for column in df.columns:
        if column == target_column:
            continue
        lower = normalize(str(column))
        if any(token in lower for token in TARGET_PARTIAL):
            cols.append(str(column))
    return cols


def find_auxiliary_domain_columns(df: pd.DataFrame, domain_column: str) -> list[str]:
    cols: list[str] = []
    for column in df.columns:
        if column == domain_column:
            continue
        lower = normalize(str(column))
        if any(token in lower for token in DOMAIN_PARTIAL):
            cols.append(str(column))
    return cols


def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    domain_column: str,
    subject_id_column: str | None,
    sample_id_column: str | None,
    auxiliary_target_columns: list[str],
    auxiliary_domain_columns: list[str],
) -> list[str]:
    excluded = {
        target_column,
        domain_column,
        *(auxiliary_target_columns),
        *(auxiliary_domain_columns),
    }
    if subject_id_column:
        excluded.add(subject_id_column)
    if sample_id_column:
        excluded.add(sample_id_column)

    feature_columns = [
        str(column)
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    return feature_columns


def encode_class_mapping(series: pd.Series) -> tuple[list[str], dict[str, int]]:
    class_labels = sorted(series.astype("string").dropna().unique().tolist())
    mapping = {label: idx for idx, label in enumerate(class_labels)}
    return class_labels, mapping


def find_exact(columns: list[str] | pd.Index, candidates: list[str]) -> str | None:
    lookup = {normalize(str(column)): str(column) for column in columns}
    for candidate in candidates:
        matched = lookup.get(normalize(candidate))
        if matched is not None:
            return matched
    return None


def find_partial(columns: list[str] | pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        candidate_norm = normalize(candidate)
        for column in columns:
            if candidate_norm in normalize(str(column)):
                return str(column)
    return None


def normalize(value: str) -> str:
    return value.strip().lower()


def _is_high_uniqueness(series: pd.Series) -> bool:
    return series.nunique(dropna=False) / max(len(series), 1) >= 0.75
