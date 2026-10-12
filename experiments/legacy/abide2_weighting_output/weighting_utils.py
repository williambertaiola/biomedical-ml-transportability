"""Transportability experiment: weighting utils."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


METHOD_ORDER = ["raw", "domain_balanced", "similarity_based"]
METHOD_COLORS = {
    "raw": "#C23B22",
    "domain_balanced": "#D9A441",
    "similarity_based": "#2E8B57",
}
DEFAULT_PERFORMANCE_CMAP = "RdYlGn"
DEFAULT_DISTANCE_CMAP = "RdYlGn_r"
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]
ID_LIKE_PATTERNS = [
    r"(^|_)id$",
    r"^id_",
    r"(^|_)(subject|sample|patient|case|record|barcode|accession)(_|$)",
]
DOMAIN_KEYWORDS = [
    "domain_id",
    "domain",
    "site",
    "hospital",
    "center",
    "centre",
    "batch",
    "source",
    "cohort",
]
TARGET_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "class",
    "disease",
    "dx",
    "status",
    "diagnosis",
    "phenotype",
    "case_control",
    "num",
    "y",
]
TARGET_PRIORITY_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "disease",
    "diagnosis",
    "phenotype",
    "case_control",
    "class",
]
NEGATIVE_LABEL_KEYWORDS = [
    "control",
    "healthy",
    "typical",
    "td",
    "negative",
    "non_case",
    "no_disease",
]
POSITIVE_LABEL_KEYWORDS = [
    "asd",
    "autism",
    "case",
    "patient",
    "disease",
    "positive",
    "affected",
]


@dataclass
class DatasetSchema:
    target_col: str
    domain_col: str
    id_cols: list[str]
    leakage_exclude_cols: list[str]
    numeric_candidates: list[str]
    categorical_candidates: list[str]
    feature_candidates: list[str]
    target_mapping: dict[str, Any]
    target_note: str
    target_binarized: bool
    domain_standardized_name: str = "domain_id"


@dataclass
class ColumnSpec:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    dropped_cols: list[dict[str, str]]
    warnings: list[str]


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def init_logger(log_path: str | Path) -> logging.Logger:
    log_file = Path(log_path)
    ensure_directory(log_file.parent)
    logger = logging.getLogger(f"weighting::{log_file.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(input_path, sep=None, engine="python")
    raise ValueError(f"Unsupported input format: {input_path}")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return cleaned.lower() or "value"


def is_integer_like(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty or not pd.api.types.is_numeric_dtype(non_missing):
        return False
    values = non_missing.to_numpy(dtype=float)
    return bool(
        np.all(np.isfinite(values)) and np.all(np.isclose(values, np.round(values)))
    )


def is_id_like_name(column_name: str) -> bool:
    sanitized = sanitize_name(column_name)
    return any(
        re.search(pattern, sanitized, flags=re.IGNORECASE)
        for pattern in ID_LIKE_PATTERNS
    )


def is_domain_like_name(column_name: str) -> bool:
    sanitized = sanitize_name(column_name)
    return any(
        re.search(rf"(^|_){re.escape(keyword)}(_|$)", sanitized, flags=re.IGNORECASE)
        for keyword in DOMAIN_KEYWORDS
    )


def is_target_like_name(column_name: str) -> bool:
    sanitized = sanitize_name(column_name)
    return any(
        re.search(rf"(^|_){re.escape(keyword)}(_|$)", sanitized, flags=re.IGNORECASE)
        for keyword in TARGET_KEYWORDS
    )


def column_name_score(column_name: str, keywords: list[str]) -> int:
    sanitized = sanitize_name(column_name)
    score = 0
    for keyword in keywords:
        if sanitized == keyword:
            score += 120
        elif re.search(rf"(^|_){re.escape(keyword)}(_|$)", sanitized):
            score += 80
        elif keyword in sanitized:
            score += 35
    return score


def has_matching_label_or_code_partner(df: pd.DataFrame, column_name: str) -> bool:
    sanitized = sanitize_name(column_name)
    candidates = {
        re.sub(r"(_code|_label)$", "", sanitized),
        sanitized.replace("_code", ""),
        sanitized.replace("_label", ""),
    }
    sanitized_columns = {sanitize_name(name): name for name in df.columns}
    for candidate in candidates:
        if not candidate:
            continue
        if (
            f"{candidate}_code" in sanitized_columns
            and f"{candidate}_label" in sanitized_columns
        ):
            return True
        if f"{candidate}_code" in sanitized_columns and candidate in sanitized_columns:
            return True
        if f"{candidate}_label" in sanitized_columns and candidate in sanitized_columns:
            return True
    return False


def find_partner_label_column(df: pd.DataFrame, column_name: str) -> str | None:
    sanitized = sanitize_name(column_name)
    candidates = [
        re.sub(r"_code$", "", sanitized),
        re.sub(r"_label$", "", sanitized),
        sanitized.replace("_code", ""),
        sanitized.replace("_label", ""),
    ]
    sanitized_columns = {sanitize_name(name): name for name in df.columns}
    for candidate in candidates:
        if not candidate:
            continue
        for option in [
            candidate,
            f"{candidate}_label",
            f"{candidate}_name",
            f"{candidate}_group",
            f"{candidate}_level",
        ]:
            if option in sanitized_columns and sanitized_columns[option] != column_name:
                partner = sanitized_columns[option]
                if pd.api.types.is_object_dtype(
                    df[partner]
                ) or pd.api.types.is_string_dtype(df[partner]):
                    return partner
    return None


def infer_binary_mapping_from_partner(
    series: pd.Series,
    *,
    full_df: pd.DataFrame | None,
    column_name: str | None,
) -> tuple[Any, Any] | None:
    if full_df is None or column_name is None:
        return None
    partner = find_partner_label_column(full_df, column_name)
    if partner is None:
        return None

    temp_df = pd.DataFrame({"code": series, "label": full_df[partner]})
    temp_df = temp_df.dropna(subset=["code", "label"]).copy()
    if temp_df.empty:
        return None

    label_map = (
        temp_df.groupby("code", dropna=False)["label"]
        .agg(
            lambda values: (
                pd.Series(values).mode().iloc[0]
                if not pd.Series(values).mode().empty
                else pd.Series(values).iloc[0]
            )
        )
        .to_dict()
    )
    if len(label_map) != 2:
        return None

    negative_label = None
    positive_label = None
    for code_value, label_value in label_map.items():
        label_str = sanitize_name(str(label_value))
        if any(keyword in label_str for keyword in NEGATIVE_LABEL_KEYWORDS):
            negative_label = code_value
        if any(keyword in label_str for keyword in POSITIVE_LABEL_KEYWORDS):
            positive_label = code_value

    if (
        negative_label is not None
        and positive_label is not None
        and negative_label != positive_label
    ):
        return negative_label, positive_label
    return None


def detect_domain_column(df: pd.DataFrame) -> str:
    n_rows = len(df)
    candidates: list[tuple[float, str]] = []
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count < 2:
            continue
        score = float(column_name_score(column, DOMAIN_KEYWORDS))
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            score += 15.0
        if 2 <= unique_count <= max(2, min(100, int(n_rows * 0.75))):
            score += 12.0
        if unique_count >= max(2, int(n_rows * 0.95)):
            score -= 75.0
        if is_id_like_name(column):
            score -= 25.0
        candidates.append((score, column))
    if not candidates:
        raise ValueError("Unable to detect a domain/site column automatically.")
    best_score, best_column = max(
        candidates, key=lambda item: (item[0], -df.columns.get_loc(item[1]))
    )
    if best_score < 20:
        raise ValueError(
            "Unable to detect a domain/site column with enough confidence. "
            f"Best candidate was '{best_column}' (score={best_score:.1f})."
        )
    return best_column


def detect_target_column(df: pd.DataFrame, domain_col: str) -> str:
    n_rows = len(df)
    candidates: list[tuple[float, str]] = []
    for column in df.columns:
        if column == domain_col:
            continue
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count < 2:
            continue
        score = float(column_name_score(column, TARGET_KEYWORDS))
        score += 1.5 * float(column_name_score(column, TARGET_PRIORITY_KEYWORDS))
        if unique_count == 2:
            score += 60.0
        elif 3 <= unique_count <= 5 and is_integer_like(series):
            score += 40.0
        elif 6 <= unique_count <= 10:
            score += 5.0
        else:
            score -= 20.0
        if has_matching_label_or_code_partner(df, column):
            score += 35.0
        sanitized = sanitize_name(column)
        if sanitized.endswith("_code"):
            score += 10.0
        if "status" in sanitized and "disease" not in sanitized:
            score -= 30.0
        if unique_count >= max(10, int(n_rows * 0.5)):
            score -= 60.0
        if is_id_like_name(column):
            score -= 40.0
        if is_domain_like_name(column):
            score -= 35.0
        candidates.append((score, column))
    if not candidates:
        raise ValueError("Unable to detect a target column automatically.")
    best_score, best_column = max(
        candidates, key=lambda item: (item[0], -df.columns.get_loc(item[1]))
    )
    if best_score < 20:
        raise ValueError(
            "Unable to detect a target column with enough confidence. "
            f"Best candidate was '{best_column}' (score={best_score:.1f})."
        )
    return best_column


def binarize_target(
    series: pd.Series,
    *,
    full_df: pd.DataFrame | None = None,
    column_name: str | None = None,
) -> tuple[pd.Series, dict[str, Any], str, bool]:
    non_missing = series.dropna()
    if non_missing.empty:
        raise ValueError("Target column contains only missing values.")

    unique_values = sorted(
        pd.Series(non_missing.unique()).tolist(), key=lambda value: str(value)
    )
    unique_set = set(unique_values)

    if unique_set == {0, 1}:
        return (
            series.astype("Int64"),
            {"negative": 0, "positive": 1},
            "Target already binary (0/1).",
            False,
        )

    if len(unique_values) == 2:
        partner_mapping = infer_binary_mapping_from_partner(
            series,
            full_df=full_df,
            column_name=column_name,
        )
        if partner_mapping is not None:
            negative_label, positive_label = partner_mapping
            mapping_note = " partner-label-aware mapping"
        else:
            negative_label, positive_label = unique_values[0], unique_values[1]
            mapping_note = " sorted-label mapping"
        mapped = series.map({negative_label: 0, positive_label: 1})
        if mapped.isna().any() and series.notna().any():
            raise ValueError("Unable to map binary target labels consistently.")
        note = (
            "Target mapped to binary using"
            f"{mapping_note}: "
            f"{negative_label!r} -> 0, {positive_label!r} -> 1."
        )
        return (
            mapped.astype("Int64"),
            {"negative": negative_label, "positive": positive_label},
            note,
            True,
        )

    if is_integer_like(non_missing):
        numeric_values = sorted(
            int(value) for value in pd.Series(non_missing).astype(int).unique().tolist()
        )
        if 0 in numeric_values and all(value >= 0 for value in numeric_values):
            positive_values = [value for value in numeric_values if value > 0]
            if positive_values and max(numeric_values) <= 4:
                mapped = series.astype("Int64").map(
                    lambda value: pd.NA if pd.isna(value) else int(value > 0)
                )
                note = (
                    "Target binarized with the thesis rule 0 vs 1-4: "
                    "0 = negative, values > 0 = positive."
                )
                mapping = {"negative": [0], "positive": positive_values}
                return mapped.astype("Int64"), mapping, note, True

    raise ValueError(
        "The detected target column is not binary and could not be safely converted with the 0-vs-1..4 rule."
    )


def detect_id_columns(df: pd.DataFrame, target_col: str, domain_col: str) -> list[str]:
    detected: list[str] = []
    for column in df.columns:
        if column in {target_col, domain_col}:
            continue
        if is_id_like_name(column):
            detected.append(column)
            continue
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count >= max(2, int(len(df) * 0.98)):
            detected.append(column)
    return sorted(set(detected))


def detect_leakage_prone_columns(
    df: pd.DataFrame, target_col: str, domain_col: str, id_cols: list[str]
) -> list[str]:
    excluded: list[str] = []
    protected = {target_col, domain_col, *id_cols}
    for column in df.columns:
        if column in protected:
            continue
        if is_target_like_name(column) or is_domain_like_name(column):
            excluded.append(column)
    return sorted(set(excluded))


def infer_feature_types(
    df: pd.DataFrame,
    *,
    target_col: str,
    domain_col: str,
    id_cols: list[str],
    leakage_exclude_cols: list[str],
    low_cardinality_threshold: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    excluded = {target_col, domain_col, *id_cols, *leakage_exclude_cols}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    feature_cols: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue
        series = df[column]
        if series.dropna().empty:
            continue
        feature_cols.append(column)
        if pd.api.types.is_bool_dtype(series):
            categorical_cols.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            unique_count = int(series.dropna().nunique())
            if is_integer_like(series) and unique_count <= low_cardinality_threshold:
                categorical_cols.append(column)
            else:
                numeric_cols.append(column)
            continue
        categorical_cols.append(column)

    return numeric_cols, categorical_cols, feature_cols


def detect_dataset_schema(
    df: pd.DataFrame,
    *,
    low_cardinality_threshold: int = 10,
) -> DatasetSchema:
    domain_col = detect_domain_column(df)
    target_col = detect_target_column(df, domain_col=domain_col)
    id_cols = detect_id_columns(df, target_col=target_col, domain_col=domain_col)
    leakage_exclude_cols = detect_leakage_prone_columns(
        df,
        target_col=target_col,
        domain_col=domain_col,
        id_cols=id_cols,
    )
    numeric_cols, categorical_cols, feature_cols = infer_feature_types(
        df,
        target_col=target_col,
        domain_col=domain_col,
        id_cols=id_cols,
        leakage_exclude_cols=leakage_exclude_cols,
        low_cardinality_threshold=low_cardinality_threshold,
    )
    binary_target, target_mapping, target_note, target_binarized = binarize_target(
        df[target_col],
        full_df=df,
        column_name=target_col,
    )
    prepared_df = df.copy()
    prepared_df[target_col] = binary_target
    if prepared_df[domain_col].nunique(dropna=True) < 2:
        raise ValueError("LODO requires at least 2 distinct domains/sites.")

    return DatasetSchema(
        target_col=target_col,
        domain_col=domain_col,
        id_cols=id_cols,
        leakage_exclude_cols=leakage_exclude_cols,
        numeric_candidates=numeric_cols,
        categorical_candidates=categorical_cols,
        feature_candidates=feature_cols,
        target_mapping=target_mapping,
        target_note=target_note,
        target_binarized=target_binarized,
    )


def prepare_modeling_dataframe(
    df: pd.DataFrame, schema: DatasetSchema, logger: logging.Logger
) -> pd.DataFrame:
    prepared = df.copy()
    binary_target, _, _, _ = binarize_target(
        prepared[schema.target_col],
        full_df=prepared,
        column_name=schema.target_col,
    )
    prepared[schema.target_col] = binary_target
    prepared[schema.domain_col] = prepared[schema.domain_col].astype("string")

    for column in schema.numeric_candidates:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in schema.categorical_candidates:
        prepared[column] = prepared[column].astype(object)
        prepared.loc[prepared[column].isna(), column] = np.nan

    before_rows = len(prepared)
    prepared = prepared.dropna(
        subset=[schema.target_col, schema.domain_col]
    ).reset_index(drop=True)
    dropped_rows = before_rows - len(prepared)
    if dropped_rows > 0:
        logger.warning(
            "Dropped %s rows with missing target or domain identifiers.", dropped_rows
        )
    prepared[schema.target_col] = prepared[schema.target_col].astype(int)

    domain_count = int(prepared[schema.domain_col].nunique(dropna=True))
    if domain_count < 2:
        raise ValueError(
            "The cleaned dataset contains fewer than 2 domains, so LODO cannot run."
        )

    return prepared


def prepare_column_spec(train_df: pd.DataFrame, schema: DatasetSchema) -> ColumnSpec:
    dropped_cols: list[dict[str, str]] = []
    warnings_out: list[str] = []
    feature_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    numeric_lookup = set(schema.numeric_candidates)
    categorical_lookup = set(schema.categorical_candidates)

    for column in schema.feature_candidates:
        series = train_df[column]
        if series.isna().all():
            dropped_cols.append(
                {"column_name": column, "reason": "all_missing_in_training"}
            )
            continue
        if int(series.dropna().nunique()) <= 1:
            dropped_cols.append(
                {"column_name": column, "reason": "constant_in_training"}
            )
            continue
        feature_cols.append(column)
        if column in numeric_lookup:
            numeric_cols.append(column)
        elif column in categorical_lookup:
            categorical_cols.append(column)
        else:
            categorical_cols.append(column)

    if not feature_cols:
        raise ValueError(
            "No usable features remain after training-only quality checks."
        )
    if not numeric_cols:
        warnings_out.append("no_numeric_features_available")
    if not categorical_cols:
        warnings_out.append("no_categorical_features_available")

    return ColumnSpec(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped_cols,
        warnings=warnings_out,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(spec: ColumnSpec) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                spec.numeric_cols,
            )
        )
    if spec.categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                spec.categorical_cols,
            )
        )
    return ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )


def transform_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: ColumnSpec,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(spec)
    X_train = preprocessor.fit_transform(train_df[spec.feature_cols])
    X_test = preprocessor.transform(test_df[spec.feature_cols])
    return (
        preprocessor,
        np.asarray(X_train, dtype=float),
        np.asarray(X_test, dtype=float),
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = math.nan
    return metrics


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    random_seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        random_state=random_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def compute_domain_balanced_weights(
    train_domains: pd.Series,
) -> tuple[np.ndarray, dict[str, float]]:
    counts = train_domains.astype(str).value_counts().to_dict()
    n_samples = len(train_domains)
    n_domains = len(counts)
    domain_weights = {
        domain: float(n_samples / (n_domains * count))
        for domain, count in counts.items()
    }
    sample_weights = np.asarray(
        [domain_weights[str(domain)] for domain in train_domains], dtype=float
    )
    return sample_weights, domain_weights


def compute_similarity_domain_distances(
    X_train: np.ndarray,
    train_domains: pd.Series,
    X_target: np.ndarray,
) -> dict[str, float]:
    target_centroid = np.mean(X_target, axis=0)
    distances: dict[str, float] = {}
    domain_series = train_domains.astype(str)
    for domain in sorted(domain_series.unique().tolist()):
        mask = domain_series == domain
        source_centroid = np.mean(X_train[mask.to_numpy()], axis=0)
        distances[domain] = float(
            np.sqrt(np.mean((source_centroid - target_centroid) ** 2))
        )
    return distances


def compute_similarity_weights(
    X_train: np.ndarray,
    train_domains: pd.Series,
    X_target: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, float], float]:
    distances = compute_similarity_domain_distances(X_train, train_domains, X_target)
    positive_distances = [value for value in distances.values() if value > 0]
    scale = float(np.median(positive_distances)) if positive_distances else 1.0
    scale = max(scale, 1e-8)

    similarity_scores = {
        domain: float(np.exp(-alpha * distance / scale))
        for domain, distance in distances.items()
    }

    counts = train_domains.astype(str).value_counts().to_dict()
    n_samples = len(train_domains)
    denominator = sum(counts[domain] * similarity_scores[domain] for domain in counts)
    denominator = max(float(denominator), 1e-12)
    domain_weights = {
        domain: float(similarity_scores[domain] * n_samples / denominator)
        for domain in counts
    }
    sample_weights = np.asarray(
        [domain_weights[str(domain)] for domain in train_domains], dtype=float
    )
    return sample_weights, domain_weights, distances, scale


def build_domain_weight_rows(
    *,
    holdout_domain: str,
    method: str,
    train_domains: pd.Series,
    domain_weights: dict[str, float],
    distances: dict[str, float] | None = None,
    similarity_scale: float | None = None,
) -> list[dict[str, Any]]:
    counts = train_domains.astype(str).value_counts().to_dict()
    rows: list[dict[str, Any]] = []
    for source_domain in sorted(counts):
        rows.append(
            {
                "holdout_domain": holdout_domain,
                "method": method,
                "source_domain": source_domain,
                "n_source_samples": int(counts[source_domain]),
                "domain_distance_to_holdout": math.nan
                if distances is None
                else float(distances[source_domain]),
                "similarity_scale": math.nan
                if similarity_scale is None
                else float(similarity_scale),
                "domain_sample_weight": float(domain_weights[source_domain]),
                "domain_total_weight": float(
                    domain_weights[source_domain] * counts[source_domain]
                ),
            }
        )
    return rows


def aggregate_metric_summary(lodo_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_df = lodo_results[lodo_results["method"] == method]
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(method_df[metric], errors="coerce").dropna()
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()) if not values.empty else math.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "median": float(values.median()) if not values.empty else math.nan,
                    "min": float(values.min()) if not values.empty else math.nan,
                    "max": float(values.max()) if not values.empty else math.nan,
                    "n_folds": int(values.shape[0]),
                }
            )
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df["rank_by_mean_desc"] = summary_df.groupby("metric")["mean"].rank(
            ascending=False,
            method="dense",
        )
    return summary_df


def frame_to_text_block(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.head(max_rows) if max_rows is not None else df
    return "```text\n" + view.to_string(index=False) + "\n```"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_colormap(cmap_name: str):
    return matplotlib.colormaps[cmap_name]


def compute_gradient_colors(values: np.ndarray, *, cmap_name: str) -> list[Any]:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return ["#808080"] * len(values)
    vmin = float(np.nanmin(values[finite_mask]))
    vmax = float(np.nanmax(values[finite_mask]))
    if np.isclose(vmin, vmax):
        normalized = np.full(values.shape, 0.85)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    cmap = get_colormap(cmap_name)
    colors: list[Any] = []
    for value, is_finite in zip(normalized, finite_mask):
        if not is_finite:
            colors.append("#808080")
        else:
            colors.append(cmap(float(value)))
    return colors


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, fmt: str = "{:.3f}") -> None:
    if matrix.size == 0:
        return
    valid_values = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(valid_values)) if valid_values.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                text = "NA"
                color = "black"
            else:
                text = fmt.format(value)
                color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )


def plot_domain_sample_distribution(
    domain_distribution_df: pd.DataFrame, output_path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(domain_distribution_df))
    colors = compute_gradient_colors(
        domain_distribution_df["n_samples"].to_numpy(dtype=float),
        cmap_name=DEFAULT_PERFORMANCE_CMAP,
    )
    bars = ax.bar(x, domain_distribution_df["n_samples"].to_numpy(), color=colors)
    ax.set_title("Sample Distribution by Domain")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of samples")
    ax.set_xticks(x)
    ax.set_xticklabels(
        domain_distribution_df["domain"].tolist(), rotation=25, ha="right"
    )
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1.0, height * 0.01),
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_path)


def plot_metric_boxplot(lodo_results: pd.DataFrame, output_path: str | Path) -> None:
    if lodo_results.empty:
        return
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_COLUMNS):
        data = [
            pd.to_numeric(
                lodo_results.loc[lodo_results["method"] == method, metric],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
            for method in METHOD_ORDER
        ]
        box = ax.boxplot(data, labels=METHOD_ORDER, patch_artist=True)
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        for position, method in enumerate(METHOD_ORDER, start=1):
            values = data[position - 1]
            if values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.04, size=values.size)
            ax.scatter(
                np.full(values.shape, position, dtype=float) + jitter,
                values,
                s=32,
                alpha=0.85,
                color=METHOD_COLORS[method],
                edgecolors="black",
                linewidths=0.3,
            )
        ax.set_title(metric)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("LODO OOD Performance by Weighting Strategy", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_mean_bars(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    if summary_df.empty:
        return
    pivot_mean = summary_df.pivot(
        index="metric", columns="method", values="mean"
    ).reindex(METRIC_COLUMNS)
    pivot_std = summary_df.pivot(
        index="metric", columns="method", values="std"
    ).reindex(METRIC_COLUMNS)

    x = np.arange(len(METRIC_COLUMNS), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(14, 6))
    center_offset = (len(METHOD_ORDER) - 1) / 2.0

    for idx, method in enumerate(METHOD_ORDER):
        ax.bar(
            x + (idx - center_offset) * width,
            pivot_mean.get(method, pd.Series(np.nan, index=METRIC_COLUMNS)).to_numpy(
                dtype=float
            ),
            yerr=pivot_std.get(
                method, pd.Series(np.nan, index=METRIC_COLUMNS)
            ).to_numpy(dtype=float),
            width=width,
            label=method,
            color=METHOD_COLORS[method],
            alpha=0.85,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLUMNS, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean OOD score")
    ax.set_title("Mean ± SD OOD Performance by Method")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, output_path)


def plot_primary_metric_by_domain(
    lodo_results: pd.DataFrame,
    output_path: str | Path,
    *,
    primary_metric: str,
) -> None:
    if lodo_results.empty or primary_metric not in lodo_results.columns:
        return
    pivot = (
        lodo_results.pivot(
            index="holdout_domain", columns="method", values=primary_metric
        )
        .reindex(columns=METHOD_ORDER)
        .sort_index()
    )
    x = np.arange(len(pivot.index), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, 3 + len(pivot.index) * 1.5), 5.5))
    center_offset = (len(METHOD_ORDER) - 1) / 2.0

    for idx, method in enumerate(METHOD_ORDER):
        values = pivot.get(method, pd.Series(np.nan, index=pivot.index)).to_numpy(
            dtype=float
        )
        bars = ax.bar(
            x + (idx - center_offset) * width,
            values,
            width=width,
            label=method,
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
        for bar in bars:
            height = float(bar.get_height())
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.012,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(f"OOD {primary_metric} by Holdout Domain")
    ax.set_xlabel("Holdout domain")
    ax.set_ylabel(primary_metric)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, output_path)


def plot_similarity_weights(
    domain_weights_df: pd.DataFrame, output_path: str | Path
) -> None:
    plot_df = domain_weights_df[
        domain_weights_df["method"] == "similarity_based"
    ].copy()
    if plot_df.empty:
        return
    holdouts = sorted(plot_df["holdout_domain"].astype(str).unique().tolist())
    n_cols = 2
    n_rows = int(math.ceil(len(holdouts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(5, 4.2 * n_rows)))
    axes = np.atleast_1d(axes).flatten()

    for ax, holdout in zip(axes, holdouts):
        sub = plot_df[plot_df["holdout_domain"].astype(str) == holdout].sort_values(
            "source_domain"
        )
        x = np.arange(len(sub), dtype=float)
        colors = compute_gradient_colors(
            sub["domain_sample_weight"].to_numpy(dtype=float),
            cmap_name=DEFAULT_PERFORMANCE_CMAP,
        )
        bars = ax.bar(
            x, sub["domain_sample_weight"].to_numpy(dtype=float), color=colors
        )
        ax.set_title(f"Similarity Weights | holdout={holdout}")
        ax.set_xlabel("Source domain")
        ax.set_ylabel("Sample weight")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["source_domain"].tolist(), rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for bar, distance in zip(
            bars, sub["domain_distance_to_holdout"].to_numpy(dtype=float)
        ):
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"w={height:.2f}\nd={distance:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes[len(holdouts) :]:
        ax.axis("off")

    fig.suptitle("Similarity-Based Domain Weights Across LODO Folds", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_from_matrix(
    matrix_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    cmap: str = DEFAULT_PERFORMANCE_CMAP,
) -> None:
    if matrix_df.empty:
        return
    values = matrix_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * matrix_df.shape[1]), max(5, 0.9 * matrix_df.shape[0]))
    )
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index.tolist())
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=90)
    annotate_matrix(ax, values)
    save_figure(fig, output_path)


def compute_domain_distance_matrix(
    df: pd.DataFrame, schema: DatasetSchema
) -> pd.DataFrame:
    spec = ColumnSpec(
        feature_cols=schema.feature_candidates,
        numeric_cols=schema.numeric_candidates,
        categorical_cols=schema.categorical_candidates,
        dropped_cols=[],
        warnings=[],
    )
    preprocessor = build_preprocessor(spec)
    X_all = np.asarray(preprocessor.fit_transform(df[spec.feature_cols]), dtype=float)
    domains = df[schema.domain_col].astype(str)
    unique_domains = sorted(domains.unique().tolist())
    centroids = {
        domain: np.mean(X_all[(domains == domain).to_numpy()], axis=0)
        for domain in unique_domains
    }
    matrix = np.zeros((len(unique_domains), len(unique_domains)), dtype=float)
    for row_idx, source_domain in enumerate(unique_domains):
        for col_idx, target_domain in enumerate(unique_domains):
            matrix[row_idx, col_idx] = float(
                np.sqrt(
                    np.mean((centroids[source_domain] - centroids[target_domain]) ** 2)
                )
            )
    return pd.DataFrame(matrix, index=unique_domains, columns=unique_domains)


def compute_pairwise_transfer(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    random_seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    domains = sorted(df[schema.domain_col].astype(str).unique().tolist())
    for source_domain in domains:
        train_df = df[df[schema.domain_col].astype(str) == source_domain].reset_index(
            drop=True
        )
        if train_df[schema.target_col].nunique() < 2:
            logger.warning(
                "Skipping source domain '%s' in pairwise transfer because only one class is present.",
                source_domain,
            )
            continue

        spec = prepare_column_spec(train_df, schema)
        preprocessor = build_preprocessor(spec)
        X_train = np.asarray(
            preprocessor.fit_transform(train_df[spec.feature_cols]), dtype=float
        )
        y_train = train_df[schema.target_col].to_numpy(dtype=int)
        model = fit_logistic_regression(
            X_train,
            y_train,
            sample_weight=np.ones(len(train_df), dtype=float),
            random_seed=random_seed,
        )

        for target_domain in domains:
            if target_domain == source_domain:
                continue
            test_df = df[
                df[schema.domain_col].astype(str) == target_domain
            ].reset_index(drop=True)
            X_test = np.asarray(
                preprocessor.transform(test_df[spec.feature_cols]), dtype=float
            )
            y_test = test_df[schema.target_col].to_numpy(dtype=int)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_prob)
            results.append(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    **metrics,
                }
            )
    return pd.DataFrame(results)


def write_report(
    output_path: str | Path,
    *,
    audit_payload: dict[str, Any],
    schema: DatasetSchema,
    lodo_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    domain_weights_df: pd.DataFrame,
    pairwise_transfer_df: pd.DataFrame,
    domain_distance_matrix: pd.DataFrame,
    primary_metric: str,
    model_name: str,
    similarity_alpha: float,
) -> None:
    pivot_summary = summary_df.pivot(
        index="metric", columns="method", values=["mean", "std"]
    )
    primary_summary = summary_df[summary_df["metric"] == primary_metric].sort_values(
        "mean", ascending=False
    )
    best_method = (
        primary_summary.iloc[0]["method"] if not primary_summary.empty else "n/a"
    )

    performance_by_domain = (
        lodo_results.pivot(
            index="holdout_domain", columns="method", values=primary_metric
        )
        .reindex(columns=METHOD_ORDER)
        .reset_index()
    )
    single_class_holdouts = (
        lodo_results.groupby("holdout_domain", dropna=False)["positive_rate_test"]
        .first()
        .loc[lambda series: (series == 0.0) | (series == 1.0)]
    )
    raw_std = summary_df.loc[
        (summary_df["metric"] == primary_metric) & (summary_df["method"] == "raw"),
        "std",
    ]
    sim_std = summary_df.loc[
        (summary_df["metric"] == primary_metric)
        & (summary_df["method"] == "similarity_based"),
        "std",
    ]
    variability_comment = "not available"
    if (
        not raw_std.empty
        and not sim_std.empty
        and pd.notna(raw_std.iloc[0])
        and pd.notna(sim_std.iloc[0])
    ):
        if sim_std.iloc[0] < raw_std.iloc[0]:
            variability_comment = (
                f"Similarity-based weighting reduced the cross-domain SD of {primary_metric} "
                f"from {raw_std.iloc[0]:.4f} to {sim_std.iloc[0]:.4f}."
            )
        elif sim_std.iloc[0] > raw_std.iloc[0]:
            variability_comment = (
                f"Similarity-based weighting improved mean {primary_metric} but increased its cross-domain SD "
                f"from {raw_std.iloc[0]:.4f} to {sim_std.iloc[0]:.4f}."
            )
        else:
            variability_comment = (
                f"Similarity-based weighting matched the raw SD of {primary_metric} "
                f"({raw_std.iloc[0]:.4f})."
            )

    method_counts = (
        domain_weights_df.groupby(["method", "holdout_domain"], dropna=False)[
            "source_domain"
        ]
        .count()
        .reset_index(name="n_weighted_source_domains")
    )

    lines = [
        "# Weighting / Reweighting Pipeline Report",
        "",
        "## Overview",
        "",
        "- Objective: evaluate whether weighting can improve cross-site transportability under strict Leave-One-Domain-Out validation.",
        f"- Input dataset: `{audit_payload['input_path']}`",
        f"- Main model: `{model_name}`",
        f"- Holdout strategy: `LODO` across `{audit_payload['n_domains']}` domains.",
        f"- Detected target column: `{schema.target_col}`",
        f"- Detected domain column: `{schema.domain_col}` (internally treated as the domain identifier role `domain_id`)",
        f"- Target mapping: `{schema.target_mapping}`",
        f"- Target preprocessing note: {schema.target_note}",
        "",
        "## Data Audit",
        "",
        f"- Dataset shape after dropping rows with missing target/domain: `{audit_payload['clean_shape']}`",
        f"- Numeric features used: `{len(schema.numeric_candidates)}`",
        f"- Categorical features used: `{len(schema.categorical_candidates)}`",
        f"- ID-like columns excluded: `{schema.id_cols}`",
        f"- Leakage-prone auxiliary columns excluded: `{schema.leakage_exclude_cols}`",
        "",
        "Target distribution:",
        frame_to_text_block(pd.DataFrame(audit_payload["target_distribution"])),
        "",
        "Domain distribution:",
        frame_to_text_block(pd.DataFrame(audit_payload["domain_distribution"])),
        "",
        "## Methodology",
        "",
        "Preprocessing was re-fit inside each outer fold using only source domains:",
        "- Median imputation for numeric features",
        "- Most-frequent imputation for categorical features",
        "- Standard scaling for numeric features",
        "- One-hot encoding for categorical features",
        "- Training-only removal of constant or all-missing columns",
        "",
        "Compared weighting strategies:",
        "",
        "```text",
        "RAW:",
        "  w_i = 1",
        "",
        "DOMAIN-BALANCED:",
        "  if sample i belongs to source domain d",
        "  w_i = N_train / (K * n_d)",
        "  where n_d is the size of source domain d and K is the number of source domains",
        "",
        "SIMILARITY-BASED:",
        "  1. Fit preprocessing on source domains only",
        "  2. Transform source X and unlabeled target X",
        "  3. Compute a centroid distance per source domain:",
        "       distance_d = sqrt(mean((mu_d - mu_target)^2))",
        "  4. Convert distance into similarity:",
        f"       similarity_d = exp(-{similarity_alpha:.2f} * distance_d / median_positive_distance)",
        "  5. Normalize to mean sample weight = 1:",
        "       w_i = similarity_d * N_train / sum_d'(n_d' * similarity_d')",
        "```",
        "",
        "Important methodological note:",
        "- Similarity-based weights use only unlabeled feature distributions from the holdout target domain.",
        "- No target labels from the holdout domain are used to estimate weights, tune preprocessing, or fit the classifier.",
        f"- Holdout domains with a single test class: `{len(single_class_holdouts)}`"
        + (
            f" ({', '.join(single_class_holdouts.index.astype(str).tolist())})"
            if len(single_class_holdouts) > 0
            else ""
        )
        + ". In these folds ROC AUC is undefined and reported as `NaN`.",
        "",
        "## Main Results",
        "",
        f"- Primary ranking metric: `{primary_metric}`",
        f"- Best average OOD method on `{primary_metric}`: `{best_method}`",
        f"- Variability comment: {variability_comment}",
        "",
        "Summary across folds:",
        frame_to_text_block(summary_df.round(4)),
        "",
        f"Per-domain `{primary_metric}` comparison:",
        frame_to_text_block(performance_by_domain.round(4)),
        "",
        "Weight diagnostics summary:",
        frame_to_text_block(method_counts),
        "",
        "Pairwise single-source transfer diagnostic (raw training only):",
        frame_to_text_block(pairwise_transfer_df.round(4)),
        "",
        "Pairwise domain-distance matrix (descriptive, unlabeled, not used for model fitting):",
        frame_to_text_block(domain_distance_matrix.round(4)),
        "",
        "## Interpretation",
        "",
        "- `raw` is the strict no-weighting reference.",
        "- `domain_balanced` tests whether forcing equal contribution from each source site is sufficient.",
        "- `similarity_based` tests whether emphasizing source domains whose covariate distribution is closer to the holdout domain improves OOD transportability.",
        "",
        "## Limitations",
        "",
        "- The main benchmark uses Logistic Regression only, to keep the weighting effect easy to interpret and fully reproducible.",
        "- The similarity metric is centroid-based in transformed feature space, so it captures mean/proportion shifts more than higher-order distributional differences.",
        "- UCI Heart Disease contains only four domains, so estimates of variability across domains remain informative but sample-limited.",
    ]

    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_source_files(output_dir: str | Path, source_paths: list[str | Path]) -> None:
    target_dir = ensure_directory(output_dir)
    for source_path in source_paths:
        source = Path(source_path)
        if source.exists():
            shutil.copy2(source, target_dir / source.name)


def schema_to_dataframe(schema: DatasetSchema) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in schema.numeric_candidates:
        rows.append(
            {"column_name": column, "role": "feature", "feature_type": "numeric"}
        )
    for column in schema.categorical_candidates:
        rows.append(
            {"column_name": column, "role": "feature", "feature_type": "categorical"}
        )
    for column in schema.id_cols:
        rows.append(
            {"column_name": column, "role": "excluded_id_like", "feature_type": "n/a"}
        )
    for column in schema.leakage_exclude_cols:
        rows.append(
            {
                "column_name": column,
                "role": "excluded_leakage_prone",
                "feature_type": "n/a",
            }
        )
    rows.append(
        {"column_name": schema.target_col, "role": "target", "feature_type": "n/a"}
    )
    rows.append(
        {"column_name": schema.domain_col, "role": "domain", "feature_type": "n/a"}
    )
    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["role", "column_name"])
        .reset_index(drop=True)
    )


def audit_payload_from_dataframe(
    original_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    schema: DatasetSchema,
    input_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_df = (
        original_df.isna()
        .sum()
        .rename("n_missing")
        .reset_index()
        .rename(columns={"index": "column_name"})
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )
    target_distribution_df = (
        clean_df[schema.target_col]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("target_value")
        .reset_index(name="n_samples")
    )
    domain_distribution_df = (
        clean_df[schema.domain_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("domain")
        .reset_index(name="n_samples")
    )
    payload = {
        "input_path": str(Path(input_path).resolve()),
        "original_shape": list(original_df.shape),
        "clean_shape": list(clean_df.shape),
        "columns": original_df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in original_df.dtypes.items()},
        "n_missing_total": int(original_df.isna().sum().sum()),
        "n_domains": int(clean_df[schema.domain_col].nunique(dropna=True)),
        "n_target_classes": int(clean_df[schema.target_col].nunique(dropna=True)),
        "target_distribution": target_distribution_df.to_dict(orient="records"),
        "domain_distribution": domain_distribution_df.to_dict(orient="records"),
        "schema": asdict(schema),
    }
    return payload, missing_df, target_distribution_df, domain_distribution_df
