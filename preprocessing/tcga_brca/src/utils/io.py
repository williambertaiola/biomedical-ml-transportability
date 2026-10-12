"""Preprocessing: io."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

MISSING_TOKENS = {
    "",
    " ",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "not reported",
    "not available",
    "not applicable",
    "unknown",
    "unspecified",
}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in MISSING_TOKENS:
            return pd.NA
        return stripped
    return value


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].map(normalize_scalar)
    return cleaned


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = "".join(handle.readline() for _ in range(5))

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        candidates = {
            delimiter: sample.count(delimiter) for delimiter in [",", "\t", ";", "|"]
        }
        return max(candidates, key=candidates.get)


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger("tcga_brca_initial_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def save_dataframe(
    df: pd.DataFrame,
    preferred_path: Path,
    *,
    index: bool = True,
) -> Path:
    ensure_directory(preferred_path.parent)
    if preferred_path.suffix == ".parquet":
        try:
            df.to_parquet(preferred_path, index=index)
            return preferred_path
        except Exception:
            fallback_path = preferred_path.with_suffix(".csv")
            df.to_csv(fallback_path, index=index)
            return fallback_path

    df.to_csv(preferred_path, index=index)
    return preferred_path


def file_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        payload.update(
            {
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "suffix": path.suffix,
            }
        )
    return payload


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    printable = df.fillna("NA").astype(str).replace({r"\|": r"\\|"}, regex=True)
    header = "| " + " | ".join(printable.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    rows = ["| " + " | ".join(values) + " |" for values in printable.to_numpy()]
    return "\n".join([header, divider, *rows])


def collapse_unique(series: pd.Series, max_values: int = 5) -> Any:
    cleaned = [
        value
        for value in series.map(normalize_scalar).dropna().tolist()
        if value is not pd.NA
    ]
    unique_values = list(dict.fromkeys(cleaned))
    if not unique_values:
        return pd.NA
    if len(unique_values) == 1:
        return unique_values[0]
    return " | ".join(map(str, unique_values[:max_values]))
