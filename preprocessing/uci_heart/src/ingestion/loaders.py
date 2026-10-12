"""Preprocessing: loaders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.specs import (
    CLASSIC_COLUMN_ORDER,
    COLUMN_RENAME_MAP,
    DOMAIN_ALIASES,
    RAW_TO_CLASSIC_INDEX_1BASED,
)
from utils.filesystem import compute_sha256


MISSING_TOKENS = {"", "?", "na", "n/a", "null", "none", "nan", "-9", "-9.", "-9.0"}
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class SourceLoadResult:
    dataframe: pd.DataFrame
    manifest_entry: dict[str, Any]
    warnings: list[str]


def standardize_column_name(column_name: str) -> str:
    cleaned = column_name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("_")


def infer_domain_from_path(path: Path) -> str:
    stem = standardize_column_name(path.stem)
    for key, value in DOMAIN_ALIASES.items():
        if key in stem:
            return value
    raise ValueError(
        f"Could not infer domain_id from filename '{path.name}'. "
        "Please provide an input filename containing a known site label."
    )


def normalize_domain_label(raw_value: str) -> str:
    normalized = standardize_column_name(raw_value)
    return DOMAIN_ALIASES.get(normalized, normalized)


def decode_text_with_fallback(path: Path) -> tuple[str, str]:
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("latin1", errors="replace"), "latin1-replace"


def looks_like_wrapped_raw(text: str) -> bool:
    return "name" in text.lower()


def looks_like_headered_table(path: Path, text: str) -> bool:
    first_lines = [line for line in text.splitlines() if line.strip()][:5]
    if not first_lines:
        return False
    if looks_like_wrapped_raw(text):
        return False
    first_line = first_lines[0]
    if path.suffix.lower() in {".csv", ".tsv"} and any(
        ch.isalpha() for ch in first_line
    ):
        return True
    return any(ch.isalpha() for ch in first_line)


def clean_wrapped_line(raw_line: str) -> str:
    printable = "".join(
        ch if ch in {" ", "\t"} or 32 <= ord(ch) <= 126 else " " for ch in raw_line
    )
    printable = re.sub(r"(?i)name", " name ", printable)
    return re.sub(r"(?<=\d)-(?=\d)", " -", printable)


def parse_wrapped_raw_to_classic(text: str) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[list[str]] = []
    current: list[str] = []
    invalid_tokens = 0
    required_idx = max(RAW_TO_CLASSIC_INDEX_1BASED.values())

    for raw_line in text.splitlines():
        for token in clean_wrapped_line(raw_line).split():
            tok = token.strip().strip(",")
            if not tok:
                continue
            if tok.lower() == "name":
                if current:
                    records.append(current)
                    current = []
                continue
            if NUMERIC_TOKEN_RE.match(tok):
                current.append(tok)
            else:
                invalid_tokens += 1

    if current:
        records.append(current)

    extracted_rows: list[dict[str, str]] = []
    short_records = 0
    for record in records:
        if len(record) < required_idx:
            short_records += 1
            continue
        extracted_rows.append(
            {
                column: record[index - 1]
                for column, index in RAW_TO_CLASSIC_INDEX_1BASED.items()
            }
        )

    summary = {
        "raw_records_found": len(records),
        "rows_extracted": len(extracted_rows),
        "rows_dropped_short": short_records,
        "invalid_tokens_ignored": invalid_tokens,
    }
    return pd.DataFrame(extracted_rows, columns=CLASSIC_COLUMN_ORDER), summary


def load_processed_14_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        dtype=str,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    if df.shape[1] != 14:
        raise ValueError(
            f"Expected 14 columns in processed UCI file {path}, found {df.shape[1]}."
        )
    df.columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target_original",
    ]
    df.insert(0, "raw_patient_id", pd.NA)
    return df


def load_headered_table(
    path: Path, domain_hint: str | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, sep=None, engine="python")
    original_columns = list(df.columns)
    df.columns = [
        COLUMN_RENAME_MAP.get(
            standardize_column_name(col), standardize_column_name(col)
        )
        for col in df.columns
    ]

    required_columns = set(
        [
            "target_original",
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    )
    missing_required = sorted(required_columns - set(df.columns))
    if missing_required:
        raise ValueError(
            f"Combined table {path} is missing required columns after standardization: {missing_required}"
        )

    if "raw_patient_id" not in df.columns:
        df["raw_patient_id"] = pd.NA

    if "domain_id" not in df.columns:
        if domain_hint is None:
            raise ValueError(
                f"Combined file {path} has no domain/site column and no domain could be inferred from filename."
            )
        df["domain_id"] = domain_hint
    else:
        df["domain_id"] = (
            df["domain_id"]
            .astype("string")
            .map(
                lambda value: (
                    normalize_domain_label(str(value)) if value is not pd.NA else value
                )
            )
        )

    out = df[
        [
            "raw_patient_id",
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target_original",
            "domain_id",
        ]
    ].copy()
    return out, {
        "original_columns": original_columns,
        "normalized_columns": list(df.columns),
    }


def find_processed_fallback(raw_path: Path) -> Path | None:
    stem = standardize_column_name(raw_path.stem)
    fallback_name_map = {
        "cleveland": "processed.cleveland.data",
        "hungarian": "processed.hungarian.data",
        "long_beach_va": "processed.va.data",
        "switzerland": "processed.switzerland.data",
    }
    domain = infer_domain_from_path(raw_path)
    fallback_name = fallback_name_map.get(domain)
    if fallback_name is None:
        return None
    candidate = raw_path.parent / fallback_name
    return candidate if candidate.exists() else None


def load_single_source(
    path: Path,
    expected_rows_by_domain: dict[str, int],
    min_valid_raw_fraction: float,
) -> SourceLoadResult:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    requested_domain = infer_domain_from_path(path)
    warnings: list[str] = []
    text, encoding = decode_text_with_fallback(path)
    manifest_entry: dict[str, Any] = {
        "requested_source_file": str(path),
        "requested_domain_id": requested_domain,
        "requested_file_sha256": compute_sha256(path),
        "requested_file_size_bytes": path.stat().st_size,
        "encoding": encoding,
    }

    if looks_like_wrapped_raw(text):
        df, parse_summary = parse_wrapped_raw_to_classic(text)
        manifest_entry["input_format"] = "wrapped_raw_76_subset"
        manifest_entry["parse_summary"] = parse_summary
        expected_rows = expected_rows_by_domain.get(requested_domain)
        if (
            expected_rows
            and (parse_summary["rows_extracted"] / expected_rows)
            < min_valid_raw_fraction
        ):
            fallback_path = find_processed_fallback(path)
            if fallback_path is None:
                warnings.append(
                    f"Raw parse for {requested_domain} extracted {parse_summary['rows_extracted']} rows out of expected {expected_rows} and no processed fallback was found."
                )
            else:
                fallback_df = load_processed_14_table(fallback_path)
                fallback_df["domain_id"] = requested_domain
                fallback_df["source_record_index"] = range(1, len(fallback_df) + 1)
                fallback_df["requested_source_file"] = str(path)
                fallback_df["actual_source_file"] = str(fallback_path)
                fallback_df["input_format"] = (
                    "processed_14_fallback_after_raw_parse_failure"
                )
                manifest_entry.update(
                    {
                        "fallback_used": True,
                        "actual_source_file": str(fallback_path),
                        "actual_file_sha256": compute_sha256(fallback_path),
                        "actual_file_size_bytes": fallback_path.stat().st_size,
                        "input_format": "processed_14_fallback_after_raw_parse_failure",
                    }
                )
                warnings.append(
                    f"Domain {requested_domain} fell back to {fallback_path.name} because the raw wrapped file yielded {parse_summary['rows_extracted']}/{expected_rows} usable rows."
                )
                return SourceLoadResult(
                    dataframe=fallback_df,
                    manifest_entry=manifest_entry,
                    warnings=warnings,
                )

        df["domain_id"] = requested_domain
        df["source_record_index"] = range(1, len(df) + 1)
        df["requested_source_file"] = str(path)
        df["actual_source_file"] = str(path)
        df["input_format"] = "wrapped_raw_76_subset"
        manifest_entry.update(
            {
                "fallback_used": False,
                "actual_source_file": str(path),
                "actual_file_sha256": compute_sha256(path),
                "actual_file_size_bytes": path.stat().st_size,
            }
        )
        return SourceLoadResult(
            dataframe=df, manifest_entry=manifest_entry, warnings=warnings
        )

    if looks_like_headered_table(path, text):
        df, table_summary = load_headered_table(path, requested_domain)
        df["source_record_index"] = range(1, len(df) + 1)
        df["requested_source_file"] = str(path)
        df["actual_source_file"] = str(path)
        df["input_format"] = "headered_combined_or_single_table"
        manifest_entry.update(
            {
                "input_format": "headered_combined_or_single_table",
                "table_summary": table_summary,
                "fallback_used": False,
                "actual_source_file": str(path),
                "actual_file_sha256": compute_sha256(path),
                "actual_file_size_bytes": path.stat().st_size,
            }
        )
        return SourceLoadResult(
            dataframe=df, manifest_entry=manifest_entry, warnings=warnings
        )

    df = load_processed_14_table(path)
    df["domain_id"] = requested_domain
    df["source_record_index"] = range(1, len(df) + 1)
    df["requested_source_file"] = str(path)
    df["actual_source_file"] = str(path)
    df["input_format"] = "headerless_processed_14_table"
    manifest_entry.update(
        {
            "input_format": "headerless_processed_14_table",
            "fallback_used": False,
            "actual_source_file": str(path),
            "actual_file_sha256": compute_sha256(path),
            "actual_file_size_bytes": path.stat().st_size,
        }
    )
    return SourceLoadResult(
        dataframe=df, manifest_entry=manifest_entry, warnings=warnings
    )


def load_sources(
    input_paths: list[Path],
    expected_rows_by_domain: dict[str, int],
    min_valid_raw_fraction: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    loaded_frames: list[pd.DataFrame] = []
    source_manifest: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in input_paths:
        result = load_single_source(
            path=path,
            expected_rows_by_domain=expected_rows_by_domain,
            min_valid_raw_fraction=min_valid_raw_fraction,
        )
        loaded_frames.append(result.dataframe.copy())
        source_manifest.append(result.manifest_entry)
        warnings.extend(result.warnings)

    if not loaded_frames:
        raise ValueError("No input files were loaded.")

    combined_df = pd.concat(loaded_frames, ignore_index=True, sort=False)
    return combined_df, source_manifest, warnings
