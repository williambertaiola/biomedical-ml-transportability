"""Preprocessing: audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ingestion.loaders import MISSING_TOKENS, NUMERIC_TOKEN_RE
from ingestion.specs import (
    CATEGORICAL_FEATURES,
    CLASSIC_FEATURES,
    EXPECTED_CATEGORY_VALUES,
    NUMERIC_FEATURES,
    PLAUSIBLE_RANGES,
)


@dataclass
class AuditResult:
    interim_df: pd.DataFrame
    analysis_ready_df: pd.DataFrame
    target_original_counts: dict[str, int]
    target_binary_counts: dict[str, int]
    domain_summary: pd.DataFrame
    target_by_domain: pd.DataFrame
    missingness_table: pd.DataFrame
    dtype_table: pd.DataFrame
    suspicious_value_table: pd.DataFrame
    inconsistent_type_table: pd.DataFrame
    quality_checks: dict[str, Any]
    warnings: list[str]
    removal_log: list[dict[str, Any]]


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    for column in clean_df.columns:
        if pd.api.types.is_object_dtype(
            clean_df[column]
        ) or pd.api.types.is_string_dtype(clean_df[column]):
            string_col = clean_df[column].astype("string").str.strip()
            lowered = string_col.str.lower()
            clean_df[column] = string_col.mask(lowered.isin(MISSING_TOKENS), pd.NA)
    return clean_df


def _find_inconsistent_numeric_tokens(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        series = df[column].dropna().astype("string").str.strip()
        invalid = series[~series.str.match(NUMERIC_TOKEN_RE)]
        rows.append(
            {
                "column": column,
                "invalid_token_count": int(invalid.shape[0]),
                "example_invalid_tokens": ", ".join(
                    sorted(invalid.unique().tolist())[:10]
                ),
            }
        )
    return pd.DataFrame(rows)


def _cast_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    clean_df["source_record_index"] = pd.to_numeric(
        clean_df["source_record_index"], errors="coerce"
    ).astype("Int64")
    clean_df["raw_patient_id"] = (
        pd.to_numeric(clean_df["raw_patient_id"], errors="coerce")
        .round()
        .astype("Int64")
    )

    for column in NUMERIC_FEATURES:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").astype(
            "Float64"
        )

    for column in CATEGORICAL_FEATURES + ["target_original"]:
        clean_df[column] = (
            pd.to_numeric(clean_df[column], errors="coerce").round().astype("Int64")
        )

    for column in [
        "domain_id",
        "requested_source_file",
        "actual_source_file",
        "input_format",
    ]:
        clean_df[column] = clean_df[column].astype("string")

    return clean_df


def _build_sample_and_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    per_domain_counter = out.groupby("domain_id").cumcount() + 1
    out["sample_id"] = (
        "uci_heart_"
        + out["domain_id"].astype("string")
        + "_"
        + per_domain_counter.astype(str).str.zfill(4)
    )
    out["subject_id_source"] = "sample_id"
    out["subject_id"] = out["sample_id"].astype("string")

    raw_id_str = out["raw_patient_id"].astype("Int64").astype("string")
    has_raw_id = raw_id_str.notna()
    out.loc[has_raw_id, "subject_id_source"] = "raw_patient_id"
    out.loc[has_raw_id, "subject_id"] = (
        out.loc[has_raw_id, "domain_id"].astype("string")
        + "_pid_"
        + raw_id_str[has_raw_id]
    ).astype("string")
    out["subject_id_source"] = out["subject_id_source"].astype("string")
    return out


def _target_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(index): int(value) for index, value in counts.items()}


def _check_suspicious_values(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column, allowed_values in EXPECTED_CATEGORY_VALUES.items():
        non_missing = df[column].dropna().astype(int)
        unexpected_mask = ~non_missing.isin(allowed_values)
        rows.append(
            {
                "column": column,
                "check_type": "unexpected_category",
                "n_flagged": int(unexpected_mask.sum()),
                "details": (
                    f"Allowed values: {sorted(allowed_values)}"
                    if int(unexpected_mask.sum()) > 0
                    else ""
                ),
            }
        )

    for column, (lower, upper) in PLAUSIBLE_RANGES.items():
        non_missing = df[column].dropna().astype(float)
        out_of_range = (non_missing < lower) | (non_missing > upper)
        rows.append(
            {
                "column": column,
                "check_type": "out_of_range",
                "n_flagged": int(out_of_range.sum()),
                "details": (
                    f"Expected range [{lower}, {upper}]"
                    if int(out_of_range.sum()) > 0
                    else ""
                ),
            }
        )

    suspicious_zero_mask = df["chol"].fillna(1).astype(float) == 0
    rows.append(
        {
            "column": "chol",
            "check_type": "zero_value_review",
            "n_flagged": int(suspicious_zero_mask.sum()),
            "details": "Zero cholesterol is likely suspicious clinically but was preserved because it is not a documented UCI missing code.",
        }
    )

    return pd.DataFrame(rows)


def prepare_analysis_dataset(
    unified_df: pd.DataFrame,
    high_missingness_warning: float,
    min_domain_samples_warning: int,
) -> AuditResult:
    normalized_df = _normalize_strings(unified_df)
    inconsistent_type_table = _find_inconsistent_numeric_tokens(
        normalized_df,
        ["raw_patient_id", *CLASSIC_FEATURES, "target_original"],
    )
    cast_df = _cast_feature_types(normalized_df)

    removal_log: list[dict[str, Any]] = []
    invalid_target_mask = cast_df["target_original"].isna() | ~cast_df[
        "target_original"
    ].isin([0, 1, 2, 3, 4])
    if int(invalid_target_mask.sum()) > 0:
        removal_log.append(
            {
                "reason": "missing_or_unexpected_target_original",
                "n_rows_removed": int(invalid_target_mask.sum()),
            }
        )
    working_df = cast_df.loc[~invalid_target_mask].copy()

    working_df["target"] = (working_df["target_original"] > 0).astype("Int64")
    working_df = _build_sample_and_subject_ids(working_df)

    interim_columns = [
        "sample_id",
        "subject_id",
        "raw_patient_id",
        "domain_id",
        "subject_id_source",
        "requested_source_file",
        "actual_source_file",
        "input_format",
        "source_record_index",
        "target_original",
        "target",
        *CLASSIC_FEATURES,
    ]
    analysis_ready_columns = [
        "subject_id",
        "domain_id",
        "target_original",
        "target",
        *CLASSIC_FEATURES,
    ]

    interim_df = working_df[interim_columns].copy()
    analysis_ready_df = working_df[analysis_ready_columns].copy()

    duplicate_subset = ["domain_id", *CLASSIC_FEATURES, "target_original"]
    exact_duplicate_mask = working_df.duplicated(subset=duplicate_subset, keep=False)
    exact_duplicate_count = int(exact_duplicate_mask.sum())

    suspicious_value_table = _check_suspicious_values(working_df)

    missingness_table = pd.DataFrame(
        [
            {
                "column": column,
                "missing_count": int(analysis_ready_df[column].isna().sum()),
                "missing_pct": round(float(analysis_ready_df[column].isna().mean()), 4),
            }
            for column in analysis_ready_df.columns
        ]
    ).sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)

    dtype_table = pd.DataFrame(
        [
            {
                "column": column,
                "dtype": str(analysis_ready_df[column].dtype),
                "non_null_count": int(analysis_ready_df[column].notna().sum()),
                "unique_non_null": int(analysis_ready_df[column].nunique(dropna=True)),
            }
            for column in analysis_ready_df.columns
        ]
    )

    domain_summary = (
        analysis_ready_df.groupby("domain_id", dropna=False)
        .agg(
            n_samples=("subject_id", "size"),
            n_controls=("target", lambda s: int((s == 0).sum())),
            n_cases=("target", lambda s: int((s == 1).sum())),
            prevalence=("target", lambda s: round(float(s.mean()), 4)),
        )
        .reset_index()
        .sort_values("domain_id", ignore_index=True)
    )

    target_by_domain = (
        analysis_ready_df.groupby(["domain_id", "target_original"], dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["domain_id", "target_original"], ignore_index=True)
    )

    zero_variance_columns = [
        column
        for column in CLASSIC_FEATURES
        if working_df[column].nunique(dropna=True) <= 1
    ]
    high_missingness_columns = missingness_table.loc[
        missingness_table["missing_pct"] >= high_missingness_warning, "column"
    ].tolist()
    low_sample_domains = domain_summary.loc[
        domain_summary["n_samples"] < min_domain_samples_warning, "domain_id"
    ].tolist()
    single_class_domains = domain_summary.loc[
        (domain_summary["n_controls"] == 0) | (domain_summary["n_cases"] == 0),
        "domain_id",
    ].tolist()
    unexpected_category_rows = suspicious_value_table.loc[
        (suspicious_value_table["n_flagged"] > 0)
        & (suspicious_value_table["check_type"] == "unexpected_category")
    ]

    warnings: list[str] = []
    if exact_duplicate_count > 0:
        warnings.append(
            f"Found {exact_duplicate_count} exact duplicate rows across domain_id + features + target_original. They were preserved and documented, not dropped."
        )
    if low_sample_domains:
        warnings.append(
            f"Domains below the sample-size warning threshold ({min_domain_samples_warning} rows): {', '.join(low_sample_domains)}."
        )
    if single_class_domains:
        warnings.append(
            f"Domains with a single binary class after conversion: {', '.join(single_class_domains)}."
        )
    if high_missingness_columns:
        warnings.append(
            f"Columns with missingness >= {high_missingness_warning:.0%}: {', '.join(high_missingness_columns)}."
        )
    if zero_variance_columns:
        warnings.append(
            f"Zero-variance raw feature columns detected: {', '.join(zero_variance_columns)}."
        )
    invalid_type_columns = inconsistent_type_table.loc[
        inconsistent_type_table["invalid_token_count"] > 0, "column"
    ].tolist()
    if invalid_type_columns:
        warnings.append(
            f"Columns with non-numeric tokens before coercion: {', '.join(invalid_type_columns)}."
        )
    if not unexpected_category_rows.empty:
        flagged_columns = unexpected_category_rows["column"].drop_duplicates().tolist()
        warnings.append(
            f"Unexpected categorical values were detected in: {', '.join(flagged_columns)}."
        )

    quality_checks: dict[str, Any] = {
        "domains_below_sample_threshold": low_sample_domains,
        "single_class_domains": single_class_domains,
        "columns_with_high_missingness": high_missingness_columns,
        "zero_variance_columns": zero_variance_columns,
        "columns_with_inconsistent_numeric_tokens": invalid_type_columns,
        "possible_leakage_columns": [],
        "unexpected_target_values_removed": int(invalid_target_mask.sum()),
        "exact_duplicate_rows_detected": exact_duplicate_count,
        "no_global_imputation_or_encoding_applied": True,
    }

    return AuditResult(
        interim_df=interim_df,
        analysis_ready_df=analysis_ready_df,
        target_original_counts=_target_counts(analysis_ready_df["target_original"]),
        target_binary_counts=_target_counts(analysis_ready_df["target"]),
        domain_summary=domain_summary,
        target_by_domain=target_by_domain,
        missingness_table=missingness_table,
        dtype_table=dtype_table,
        suspicious_value_table=suspicious_value_table,
        inconsistent_type_table=inconsistent_type_table,
        quality_checks=quality_checks,
        warnings=warnings,
        removal_log=removal_log,
    )
