"""Preprocessing: reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.specs import FEATURE_DICTIONARY


def _format_cell(value: Any) -> str:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(_format_cell(row[col]).replace("\n", " ") for col in columns)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def build_report(
    run_timestamp: datetime,
    config_payload: dict[str, Any],
    source_manifest: list[dict[str, Any]],
    ingestion_warnings: list[str],
    audit_result: Any,
    feature_schema: dict[str, Any],
    split_manifest: list[dict[str, Any]],
    output_files: list[Path],
) -> str:
    overview_lines = [
        "# UCI Heart Disease Preprocessing Report",
        "",
        "## Dataset Overview",
        "",
        f"- Run timestamp: {run_timestamp.isoformat()}",
        f"- Project root: `{config_payload['project_root']}`",
        f"- Number of samples: {audit_result.analysis_ready_df.shape[0]}",
        f"- Number of domains: {audit_result.analysis_ready_df['domain_id'].nunique()}",
        f"- Domain names: {', '.join(sorted(audit_result.analysis_ready_df['domain_id'].dropna().astype(str).unique().tolist()))}",
        f"- Number of predictive features: {len(feature_schema['predictive_feature_columns'])}",
        "- Dataset type: tabular clinical data from multiple hospitals/centers.",
        "- Domain identifier: `domain_id` encodes site/hospital and is preserved for downstream leave-one-domain-out analyses.",
        "",
        "## Source Files Used",
        "",
    ]

    source_rows = []
    for entry in source_manifest:
        source_rows.append(
            {
                "requested_source_file": entry["requested_source_file"],
                "requested_domain_id": entry["requested_domain_id"],
                "actual_source_file": entry["actual_source_file"],
                "input_format": entry["input_format"],
                "fallback_used": entry.get("fallback_used", False),
            }
        )
    overview_lines.append(dataframe_to_markdown(pd.DataFrame(source_rows)))
    overview_lines.append("")

    final_columns = audit_result.analysis_ready_df.columns.tolist()

    overview_lines.extend(
        [
            "## Target Construction",
            "",
            "- Original UCI target `target_original` takes values 0, 1, 2, 3, and 4.",
            "- Binary `target` was created as follows:",
            "  - `target = 0` when `target_original == 0`",
            "  - `target = 1` when `target_original in {1, 2, 3, 4}`",
            f"- Original target counts: `{audit_result.target_original_counts}`",
            f"- Binary target counts: `{audit_result.target_binary_counts}`",
            "",
            "## Dataset-Level Cleaning Performed Safely Before Any Train/Test Split",
            "",
            "- Standardized column names to lowercase snake_case where needed.",
            "- Inferred or preserved `domain_id` from filenames or combined-file site columns.",
            "- Added deterministic `subject_id` values for the final analysis-ready dataset.",
            "- Generated `sample_id` plus file-provenance fields for traceability in the interim dataset and source manifest.",
            "- Converted documented missing-value codings such as `?`, `-9`, `-9.`, empty strings, `NA`, `N/A`, `null`, and `nan` to proper missing values.",
            "- Converted feature columns to explicit numeric or integer-coded categorical dtypes.",
            "- Kept the final processed dataset restricted to the user-requested modeling columns.",
            "- Did not apply global imputation, scaling, one-hot encoding, or variance filtering.",
            "",
            "## Global vs Train-Only Transformations",
            "",
            "Global steps that were done safely:",
            "- ingestion, row-level harmonization, missing-code normalization, target construction, duplicate inspection, feature typing, metadata preservation, and split-manifest creation.",
            "",
            "Transformations that must be fit only on training folds later:",
            "- median imputation for numeric variables",
            "- most-frequent imputation for categorical variables",
            "- numeric standardization",
            "- one-hot encoding",
            "- zero-variance filtering after transformation",
            "",
            "Critical anti-leakage rule:",
            "- Imputers, scalers, one-hot encoders, and variance filters must be fit only on the training partition inside each future LODO fold, never on the full dataset.",
            "",
            "## Duplicates and Removals",
            "",
            f"- Removal log: `{audit_result.removal_log}`",
            f"- Exact duplicate rows detected on domain + features + original target: {audit_result.quality_checks['exact_duplicate_rows_detected']}",
            "",
            "## Feature Typing Decisions",
            "",
            f"- Final analysis-ready columns: {final_columns}",
            f"- Numeric features: {feature_schema['numeric_features']}",
            f"- Categorical features: {feature_schema['categorical_features']}",
            f"- ID columns: {feature_schema['id_columns']}",
            f"- Metadata columns: {feature_schema['metadata_columns']}",
            f"- Excluded columns: {feature_schema['excluded_columns']}",
            "",
            "## Column Dictionary",
            "",
        ]
    )

    dictionary_rows = pd.DataFrame(
        [
            {"column": key, "description": value}
            for key, value in FEATURE_DICTIONARY.items()
            if key in final_columns
        ]
    )
    overview_lines.append(dataframe_to_markdown(dictionary_rows))
    overview_lines.append("")

    overview_lines.extend(
        [
            "## Summary Tables",
            "",
            "### Samples by Domain",
            "",
            dataframe_to_markdown(audit_result.domain_summary),
            "",
            "### Target Prevalence by Domain",
            "",
            dataframe_to_markdown(audit_result.target_by_domain),
            "",
            "### Missingness per Column",
            "",
            dataframe_to_markdown(audit_result.missingness_table),
            "",
            "### Data Types per Column",
            "",
            dataframe_to_markdown(audit_result.dtype_table),
            "",
            "## Missing Data Analysis",
            "",
            "- Missing values were standardized at the dataset level but left unimputed.",
            "- This keeps the dataset leakage-safe for later fold-specific model fitting.",
            "- Columns with elevated missingness were flagged in the warnings section below.",
            "",
            "## Anomalous Coding Cleanup Performed",
            "",
            "- Converted `?`, empty strings, `NA`, `N/A`, `null`, `none`, `nan`, `-9`, `-9.`, and `-9.0` to missing values.",
            "- Preserved suspicious but undocumented values such as `chol = 0` and reported them rather than silently recoding them.",
            "",
            "## Quality Checks and Warnings",
            "",
        ]
    )

    combined_warnings = list(ingestion_warnings) + list(audit_result.warnings)
    if combined_warnings:
        for warning in combined_warnings:
            overview_lines.append(f"- {warning}")
    else:
        overview_lines.append(
            "- No high-priority warnings were triggered by the implemented checks."
        )

    overview_lines.extend(
        [
            "",
            "### Suspicious Value Checks",
            "",
            dataframe_to_markdown(audit_result.suspicious_value_table),
            "",
            "### Inconsistent Type Checks",
            "",
            dataframe_to_markdown(audit_result.inconsistent_type_table),
            "",
            "## LODO Split Manifest Summary",
            "",
            dataframe_to_markdown(pd.DataFrame(split_manifest)),
            "",
            "## Recommendations for Downstream Modeling",
            "",
            "- Keep `domain_id` out of the predictive feature matrix unless you intentionally run a domain-aware experiment.",
            "- Fit preprocessing only on the training partition within each LODO fold.",
            "- Monitor domain-level class imbalance because some held-out sites are much more case-heavy than others.",
            "- Consider reporting both pooled and per-domain missingness in downstream models because missingness patterns differ across sites.",
            "- Use the interim dataset and source manifest when you need row provenance or raw patient IDs; use the analysis-ready dataset for modeling inputs.",
            "",
            "## Produced Output Files",
            "",
        ]
    )

    for output_file in output_files:
        overview_lines.append(f"- `{output_file}`")

    overview_lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- The provided `*.data` files were treated as the official UCI Heart Disease multi-center inputs.",
            "- Wrapped raw files were converted to the classic 14-variable subset using the documented attribute indices from `heart-disease.names`.",
            "- The Cleveland raw file appeared corrupted or partially non-textual in this local copy, so the pipeline transparently fell back to `processed.cleveland.data` for that domain and documented the fallback.",
            "- `subject_id` uses the recoverable raw patient ID when available and otherwise falls back to `sample_id`.",
            "- The requested final column `thai` was interpreted as the standard UCI feature `thal`.",
            "- `domain_id` was inferred from the provided filenames for the four separate site files.",
        ]
    )

    return "\n".join(overview_lines) + "\n"
