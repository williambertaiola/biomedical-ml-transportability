"""Preprocessing: abide2 transportability rich."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.abide2_core_transportability import (
    DOMAIN_COLUMNS,
    ID_COLUMNS,
    RENAME_MAP,
    TARGET_COLUMNS,
    build_blueprint as build_strict_blueprint,
    build_final_dataset as build_strict_dataset,
    filter_to_cross_sectional,
    init_logger,
    load_master_table,
    save_dataframe,
    write_json,
)


RICH_FEATURES = [
    "age_at_scan",
    "sex",
    "handedness_category",
    "eye_status_at_scan",
    "fiq",
    "viq",
    "piq",
    "fiq_test_type",
    "viq_test_type",
    "piq_test_type",
    "current_med_status",
    "srs_edition",
    "srs_version",
    "srs_informant",
    "srs_total_raw",
    "srs_total_t",
    "srs_awareness_raw",
    "srs_awareness_t",
    "srs_cognition_raw",
    "srs_cognition_t",
    "srs_communication_raw",
    "srs_communication_t",
    "srs_motivation_raw",
    "srs_motivation_t",
    "srs_mannerisms_raw",
    "srs_mannerisms_t",
]

RICH_RENAME_MAP = {
    **RENAME_MAP,
    "current_med_status": "current_med_status_code",
}


@dataclass
class RichTransportabilityArtifacts:
    strict_df: pd.DataFrame
    rich_df: pd.DataFrame
    rich_feature_manifest: pd.DataFrame
    rich_row_missingness_summary: pd.DataFrame
    rich_site_completeness: pd.DataFrame
    blueprint: dict[str, Any]
    report_text: str


def build_rich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = ID_COLUMNS + DOMAIN_COLUMNS + TARGET_COLUMNS + RICH_FEATURES
    available_columns = [column for column in keep_columns if column in df.columns]
    out = df[available_columns].copy().rename(columns=RICH_RENAME_MAP)

    for column in [
        "disease_level_code",
        "sex_code",
        "handedness_category_code",
        "eye_status_at_scan_code",
        "current_med_status_code",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    for column in ["age_at_scan", "fiq", "viq", "piq"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out.sort_values(["site", "subject_id"]).reset_index(drop=True)


def build_rich_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in df.columns
        if column not in [*ID_COLUMNS, *DOMAIN_COLUMNS, *TARGET_COLUMNS]
    ]
    site_n = int(df["site"].nunique())
    rows = []
    for column in feature_columns:
        rows.append(
            {
                "column_name": column,
                "dtype": str(df[column].dtype),
                "missing_n": int(df[column].isna().sum()),
                "non_missing_n": int(df[column].notna().sum()),
                "missing_rate": float(df[column].isna().mean()),
                "sites_with_any": int(
                    df.groupby("site")[column].apply(lambda s: s.notna().any()).sum()
                ),
                "sites_total": site_n,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["missing_rate", "column_name"])
        .reset_index(drop=True)
    )


def build_rich_row_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in df.columns
        if column not in [*ID_COLUMNS, *DOMAIN_COLUMNS, *TARGET_COLUMNS]
    ]
    missing_per_row = df[feature_columns].isna().sum(axis=1)
    summary = (
        missing_per_row.value_counts()
        .rename_axis("n_missing_features")
        .reset_index(name="n_subjects")
        .sort_values("n_missing_features")
        .reset_index(drop=True)
    )
    summary["share_subjects"] = summary["n_subjects"] / len(df)
    return summary


def build_rich_site_completeness(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in df.columns
        if column not in [*ID_COLUMNS, *DOMAIN_COLUMNS, *TARGET_COLUMNS]
    ]
    rows = []
    for site, group in df.groupby("site"):
        row: dict[str, Any] = {"site": site, "n_subjects": int(len(group))}
        for column in feature_columns:
            row[f"{column}_non_missing_rate"] = float(group[column].notna().mean())
        row["mean_feature_completeness"] = float(
            group[feature_columns].notna().mean().mean()
        )
        row["median_feature_completeness"] = float(
            group[feature_columns].notna().mean().median()
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("n_subjects", ascending=False)
        .reset_index(drop=True)
    )


def build_blueprint(strict_df: pd.DataFrame, rich_df: pd.DataFrame) -> dict[str, Any]:
    strict_blueprint = build_strict_blueprint(strict_df)
    return {
        "dataset_name": "abide2_core_multisite_transportability_dataset",
        "description": "Cross-sectional non-imaging ABIDE II transportability bundle with strict and rich variants.",
        "subject_filtering": {
            "n_subjects": int(len(rich_df)),
            "n_sites": int(rich_df["site"].nunique()),
            "excluded_sites": ["ABIDEII-UCLA_Long", "ABIDEII-UPSM_Long"],
        },
        "strict_variant": {
            "path": "data/processed/abide2_core_multisite_transportability_dataset_strict.csv",
            "recommended_for": "primary non-circular baseline and transportability analyses",
            "feature_columns": strict_blueprint["recommended_feature_columns"],
        },
        "rich_variant": {
            "path": "data/processed/abide2_core_multisite_transportability_dataset.csv",
            "recommended_for": "richer clinical experiments after careful leakage review",
            "feature_columns": [
                column
                for column in rich_df.columns
                if column not in [*ID_COLUMNS, *DOMAIN_COLUMNS, *TARGET_COLUMNS]
            ],
            "high_target_proximity_block": [
                "srs_edition",
                "srs_version",
                "srs_informant",
                "srs_total_raw",
                "srs_total_t",
                "srs_awareness_raw",
                "srs_awareness_t",
                "srs_cognition_raw",
                "srs_cognition_t",
                "srs_communication_raw",
                "srs_communication_t",
                "srs_motivation_raw",
                "srs_motivation_t",
                "srs_mannerisms_raw",
                "srs_mannerisms_t",
            ],
        },
        "train_fold_only_preprocessing": {
            "split_strategy": "Leave-One-Site-Out or GroupKFold with site as group",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            "scaling": "standardize_numeric_features_after_imputation_using_training_fold_only",
        },
    }


def render_report(
    *,
    master_df: pd.DataFrame,
    cross_sectional_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    rich_df: pd.DataFrame,
    rich_feature_manifest: pd.DataFrame,
    rich_row_missingness_summary: pd.DataFrame,
    blueprint: dict[str, Any],
) -> str:
    target_dist = (
        rich_df["disease_level"]
        .value_counts(dropna=False)
        .rename_axis("disease_level")
        .reset_index(name="n_subjects")
    )
    top_features = rich_feature_manifest[
        ["column_name", "non_missing_n", "missing_rate", "sites_with_any"]
    ].to_string(index=False)

    lines = [
        "# ABIDE II Transportability Dataset Bundle",
        "",
        "## Overview",
        f"- Master composite table: `{master_df.shape[0]} rows x {master_df.shape[1]} columns`.",
        f"- Cross-sectional multisite cohort: `{cross_sectional_df.shape[0]} subjects` across `{cross_sectional_df['site'].nunique()}` sites.",
        f"- Strict dataset: `{strict_df.shape[0]} rows x {strict_df.shape[1]} columns`.",
        f"- Rich dataset: `{rich_df.shape[0]} rows x {rich_df.shape[1]} columns`.",
        "",
        "## Why the previous file looked empty",
        "- The previous strict version intentionally kept only the safest low-missing, low-circularity variables.",
        "- That was useful methodologically, but too thin as a working clinical table.",
        "- The main output path now contains the richer phenotype-preserving version, while the strict version is still saved separately as a backup.",
        "",
        "## Current outputs",
        "- `data/processed/abide2_core_multisite_transportability_dataset.csv`: richer clinical version.",
        "- `data/processed/abide2_core_multisite_transportability_dataset_strict.csv`: previous strict minimal version.",
        "",
        "## Target distribution",
        target_dist.to_string(index=False),
        "",
        "## Rich feature coverage",
        top_features,
        "",
        "## Row missingness in the rich dataset",
        rich_row_missingness_summary.to_string(index=False),
        "",
        "## Recommended usage",
        "- Use the strict dataset for your main non-circular baseline and transportability analyses.",
        "- Use the rich dataset when you want a more informative clinical table and are willing to explicitly discuss that the SRS block is behaviorally close to the ASD/control label.",
        "- In both cases, keep all imputation and encoding inside the training folds only.",
        "",
        "## Practical note",
        "- The rich dataset is no longer empty: it now includes demographics, IQ variables, test-type variables, medication status, and the SRS behavioral block.",
    ]
    return "\n".join(lines) + "\n"


def run_rich_transportability_processing(
    master_path: Path, logger
) -> RichTransportabilityArtifacts:
    master_df = load_master_table(master_path)
    cross_sectional_df, _ = filter_to_cross_sectional(master_df)
    logger.info(
        "Cross-sectional cohort has %s subjects across %s sites",
        len(cross_sectional_df),
        cross_sectional_df["site"].nunique(),
    )

    strict_df = build_strict_dataset(cross_sectional_df)
    rich_df = build_rich_dataset(cross_sectional_df)
    rich_feature_manifest = build_rich_feature_manifest(rich_df)
    rich_row_missingness_summary = build_rich_row_missingness_summary(rich_df)
    rich_site_completeness = build_rich_site_completeness(rich_df)
    blueprint = build_blueprint(strict_df, rich_df)
    report_text = render_report(
        master_df=master_df,
        cross_sectional_df=cross_sectional_df,
        strict_df=strict_df,
        rich_df=rich_df,
        rich_feature_manifest=rich_feature_manifest,
        rich_row_missingness_summary=rich_row_missingness_summary,
        blueprint=blueprint,
    )

    return RichTransportabilityArtifacts(
        strict_df=strict_df,
        rich_df=rich_df,
        rich_feature_manifest=rich_feature_manifest,
        rich_row_missingness_summary=rich_row_missingness_summary,
        rich_site_completeness=rich_site_completeness,
        blueprint=blueprint,
        report_text=report_text,
    )
