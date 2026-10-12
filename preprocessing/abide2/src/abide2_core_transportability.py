"""Preprocessing: abide2 core transportability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RECOMMENDED_FEATURES = [
    "age_at_scan",
    "sex",
    "handedness_category",
    "eye_status_at_scan",
    "fiq",
]

OPTIONAL_FEATURES = [
    "current_med_status",
    "viq",
    "piq",
    "fiq_test_type",
    "viq_test_type",
    "piq_test_type",
]

ID_COLUMNS = ["subject_id"]
DOMAIN_COLUMNS = ["site"]
TARGET_COLUMNS = ["disease_level", "disease_level_code"]

RENAME_MAP = {
    "sex": "sex_code",
    "handedness_category": "handedness_category_code",
    "eye_status_at_scan": "eye_status_at_scan_code",
}

EXCLUDE_PREFIX_GROUPS = {
    "imaging_qap_block_removed_per_user_scope": ("anat_", "func_", "dti_", "raw_bni_"),
    "diagnostic_scale_removed_to_avoid_target_circularity": (
        "adi_",
        "ados_",
        "aq_",
        "asd_",
        "basc2_",
        "bdi_",
        "brief_",
        "casi_",
        "cbcl_",
        "celf_",
        "cprs_",
        "nonasd_",
        "pdd_",
        "rbsr_",
        "scq_",
        "srs_",
        "vineland_",
        "wiat_",
    ),
}

EXPLICIT_EXCLUSIONS = {
    "dx_group": "duplicate_of_target_code",
    "disease_level_source": "metadata_not_feature",
    "site_id": "duplicate_of_site",
    "sub_id": "identifier_not_feature",
    "phenotypic_available": "source_availability_flag_not_feature",
    "phenotypic_composite_available": "source_availability_flag_not_feature",
    "phenotypic_long_available": "source_availability_flag_not_feature",
    "anat_available": "source_availability_flag_not_feature",
    "dti_available": "source_availability_flag_not_feature",
    "func_available": "source_availability_flag_not_feature",
    "raw_bni_available": "source_availability_flag_not_feature",
    "current_medication_name": "high_cardinality_medication_string",
    "handedness_scores": "optional_but_sparse_site_coverage",
}


@dataclass
class CoreTransportabilityArtifacts:
    final_df: pd.DataFrame
    feature_manifest: pd.DataFrame
    site_completeness: pd.DataFrame
    row_missingness_summary: pd.DataFrame
    blueprint: dict[str, Any]
    report_text: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger("abide2_core_transportability")
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


def save_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
    return path


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def load_master_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def filter_to_cross_sectional(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = ~df["site"].astype("string").str.endswith("_Long", na=False)
    excluded = (
        df.loc[~mask, ["subject_id", "site", "disease_level"]]
        .copy()
        .assign(
            exclusion_reason="followup_only_longitudinal_site_removed_from_main_cross_sectional_dataset"
        )
    )
    return df.loc[mask].copy(), excluded


def infer_column_role(column: str) -> tuple[str, str]:
    if column in ID_COLUMNS:
        return "id", "identifier"
    if column in DOMAIN_COLUMNS:
        return "domain", "domain_id"
    if column in TARGET_COLUMNS:
        return "target", "classification_target"
    if column in RECOMMENDED_FEATURES:
        return "recommended_feature", "kept_for_main_multisite_modeling_dataset"
    if column in OPTIONAL_FEATURES:
        return (
            "optional_feature",
            "kept_in_master_only_optional_for_sensitivity_analysis",
        )
    if column in EXPLICIT_EXCLUSIONS:
        return "excluded", EXPLICIT_EXCLUSIONS[column]
    for reason, prefixes in EXCLUDE_PREFIX_GROUPS.items():
        if column.startswith(prefixes):
            return "excluded", reason
    return "excluded", "not_selected_for_main_non_imaging_core_dataset"


def build_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    site_n = int(df["site"].nunique())
    rows = []
    for column in df.columns:
        role, reason = infer_column_role(column)
        rows.append(
            {
                "column_name": column,
                "role": role,
                "selection_reason": reason,
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
        pd.DataFrame(rows).sort_values(["role", "column_name"]).reset_index(drop=True)
    )


def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = ID_COLUMNS + DOMAIN_COLUMNS + TARGET_COLUMNS + RECOMMENDED_FEATURES
    out = df[keep_columns].copy().rename(columns=RENAME_MAP)
    for column in [
        "disease_level_code",
        "sex_code",
        "handedness_category_code",
        "eye_status_at_scan_code",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    out = out.sort_values(["site", "subject_id"]).reset_index(drop=True)
    return out


def build_site_completeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_columns = [
        RENAME_MAP.get(column, column) for column in RECOMMENDED_FEATURES
    ]
    for site, group in df.groupby("site"):
        row: dict[str, Any] = {"site": site, "n_subjects": int(len(group))}
        for column in feature_columns:
            row[f"{column}_non_missing_n"] = int(group[column].notna().sum())
            row[f"{column}_non_missing_rate"] = float(group[column].notna().mean())
        row["complete_rows_n"] = int(group[feature_columns].notna().all(axis=1).sum())
        row["complete_rows_rate"] = float(
            group[feature_columns].notna().all(axis=1).mean()
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("n_subjects", ascending=False)
        .reset_index(drop=True)
    )


def build_row_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        RENAME_MAP.get(column, column) for column in RECOMMENDED_FEATURES
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


def build_blueprint(final_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset_name": "abide2_core_multisite_transportability_dataset",
        "description": "Cross-sectional non-imaging ABIDE II core dataset for baseline and transportability experiments.",
        "n_subjects": int(len(final_df)),
        "n_sites": int(final_df["site"].nunique()),
        "id_column": "subject_id",
        "domain_column": "site",
        "target_column": "disease_level",
        "target_code_column": "disease_level_code",
        "numeric_feature_columns": ["age_at_scan", "fiq"],
        "categorical_feature_columns": [
            "sex_code",
            "handedness_category_code",
            "eye_status_at_scan_code",
        ],
        "recommended_feature_columns": [
            "age_at_scan",
            "sex_code",
            "handedness_category_code",
            "eye_status_at_scan_code",
            "fiq",
        ],
        "optional_master_only_columns": OPTIONAL_FEATURES,
        "excluded_from_main_dataset": {
            "followup_sites": ["ABIDEII-UCLA_Long", "ABIDEII-UPSM_Long"],
            "imaging_blocks": ["anat_*", "func_*", "dti_*", "raw_bni_*"],
            "high_circularity_blocks": [
                "adi_*",
                "ados_*",
                "aq_*",
                "asd_*",
                "basc2_*",
                "bdi_*",
                "brief_*",
                "casi_*",
                "cbcl_*",
                "celf_*",
                "cprs_*",
                "nonasd_*",
                "pdd_*",
                "rbsr_*",
                "scq_*",
                "srs_*",
                "vineland_*",
                "wiat_*",
            ],
        },
        "train_fold_only_preprocessing": {
            "split_strategy": "Leave-One-Site-Out or GroupKFold with site as group",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            "scaling": "standardize_numeric_features_after_imputation_using_training_fold_only",
            "missing_indicators": "not recommended in the main analysis because site-structured missingness can act as a domain proxy",
        },
    }


def render_report(
    *,
    master_df: pd.DataFrame,
    cross_sectional_df: pd.DataFrame,
    excluded_followup_df: pd.DataFrame,
    final_df: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    site_completeness: pd.DataFrame,
    row_missingness_summary: pd.DataFrame,
    blueprint: dict[str, Any],
) -> str:
    recommended = feature_manifest[
        feature_manifest["role"] == "recommended_feature"
    ].copy()
    optional = feature_manifest[feature_manifest["role"] == "optional_feature"].copy()
    target_dist = (
        final_df["disease_level"]
        .value_counts(dropna=False)
        .rename_axis("disease_level")
        .reset_index(name="n_subjects")
    )

    lines = [
        "# ABIDE II Core Multisite Transportability Dataset",
        "",
        "## Goal",
        "- Build a compact ABIDE II phenotypic dataset for downstream baseline and transportability experiments.",
        "- Keep the dataset cross-sectional, multisite, non-imaging, and pre-imputation to avoid global leakage.",
        "",
        "## Cohort choice",
        f"- Master composite-enriched table: `{master_df.shape[0]} rows x {master_df.shape[1]} columns` across `{master_df['site'].nunique()}` sites.",
        f"- Main modeling cohort after removing `*_Long` follow-up sites: `{cross_sectional_df.shape[0]} subjects` across `{cross_sectional_df['site'].nunique()}` sites.",
        f"- Excluded follow-up-only rows: `{len(excluded_followup_df)}`.",
        "",
        "## Final dataset",
        f"- Final dataset shape: `{final_df.shape[0]} rows x {final_df.shape[1]} columns`.",
        "- One row per subject.",
        "- `site` is the domain variable for leave-one-site-out transportability.",
        "- `disease_level` is the target variable.",
        "",
        "## Target distribution",
        target_dist.to_string(index=False),
        "",
        "## Recommended feature set",
        recommended[
            [
                "column_name",
                "non_missing_n",
                "missing_rate",
                "sites_with_any",
                "selection_reason",
            ]
        ].to_string(index=False),
        "",
        "## Optional master-only candidates",
        optional[
            [
                "column_name",
                "non_missing_n",
                "missing_rate",
                "sites_with_any",
                "selection_reason",
            ]
        ].to_string(index=False)
        if not optional.empty
        else "None.",
        "",
        "## Site completeness",
        site_completeness.to_string(index=False),
        "",
        "## Row missingness in the final dataset",
        row_missingness_summary.to_string(index=False),
        "",
        "## Why these columns were kept",
        "- `age_at_scan`, `sex`, `handedness_category`, and `eye_status_at_scan` are almost complete across the 19 cross-sectional sites.",
        "- `fiq` adds clinically meaningful non-imaging information while still covering most of the cohort.",
        "- Higher-missing columns such as `viq`, `piq`, and `current_med_status` are preserved in the master table but left out of the main transportability dataset.",
        "- Diagnostic scales such as `ADOS`, `SRS`, `SCQ`, `AQ`, and `ADI-R` were excluded from the main dataset because they are too close to the ASD/control target and can make the task circular.",
        "",
        "## Preprocessing blueprint",
        f"- Numeric features: `{', '.join(blueprint['numeric_feature_columns'])}`.",
        f"- Categorical features: `{', '.join(blueprint['categorical_feature_columns'])}`.",
        "- Apply imputation, one-hot encoding, and scaling inside each training fold only.",
        "- Recommended split strategy: leave-one-site-out or grouped CV by `site`.",
        "- Missingness indicators are not recommended in the main analysis because structural missingness can encode site/domain information.",
        "",
        "## Recommendation",
        "- Use this dataset as the main ABIDE II input for baseline models and DASHI-style transportability experiments.",
        "- Keep the broader composite master table for sensitivity analyses, optional covariates, and ablation studies.",
    ]
    return "\n".join(lines) + "\n"


def run_core_transportability_processing(
    master_path: Path, logger: logging.Logger
) -> CoreTransportabilityArtifacts:
    master_df = load_master_table(master_path)
    cross_sectional_df, excluded_followup_df = filter_to_cross_sectional(master_df)
    logger.info(
        "Cross-sectional cohort has %s subjects across %s sites after excluding %s follow-up rows",
        len(cross_sectional_df),
        cross_sectional_df["site"].nunique(),
        len(excluded_followup_df),
    )

    feature_manifest = build_feature_manifest(cross_sectional_df)
    final_df = build_final_dataset(cross_sectional_df)
    site_completeness = build_site_completeness(final_df)
    row_missingness_summary = build_row_missingness_summary(final_df)
    blueprint = build_blueprint(final_df)
    report_text = render_report(
        master_df=master_df,
        cross_sectional_df=cross_sectional_df,
        excluded_followup_df=excluded_followup_df,
        final_df=final_df,
        feature_manifest=feature_manifest,
        site_completeness=site_completeness,
        row_missingness_summary=row_missingness_summary,
        blueprint=blueprint,
    )

    return CoreTransportabilityArtifacts(
        final_df=final_df,
        feature_manifest=feature_manifest,
        site_completeness=site_completeness,
        row_missingness_summary=row_missingness_summary,
        blueprint=blueprint,
        report_text=report_text,
    )
