"""Preprocessing: stage ready dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.barcodes import build_sample_metadata
from src.audit.clinical import load_subject_level_clinical
from src.audit.summaries import compute_expression_qc
from src.ingestion.expression import load_and_standardize_expression
from src.utils.io import save_dataframe


@dataclass
class StageReadyArtifacts:
    stage_annotated_df: pd.DataFrame
    transportability_df: pd.DataFrame
    site_summary: pd.DataFrame
    duplicate_primary_subjects: pd.DataFrame
    metadata_completeness: pd.DataFrame
    summary_payload: dict[str, Any]


def collapse_pathologic_stage(raw_value: Any) -> str | None:
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip()
    if re.match(r"^Stage\s+IV(?:\b|[A-C])", value):
        return "Stage III-IV"
    if re.match(r"^Stage\s+III(?:\b|[A-C])", value):
        return "Stage III-IV"
    if re.match(r"^Stage\s+II(?:\b|[A-C])", value):
        return "Stage I-II"
    if re.match(r"^Stage\s+I(?:\b|[A-C])", value):
        return "Stage I-II"
    return None


def _build_primary_subject_table(
    expression_matrix: Path,
    clinical_dir: Path,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression_result = load_and_standardize_expression(
        expression_matrix, logger, seed=42
    )
    expression_df = expression_result.standardized_df.copy()
    expression_export = expression_df.reset_index().rename(
        columns={"index": "sample_barcode"}
    )

    clinical_subject_df, _ = load_subject_level_clinical(clinical_dir)
    samples_df, _ = build_sample_metadata(
        list(expression_df.index), clinical_subject_df
    )
    sample_qc, _, _ = compute_expression_qc(expression_df)
    samples_df = samples_df.merge(sample_qc, how="left", on="sample_barcode")

    primary_df = samples_df.loc[samples_df["sample_type_code"] == "01"].copy()
    duplicate_primary_subjects = (
        primary_df.groupby("subject_id", dropna=False)
        .agg(
            n_primary_samples=("sample_barcode", "size"),
            sample_barcodes=(
                "sample_barcode",
                lambda series: "|".join(map(str, series)),
            ),
        )
        .reset_index()
    )
    duplicate_primary_subjects = duplicate_primary_subjects.loc[
        duplicate_primary_subjects["n_primary_samples"] > 1
    ].sort_values(["n_primary_samples", "subject_id"], ascending=[False, True])

    ranked_primary = primary_df.sort_values(
        ["subject_id", "zero_fraction", "total_signal", "sample_barcode"],
        ascending=[True, True, False, True],
    )
    subject_level_primary = ranked_primary.drop_duplicates(
        "subject_id", keep="first"
    ).copy()
    subject_level_primary["site"] = subject_level_primary["tss"].astype(str)
    subject_level_primary["ajcc_pathologic_stage_raw"] = subject_level_primary[
        "clinical_ajcc_pathologic_stage"
    ]
    subject_level_primary["target_label"] = subject_level_primary[
        "ajcc_pathologic_stage_raw"
    ].map(collapse_pathologic_stage)
    subject_level_primary["target"] = subject_level_primary["target_label"].map(
        {"Stage I-II": 0, "Stage III-IV": 1}
    )

    stage_annotated_df = subject_level_primary.merge(
        expression_export, how="left", on="sample_barcode"
    )
    return stage_annotated_df, duplicate_primary_subjects


def build_stage_ready_artifacts(
    *,
    expression_matrix: Path,
    clinical_dir: Path,
    output_root: Path,
    min_per_class_per_site: int,
    logger,
) -> StageReadyArtifacts:
    stage_annotated_df, duplicate_primary_subjects = _build_primary_subject_table(
        expression_matrix=expression_matrix,
        clinical_dir=clinical_dir,
        logger=logger,
    )

    stage_available_df = stage_annotated_df.dropna(subset=["target"]).copy()
    stage_available_df["target"] = stage_available_df["target"].astype(int)

    site_target_counts = pd.crosstab(
        stage_available_df["site"], stage_available_df["target_label"]
    )
    valid_sites = site_target_counts.index[
        site_target_counts.min(axis=1) >= min_per_class_per_site
    ].tolist()
    transportability_df = stage_available_df.loc[
        stage_available_df["site"].isin(valid_sites)
    ].copy()
    transportability_df["target"] = transportability_df["target"].astype(int)

    site_summary = (
        transportability_df.groupby("site", dropna=False)
        .agg(
            n_samples=("sample_barcode", "size"),
            n_subjects=("subject_id", "nunique"),
            n_early=("target", lambda series: int((series == 0).sum())),
            n_late=("target", lambda series: int((series == 1).sum())),
            positive_rate=("target", "mean"),
        )
        .reset_index()
        .sort_values(["n_samples", "site"], ascending=[False, True])
        .reset_index(drop=True)
    )

    metadata_columns = [
        "clinical_primary_site",
        "clinical_primary_diagnosis",
        "clinical_site_of_resection_or_biopsy",
        "ajcc_pathologic_stage_raw",
        "target_label",
    ]
    available_metadata_columns = [
        column for column in metadata_columns if column in transportability_df.columns
    ]
    metadata_completeness = (
        transportability_df.groupby("site", dropna=False)[available_metadata_columns]
        .agg(lambda series: float(series.notna().mean()))
        .sort_index()
    )

    summary_payload = {
        "n_samples_stage_annotated": int(len(stage_annotated_df)),
        "n_primary_subject_level_rows": int(stage_annotated_df["subject_id"].nunique()),
        "n_samples_with_stage_labels": int(len(stage_available_df)),
        "n_sites_with_stage_labels": int(stage_available_df["site"].nunique()),
        "n_valid_sites_after_min_class_filter": int(len(valid_sites)),
        "valid_sites": valid_sites,
        "class_counts_transportability": transportability_df["target_label"]
        .value_counts()
        .to_dict(),
        "duplicate_primary_subjects": int(len(duplicate_primary_subjects)),
        "output_root": str(output_root),
    }

    save_dataframe(
        stage_annotated_df,
        output_root
        / "data"
        / "processed"
        / "tcga_brca_stage_annotated_primary_tumors.csv",
        index=False,
    )
    save_dataframe(
        transportability_df,
        output_root
        / "data"
        / "processed"
        / "tcga_brca_stage_transportability_dataset.csv",
        index=False,
    )
    save_dataframe(
        site_summary,
        output_root
        / "reports"
        / "tables_standard"
        / "tcga_brca_stage_site_summary.csv",
        index=False,
    )
    save_dataframe(
        duplicate_primary_subjects,
        output_root
        / "reports"
        / "tables_standard"
        / "tcga_brca_duplicate_primary_subjects.csv",
        index=False,
    )
    save_dataframe(
        metadata_completeness.reset_index(),
        output_root
        / "reports"
        / "tables_standard"
        / "tcga_brca_site_metadata_completeness.csv",
        index=False,
    )

    return StageReadyArtifacts(
        stage_annotated_df=stage_annotated_df,
        transportability_df=transportability_df,
        site_summary=site_summary,
        duplicate_primary_subjects=duplicate_primary_subjects,
        metadata_completeness=metadata_completeness,
        summary_payload=summary_payload,
    )
