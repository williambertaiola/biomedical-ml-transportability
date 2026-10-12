"""Transportability experiment: summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.audit.plots import plot_bar, plot_histogram
from src.utils.io import save_dataframe

FOCAL_SAMPLE_TYPES = ["01", "11", "06"]


@dataclass
class AuditArtifacts:
    summary_metrics: dict[str, Any]
    table_paths: dict[str, str]
    figure_paths: dict[str, str]


def _outlier_flags(values: pd.Series) -> pd.Series:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    return values < lower_bound


def compute_expression_qc(
    expression_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix = expression_df.to_numpy(copy=False)
    n_samples, n_genes = matrix.shape

    nonzero_per_sample = np.count_nonzero(matrix, axis=1)
    nonzero_per_gene = np.count_nonzero(matrix, axis=0)
    total_signal = matrix.sum(axis=1)
    gene_variance = matrix.var(axis=0)
    missing_cells = int(np.isnan(matrix).sum())
    zero_entries = int(matrix.size - np.count_nonzero(matrix))

    sample_qc = pd.DataFrame(
        {
            "sample_barcode": expression_df.index,
            "total_signal": total_signal,
            "nonzero_gene_count": nonzero_per_sample,
            "zero_fraction": 1.0 - (nonzero_per_sample / n_genes),
        }
    )
    sample_qc["low_total_signal_outlier"] = _outlier_flags(
        sample_qc["total_signal"]
    ).astype(bool)

    gene_qc = pd.DataFrame(
        {
            "gene_id": expression_df.columns,
            "nonzero_sample_count": nonzero_per_gene,
            "zero_sample_count": n_samples - nonzero_per_gene,
            "variance": gene_variance,
            "all_zero": nonzero_per_gene == 0,
        }
    )

    qc_summary = {
        "expression_missing_cells": missing_cells,
        "expression_missing_rate": float(missing_cells / matrix.size),
        "expression_zero_entries": zero_entries,
        "expression_zero_rate": float(zero_entries / matrix.size),
        "genes_all_zero_count": int((nonzero_per_gene == 0).sum()),
        "samples_all_zero_count": int((nonzero_per_sample == 0).sum()),
        "low_total_signal_outlier_count": int(
            sample_qc["low_total_signal_outlier"].sum()
        ),
    }
    return sample_qc, gene_qc, qc_summary


def create_audit_tables_and_figures(
    samples_df: pd.DataFrame,
    sample_qc: pd.DataFrame,
    gene_qc: pd.DataFrame,
    qc_summary: dict[str, Any],
    reports_tables_dir: Path,
    reports_figures_dir: Path,
    *,
    value_type_guess: str,
) -> AuditArtifacts:
    table_paths: dict[str, str] = {}
    figure_paths: dict[str, str] = {}
    samples_augmented = samples_df.merge(sample_qc, how="left", on="sample_barcode")

    samples_by_type = (
        samples_augmented[
            samples_augmented["sample_type_code"].isin(FOCAL_SAMPLE_TYPES)
        ]
        .groupby(["sample_type_code", "sample_type_label"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_type_code")
    )
    subjects_by_type = (
        samples_augmented[
            samples_augmented["sample_type_code"].isin(FOCAL_SAMPLE_TYPES)
        ]
        .groupby(["sample_type_code", "sample_type_label"], dropna=False)["subject_id"]
        .nunique()
        .reset_index(name="subject_count")
        .sort_values("sample_type_code")
    )
    samples_by_tss = (
        samples_augmented.groupby("tss", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )
    subjects_by_tss = (
        samples_augmented.groupby("tss", dropna=False)["subject_id"]
        .nunique()
        .reset_index(name="subject_count")
        .sort_values("subject_count", ascending=False)
    )
    samples_by_center = (
        samples_augmented[samples_augmented["center"].notna()]
        .groupby("center", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )

    repeated_subjects = (
        samples_augmented.groupby("subject_id", dropna=False)
        .agg(
            sample_count=("sample_barcode", "size"),
            sample_types=(
                "sample_type_code",
                lambda s: "|".join(sorted({str(value) for value in s.dropna()})),
            ),
            sample_barcodes=("sample_barcode", lambda s: "|".join(map(str, s))),
        )
        .reset_index()
    )
    repeated_subjects = repeated_subjects[
        repeated_subjects["sample_count"] > 1
    ].sort_values(["sample_count", "subject_id"], ascending=[False, True])

    repeated_subject_sampletype = (
        samples_augmented.groupby(["subject_id", "sample_type_code"], dropna=False)
        .agg(
            sample_count=("sample_barcode", "size"),
            sample_barcodes=("sample_barcode", lambda s: "|".join(map(str, s))),
            domain_primary_values=(
                "domain_primary",
                lambda s: "|".join(sorted({str(value) for value in s.dropna()})),
            ),
        )
        .reset_index()
    )
    repeated_subject_sampletype = repeated_subject_sampletype[
        repeated_subject_sampletype["sample_count"] > 1
    ].sort_values(
        ["sample_count", "subject_id", "sample_type_code"],
        ascending=[False, True, True],
    )

    malformed_barcodes = samples_augmented[
        samples_augmented["barcode_parse_status"] != "parsed"
    ].copy()

    missingness_summary = pd.DataFrame(
        [
            {
                "metric": "expression_missing_cells",
                "value": qc_summary["expression_missing_cells"],
            },
            {
                "metric": "expression_missing_rate",
                "value": qc_summary["expression_missing_rate"],
            },
            {
                "metric": "expression_zero_entries",
                "value": qc_summary["expression_zero_entries"],
            },
            {
                "metric": "expression_zero_rate",
                "value": qc_summary["expression_zero_rate"],
            },
            {"metric": "malformed_barcodes", "value": int(len(malformed_barcodes))},
            {
                "metric": "samples_missing_tss",
                "value": int(samples_augmented["tss"].isna().sum()),
            },
            {
                "metric": "samples_missing_center",
                "value": int(samples_augmented["center"].isna().sum()),
            },
            {
                "metric": "samples_missing_plate",
                "value": int(samples_augmented["plate"].isna().sum()),
            },
            {
                "metric": "samples_missing_domain_primary",
                "value": int(samples_augmented["domain_primary"].isna().sum()),
            },
        ]
    )

    gene_zero_variance_summary = pd.DataFrame(
        [
            {"metric": "gene_count_total", "value": int(len(gene_qc))},
            {"metric": "genes_all_zero_count", "value": int(gene_qc["all_zero"].sum())},
            {
                "metric": "genes_variance_eq_0",
                "value": int((gene_qc["variance"] == 0).sum()),
            },
            {
                "metric": "genes_variance_le_1e-08",
                "value": int((gene_qc["variance"] <= 1e-8).sum()),
            },
            {
                "metric": "genes_variance_le_1e-06",
                "value": int((gene_qc["variance"] <= 1e-6).sum()),
            },
            {
                "metric": "genes_variance_le_1e-04",
                "value": int((gene_qc["variance"] <= 1e-4).sum()),
            },
            {
                "metric": "genes_variance_le_1e-03",
                "value": int((gene_qc["variance"] <= 1e-3).sum()),
            },
        ]
    )

    outputs = {
        "samples_by_sample_type.csv": samples_by_type,
        "subjects_by_sample_type.csv": subjects_by_type,
        "samples_by_tss.csv": samples_by_tss,
        "subjects_by_tss.csv": subjects_by_tss,
        "samples_by_center.csv": samples_by_center,
        "repeated_subjects.csv": repeated_subjects,
        "repeated_subject_sampletype_combinations.csv": repeated_subject_sampletype,
        "malformed_barcodes.csv": malformed_barcodes,
        "missingness_summary.csv": missingness_summary,
        "gene_zero_variance_summary.csv": gene_zero_variance_summary,
        "sample_qc_metrics.csv": sample_qc,
        "gene_qc_metrics.csv": gene_qc,
    }

    for file_name, dataframe in outputs.items():
        output_path = reports_tables_dir / file_name
        saved_path = save_dataframe(dataframe, output_path, index=False)
        table_paths[file_name] = str(saved_path)

    plot_bar(
        samples_by_type,
        x="sample_type_code",
        y="sample_count",
        title="TCGA-BRCA samples by sample type",
        ylabel="Samples",
        output_path=reports_figures_dir / "samples_by_sample_type.png",
        rotation=0,
    )
    figure_paths["samples_by_sample_type.png"] = str(
        reports_figures_dir / "samples_by_sample_type.png"
    )

    plot_bar(
        subjects_by_type,
        x="sample_type_code",
        y="subject_count",
        title="TCGA-BRCA subjects by sample type",
        ylabel="Subjects",
        output_path=reports_figures_dir / "subjects_by_sample_type.png",
        rotation=0,
    )
    figure_paths["subjects_by_sample_type.png"] = str(
        reports_figures_dir / "subjects_by_sample_type.png"
    )

    plot_bar(
        samples_by_tss.head(25),
        x="tss",
        y="sample_count",
        title="TCGA-BRCA samples by TSS",
        ylabel="Samples",
        output_path=reports_figures_dir / "samples_by_tss.png",
        rotation=60,
    )
    figure_paths["samples_by_tss.png"] = str(reports_figures_dir / "samples_by_tss.png")

    library_axis_label = (
        "Per-sample total counts"
        if value_type_guess == "likely_raw_counts"
        else "Per-sample total expression signal"
    )
    plot_histogram(
        sample_qc["total_signal"],
        title="Per-sample total signal distribution",
        xlabel=library_axis_label,
        output_path=reports_figures_dir / "per_sample_total_signal_histogram.png",
    )
    figure_paths["per_sample_total_signal_histogram.png"] = str(
        reports_figures_dir / "per_sample_total_signal_histogram.png"
    )

    plot_histogram(
        gene_qc["variance"],
        title="Gene-wise variance distribution",
        xlabel="Variance across samples",
        output_path=reports_figures_dir / "gene_variance_histogram.png",
    )
    figure_paths["gene_variance_histogram.png"] = str(
        reports_figures_dir / "gene_variance_histogram.png"
    )

    subject_type_matrix = (
        samples_augmented[
            samples_augmented["sample_type_code"].isin(FOCAL_SAMPLE_TYPES)
        ]
        .assign(flag=1)
        .pivot_table(
            index="subject_id",
            columns="sample_type_code",
            values="flag",
            fill_value=0,
            aggfunc="max",
        )
        .reset_index()
    )
    for code in FOCAL_SAMPLE_TYPES:
        if code not in subject_type_matrix.columns:
            subject_type_matrix[code] = 0

    summary_metrics = {
        "total_samples": int(len(samples_augmented)),
        "total_subjects": int(samples_augmented["subject_id"].nunique(dropna=True)),
        "total_genes": int(len(gene_qc)),
        "duplicate_sample_barcodes": int(
            samples_augmented["sample_barcode"].duplicated().sum()
        ),
        "subjects_with_multiple_samples": int(len(repeated_subjects)),
        "subjects_with_multiple_sample_types": int(
            (
                subject_type_matrix[FOCAL_SAMPLE_TYPES]
                .astype(int)
                .sum(axis=1)
                .gt(1)
                .sum()
            )
        ),
        "sample_type_counts": {
            row["sample_type_code"]: int(row["sample_count"])
            for _, row in samples_by_type.iterrows()
        },
        "subject_type_counts": {
            row["sample_type_code"]: int(row["subject_count"])
            for _, row in subjects_by_type.iterrows()
        },
        "subjects_with_01_and_11": int(
            ((subject_type_matrix["01"] > 0) & (subject_type_matrix["11"] > 0)).sum()
        ),
        "subjects_with_01_and_06": int(
            ((subject_type_matrix["01"] > 0) & (subject_type_matrix["06"] > 0)).sum()
        ),
        "subjects_with_11_and_06": int(
            ((subject_type_matrix["11"] > 0) & (subject_type_matrix["06"] > 0)).sum()
        ),
        "value_type_guess": value_type_guess,
        **qc_summary,
    }

    return AuditArtifacts(
        summary_metrics=summary_metrics,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )
