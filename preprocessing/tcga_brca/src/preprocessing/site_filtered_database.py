"""Preprocessing: site filtered database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe


@dataclass
class SiteFilterResult:
    site_counts: pd.DataFrame
    kept_sites: list[str]
    site_threshold: int
    filtered_samples: pd.DataFrame


@dataclass
class GeneFilterResult:
    gene_manifest: pd.DataFrame
    filtered_expression: pd.DataFrame
    summary: dict[str, Any]


def determine_kept_sites(
    samples_df: pd.DataFrame,
    *,
    allowed_tumor_levels: list[str],
    site_anchor: str,
) -> SiteFilterResult:
    subset = samples_df[
        samples_df["sample_type_code"].isin(allowed_tumor_levels)
    ].copy()
    site_counts = (
        subset.groupby("tss", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
        .reset_index(drop=True)
    )
    anchor_row = site_counts.loc[site_counts["tss"] == site_anchor]
    if anchor_row.empty:
        raise ValueError(
            f"Anchor site '{site_anchor}' was not found in the sample metadata."
        )

    site_threshold = int(anchor_row["sample_count"].iloc[0])
    kept_sites = (
        site_counts.loc[site_counts["sample_count"] >= site_threshold, "tss"]
        .astype(str)
        .tolist()
    )
    filtered_samples = subset[subset["tss"].isin(kept_sites)].copy()
    return SiteFilterResult(
        site_counts=site_counts,
        kept_sites=kept_sites,
        site_threshold=site_threshold,
        filtered_samples=filtered_samples,
    )


def load_expression_subset(
    expression_path: Path,
    keep_barcodes: list[str],
    *,
    chunksize: int = 128,
) -> pd.DataFrame:
    keep_set = set(keep_barcodes)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(expression_path, chunksize=chunksize):
        retained = chunk[chunk["sample_barcode"].isin(keep_set)].copy()
        if not retained.empty:
            chunks.append(retained)

    if not chunks:
        raise ValueError(
            "No samples matched between the standardized expression matrix and the requested filtered barcode set."
        )

    expression_df = pd.concat(chunks, ignore_index=True)
    expression_df = expression_df.set_index("sample_barcode")
    expression_df.index.name = "sample_barcode"

    gene_columns = expression_df.columns.tolist()
    expression_df[gene_columns] = expression_df[gene_columns].apply(
        pd.to_numeric, errors="coerce", downcast="float"
    )
    return expression_df


def filter_genes(
    expression_df: pd.DataFrame,
    *,
    min_detected_samples: int,
    min_variance: float,
) -> GeneFilterResult:
    matrix = expression_df.to_numpy(copy=False)
    n_samples = expression_df.shape[0]
    nonzero_counts = np.count_nonzero(matrix, axis=0)
    variance = matrix.var(axis=0)
    mean_expression = matrix.mean(axis=0)
    median_expression = np.median(matrix, axis=0)

    gene_manifest = pd.DataFrame(
        {
            "gene_id": expression_df.columns,
            "nonzero_sample_count": nonzero_counts,
            "detection_rate": nonzero_counts / n_samples,
            "variance": variance,
            "mean_expression": mean_expression,
            "median_expression": median_expression,
        }
    )
    gene_manifest["pass_detection_filter"] = (
        gene_manifest["nonzero_sample_count"] >= min_detected_samples
    )
    gene_manifest["pass_variance_filter"] = gene_manifest["variance"] > min_variance
    gene_manifest["keep_gene"] = (
        gene_manifest["pass_detection_filter"] & gene_manifest["pass_variance_filter"]
    )

    kept_gene_ids = gene_manifest.loc[gene_manifest["keep_gene"], "gene_id"].tolist()
    filtered_expression = expression_df.loc[:, kept_gene_ids].copy()

    summary = {
        "samples_retained": int(n_samples),
        "genes_before_filtering": int(expression_df.shape[1]),
        "genes_after_filtering": int(filtered_expression.shape[1]),
        "genes_removed": int(expression_df.shape[1] - filtered_expression.shape[1]),
        "min_detected_samples": int(min_detected_samples),
        "min_variance": float(min_variance),
        "genes_passing_detection": int(gene_manifest["pass_detection_filter"].sum()),
        "genes_passing_variance": int(gene_manifest["pass_variance_filter"].sum()),
    }
    return GeneFilterResult(
        gene_manifest=gene_manifest.sort_values(
            ["keep_gene", "variance", "nonzero_sample_count"],
            ascending=[False, False, False],
        ),
        filtered_expression=filtered_expression,
        summary=summary,
    )


def build_subject_gene_database(
    filtered_samples: pd.DataFrame,
    filtered_expression: pd.DataFrame,
) -> pd.DataFrame:
    metadata = filtered_samples.copy()
    metadata["site"] = metadata["tss"]
    metadata["tumor_level"] = metadata["sample_type_code"]
    metadata = metadata[
        [
            "sample_barcode",
            "subject_id",
            "site",
            "tumor_level",
            "sample_type_label",
            "domain_primary",
            "domain_source",
        ]
    ].drop_duplicates()

    gene_database = metadata.merge(
        filtered_expression.reset_index(), how="inner", on="sample_barcode"
    )
    return gene_database.sort_values(
        ["site", "tumor_level", "subject_id", "sample_barcode"]
    )


def save_site_filtered_outputs(
    *,
    output_root: Path,
    site_filter_result: SiteFilterResult,
    gene_filter_result: GeneFilterResult,
    subject_gene_database: pd.DataFrame,
) -> dict[str, str]:
    reports_tables_dir = output_root / "reports" / "tables"
    data_processed_dir = output_root / "data" / "processed"

    saved_paths = {
        "site_filter_summary": str(
            save_dataframe(
                site_filter_result.site_counts,
                reports_tables_dir / "tcga_brca_site_filter_summary.csv",
                index=False,
            )
        ),
        "gene_manifest": str(
            save_dataframe(
                gene_filter_result.gene_manifest,
                reports_tables_dir / "tcga_brca_site_filtered_gene_manifest.csv",
                index=False,
            )
        ),
        "subject_gene_database": str(
            save_dataframe(
                subject_gene_database,
                data_processed_dir
                / "tcga_brca_site_filtered_subject_gene_database.csv",
                index=False,
            )
        ),
    }

    gene_filter_summary_df = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in gene_filter_result.summary.items()
        ]
    )
    saved_paths["gene_filter_summary"] = str(
        save_dataframe(
            gene_filter_summary_df,
            reports_tables_dir / "tcga_brca_site_filtered_gene_filter_summary.csv",
            index=False,
        )
    )

    top_variable = gene_filter_result.gene_manifest.loc[
        gene_filter_result.gene_manifest["keep_gene"]
    ].head(100)
    saved_paths["top_variable_genes"] = str(
        save_dataframe(
            top_variable,
            reports_tables_dir / "tcga_brca_site_filtered_top_variable_genes.csv",
            index=False,
        )
    )
    return saved_paths


def render_site_filtered_report(
    *,
    site_filter_result: SiteFilterResult,
    gene_filter_result: GeneFilterResult,
    subject_gene_database: pd.DataFrame,
    saved_paths: dict[str, str],
) -> str:
    site_counts_kept = site_filter_result.site_counts.loc[
        site_filter_result.site_counts["tss"].isin(site_filter_result.kept_sites)
    ].copy()
    tumor_counts = (
        site_filter_result.filtered_samples.groupby("sample_type_code")
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_type_code")
    )
    top_variable = gene_filter_result.gene_manifest.loc[
        gene_filter_result.gene_manifest["keep_gene"]
    ].head(20)
    top_mean = (
        gene_filter_result.gene_manifest.loc[
            gene_filter_result.gene_manifest["keep_gene"]
        ]
        .sort_values("mean_expression", ascending=False)
        .head(20)
    )

    lines = [
        "# TCGA-BRCA Site-Filtered Processing",
        "",
        "## Site filtering",
        f"- Anchor site: `AN` with threshold `{site_filter_result.site_threshold}` samples.",
        f"- Kept sites: `{', '.join(site_filter_result.kept_sites)}`.",
        f"- Samples retained after site + tumor-level filtering: `{len(site_filter_result.filtered_samples)}`.",
        f"- Subjects retained: `{site_filter_result.filtered_samples['subject_id'].nunique()}`.",
        "",
        "### Retained site counts",
        site_counts_kept.to_string(index=False),
        "",
        "### Retained tumor-level counts",
        tumor_counts.to_string(index=False),
        "",
        "## Gene filtering",
        f"- Genes before filtering: `{gene_filter_result.summary['genes_before_filtering']}`.",
        f"- Genes after filtering: `{gene_filter_result.summary['genes_after_filtering']}`.",
        f"- Genes removed: `{gene_filter_result.summary['genes_removed']}`.",
        f"- Detection threshold: nonzero expression in at least `{gene_filter_result.summary['min_detected_samples']}` retained samples.",
        f"- Variance threshold: strictly greater than `{gene_filter_result.summary['min_variance']}`.",
        "",
        "### Top variable retained genes",
        top_variable[
            ["gene_id", "variance", "nonzero_sample_count", "mean_expression"]
        ].to_string(index=False),
        "",
        "### Top mean-expression retained genes",
        top_mean[
            ["gene_id", "mean_expression", "variance", "nonzero_sample_count"]
        ].to_string(index=False),
        "",
        "## Final database",
        f"- Final sample-level database shape: `{subject_gene_database.shape[0]} rows x {subject_gene_database.shape[1]} columns`.",
        "- Metadata columns added to the gene matrix: `subject_id`, `site`, `tumor_level`, `sample_type_label`, `domain_primary`, `domain_source`.",
        f"- Database file: `{saved_paths['subject_gene_database']}`.",
    ]
    return "\n".join(lines) + "\n"
