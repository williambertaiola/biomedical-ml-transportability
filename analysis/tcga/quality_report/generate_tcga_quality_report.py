"""Post-experiment analysis: generate tcga quality report."""

from __future__ import annotations

# Locate the repository without depending on the current working directory.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys

_REPOSITORY_ROOT = next(
    parent
    for parent in _RepositoryPath(__file__).resolve().parents
    if (parent / ".thesis-root").is_file()
)
_repository_sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
from thesis_tools.paths import configured_path as _configured_path


import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    _configured_path(
        "results_data_tcga_brca_preprocessing_data_processed_tcga_brca_stage_transportability_dataset.csv"
    )
)
STAGE_SUMMARY_JSON = Path(
    _configured_path(
        "results_data_tcga_brca_preprocessing_data_processed_tcga_brca_stage_ready_summary.json"
    )
)
MISSINGNESS_CSV = Path(
    _configured_path(
        "results_data_tcga_brca_preprocessing_reports_tables_missingness_summary.csv"
    )
)
OUTPUT_DIR = Path(_configured_path("results_tcga_quality_report"))


COLUMN_DESCRIPTIONS = {
    "sample_barcode": {
        "role": "identifier",
        "group": "sample_identifier",
        "conceptual_type": "TCGA aliquot barcode",
        "description": "Original TCGA sample barcode carried into the final dataset.",
        "notes": "Useful for audit trails and joins, not intended as a predictive feature.",
    },
    "project_code": {
        "role": "metadata",
        "group": "project_metadata",
        "conceptual_type": "project label",
        "description": "TCGA project code for the cohort.",
        "notes": "Constant in this dataset because all rows belong to TCGA-BRCA.",
    },
    "tss": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "tissue source site code",
        "description": "TCGA tissue source site code parsed from the barcode.",
        "notes": "Closely related to the site/domain definition used downstream.",
    },
    "participant_id": {
        "role": "identifier",
        "group": "subject_identifier",
        "conceptual_type": "participant identifier",
        "description": "TCGA participant-level identifier.",
        "notes": "Useful to trace subject-level provenance before deriving subject_id.",
    },
    "subject_id": {
        "role": "identifier",
        "group": "subject_identifier",
        "conceptual_type": "subject identifier",
        "description": "Subject-level identifier used in the final transportability table.",
        "notes": "Should be unique in the final CSV and is excluded from modeling.",
    },
    "sample_id": {
        "role": "identifier",
        "group": "sample_identifier",
        "conceptual_type": "sample identifier",
        "description": "Intermediate sample identifier used during preprocessing.",
        "notes": "Useful for sanity checks and joins across preprocessing steps.",
    },
    "sample_type_code": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "coded categorical",
        "description": "TCGA sample type code parsed from the barcode.",
        "notes": "Expected to be low-cardinality and tightly linked to the selected tumor sample subset.",
    },
    "sample_type_label": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "categorical string",
        "description": "Human-readable TCGA sample type label.",
        "notes": "Usually constant or nearly constant after filtering to primary tumors.",
    },
    "vial": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "barcode component",
        "description": "TCGA vial component parsed from the barcode.",
        "notes": "Retained for provenance; often low-information for modeling.",
    },
    "portion": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "barcode component",
        "description": "TCGA portion component parsed from the barcode.",
        "notes": "Completely missing in the final CSV.",
    },
    "analyte": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "barcode component",
        "description": "TCGA analyte component parsed from the barcode.",
        "notes": "Completely missing in the final CSV.",
    },
    "plate": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "barcode component",
        "description": "TCGA plate component parsed from the barcode.",
        "notes": "Completely missing in the final CSV; plate missingness was already noted during preprocessing.",
    },
    "center": {
        "role": "metadata",
        "group": "sample_metadata",
        "conceptual_type": "barcode component",
        "description": "TCGA center component parsed from the barcode.",
        "notes": "Completely missing in the final CSV; center missingness was already noted during preprocessing.",
    },
    "barcode_scope": {
        "role": "metadata",
        "group": "qc_metadata",
        "conceptual_type": "processing flag",
        "description": "Scope used while parsing TCGA barcodes.",
        "notes": "Useful for auditing barcode parsing choices rather than modeling.",
    },
    "barcode_parse_status": {
        "role": "metadata",
        "group": "qc_metadata",
        "conceptual_type": "processing flag",
        "description": "Status flag from the barcode parser.",
        "notes": "Should be constant if barcode parsing completed cleanly.",
    },
    "barcode_parse_note": {
        "role": "metadata",
        "group": "qc_metadata",
        "conceptual_type": "processing note",
        "description": "Free-text note emitted by the barcode parser.",
        "notes": "Useful for audit, not for predictive modeling.",
    },
    "clinical_case_id": {
        "role": "identifier",
        "group": "clinical_identifier",
        "conceptual_type": "clinical case identifier",
        "description": "Clinical case identifier linked to TCGA metadata.",
        "notes": "High-cardinality join key; excluded from modeling.",
    },
    "clinical_primary_site": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical primary site label.",
        "notes": "Expected to be constant for the BRCA cohort.",
    },
    "clinical_ethnicity": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical ethnicity annotation.",
        "notes": "Contains moderate missingness and should be handled as categorical if used.",
    },
    "clinical_gender": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical gender annotation.",
        "notes": "Strongly imbalanced toward female cases in this BRCA cohort.",
    },
    "clinical_race": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical race annotation.",
        "notes": "Contains non-negligible missingness and site-dependent composition.",
    },
    "clinical_ajcc_pathologic_stage": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "ordinal categorical string",
        "description": "Original AJCC pathologic stage annotation from the clinical table.",
        "notes": "Kept for interpretation; the modeling target is the simplified binary stage endpoint.",
    },
    "clinical_primary_diagnosis": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical primary diagnosis text field.",
        "notes": "Potentially informative but high-cardinality for direct modeling.",
    },
    "clinical_site_of_resection_or_biopsy": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical site of resection or biopsy.",
        "notes": "Usually constant or low-information within a single cancer type cohort.",
    },
    "clinical_tissue_or_organ_of_origin": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "categorical string",
        "description": "Clinical tissue or organ of origin annotation.",
        "notes": "Useful for audit and biological interpretation.",
    },
    "clinical_record_count": {
        "role": "metadata",
        "group": "clinical_metadata",
        "conceptual_type": "numeric count",
        "description": "Count of clinical records linked to the sample during preprocessing.",
        "notes": "Quality/provenance variable rather than a biological predictor.",
    },
    "domain_primary": {
        "role": "metadata",
        "group": "domain_metadata",
        "conceptual_type": "site/domain label",
        "description": "Primary site/domain assignment derived during preprocessing.",
        "notes": "Matches the multi-site transportability framing.",
    },
    "domain_source": {
        "role": "metadata",
        "group": "domain_metadata",
        "conceptual_type": "source description",
        "description": "String describing how the domain assignment was obtained.",
        "notes": "Useful for provenance, not typically for modeling.",
    },
    "domain_secondary_candidates": {
        "role": "metadata",
        "group": "domain_metadata",
        "conceptual_type": "candidate site notes",
        "description": "Alternative site/domain candidates retained during preprocessing.",
        "notes": "Mostly useful for audit and ambiguity tracking.",
    },
    "total_signal": {
        "role": "qc_feature",
        "group": "expression_qc",
        "conceptual_type": "numeric continuous",
        "description": "Per-sample total expression signal across the retained gene matrix.",
        "notes": "Useful as a global sample-level quality indicator.",
    },
    "nonzero_gene_count": {
        "role": "qc_feature",
        "group": "expression_qc",
        "conceptual_type": "numeric count",
        "description": "Number of genes with non-zero expression in each sample.",
        "notes": "Captures sample sparsity and sequencing complexity.",
    },
    "zero_fraction": {
        "role": "qc_feature",
        "group": "expression_qc",
        "conceptual_type": "numeric proportion",
        "description": "Fraction of zero-valued gene entries within each sample.",
        "notes": "A compact summary of sample sparsity; higher values may indicate lower complexity.",
    },
    "low_total_signal_outlier": {
        "role": "qc_feature",
        "group": "expression_qc",
        "conceptual_type": "boolean flag",
        "description": "Flag marking samples with unusually low total signal.",
        "notes": "Should be reviewed before strong conclusions about transportability.",
    },
    "site": {
        "role": "metadata",
        "group": "domain_metadata",
        "conceptual_type": "site/domain label",
        "description": "Final site variable used as the domain identifier in cross-site experiments.",
        "notes": "This is the key domain variable for the transportability protocol.",
    },
    "ajcc_pathologic_stage_raw": {
        "role": "label",
        "group": "target",
        "conceptual_type": "ordinal categorical string",
        "description": "Raw AJCC pathologic stage string retained before binary simplification.",
        "notes": "Useful for interpretation and audit; not the final modeling label.",
    },
    "target_label": {
        "role": "label",
        "group": "target",
        "conceptual_type": "binary categorical string",
        "description": "Human-readable binary stage endpoint used downstream.",
        "notes": "Observed values: Stage I-II vs Stage III-IV.",
    },
    "target": {
        "role": "label",
        "group": "target",
        "conceptual_type": "binary target",
        "description": "Binary target used for transportability modeling.",
        "notes": "0 = Stage I-II, 1 = Stage III-IV.",
    },
}


def first_values(series: pd.Series, n: int = 6) -> str:
    values = series.dropna().unique().tolist()[:n]
    rendered: list[str] = []
    for value in values:
        if isinstance(value, (np.integer, int)):
            rendered.append(str(int(value)))
        elif isinstance(value, (np.floating, float)) and float(value).is_integer():
            rendered.append(str(int(value)))
        else:
            rendered.append(str(value))
    return ", ".join(rendered)


def markdown_table(
    df: pd.DataFrame,
    round_cols: dict[str, int] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    if round_cols:
        for column, digits in round_cols.items():
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda x: round(float(x), digits) if pd.notna(x) else x
                )

    df = df.fillna("NA")
    headers = [str(col) for col in df.columns]
    rows = [
        [str(value) for value in row] for row in df.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        if rows
        else len(headers[idx])
        for idx in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    return lines


def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    stage_summary = json.loads(STAGE_SUMMARY_JSON.read_text(encoding="utf-8"))
    missingness = pd.read_csv(MISSINGNESS_CSV)
    return df, stage_summary, missingness


def build_metadata_inventory(
    df: pd.DataFrame, metadata_cols: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in metadata_cols:
        series = df[column]
        desc = COLUMN_DESCRIPTIONS.get(
            column,
            {
                "role": "metadata",
                "group": "other_metadata",
                "conceptual_type": "unknown",
                "description": "",
                "notes": "",
            },
        )

        row = {
            "column_name": column,
            "role": desc["role"],
            "variable_group": desc["group"],
            "conceptual_type": desc["conceptual_type"],
            "pandas_dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100.0),
            "unique_non_null": int(series.nunique(dropna=True)),
            "example_values": first_values(series),
            "description": desc["description"],
            "notes": desc["notes"],
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            row.update(
                {
                    "min": float(series.min()) if series.notna().any() else np.nan,
                    "q25": float(series.quantile(0.25))
                    if series.notna().any()
                    else np.nan,
                    "median": float(series.median())
                    if series.notna().any()
                    else np.nan,
                    "q75": float(series.quantile(0.75))
                    if series.notna().any()
                    else np.nan,
                    "max": float(series.max()) if series.notna().any() else np.nan,
                    "mean": float(series.mean()) if series.notna().any() else np.nan,
                    "std": float(series.std()) if series.notna().any() else np.nan,
                }
            )
        else:
            row.update(
                {
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
        rows.append(row)

    inventory = (
        pd.DataFrame(rows)
        .sort_values(["role", "variable_group", "column_name"])
        .reset_index(drop=True)
    )
    return inventory


def build_metadata_missingness(metadata_inventory: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata_inventory.loc[
            :,
            [
                "column_name",
                "role",
                "variable_group",
                "missing_count",
                "missing_pct",
                "unique_non_null",
            ],
        ]
        .sort_values(["missing_count", "column_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_site_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("site", dropna=False)
    summary = grouped.apply(
        lambda x: pd.Series(
            {
                "n_samples": len(x),
                "n_subjects": x["subject_id"].nunique(dropna=True),
                "n_early": int((x["target"] == 0).sum()),
                "n_late": int((x["target"] == 1).sum()),
                "positive_rate": float(x["target"].mean()),
                "mean_total_signal": float(x["total_signal"].mean()),
                "sd_total_signal": float(x["total_signal"].std()),
                "mean_nonzero_gene_count": float(x["nonzero_gene_count"].mean()),
                "mean_zero_fraction": float(x["zero_fraction"].mean()),
                "low_total_signal_outliers": int(
                    x["low_total_signal_outlier"].astype(bool).sum()
                ),
            }
        ),
        include_groups=False,
    )
    summary = (
        summary.reset_index()
        .sort_values("n_samples", ascending=False)
        .reset_index(drop=True)
    )
    summary["n_samples"] = summary["n_samples"].astype(int)
    summary["n_subjects"] = summary["n_subjects"].astype(int)
    summary["n_early"] = summary["n_early"].astype(int)
    summary["n_late"] = summary["n_late"].astype(int)
    summary["low_total_signal_outliers"] = summary["low_total_signal_outliers"].astype(
        int
    )
    return summary


def build_selected_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "site",
        "target_label",
        "clinical_gender",
        "clinical_race",
        "clinical_ethnicity",
        "sample_type_label",
        "domain_primary",
        "low_total_signal_outlier",
    ]
    rows: list[dict[str, object]] = []
    for column in selected_columns:
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False)
        rendered = []
        for key, value in counts.items():
            rendered_key = "NA" if pd.isna(key) else str(key)
            rendered.append(f"{rendered_key}: {int(value)}")
        rows.append(
            {
                "column_name": column,
                "missing_count": int(df[column].isna().sum()),
                "unique_non_null": int(df[column].nunique(dropna=True)),
                "levels_and_counts": "; ".join(rendered),
            }
        )
    return pd.DataFrame(rows)


def build_gene_matrix_summary(
    df: pd.DataFrame, gene_cols: list[str], missingness_lookup: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_df = df[gene_cols]
    variances = gene_df.var(axis=0)
    variance_quantiles = variances.quantile(
        [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    )

    gene_summary = pd.DataFrame(
        [
            {
                "metric": "n_samples",
                "value": int(len(df)),
                "notes": "Number of rows in the final transportability dataset.",
            },
            {
                "metric": "n_gene_features",
                "value": int(len(gene_cols)),
                "notes": "Count of ENSG gene-expression columns used as predictors.",
            },
            {
                "metric": "expression_missing_cells",
                "value": float(
                    missingness_lookup.get(
                        "expression_missing_cells", float(gene_df.isna().sum().sum())
                    )
                ),
                "notes": "Missing gene-expression cells reported by preprocessing QC.",
            },
            {
                "metric": "expression_missing_rate",
                "value": float(
                    missingness_lookup.get(
                        "expression_missing_rate", float(gene_df.isna().mean().mean())
                    )
                ),
                "notes": "Overall missing-value rate within the gene-expression matrix.",
            },
            {
                "metric": "expression_zero_entries",
                "value": float(
                    missingness_lookup.get(
                        "expression_zero_entries", float((gene_df == 0).sum().sum())
                    )
                ),
                "notes": "Total number of zero-valued entries in the gene-expression matrix.",
            },
            {
                "metric": "expression_zero_rate",
                "value": float(
                    missingness_lookup.get(
                        "expression_zero_rate", float((gene_df == 0).mean().mean())
                    )
                ),
                "notes": "Overall zero-entry rate in the gene-expression matrix.",
            },
            {
                "metric": "zero_variance_genes",
                "value": int((variances <= 0).sum()),
                "notes": "Genes with no variance across the final cohort and therefore no discriminative value.",
            },
            {
                "metric": "mean_total_signal",
                "value": float(df["total_signal"].mean()),
                "notes": "Average per-sample total expression signal.",
            },
            {
                "metric": "mean_nonzero_gene_count",
                "value": float(df["nonzero_gene_count"].mean()),
                "notes": "Average number of genes with non-zero expression per sample.",
            },
            {
                "metric": "mean_zero_fraction",
                "value": float(df["zero_fraction"].mean()),
                "notes": "Average per-sample gene-expression sparsity.",
            },
            {
                "metric": "low_total_signal_outlier_count",
                "value": int(df["low_total_signal_outlier"].astype(bool).sum()),
                "notes": "Samples flagged as unusually low-total-signal outliers.",
            },
        ]
    )

    variance_summary = pd.DataFrame(
        {
            "variance_quantile": variance_quantiles.index.astype(str),
            "variance_value": variance_quantiles.values,
        }
    )
    top_variable_genes = (
        variances.sort_values(ascending=False)
        .head(25)
        .rename("variance")
        .reset_index()
        .rename(columns={"index": "gene"})
    )
    combined = pd.DataFrame(
        {
            "variance_quantile": variance_summary["variance_quantile"],
            "variance_value": variance_summary["variance_value"],
            "top_variable_gene": pd.Series(top_variable_genes["gene"]),
            "top_gene_variance": pd.Series(top_variable_genes["variance"]),
        }
    )
    return gene_summary, combined


def build_quality_flags(
    df: pd.DataFrame,
    metadata_missingness: pd.DataFrame,
    gene_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    stage_summary: dict,
) -> pd.DataFrame:
    gene_lookup = dict(zip(gene_summary["metric"], gene_summary["value"]))
    fully_missing_metadata = metadata_missingness.loc[
        metadata_missingness["missing_pct"] == 100.0, "column_name"
    ].tolist()
    site_prevalence_range = float(
        site_summary["positive_rate"].max() - site_summary["positive_rate"].min()
    )

    rows = [
        {
            "flag": "final_csv_exact_duplicate_rows",
            "value": int(df.duplicated().sum()),
            "severity": "info" if int(df.duplicated().sum()) == 0 else "warning",
            "notes": "Direct duplicate-row check on the final stage transportability CSV.",
        },
        {
            "flag": "duplicated_subject_id_rows",
            "value": int(df["subject_id"].duplicated().sum()),
            "severity": "info"
            if int(df["subject_id"].duplicated().sum()) == 0
            else "warning",
            "notes": "Duplicated subject_id values in the final CSV.",
        },
        {
            "flag": "upstream_duplicate_primary_subjects",
            "value": int(stage_summary.get("duplicate_primary_subjects", 0)),
            "severity": "review"
            if int(stage_summary.get("duplicate_primary_subjects", 0)) > 0
            else "info",
            "notes": "Duplicate primary-subject count reported upstream before the final filtered dataset.",
        },
        {
            "flag": "metadata_columns_fully_missing",
            "value": int(len(fully_missing_metadata)),
            "severity": "warning" if fully_missing_metadata else "info",
            "notes": "Metadata columns with 100% missingness: "
            + (", ".join(fully_missing_metadata) if fully_missing_metadata else "none"),
        },
        {
            "flag": "clinical_ethnicity_missing",
            "value": int(df["clinical_ethnicity"].isna().sum()),
            "severity": "review"
            if int(df["clinical_ethnicity"].isna().sum()) > 0
            else "info",
            "notes": "Missing ethnicity labels in the final dataset.",
        },
        {
            "flag": "clinical_race_missing",
            "value": int(df["clinical_race"].isna().sum()),
            "severity": "review"
            if int(df["clinical_race"].isna().sum()) > 0
            else "info",
            "notes": "Missing race labels in the final dataset.",
        },
        {
            "flag": "zero_variance_genes",
            "value": int(gene_lookup["zero_variance_genes"]),
            "severity": "review"
            if int(gene_lookup["zero_variance_genes"]) > 0
            else "info",
            "notes": "Gene columns with no variance across samples.",
        },
        {
            "flag": "low_total_signal_outliers",
            "value": int(df["low_total_signal_outlier"].astype(bool).sum()),
            "severity": "review"
            if int(df["low_total_signal_outlier"].astype(bool).sum()) > 0
            else "info",
            "notes": "Samples flagged by the preprocessing QC as low-total-signal outliers.",
        },
        {
            "flag": "high_zero_fraction_samples_gt_0_55",
            "value": int((df["zero_fraction"] > 0.55).sum()),
            "severity": "review"
            if int((df["zero_fraction"] > 0.55).sum()) > 0
            else "info",
            "notes": "Samples with especially sparse expression profiles.",
        },
        {
            "flag": "site_positive_rate_range",
            "value": round(site_prevalence_range, 6),
            "severity": "review" if site_prevalence_range >= 0.15 else "info",
            "notes": "Difference between maximum and minimum site-level positive rate.",
        },
    ]
    return pd.DataFrame(rows)


def build_gene_block_dictionary_row(gene_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column_name": "GENE_EXPRESSION_BLOCK",
                "role": "predictive_feature_block",
                "variable_group": "gene_expression",
                "conceptual_type": "high-dimensional numeric matrix",
                "pandas_dtype": "float64",
                "missing_count": 0,
                "missing_pct": 0.0,
                "unique_non_null": len(gene_cols),
                "description": "All ENSG-prefixed RNA-seq expression features retained in the final TCGA BRCA transportability dataset.",
                "notes": "There are 60k+ gene columns; they are summarized as a block rather than documented one by one in the report.",
            }
        ]
    )


def build_report(
    df: pd.DataFrame,
    metadata_inventory: pd.DataFrame,
    metadata_missingness: pd.DataFrame,
    site_summary: pd.DataFrame,
    selected_categorical_summary: pd.DataFrame,
    gene_summary: pd.DataFrame,
    gene_variance_summary: pd.DataFrame,
    quality_flags: pd.DataFrame,
    stage_summary: dict,
    gene_cols: list[str],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    gene_lookup = dict(zip(gene_summary["metric"], gene_summary["value"]))
    max_site = site_summary.loc[site_summary["positive_rate"].idxmax(), "site"]
    min_site = site_summary.loc[site_summary["positive_rate"].idxmin(), "site"]
    gene_block_row = build_gene_block_dictionary_row(gene_cols)
    dictionary_view = pd.concat(
        [
            metadata_inventory.loc[
                :,
                [
                    "column_name",
                    "role",
                    "variable_group",
                    "conceptual_type",
                    "pandas_dtype",
                    "missing_count",
                    "missing_pct",
                    "unique_non_null",
                    "description",
                    "notes",
                ],
            ],
            gene_block_row,
        ],
        ignore_index=True,
    )

    lines: list[str] = [
        "# TCGA BRCA quality report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Dataset audited",
        f"- Input CSV: `{INPUT_CSV}`",
        f"- Stage summary JSON: `{STAGE_SUMMARY_JSON}`",
        f"- Missingness summary CSV: `{MISSINGNESS_CSV}`",
        f"- Rows: `{len(df)}`",
        f"- Columns: `{df.shape[1]}`",
        f"- Metadata columns: `{len([c for c in df.columns if not c.startswith('ENSG')])}`",
        f"- Gene-expression columns: `{len(gene_cols)}`",
        f"- Unique subject_id values: `{df['subject_id'].nunique(dropna=True)}`",
        f"- Valid sites/domains: `{df['site'].nunique(dropna=True)}`",
        f"- Binary target prevalence: `{df['target'].mean():.4f}`",
        "",
        "## What this file contains",
        "- A plain-language description of the final TCGA BRCA stage transportability dataset.",
        "- Variable roles and semantic types for all metadata columns in the final CSV.",
        "- A dedicated summary of the high-dimensional gene-expression block.",
        "- Site-level class balance and sample-quality asymmetries that may affect cross-site transportability.",
        "- Quality flags worth mentioning before extending M0-M4 to TCGA.",
        "",
        "## Variable groups",
        "- Identifiers: sample and subject linkage columns retained for provenance only.",
        "- Sample metadata: barcode-derived descriptors such as `tss`, `sample_type`, `vial`, `plate`, and `center`.",
        "- Clinical metadata: demographic and staging context carried over from the clinical table.",
        "- Domain metadata: site/domain labels used to define cross-site experiments.",
        "- Expression QC features: per-sample summaries such as `total_signal`, `nonzero_gene_count`, and `zero_fraction`.",
        "- Target columns: raw stage labels plus the binary endpoint used in transportability modeling.",
        "- Gene expression block: all ENSG-prefixed RNA-seq features used as predictors.",
        "",
        "## Dataset content summary",
        "- The final table is a subject-level, stage-labeled TCGA BRCA dataset prepared for cross-site transportability experiments.",
        "- Each row corresponds to one final subject/sample entry with one site/domain label and one binary stage target.",
        "- The predictive signal is dominated by a high-dimensional RNA-seq matrix with more than sixty thousand gene columns.",
        "- Sample-level QC summaries are already included in the table and are useful for descriptive auditing even if not always used in predictive models.",
        "",
        "## Site summary",
        "",
    ]
    lines.extend(
        markdown_table(
            site_summary,
            round_cols={
                "positive_rate": 4,
                "mean_total_signal": 1,
                "sd_total_signal": 1,
                "mean_nonzero_gene_count": 1,
                "mean_zero_fraction": 4,
            },
        )
    )
    lines.extend(
        [
            "",
            "## Metadata missingness",
            "",
        ]
    )
    lines.extend(markdown_table(metadata_missingness, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Gene-expression block summary",
            "",
        ]
    )
    lines.extend(markdown_table(gene_summary, round_cols={"value": 6}))
    lines.extend(
        [
            "",
            "## Gene variance quantiles and top variable genes",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            gene_variance_summary,
            round_cols={"variance_value": 6, "top_gene_variance": 6},
            max_rows=25,
        )
    )
    lines.extend(
        [
            "",
            "## Selected categorical summaries",
            "",
        ]
    )
    lines.extend(markdown_table(selected_categorical_summary))
    lines.extend(
        [
            "",
            "## Main quality findings",
            f"- The final CSV contains `{int(df.duplicated().sum())}` exact duplicated rows and `{int(df['subject_id'].duplicated().sum())}` duplicated `subject_id` values, so the final transportability table is clean at subject level.",
            f"- Upstream preprocessing reported `{int(stage_summary.get('duplicate_primary_subjects', 0))}` duplicate primary-subject entries before the final filtering and subject-level resolution step, so the current CSV is already a cleaned downstream view.",
            f"- The final cohort contains `{len(df)}` samples from `{int(stage_summary.get('n_valid_sites_after_min_class_filter', df['site'].nunique()))}` valid sites, after reducing the broader stage-labeled pool to the subset suitable for cross-site experiments.",
            f"- Class balance is heterogeneous across sites: the lowest positive rate is `{site_summary['positive_rate'].min():.3f}` at `{min_site}`, and the highest is `{site_summary['positive_rate'].max():.3f}` at `{max_site}`.",
            f"- The gene-expression matrix has `{int(gene_lookup['zero_variance_genes'])}` zero-variance genes and an overall zero-entry rate of `{float(gene_lookup['expression_zero_rate']):.4f}`, indicating strong sparsity but no missing expression cells.",
            f"- The average sample has `{float(gene_lookup['mean_nonzero_gene_count']):.1f}` non-zero genes and a mean zero fraction of `{float(gene_lookup['mean_zero_fraction']):.4f}`.",
            f"- `{int(df['low_total_signal_outlier'].astype(bool).sum())}` samples are flagged as low-total-signal outliers, which is a modest but non-zero subset worth keeping in mind during model interpretation.",
            f"- Four barcode-derived metadata columns are fully missing in the final CSV (`portion`, `analyte`, `plate`, `center`), so they should be treated as provenance artifacts rather than usable predictors.",
            f"- Clinical metadata are mostly complete, but `clinical_ethnicity` is missing in `{int(df['clinical_ethnicity'].isna().sum())}` rows and `clinical_race` in `{int(df['clinical_race'].isna().sum())}` rows.",
            f"- `clinical_gender` is extremely imbalanced (`female={int((df['clinical_gender'] == 'female').sum())}`, `male={int((df['clinical_gender'] == 'male').sum())}`), which is expected for BRCA but should still be acknowledged descriptively.",
            "",
            "## Variable dictionary",
            "",
        ]
    )
    lines.extend(markdown_table(dictionary_view, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
        ]
    )
    lines.extend(markdown_table(quality_flags))
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "- This report describes the final analysis-ready TCGA BRCA transportability dataset, not the raw GDC download.",
            "- The `site` column is the domain variable to use for the same cross-site split protocol already applied to UCI: target-site 50/50 split into `train_1` and `test_1`, all remaining sites used as sources.",
            "- The gene-expression matrix is high dimensional and sparse, so train-only feature filtering and selection remain essential for leakage-safe modeling.",
            "- Metadata completeness is good overall, but some barcode fragments are structurally missing and should not be over-interpreted.",
            "",
            "## Companion files",
            "- `TCGA_BRCA_quality_report.rdm`: requested human-readable report file.",
            "- `TCGA_BRCA_quality_report.md`: markdown twin of the same report.",
            "- `tcga_metadata_inventory.csv`: metadata dictionary with roles, types, missingness, and notes.",
            "- `tcga_metadata_missingness.csv`: compact metadata missingness table.",
            "- `tcga_site_quality_summary.csv`: site-level counts, prevalence, and expression QC summaries.",
            "- `tcga_selected_categorical_summary.csv`: readable counts for the main categorical variables.",
            "- `tcga_gene_matrix_summary.csv`: overall summary metrics for the gene-expression matrix.",
            "- `tcga_gene_variance_summary.csv`: variance quantiles plus top variable genes.",
            "- `tcga_quality_flags.csv`: compact list of QC attention points.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, stage_summary, missingness = load_inputs()

    gene_cols = [column for column in df.columns if column.startswith("ENSG")]
    metadata_cols = [column for column in df.columns if column not in gene_cols]
    missingness_lookup = dict(zip(missingness["metric"], missingness["value"]))

    metadata_inventory = build_metadata_inventory(df, metadata_cols)
    metadata_missingness = build_metadata_missingness(metadata_inventory)
    site_summary = build_site_quality_summary(df)
    selected_categorical_summary = build_selected_categorical_summary(df)
    gene_summary, gene_variance_summary = build_gene_matrix_summary(
        df, gene_cols, missingness_lookup
    )
    quality_flags = build_quality_flags(
        df, metadata_missingness, gene_summary, site_summary, stage_summary
    )

    metadata_inventory.to_csv(OUTPUT_DIR / "tcga_metadata_inventory.csv", index=False)
    metadata_missingness.to_csv(
        OUTPUT_DIR / "tcga_metadata_missingness.csv", index=False
    )
    site_summary.to_csv(OUTPUT_DIR / "tcga_site_quality_summary.csv", index=False)
    selected_categorical_summary.to_csv(
        OUTPUT_DIR / "tcga_selected_categorical_summary.csv", index=False
    )
    gene_summary.to_csv(OUTPUT_DIR / "tcga_gene_matrix_summary.csv", index=False)
    gene_variance_summary.to_csv(
        OUTPUT_DIR / "tcga_gene_variance_summary.csv", index=False
    )
    quality_flags.to_csv(OUTPUT_DIR / "tcga_quality_flags.csv", index=False)

    report_text = build_report(
        df=df,
        metadata_inventory=metadata_inventory,
        metadata_missingness=metadata_missingness,
        site_summary=site_summary,
        selected_categorical_summary=selected_categorical_summary,
        gene_summary=gene_summary,
        gene_variance_summary=gene_variance_summary,
        quality_flags=quality_flags,
        stage_summary=stage_summary,
        gene_cols=gene_cols,
    )
    (OUTPUT_DIR / "TCGA_BRCA_quality_report.rdm").write_text(
        report_text, encoding="utf-8"
    )
    (OUTPUT_DIR / "TCGA_BRCA_quality_report.md").write_text(
        report_text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
