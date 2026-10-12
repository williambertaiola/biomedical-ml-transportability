"""Transportability experiment: reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.io import dataframe_to_markdown


def render_audit_report(
    *,
    expression_result,
    samples_df: pd.DataFrame,
    clinical_summary: dict[str, int],
    metadata_inventory: dict[str, list[str]],
    audit_artifacts,
    blueprint: dict[str, Any],
) -> str:
    metrics = audit_artifacts.summary_metrics

    samples_by_type = pd.read_csv(
        audit_artifacts.table_paths["samples_by_sample_type.csv"],
        dtype={"sample_type_code": str},
    )
    subjects_by_type = pd.read_csv(
        audit_artifacts.table_paths["subjects_by_sample_type.csv"],
        dtype={"sample_type_code": str},
    )
    samples_by_tss = pd.read_csv(
        audit_artifacts.table_paths["samples_by_tss.csv"], dtype={"tss": str}
    ).head(15)
    subjects_by_tss = pd.read_csv(
        audit_artifacts.table_paths["subjects_by_tss.csv"], dtype={"tss": str}
    ).head(15)
    repeated_subjects = pd.read_csv(
        audit_artifacts.table_paths["repeated_subjects.csv"]
    ).head(15)
    gene_zero_variance_summary = pd.read_csv(
        audit_artifacts.table_paths["gene_zero_variance_summary.csv"]
    )

    missing_technical_fields = [
        field
        for field in ["portion", "analyte", "plate", "center"]
        if field in samples_df.columns and samples_df[field].notna().sum() == 0
    ]
    if "clinical_record_count" in samples_df.columns:
        unmatched_subjects = (
            samples_df.loc[samples_df["clinical_record_count"].isna(), "subject_id"]
            .dropna()
            .nunique()
        )
    else:
        unmatched_subjects = samples_df["subject_id"].dropna().nunique()

    report_lines = [
        "# TCGA-BRCA Initial Audit",
        "",
        "## Dataset overview",
        f"- Input file: `{expression_result.original_shape[0]} x {expression_result.original_shape[1]}` raw matrix loaded from the supplied STAR-derived expression file.",
        f"- Original raw shape: `{expression_result.original_shape[0]} rows x {expression_result.original_shape[1]} columns`.",
        f"- Detected delimiter: `{expression_result.delimiter}`.",
        f"- Detected orientation: `{expression_result.original_orientation}`.",
        f"- Final standardized shape: `{expression_result.final_shape[0]} samples x {expression_result.final_shape[1]} genes`.",
        f"- Transpose applied: `{expression_result.transpose_applied}`.",
        f"- Matrix value profile guess: `{expression_result.value_profile['value_type_guess']}`.",
        f"- Fractional value rate in sampled cells: `{expression_result.value_profile['fractional_value_rate']:.4f}`.",
        "",
        "## Barcode parsing logic",
        "- Expected TCGA format: `TCGA-TSS-PARTICIPANT-SAMPLE[VIAL]` with optional aliquot extensions for portion, analyte, plate, and center.",
        "- Supplied sample identifiers are all short TCGA sample barcodes such as `TCGA-ZZ-0000-01A`.",
        "- Recoverable fields from the supplied matrix: `project_code`, `tss`, `participant_id`, `subject_id`, `sample_id`, `sample_type_code`, `sample_type_label`, and `vial`.",
        f"- Not recoverable from the supplied barcodes: `{', '.join(missing_technical_fields) if missing_technical_fields else 'none'}`.",
        "",
        "## Dataset structure",
        f"- Total samples: `{metrics['total_samples']}`.",
        f"- Total unique subjects: `{metrics['total_subjects']}`.",
        f"- Total genes/features: `{metrics['total_genes']}`.",
        f"- Duplicated sample barcodes: `{metrics['duplicate_sample_barcodes']}`.",
        f"- Subjects with multiple samples: `{metrics['subjects_with_multiple_samples']}`.",
        f"- Subjects with multiple sample types: `{metrics['subjects_with_multiple_sample_types']}`.",
        "",
        "## Sample type composition",
        dataframe_to_markdown(samples_by_type),
        "",
        "### Unique subjects by focal sample type",
        dataframe_to_markdown(subjects_by_type),
        "",
        f"- Subjects with both sample type `01` and `11`: `{metrics['subjects_with_01_and_11']}`.",
        f"- Subjects with both sample type `01` and `06`: `{metrics['subjects_with_01_and_06']}`.",
        f"- Subjects with both sample type `11` and `06`: `{metrics['subjects_with_11_and_06']}`.",
        "",
        "## Site and domain composition",
        "- No explicit sequencing center, plate, or batch metadata linked to the expression aliquots were present in the supplied files.",
        "- `cases.primary_site` is available in the clinical file, but it is cohort-level disease anatomy (`Breast`) rather than an acquisition domain variable.",
        "- The strongest recoverable domain proxy is the TCGA tissue source site code (`tss`) parsed from the sample barcode.",
        "",
        "### Samples by TSS (top 15)",
        dataframe_to_markdown(samples_by_tss),
        "",
        "### Subjects by TSS (top 15)",
        dataframe_to_markdown(subjects_by_tss),
        "",
        "## Metadata inventory scanned from supplied TSV files",
    ]

    if metadata_inventory:
        for file_name, columns in metadata_inventory.items():
            report_lines.append(
                f"- `{file_name}`: {', '.join(columns) if columns else 'no site/batch-like columns detected'}."
            )
    else:
        report_lines.append("- No metadata directory was available.")

    report_lines.extend(
        [
            "",
            "## Data integrity",
            f"- Missing values in expression matrix: `{metrics['expression_missing_cells']}` cells (`{metrics['expression_missing_rate']:.6f}` of the matrix).",
            f"- Zero-valued entries in expression matrix: `{metrics['expression_zero_entries']}` (`{metrics['expression_zero_rate']:.6f}` of the matrix).",
            f"- Genes with all zeros: `{metrics['genes_all_zero_count']}`.",
            f"- Samples with all zeros: `{metrics['samples_all_zero_count']}`.",
            f"- Low total-signal outlier samples: `{metrics['low_total_signal_outlier_count']}`.",
            f"- Clinical subject table rows: `{clinical_summary['clinical_rows']}` across `{clinical_summary['clinical_subjects']}` unique subjects.",
            f"- Expression subjects without a subject-level clinical match: `{unmatched_subjects}`.",
            "",
            "### Gene zero / low variance summary",
            dataframe_to_markdown(gene_zero_variance_summary),
            "",
            "## Repeated subjects",
            dataframe_to_markdown(repeated_subjects),
            "",
            "## Distribution summaries",
            "- The supplied matrix is numerically continuous rather than integer raw counts, so per-sample row sums should be interpreted as total expression signal rather than literal sequencing library sizes.",
            "- Plots saved under `reports/figures/` include sample-type counts, subject counts, TSS composition, per-sample total signal, and gene-wise variance.",
            "",
            "## Warnings and ambiguities",
            "- The file name suggests STAR counts, but the matrix values are fractional and log-like; this is not a raw integer count matrix.",
            "- All supplied sample identifiers are short TCGA barcodes, so aliquot-level fields such as plate and center are unavailable from the inputs currently provided.",
            "- TSS is a strong site proxy for transportability analyses, but it should not be over-interpreted as a confirmed sequencing batch or sequencing center without richer biospecimen metadata.",
            "",
            "## Preprocessing recommendations",
        ]
    )

    for item in blueprint["sample_selection_policy"]:
        report_lines.append(f"- {item}")
    report_lines.append(
        f"- Recommended domain ID: `{blueprint['domain_recommendation']['domain_id']}`. {blueprint['domain_recommendation']['domain_source']}"
    )
    report_lines.append(
        f"- Subject deduplication rule: {blueprint['subject_deduplication_rule']}"
    )

    report_lines.extend(
        [
            "- Expression preprocessing plan:",
            *[f"  - {item}" for item in blueprint["expression_preprocessing_plan"]],
            "- Feature filtering plan:",
            *[f"  - {item}" for item in blueprint["feature_filtering_plan"]],
            "- Preserve the following metadata for future ComBat, DASHI, and leave-one-domain-out experiments:",
            *[f"  - `{item}`" for item in blueprint["metadata_to_preserve"]],
            "",
            "## Next-stage blueprint",
            f"- {blueprint['next_stage_warning']}",
        ]
    )

    return "\n".join(report_lines) + "\n"
