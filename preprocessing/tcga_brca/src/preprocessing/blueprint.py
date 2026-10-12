"""Preprocessing: blueprint."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_preprocessing_blueprint(
    samples_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
    value_profile: dict[str, Any],
) -> dict[str, Any]:
    sample_type_counts = summary_metrics["sample_type_counts"]
    subject_type_counts = summary_metrics["subject_type_counts"]
    repeated_same_type = (
        samples_df.groupby(["subject_id", "sample_type_code"], dropna=False)
        .size()
        .gt(1)
        .sum()
    )

    if samples_df["domain_source"].eq("barcode_center").any():
        domain_recommendation = {
            "domain_id": "center",
            "domain_source": "Recovered from barcode center field.",
        }
    elif samples_df["domain_source"].eq("barcode_tss").any():
        domain_recommendation = {
            "domain_id": "tss",
            "domain_source": "Best available domain proxy recovered from the TCGA barcode.",
        }
    elif samples_df["domain_source"].eq("barcode_plate").any():
        domain_recommendation = {
            "domain_id": "plate",
            "domain_source": "Technical proxy recovered from the TCGA barcode.",
        }
    else:
        domain_recommendation = {
            "domain_id": "unknown_domain",
            "domain_source": "No robust site or batch proxy was recoverable from the supplied inputs.",
        }

    value_type_guess = value_profile["value_type_guess"]
    if value_type_guess == "likely_raw_counts":
        expression_plan = [
            "Treat the matrix as raw counts after independent provenance confirmation.",
            "Apply low-expression filtering within training folds only.",
            "Use a count-aware transformation later, such as log-CPM or voom-like scaling computed without leaking test-fold information.",
        ]
    else:
        expression_plan = [
            "Treat the matrix as already transformed or normalized continuous expression unless raw-count provenance is independently confirmed.",
            "Do not apply a blind global log transform.",
            "Restrict later train-only preprocessing to feature filtering and train-fold scaling.",
        ]

    return {
        "sample_selection_policy": [
            f"Use sample type 01 as the main analysis cohort because it dominates the dataset ({sample_type_counts.get('01', 0)} samples; {subject_type_counts.get('01', 0)} subjects) and best matches the primary tumor transportability question.",
            f"Keep sample type 11 as a comparison/QC cohort rather than pooling it into the main modeling set ({sample_type_counts.get('11', 0)} samples; {subject_type_counts.get('11', 0)} subjects).",
            f"Keep sample type 06 separate for sensitivity analyses only because it is extremely sparse ({sample_type_counts.get('06', 0)} samples; {subject_type_counts.get('06', 0)} subjects).",
        ],
        "subject_deduplication_rule": (
            "When multiple samples exist for the same subject and sample type, retain one sample deterministically using: "
            "lowest zero fraction, then highest total signal, then lexicographically smallest barcode. "
            f"This rule is motivated by {int(repeated_same_type)} repeated subject-sample-type combinations."
        ),
        "domain_recommendation": domain_recommendation,
        "expression_preprocessing_plan": expression_plan,
        "feature_filtering_plan": [
            "Always remove genes with all-zero expression.",
            "Apply near-zero variance filtering within training data only.",
            "Optionally apply low-expression-frequency filtering only after confirming whether the matrix is raw counts or already transformed.",
        ],
        "metadata_to_preserve": [
            "sample_barcode",
            "subject_id",
            domain_recommendation["domain_id"],
            "domain_primary",
            "domain_source",
            "sample_type_code",
            "sample_type_label",
            "tss",
            "vial",
            "total_signal",
            "zero_fraction",
            "clinical_primary_diagnosis",
            "clinical_primary_site",
            "clinical_site_of_resection_or_biopsy",
            "clinical_ajcc_pathologic_stage",
        ],
        "next_stage_warning": (
            "All train-learned transforms for downstream ML, including scaling, feature selection, or harmonization parameter estimation, "
            "must be fit on training partitions only to preserve anti-leakage guarantees."
        ),
    }


def build_downstream_metadata_table(
    samples_df: pd.DataFrame, blueprint: dict[str, Any]
) -> pd.DataFrame:
    preferred_domain_column = blueprint["domain_recommendation"]["domain_id"]
    downstream = samples_df.copy()
    if preferred_domain_column in downstream.columns:
        downstream["domain_id"] = downstream[preferred_domain_column]
    else:
        downstream["domain_id"] = downstream["domain_primary"]

    downstream["domain_source"] = blueprint["domain_recommendation"]["domain_source"]
    keep_columns = [
        "sample_barcode",
        "subject_id",
        "domain_id",
        "domain_source",
        "domain_primary",
        "sample_type_code",
        "sample_type_label",
        "tss",
        "center",
        "plate",
        "vial",
        "total_signal",
        "zero_fraction",
        "clinical_primary_site",
        "clinical_primary_diagnosis",
        "clinical_site_of_resection_or_biopsy",
        "clinical_ajcc_pathologic_stage",
        "clinical_gender",
        "clinical_race",
        "clinical_ethnicity",
    ]
    existing_columns = [
        column for column in keep_columns if column in downstream.columns
    ]
    return downstream[existing_columns].sort_values(
        ["sample_type_code", "subject_id", "sample_barcode"]
    )
