"""Preprocessing: schema."""

from __future__ import annotations

from typing import Any

from ingestion.specs import CATEGORICAL_FEATURES, CLASSIC_FEATURES, NUMERIC_FEATURES


def build_feature_schema() -> dict[str, Any]:
    id_columns = ["subject_id"]
    metadata_columns = ["domain_id"]
    excluded_columns = id_columns + metadata_columns + ["target_original", "target"]

    return {
        "id_columns": id_columns,
        "metadata_columns": metadata_columns,
        "domain_column": "domain_id",
        "target_column": "target",
        "target_original_column": "target_original",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "predictive_feature_columns": CLASSIC_FEATURES,
        "excluded_columns": excluded_columns,
        "notes_on_exclusions": [
            "domain_id is preserved as metadata for later LODO/LOSO evaluation and is excluded from predictive features by default.",
            "subject_id is an identifier and is excluded from modeling.",
            "target and target_original are labels, not predictive inputs.",
            "Additional provenance columns and raw_patient_id are preserved in the interim dataset and source manifest, not in the final analysis-ready table.",
        ],
    }
