"""Transportability experiment: clinical."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import clean_object_columns, collapse_unique

CLINICAL_SUBJECT_COLUMNS = [
    "cases.submitter_id",
    "cases.case_id",
    "cases.primary_site",
    "diagnoses.primary_diagnosis",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.ajcc_pathologic_stage",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
]

DOMAIN_KEYWORDS = [
    "center",
    "site",
    "tss",
    "plate",
    "batch",
    "sample",
    "submitter",
    "analyte",
    "barcode",
    "portion",
    "protocol",
]


def scan_metadata_inventory(metadata_dir: Path | None) -> dict[str, list[str]]:
    if metadata_dir is None or not metadata_dir.exists():
        return {}

    inventory: dict[str, list[str]] = {}
    for path in sorted(metadata_dir.glob("*.tsv")):
        header = pd.read_csv(path, sep="\t", nrows=0)
        inventory[path.name] = [
            column
            for column in header.columns
            if any(keyword in column.lower() for keyword in DOMAIN_KEYWORDS)
        ]
    return inventory


def load_subject_level_clinical(
    metadata_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if metadata_dir is None:
        return pd.DataFrame(), {"clinical_rows": 0, "clinical_subjects": 0}

    clinical_path = metadata_dir / "clinical.tsv"
    if not clinical_path.exists():
        return pd.DataFrame(), {"clinical_rows": 0, "clinical_subjects": 0}

    header = pd.read_csv(clinical_path, sep="\t", nrows=0)
    usecols = [
        column for column in CLINICAL_SUBJECT_COLUMNS if column in header.columns
    ]
    clinical_df = pd.read_csv(clinical_path, sep="\t", usecols=usecols)
    clinical_df = clean_object_columns(clinical_df)

    rename_map = {
        "cases.submitter_id": "subject_id",
        "cases.case_id": "clinical_case_id",
        "cases.primary_site": "clinical_primary_site",
        "diagnoses.primary_diagnosis": "clinical_primary_diagnosis",
        "diagnoses.tissue_or_organ_of_origin": "clinical_tissue_or_organ_of_origin",
        "diagnoses.site_of_resection_or_biopsy": "clinical_site_of_resection_or_biopsy",
        "diagnoses.ajcc_pathologic_stage": "clinical_ajcc_pathologic_stage",
        "demographic.gender": "clinical_gender",
        "demographic.race": "clinical_race",
        "demographic.ethnicity": "clinical_ethnicity",
    }
    clinical_df = clinical_df.rename(columns=rename_map)

    grouped = (
        clinical_df.groupby("subject_id", dropna=False)
        .agg(
            {
                column: collapse_unique
                for column in clinical_df.columns
                if column != "subject_id"
            }
        )
        .reset_index()
    )
    grouped["clinical_record_count"] = (
        clinical_df.groupby("subject_id")
        .size()
        .reindex(grouped["subject_id"])
        .to_numpy()
    )

    summary = {
        "clinical_rows": int(len(clinical_df)),
        "clinical_subjects": int(grouped["subject_id"].nunique()),
    }
    return grouped, summary
