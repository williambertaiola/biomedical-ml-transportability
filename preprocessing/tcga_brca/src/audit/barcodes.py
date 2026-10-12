"""Preprocessing: barcodes."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

SAMPLE_TYPE_MAP = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "05": "Additional - New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "08": "Human Tumor Original Cells",
    "09": "Primary Blood Derived Cancer - Bone Marrow",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
    "12": "Buccal Cell Normal",
    "13": "EBV Immortalized Normal",
    "14": "Bone Marrow Normal",
}

TCGA_BARCODE_PATTERN = re.compile(
    r"^(?P<project>TCGA)-(?P<tss>[A-Z0-9]{2})-(?P<participant>[A-Z0-9]{4})-"
    r"(?P<sample_type_code>\d{2})(?P<vial>[A-Z])"
    r"(?:-(?P<portion>\d{2})(?P<analyte>[A-Z])"
    r"(?:-(?P<plate>[A-Z0-9]{4})(?:-(?P<center>\d{2}))?)?)?$"
)


def parse_tcga_barcode(barcode: str) -> dict[str, Any]:
    barcode = str(barcode).strip()
    match = TCGA_BARCODE_PATTERN.match(barcode)
    payload: dict[str, Any] = {
        "sample_barcode": barcode,
        "project_code": pd.NA,
        "tss": pd.NA,
        "participant_id": pd.NA,
        "subject_id": pd.NA,
        "sample_id": pd.NA,
        "sample_type_code": pd.NA,
        "sample_type_label": pd.NA,
        "vial": pd.NA,
        "portion": pd.NA,
        "analyte": pd.NA,
        "plate": pd.NA,
        "center": pd.NA,
        "barcode_scope": "malformed",
        "barcode_parse_status": "malformed",
        "barcode_parse_note": "Barcode did not match the expected TCGA sample or aliquot pattern.",
    }
    if not match:
        return payload

    parsed = match.groupdict()
    subject_id = f"{parsed['project']}-{parsed['tss']}-{parsed['participant']}"
    sample_segment = f"{parsed['sample_type_code']}{parsed['vial']}"
    has_aliquot_fields = any(
        parsed.get(field) for field in ["portion", "analyte", "plate", "center"]
    )
    barcode_scope = (
        "full_aliquot_barcode" if has_aliquot_fields else "short_sample_barcode"
    )

    payload.update(
        {
            "project_code": parsed["project"],
            "tss": parsed["tss"],
            "participant_id": parsed["participant"],
            "subject_id": subject_id,
            "sample_id": sample_segment,
            "sample_type_code": parsed["sample_type_code"],
            "sample_type_label": SAMPLE_TYPE_MAP.get(
                parsed["sample_type_code"], "Unknown sample type"
            ),
            "vial": parsed["vial"],
            "portion": parsed.get("portion") or pd.NA,
            "analyte": parsed.get("analyte") or pd.NA,
            "plate": parsed.get("plate") or pd.NA,
            "center": parsed.get("center") or pd.NA,
            "barcode_scope": barcode_scope,
            "barcode_parse_status": "parsed",
            "barcode_parse_note": "Parsed from supplied TCGA barcode.",
        }
    )
    if not has_aliquot_fields:
        payload["barcode_parse_note"] = (
            "Parsed from a short TCGA sample barcode. Portion, analyte, plate, and center are unavailable in the supplied barcode."
        )
    return payload


def build_sample_metadata(
    sample_barcodes: list[str],
    clinical_subject_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = [parse_tcga_barcode(barcode) for barcode in sample_barcodes]
    samples_df = pd.DataFrame.from_records(records)

    if clinical_subject_df is not None and not clinical_subject_df.empty:
        samples_df = samples_df.merge(
            clinical_subject_df, how="left", left_on="subject_id", right_on="subject_id"
        )

    candidate_fields = {
        "true_site_metadata": [],
        "barcode_domain_fields": [],
        "unavailable_from_inputs": [],
    }

    if "center" in samples_df and samples_df["center"].notna().sum() > 0:
        samples_df["domain_primary"] = samples_df["center"]
        samples_df["domain_source"] = "barcode_center"
        candidate_fields["barcode_domain_fields"].append("center")
        secondary = ["tss", "plate"]
    elif "tss" in samples_df and samples_df["tss"].notna().sum() > 0:
        samples_df["domain_primary"] = samples_df["tss"]
        samples_df["domain_source"] = "barcode_tss"
        candidate_fields["barcode_domain_fields"].append("tss")
        secondary = ["center", "plate"]
    elif "plate" in samples_df and samples_df["plate"].notna().sum() > 0:
        samples_df["domain_primary"] = samples_df["plate"]
        samples_df["domain_source"] = "barcode_plate"
        candidate_fields["barcode_domain_fields"].append("plate")
        secondary = ["tss", "center"]
    else:
        samples_df["domain_primary"] = "unknown_domain"
        samples_df["domain_source"] = "unavailable"
        secondary = []

    unavailable = []
    for field in ["portion", "analyte", "plate", "center"]:
        if field not in samples_df or samples_df[field].notna().sum() == 0:
            unavailable.append(field)
    candidate_fields["unavailable_from_inputs"] = unavailable

    available_secondary = [
        field
        for field in secondary
        if field in samples_df
        and samples_df[field].notna().sum() > 0
        and field != samples_df["domain_source"].iat[0].replace("barcode_", "")
    ]
    secondary_value = (
        "|".join(available_secondary)
        if available_secondary
        else "none_recovered_from_inputs"
    )
    samples_df["domain_secondary_candidates"] = secondary_value

    return samples_df, candidate_fields
