"""Preprocessing: abide2 processing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ProcessingArtifacts:
    final_df: pd.DataFrame
    file_overview: pd.DataFrame
    qap_coverage: pd.DataFrame
    missingness: pd.DataFrame
    dropped_columns: pd.DataFrame
    report_text: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger("abide2_preprocessing")
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


def file_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        payload["size_bytes"] = stat.st_size
    return payload


def clean_column_name(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "unnamed"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    for column in out.columns:
        base = clean_column_name(column)
        seen[base] = seen.get(base, 0) + 1
        new_columns.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    out.columns = new_columns
    return out


def normalize_sub_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def collapse_unique(series: pd.Series) -> Any:
    values = [value for value in series.dropna().tolist() if str(value).strip() != ""]
    unique_values = list(dict.fromkeys(map(str, values)))
    if not unique_values:
        return pd.NA
    if len(unique_values) == 1:
        return unique_values[0]
    return " | ".join(unique_values)


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return clean_columns(df)


def build_subject_id(site: Any, sub_id: Any) -> str | pd._libs.missing.NAType:
    if pd.isna(site) or pd.isna(sub_id):
        return pd.NA
    return f"{site}_{int(sub_id)}"


def derive_disease_level(dx_group: Any) -> str | pd._libs.missing.NAType:
    if pd.isna(dx_group):
        return pd.NA
    mapping = {1: "ASD", 2: "Control"}
    try:
        return mapping.get(int(dx_group), pd.NA)
    except Exception:
        return pd.NA


def load_phenotypic(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pheno = read_csv_clean(path)
    if "site_id" not in pheno.columns or "sub_id" not in pheno.columns:
        raise RuntimeError("Phenotypic file must contain SITE_ID and SUB_ID.")

    pheno["sub_id"] = normalize_sub_id(pheno["sub_id"])
    pheno["site"] = pheno["site_id"].astype("string")
    pheno["subject_id"] = [
        build_subject_id(site, sub) for site, sub in zip(pheno["site"], pheno["sub_id"])
    ]

    if "dx_group" in pheno.columns:
        pheno["disease_level_code"] = pd.to_numeric(
            pheno["dx_group"], errors="coerce"
        ).astype("Int64")
        pheno["disease_level"] = (
            pheno["disease_level_code"].map({1: "ASD", 2: "Control"}).astype("string")
        )
        pheno["disease_level_source"] = "dx_group"
    else:
        pheno["disease_level"] = pd.Series(pd.NA, index=pheno.index, dtype="string")
        pheno["disease_level_source"] = pd.NA

    keep_non_empty = pheno.notna().sum()
    dropped = keep_non_empty[keep_non_empty == 0].index.tolist()
    keep_columns = [column for column in pheno.columns if column not in dropped]
    pheno = pheno[keep_columns].copy()

    dropped_df = pd.DataFrame(
        {
            "column_name": dropped,
            "drop_reason": "all_values_missing_in_phenotypic_file",
            "source_table": "phenotypic",
        }
    )
    return pheno, dropped_df


def aggregate_qap(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in df.columns
        if column not in {"site_id", "sub_id"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    non_numeric_columns = [
        column
        for column in df.columns
        if column not in {"site_id", "sub_id", *numeric_columns}
    ]

    aggregated = (
        df.groupby(["site_id", "sub_id"], dropna=False)
        .agg(
            {
                **{column: "mean" for column in numeric_columns},
                **{column: collapse_unique for column in non_numeric_columns},
            }
        )
        .reset_index()
    )

    rename_map = {
        column: f"{prefix}_{column}"
        for column in aggregated.columns
        if column not in {"site_id", "sub_id"}
    }
    aggregated = aggregated.rename(columns=rename_map)
    aggregated[f"{prefix}_available"] = True
    return aggregated


def load_qap_subset(
    path: Path, *, prefix: str, site: str, subject_ids: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_clean(path)
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError(f"QAP file missing site_id/sub_id: {path}")

    df["sub_id"] = normalize_sub_id(df["sub_id"])
    df = df.loc[
        (df["site_id"].astype("string") == site) & (df["sub_id"].isin(subject_ids))
    ].copy()

    dropped_columns: list[dict[str, str]] = []
    for artifact_column in ["unique_id_1", "unnamed_17"]:
        if artifact_column in df.columns:
            df = df.drop(columns=[artifact_column])
            dropped_columns.append(
                {
                    "column_name": artifact_column,
                    "drop_reason": "apparent_duplicate_or_artifact_qap_column",
                    "source_table": prefix,
                }
            )

    keep_non_empty = df.notna().sum()
    empty_columns = [
        column
        for column in keep_non_empty[keep_non_empty == 0].index
        if column not in {"site_id", "sub_id"}
    ]
    if empty_columns:
        df = df.drop(columns=empty_columns)
        for column in empty_columns:
            dropped_columns.append(
                {
                    "column_name": column,
                    "drop_reason": "all_values_missing_after_site_subject_filter",
                    "source_table": prefix,
                }
            )

    aggregated = aggregate_qap(df, prefix=prefix)
    dropped_df = pd.DataFrame(dropped_columns)
    return aggregated, dropped_df


def build_file_overview(
    pheno: pd.DataFrame,
    anat: pd.DataFrame,
    dti: pd.DataFrame,
    func: pd.DataFrame,
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, df in [
        ("phenotypic_clean", pheno),
        ("anat_qap_filtered", anat),
        ("dti_qap_filtered", dti),
        ("functional_qap_filtered", func),
        ("final_subject_level_database", final_df),
    ]:
        rows.append(
            {
                "table_name": name,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "unique_subjects": int(df["subject_id"].nunique())
                if "subject_id" in df.columns
                else int(df["sub_id"].nunique()),
                "missing_cells": int(df.isna().sum().sum()),
                "missing_rate": float(
                    df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_qap_coverage(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix in ["anat", "dti", "func"]:
        availability_col = f"{prefix}_available"
        rows.append(
            {
                "modality": prefix,
                "subjects_with_data": int(
                    final_df[availability_col].fillna(False).astype(bool).sum()
                )
                if availability_col in final_df.columns
                else 0,
                "subjects_total": int(len(final_df)),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_table(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in final_df.columns:
        missing_n = int(final_df[column].isna().sum())
        rows.append(
            {
                "column_name": column,
                "missing_n": missing_n,
                "missing_rate": float(missing_n / len(final_df)),
                "non_missing_n": int(len(final_df) - missing_n),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "column_name"], ascending=[False, True]
    )


def find_best_severity_candidates(pheno: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "ados_2_severity_total",
        "ados_2_total",
        "ados_g_total",
        "ados_g_social",
        "srs_total_t",
        "srs_total_raw",
        "bdi_total",
        "vineland_sum_scores",
    ]
    rows = []
    for column in candidate_columns:
        if column not in pheno.columns:
            continue
        rows.append(
            {
                "column_name": column,
                "non_missing_n": int(pheno[column].notna().sum()),
                "missing_rate": float(pheno[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["non_missing_n", "column_name"], ascending=[False, True]
    )


def render_report(
    *,
    final_df: pd.DataFrame,
    pheno: pd.DataFrame,
    qap_coverage: pd.DataFrame,
    file_overview: pd.DataFrame,
    dropped_columns: pd.DataFrame,
    severity_candidates: pd.DataFrame,
) -> str:
    site_value = (
        final_df["site"].dropna().iloc[0]
        if "site" in final_df.columns and final_df["site"].notna().any()
        else "unknown"
    )
    disease_level_present = (
        "disease_level" in final_df.columns and final_df["disease_level"].notna().any()
    )
    disease_counts = (
        final_df["disease_level"].value_counts(dropna=False).to_string()
        if disease_level_present
        else "No disease-level column available."
    )
    lines = [
        "# ABIDE II Processing Report",
        "",
        "## Overview",
        f"- Final table shape: `{final_df.shape[0]} rows x {final_df.shape[1]} columns`.",
        f"- Site retained from the supplied phenotypic file: `{site_value}`.",
        f"- Unique subjects: `{final_df['subject_id'].nunique()}`.",
        "",
        "## Added explicit columns",
        "- `subject_id`: constructed as `site_subid`.",
        "- `site`: explicit site/domain column.",
        "- `disease_level`: derived from `DX_GROUP` when available.",
        "- `disease_level_code`: original numeric diagnosis code from `DX_GROUP`.",
        "",
        "## Disease-level information",
        f"- Disease-level column present: `{disease_level_present}`.",
        "- Mapping used: `DX_GROUP = 1 -> ASD`, `DX_GROUP = 2 -> Control`.",
        "- No robust ordinal severity variable with complete coverage was imposed as a new mandatory column.",
        "",
        "### Disease-level counts",
        disease_counts,
        "",
        "## QAP merge coverage",
        qap_coverage.to_string(index=False),
        "",
        "## File overview",
        file_overview.to_string(index=False),
        "",
    ]

    if not severity_candidates.empty:
        lines.extend(
            [
                "## Best-covered disease-associated columns in the phenotypic file",
                severity_candidates.to_string(index=False),
                "",
            ]
        )

    if not dropped_columns.empty:
        lines.extend(
            [
                "## Dropped columns",
                dropped_columns.to_string(index=False),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def run_processing(
    config: dict[str, Any], logger: logging.Logger
) -> ProcessingArtifacts:
    phenotypic_path = Path(config["phenotypic_file"])
    qap_dir = Path(config["qap_dir"])

    pheno, dropped_pheno = load_phenotypic(phenotypic_path)
    logger.info("Loaded phenotypic table: %s rows x %s columns", *pheno.shape)

    if pheno["site"].nunique(dropna=True) != 1:
        logger.warning(
            "Phenotypic file contains %s distinct sites.",
            pheno["site"].nunique(dropna=True),
        )
    site = str(pheno["site"].dropna().iloc[0])
    subject_ids = set(
        int(value) for value in pheno["sub_id"].dropna().astype(int).tolist()
    )

    anat, dropped_anat = load_qap_subset(
        qap_dir / "anat_qap.csv", prefix="anat", site=site, subject_ids=subject_ids
    )
    dti, dropped_dti = load_qap_subset(
        qap_dir / "dti_qap.csv", prefix="dti", site=site, subject_ids=subject_ids
    )
    func, dropped_func = load_qap_subset(
        qap_dir / "functional_qap.csv",
        prefix="func",
        site=site,
        subject_ids=subject_ids,
    )
    logger.info(
        "Loaded QAP subsets: anat=%s, dti=%s, func=%s", len(anat), len(dti), len(func)
    )

    final_df = pheno.merge(anat, on=["site_id", "sub_id"], how="left")
    final_df = final_df.merge(dti, on=["site_id", "sub_id"], how="left")
    final_df = final_df.merge(func, on=["site_id", "sub_id"], how="left")

    ordered_front = [
        "subject_id",
        "site",
        "sub_id",
        "site_id",
        "disease_level",
        "disease_level_code",
        "disease_level_source",
        "dx_group",
    ]
    ordered_front = [column for column in ordered_front if column in final_df.columns]
    remaining = [column for column in final_df.columns if column not in ordered_front]
    final_df = final_df[ordered_front + remaining].copy()

    file_overview = build_file_overview(pheno, anat, dti, func, final_df)
    qap_coverage = build_qap_coverage(final_df)
    missingness = build_missingness_table(final_df)
    dropped_columns = (
        pd.concat(
            [
                frame
                for frame in [dropped_pheno, dropped_anat, dropped_dti, dropped_func]
                if not frame.empty
            ],
            ignore_index=True,
        )
        if any(
            not frame.empty
            for frame in [dropped_pheno, dropped_anat, dropped_dti, dropped_func]
        )
        else pd.DataFrame(columns=["column_name", "drop_reason", "source_table"])
    )
    severity_candidates = find_best_severity_candidates(pheno)
    report_text = render_report(
        final_df=final_df,
        pheno=pheno,
        qap_coverage=qap_coverage,
        file_overview=file_overview,
        dropped_columns=dropped_columns,
        severity_candidates=severity_candidates,
    )

    return ProcessingArtifacts(
        final_df=final_df,
        file_overview=file_overview,
        qap_coverage=qap_coverage,
        missingness=missingness,
        dropped_columns=dropped_columns,
        report_text=report_text,
    )
