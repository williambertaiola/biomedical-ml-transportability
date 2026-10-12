"""Preprocessing: abide2 folder unified."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FolderUnifiedArtifacts:
    final_df: pd.DataFrame
    source_coverage: pd.DataFrame
    site_summary: pd.DataFrame
    disease_summary: pd.DataFrame
    missingness: pd.DataFrame
    dropped_columns: pd.DataFrame
    file_inventory: pd.DataFrame
    report_text: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger("abide2_folder_unified")
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
        payload["size_bytes"] = path.stat().st_size
    return payload


def clean_column_name(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "unnamed"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in out.columns:
        base = clean_column_name(column)
        seen[base] = seen.get(base, 0) + 1
        columns.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    out.columns = columns
    return out


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return clean_columns(pd.read_csv(path, encoding=encoding, low_memory=False))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cannot read {path}: {last_error}")


def normalize_sub_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_subject_id(site: Any, sub_id: Any) -> str | pd._libs.missing.NAType:
    if pd.isna(site) or pd.isna(sub_id):
        return pd.NA
    return f"{site}_{int(sub_id)}"


def collapse_unique(series: pd.Series) -> Any:
    values = [value for value in series.dropna().tolist() if str(value).strip() != ""]
    unique_values = list(dict.fromkeys(map(str, values)))
    if not unique_values:
        return pd.NA
    if len(unique_values) == 1:
        return unique_values[0]
    return " | ".join(unique_values)


def inventory_folder(input_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_root.rglob("*")):
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(input_root)),
                "is_file": path.is_file(),
                "suffix": path.suffix.lower() if path.is_file() else "",
                "size_bytes": path.stat().st_size if path.is_file() else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def load_bni_phenotypic(phenotypic_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_fallback(phenotypic_path)
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError("ABIDEII-BNI_1.csv must contain SITE_ID and SUB_ID.")

    df["site"] = df["site_id"].astype("string")
    df["sub_id"] = normalize_sub_id(df["sub_id"])
    df["subject_id"] = [
        build_subject_id(site, sub_id) for site, sub_id in zip(df["site"], df["sub_id"])
    ]

    if "dx_group" in df.columns:
        df["disease_level_code"] = pd.to_numeric(
            df["dx_group"], errors="coerce"
        ).astype("Int64")
        df["disease_level"] = (
            df["disease_level_code"].map({1: "ASD", 2: "Control"}).astype("string")
        )
        df["disease_level_source"] = "bni_dx_group"
    else:
        df["disease_level"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["disease_level_source"] = pd.NA

    keep_counts = df.notna().sum()
    dropped = [
        {
            "column_name": column,
            "drop_reason": "all_values_missing_in_bni_phenotypic",
            "source_table": "bni_phenotypic",
        }
        for column in keep_counts[keep_counts == 0].index.tolist()
    ]
    df = df[[column for column in df.columns if keep_counts[column] > 0]].copy()
    df["phenotypic_bni_available"] = True
    return df, pd.DataFrame(dropped)


def aggregate_qap_table(path: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_fallback(path)
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError(f"QAP file missing SITE_ID/SUB_ID: {path}")

    df["site"] = df["site_id"].astype("string")
    df["sub_id"] = normalize_sub_id(df["sub_id"])
    df["subject_id"] = [
        build_subject_id(site, sub_id) for site, sub_id in zip(df["site"], df["sub_id"])
    ]

    dropped_rows: list[dict[str, str]] = []
    for artifact_column in ["unique_id", "unique_id_1", "unnamed_17"]:
        if artifact_column in df.columns:
            df = df.drop(columns=[artifact_column])
            dropped_rows.append(
                {
                    "column_name": artifact_column,
                    "drop_reason": "identifier_or_artifact_qap_column",
                    "source_table": prefix,
                }
            )

    numeric_columns = [
        column
        for column in df.columns
        if column not in {"site", "site_id", "sub_id", "subject_id"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in df.columns
        if column not in {"site", "site_id", "sub_id", "subject_id", *numeric_columns}
    ]

    grouped = (
        df.groupby(["subject_id", "site", "site_id", "sub_id"], dropna=False)
        .agg(
            {
                **{column: "mean" for column in numeric_columns},
                **{column: collapse_unique for column in categorical_columns},
            }
        )
        .reset_index()
    )
    grouped[f"{prefix}_record_count"] = (
        df.groupby(["subject_id", "site", "site_id", "sub_id"])
        .size()
        .reset_index(name="n")["n"]
        .to_numpy()
    )
    grouped[f"{prefix}_available"] = True

    rename_map = {
        column: f"{prefix}_{column}"
        for column in grouped.columns
        if column not in {"subject_id", "site", "site_id", "sub_id"}
        and not column.startswith(f"{prefix}_")
    }
    grouped = grouped.rename(columns=rename_map)
    return grouped, pd.DataFrame(dropped_rows)


def inventory_bni_raw(raw_root: Path) -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(
        path for path in raw_root.iterdir() if path.is_dir() and path.name.isdigit()
    ):
        sub_id = int(subject_dir.name)
        files = [path for path in subject_dir.rglob("*") if path.is_file()]
        relative_paths = [path.relative_to(subject_dir) for path in files]
        relative_files = [str(path) for path in relative_paths]
        rows.append(
            {
                "subject_id": f"ABIDEII-BNI_1_{sub_id}",
                "site": "ABIDEII-BNI_1",
                "site_id": "ABIDEII-BNI_1",
                "sub_id": sub_id,
                "raw_bni_available": True,
                "raw_bni_session_count": len(
                    {path.parts[0] for path in relative_paths if path.parts}
                ),
                "raw_bni_file_count": len(relative_files),
                "raw_bni_has_anat_nifti": any(
                    path.endswith("anat.nii.gz") for path in relative_files
                ),
                "raw_bni_has_rest_nifti": any(
                    path.endswith("rest.nii.gz") for path in relative_files
                ),
                "raw_bni_has_dti_nifti": any(
                    path.endswith("dti.nii.gz") for path in relative_files
                ),
                "raw_bni_has_dti_bval": any(
                    path.endswith("dti.bval") for path in relative_files
                ),
                "raw_bni_has_dti_bvec": any(
                    path.endswith("dti.bvec") for path in relative_files
                ),
            }
        )
    return pd.DataFrame(rows)


def build_subject_base(*frames: pd.DataFrame) -> pd.DataFrame:
    minimal_frames = []
    for frame in frames:
        if frame.empty:
            continue
        cols = [
            column
            for column in ["subject_id", "site", "site_id", "sub_id"]
            if column in frame.columns
        ]
        minimal_frames.append(frame[cols].drop_duplicates())
    base = (
        pd.concat(minimal_frames, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return base


def build_source_coverage(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [
        "phenotypic_bni_available",
        "anat_available",
        "dti_available",
        "func_available",
        "raw_bni_available",
    ]:
        if column in final_df.columns:
            rows.append(
                {
                    "source_block": column,
                    "subjects_with_data": int(
                        final_df[column].fillna(False).astype(bool).sum()
                    ),
                    "subjects_total": int(len(final_df)),
                }
            )
    return pd.DataFrame(rows)


def build_site_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        final_df.groupby("site", dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            phenotypic_available=(
                "phenotypic_bni_available",
                lambda s: int(s.fillna(False).astype(bool).sum()),
            )
            if "phenotypic_bni_available" in final_df.columns
            else ("subject_id", "size"),
            anat_available=(
                "anat_available",
                lambda s: int(s.fillna(False).astype(bool).sum()),
            ),
            dti_available=(
                "dti_available",
                lambda s: int(s.fillna(False).astype(bool).sum()),
            ),
            func_available=(
                "func_available",
                lambda s: int(s.fillna(False).astype(bool).sum()),
            ),
        )
        .reset_index()
        .sort_values("n_subjects", ascending=False)
    )
    return summary


def build_disease_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    if "disease_level" not in final_df.columns:
        return pd.DataFrame(columns=["disease_level", "subject_count"])
    return (
        final_df.groupby("disease_level", dropna=False)
        .size()
        .reset_index(name="subject_count")
        .sort_values("subject_count", ascending=False)
    )


def build_missingness(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in final_df.columns:
        missing_n = int(final_df[column].isna().sum())
        rows.append(
            {
                "column_name": column,
                "missing_n": missing_n,
                "non_missing_n": int(len(final_df) - missing_n),
                "missing_rate": float(missing_n / len(final_df)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "column_name"], ascending=[False, True]
    )


def render_report(
    *,
    final_df: pd.DataFrame,
    source_coverage: pd.DataFrame,
    site_summary: pd.DataFrame,
    disease_summary: pd.DataFrame,
    dropped_columns: pd.DataFrame,
) -> str:
    lines = [
        "# ABIDE II Folder-Wide Unified Database",
        "",
        "## Overview",
        f"- Final unified database shape: `{final_df.shape[0]} rows x {final_df.shape[1]} columns`.",
        f"- Unique subjects: `{final_df['subject_id'].nunique()}`.",
        f"- Unique sites: `{final_df['site'].nunique(dropna=True)}`.",
        "- One row per `(site, subject)` pair.",
        "",
        "## What was merged",
        "- Folder-wide QAP tables: `anat_qap.csv`, `dti_qap.csv`, `functional_qap.csv`.",
        "- BNI-only phenotypic table: `ABIDEII-BNI_1.csv`.",
        "- BNI raw-folder inventory: `ABIDEII-BNI_1/`.",
        "- Derived outputs in `abide_ii_analysis_output_full/` were inspected but not used as primary source tables.",
        "",
        "## Added / standardized columns",
        "- `subject_id`: `site_subid` key.",
        "- `site`: explicit site/domain column.",
        "- `disease_level`: derived from `DX_GROUP` where available.",
        "- Availability flags: `phenotypic_bni_available`, `anat_available`, `dti_available`, `func_available`, `raw_bni_available`.",
        "",
        "## Source coverage",
        source_coverage.to_string(index=False),
        "",
        "## Site summary",
        site_summary.to_string(index=False),
        "",
        "## Disease-level coverage",
        disease_summary.to_string(index=False)
        if not disease_summary.empty
        else "No disease-level information available.",
        "",
    ]

    if not dropped_columns.empty:
        lines.extend(
            [
                "## Dropped columns",
                dropped_columns.to_string(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## Important limitation",
            "- Within the folder itself, phenotypic diagnosis information is only available for `ABIDEII-BNI_1.csv`.",
            "- Other sites contribute technical/QAP information but not matched phenotypic diagnosis inside this folder.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_folder_unified_processing(
    config: dict[str, Any], logger: logging.Logger
) -> FolderUnifiedArtifacts:
    input_root = Path(config["input_root"])
    phenotypic_path = input_root / "ABIDEII-BNI_1.csv"
    qap_root = input_root / "ABIDEII_MRI_Quality_Metrics"
    raw_bni_root = input_root / "ABIDEII-BNI_1"

    file_inventory = inventory_folder(input_root)

    bni_pheno, dropped_bni = load_bni_phenotypic(phenotypic_path)
    anat, dropped_anat = aggregate_qap_table(qap_root / "anat_qap.csv", "anat")
    dti, dropped_dti = aggregate_qap_table(qap_root / "dti_qap.csv", "dti")
    func, dropped_func = aggregate_qap_table(qap_root / "functional_qap.csv", "func")
    raw_bni = inventory_bni_raw(raw_bni_root)

    logger.info(
        "Loaded source blocks: bni_pheno=%s, anat=%s, dti=%s, func=%s, raw_bni=%s",
        len(bni_pheno),
        len(anat),
        len(dti),
        len(func),
        len(raw_bni),
    )

    subject_base = build_subject_base(bni_pheno, anat, dti, func, raw_bni)
    logger.info("Subject base contains %s unique site-subject pairs", len(subject_base))

    final_df = subject_base.merge(
        bni_pheno, on=["subject_id", "site", "site_id", "sub_id"], how="left"
    )
    final_df = final_df.merge(
        anat, on=["subject_id", "site", "site_id", "sub_id"], how="left"
    )
    final_df = final_df.merge(
        dti, on=["subject_id", "site", "site_id", "sub_id"], how="left"
    )
    final_df = final_df.merge(
        func, on=["subject_id", "site", "site_id", "sub_id"], how="left"
    )
    final_df = final_df.merge(
        raw_bni, on=["subject_id", "site", "site_id", "sub_id"], how="left"
    )

    ordered_front = [
        "subject_id",
        "site",
        "sub_id",
        "site_id",
        "disease_level",
        "disease_level_code",
        "disease_level_source",
        "dx_group",
        "phenotypic_bni_available",
        "anat_available",
        "dti_available",
        "func_available",
        "raw_bni_available",
    ]
    ordered_front = [column for column in ordered_front if column in final_df.columns]
    final_df = final_df[
        ordered_front
        + [column for column in final_df.columns if column not in ordered_front]
    ].copy()

    source_coverage = build_source_coverage(final_df)
    site_summary = build_site_summary(final_df)
    disease_summary = build_disease_summary(final_df)
    missingness = build_missingness(final_df)
    dropped_columns = (
        pd.concat(
            [
                frame
                for frame in [dropped_bni, dropped_anat, dropped_dti, dropped_func]
                if not frame.empty
            ],
            ignore_index=True,
        )
        if any(
            not frame.empty
            for frame in [dropped_bni, dropped_anat, dropped_dti, dropped_func]
        )
        else pd.DataFrame(columns=["column_name", "drop_reason", "source_table"])
    )
    report_text = render_report(
        final_df=final_df,
        source_coverage=source_coverage,
        site_summary=site_summary,
        disease_summary=disease_summary,
        dropped_columns=dropped_columns,
    )

    return FolderUnifiedArtifacts(
        final_df=final_df,
        source_coverage=source_coverage,
        site_summary=site_summary,
        disease_summary=disease_summary,
        missingness=missingness,
        dropped_columns=dropped_columns,
        file_inventory=file_inventory,
        report_text=report_text,
    )
