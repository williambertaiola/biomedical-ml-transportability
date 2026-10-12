"""Preprocessing: abide2 composite unified."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


QAP_SITE_ALIASES = {
    "ABIDEII-ONRC_2": "ABIDEII-OILH_2",
}


@dataclass
class CompositeUnifiedArtifacts:
    final_df: pd.DataFrame
    source_coverage: pd.DataFrame
    site_summary: pd.DataFrame
    disease_summary: pd.DataFrame
    missingness: pd.DataFrame
    dropped_columns: pd.DataFrame
    file_inventory: pd.DataFrame
    site_aliases: pd.DataFrame
    report_text: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger("abide2_composite_unified")
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
    new_columns: list[str] = []
    for column in out.columns:
        base = clean_column_name(column)
        seen[base] = seen.get(base, 0) + 1
        new_columns.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    out.columns = new_columns
    return out


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
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


def drop_all_missing_columns(
    df: pd.DataFrame, source_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_counts = df.notna().sum()
    dropped = pd.DataFrame(
        [
            {
                "column_name": column,
                "drop_reason": "all_values_missing",
                "source_table": source_name,
            }
            for column in keep_counts[keep_counts == 0].index.tolist()
        ]
    )
    kept_columns = [column for column in df.columns if keep_counts[column] > 0]
    return df[kept_columns].copy(), dropped


def normalize_site_name(
    series: pd.Series, aliases: dict[str, str] | None = None
) -> pd.Series:
    aliases = aliases or {}
    normalized = series.astype("string").str.strip()
    return normalized.replace(aliases)


def load_phenotypic_table(
    path: Path,
    *,
    source_name: str,
    site_filter_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_fallback(path)
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError(f"{path} must contain SITE_ID and SUB_ID.")

    df["site_id"] = normalize_site_name(df["site_id"])
    if site_filter_prefix is not None:
        df = df[
            df["site_id"].astype("string").str.startswith(site_filter_prefix, na=False)
        ].copy()

    df["site"] = df["site_id"]
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
        df["disease_level_source"] = f"{source_name}_dx_group"
    else:
        df["disease_level_code"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["disease_level"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["disease_level_source"] = pd.Series(pd.NA, index=df.index, dtype="string")

    df[f"phenotypic_{source_name}_available"] = True
    df["phenotypic_available"] = True
    return drop_all_missing_columns(df, source_name)


def aggregate_qap_table(path: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_fallback(path)
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError(f"QAP file missing SITE_ID/SUB_ID: {path}")

    df["site_id"] = normalize_site_name(df["site_id"], QAP_SITE_ALIASES)
    df["site"] = df["site_id"]
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


def combine_phenotypic_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available_frames = [frame for frame in frames if not frame.empty]
    if not available_frames:
        return pd.DataFrame(columns=["subject_id", "site", "site_id", "sub_id"])
    combined = pd.concat(available_frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["subject_id"]).reset_index(drop=True)
    return combined


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
    if not minimal_frames:
        return pd.DataFrame(columns=["subject_id", "site", "site_id", "sub_id"])
    return (
        pd.concat(minimal_frames, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_source_coverage(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [
        "phenotypic_available",
        "phenotypic_composite_available",
        "phenotypic_long_available",
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
    def count_flag(column: str) -> tuple[str, Any]:
        if column not in final_df.columns:
            return ("subject_id", "size")
        return (column, lambda s: int(s.fillna(False).astype(bool).sum()))

    summary = (
        final_df.groupby("site", dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            phenotypic_available=count_flag("phenotypic_available"),
            anat_available=count_flag("anat_available"),
            dti_available=count_flag("dti_available"),
            func_available=count_flag("func_available"),
        )
        .reset_index()
        .sort_values(["n_subjects", "site"], ascending=[False, True])
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
    site_aliases: pd.DataFrame,
) -> str:
    lines = [
        "# ABIDE II Composite-Enriched Unified Database",
        "",
        "## Overview",
        f"- Final unified database shape: `{final_df.shape[0]} rows x {final_df.shape[1]} columns`.",
        f"- Unique subjects: `{final_df['subject_id'].nunique()}`.",
        f"- Unique sites: `{final_df['site'].nunique(dropna=True)}`.",
        "- One row per `(site, subject)` pair.",
        "",
        "## What was merged",
        "- ABIDE II composite phenotypic table: `ABIDEII_Composite_Phenotypic.csv`.",
        "- ABIDE II longitudinal phenotypic table: `ABIDEII_Long_Composite_Phenotypic.csv` filtered to `ABIDEII-*` sites only.",
        "- Folder-wide QAP tables: `anat_qap.csv`, `dti_qap.csv`, `functional_qap.csv`.",
        "- BNI raw-folder inventory: `ABIDEII-BNI_1/`.",
        "- Local `ABIDEII-BNI_1.csv` remains in the project folder but is effectively subsumed by the composite phenotypic file.",
        "",
        "## Added / standardized columns",
        "- `subject_id`: `site_subid` key.",
        "- `site`: explicit site/domain column after site-name harmonization.",
        "- `disease_level`: derived from `DX_GROUP` where available.",
        "- Availability flags: `phenotypic_available`, `phenotypic_composite_available`, `phenotypic_long_available`, `anat_available`, `dti_available`, `func_available`, `raw_bni_available`.",
        "",
        "## Site harmonization",
        site_aliases.to_string(index=False)
        if not site_aliases.empty
        else "No site aliases were needed.",
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
            "## Important notes",
            "- Composite phenotypic coverage is much broader than the original BNI-only file, so most key phenotype fields now cover the multi-site cohort.",
            "- `DTI` remains incomplete because the local `dti_qap.csv` only covers a subset of ABIDE II sites.",
            "- `ABIDEII-U_MIA_1` and `ABIDEII-SU_2` are present in the composite phenotypic file but have no matching QAP rows in the provided folder.",
            "- Longitudinal phenotypic rows were restricted to `ABIDEII-*` sites, excluding `ABIDEI-*` rows to keep the final table ABIDE II only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_composite_unified_processing(
    config: dict[str, Any], logger: logging.Logger
) -> CompositeUnifiedArtifacts:
    input_root = Path(config["input_root"])
    composite_path = Path(config["composite_phenotypic_path"])
    long_path = Path(config["long_phenotypic_path"])
    qap_root = input_root / "ABIDEII_MRI_Quality_Metrics"
    raw_bni_root = input_root / "ABIDEII-BNI_1"

    file_inventory = inventory_folder(input_root)

    composite_pheno, dropped_composite = load_phenotypic_table(
        composite_path,
        source_name="composite",
        site_filter_prefix="ABIDEII-",
    )
    long_pheno, dropped_long = load_phenotypic_table(
        long_path,
        source_name="long",
        site_filter_prefix="ABIDEII-",
    )
    phenotypic = combine_phenotypic_tables([composite_pheno, long_pheno])
    anat, dropped_anat = aggregate_qap_table(qap_root / "anat_qap.csv", "anat")
    dti, dropped_dti = aggregate_qap_table(qap_root / "dti_qap.csv", "dti")
    func, dropped_func = aggregate_qap_table(qap_root / "functional_qap.csv", "func")
    raw_bni = inventory_bni_raw(raw_bni_root)

    logger.info(
        "Loaded source blocks: composite_pheno=%s, long_pheno=%s, anat=%s, dti=%s, func=%s, raw_bni=%s",
        len(composite_pheno),
        len(long_pheno),
        len(anat),
        len(dti),
        len(func),
        len(raw_bni),
    )

    subject_base = build_subject_base(phenotypic, anat, dti, func, raw_bni)
    logger.info("Subject base contains %s unique site-subject pairs", len(subject_base))

    final_df = subject_base.merge(
        phenotypic, on=["subject_id", "site", "site_id", "sub_id"], how="left"
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
        "phenotypic_available",
        "phenotypic_composite_available",
        "phenotypic_long_available",
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
    dropped_candidates = [
        dropped_composite,
        dropped_long,
        dropped_anat,
        dropped_dti,
        dropped_func,
    ]
    dropped_columns = (
        pd.concat(
            [frame for frame in dropped_candidates if not frame.empty],
            ignore_index=True,
        )
        if any(not frame.empty for frame in dropped_candidates)
        else pd.DataFrame(columns=["column_name", "drop_reason", "source_table"])
    )
    site_aliases = pd.DataFrame(
        [
            {"source_site_id": source, "canonical_site_id": target}
            for source, target in QAP_SITE_ALIASES.items()
        ]
    )
    report_text = render_report(
        final_df=final_df,
        source_coverage=source_coverage,
        site_summary=site_summary,
        disease_summary=disease_summary,
        dropped_columns=dropped_columns,
        site_aliases=site_aliases,
    )

    return CompositeUnifiedArtifacts(
        final_df=final_df,
        source_coverage=source_coverage,
        site_summary=site_summary,
        disease_summary=disease_summary,
        missingness=missingness,
        dropped_columns=dropped_columns,
        file_inventory=file_inventory,
        site_aliases=site_aliases,
        report_text=report_text,
    )
