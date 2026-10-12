"""Transportability experiment: preprocess."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_load import find_column_candidates, find_qap_column_candidates
from .utils import log


QAP_QUALITY_DIRECTIONS = {
    "anat_qap": {
        "higher_better": ["CNR", "SNR", "FBER", "Cortical Contrast"],
        "lower_better": ["Qi1", "EFC", "Smoothness_of_Voxels"],
    },
    "functional_qap": {
        "higher_better": ["SNR", "FBER", "GCORR"],
        "lower_better": [
            "mFD",
            "PercentFD_greater_than_0.20",
            "NumFD_greater_than_0.20",
            "DVARS",
            "Outliers",
            "EFC",
            "Smoothness_of_Voxels",
            "Median_Distance_Index",
        ],
    },
    "dti_qap": {
        "higher_better": ["SNR", "FBER", "GCORR"],
        "lower_better": [
            "mFD",
            "PercentFD_greater_than_0.20",
            "NumFD_greater_than_0.20",
            "DVARS",
            "Outliers",
            "EFC",
            "Smoothness_of_Voxels",
            "Median_Distance_Index",
        ],
    },
}


@dataclass
class SchemaMapping:
    phen_subject: str
    phen_site: Optional[str]
    phen_diagnosis: Optional[str]
    phen_age: Optional[str]
    phen_sex: Optional[str]
    qap_mapping: Dict[str, Dict[str, Optional[str]]]


def _normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_subject_id_series(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().mean() >= 0.5:
        out = num.round().astype("Int64").astype("string")
        return out
    return raw


def _best_override_or_detect(
    override_value: Optional[str],
    detected_value: Optional[str],
    available_columns: Iterable[str],
) -> Optional[str]:
    cols = set(available_columns)
    if override_value and override_value in cols:
        return override_value
    return detected_value


def infer_schema(tables: Dict[str, pd.DataFrame], config: Dict) -> SchemaMapping:
    schema_cfg = config.get("schema", {})
    phen_cfg = schema_cfg.get("phenotypic", {})
    qap_cfg = schema_cfg.get("qap", {})

    phen = tables["phenotypic"]
    phen_detected = find_column_candidates(phen)

    phen_subject = _best_override_or_detect(
        phen_cfg.get("subject_id"), phen_detected["subject_id"], phen.columns
    )
    if not phen_subject:
        raise ValueError(
            f"Could not infer phenotypic subject ID column. Available columns: {list(phen.columns)[:50]}"
        )
    phen_site = _best_override_or_detect(
        phen_cfg.get("site_id"), phen_detected["site_id"], phen.columns
    )
    phen_diagnosis = _best_override_or_detect(
        phen_cfg.get("diagnosis"), phen_detected["diagnosis"], phen.columns
    )
    phen_age = _best_override_or_detect(
        phen_cfg.get("age"), phen_detected["age"], phen.columns
    )
    phen_sex = _best_override_or_detect(
        phen_cfg.get("sex"), phen_detected["sex"], phen.columns
    )

    qap_mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for name in ["anat_qap", "dti_qap", "functional_qap"]:
        df = tables[name]
        detected = find_qap_column_candidates(df)
        cfg = qap_cfg.get(name, {})
        qap_mapping[name] = {
            "subject_id": _best_override_or_detect(
                cfg.get("subject_id"), detected["subject_id"], df.columns
            ),
            "site_id": _best_override_or_detect(
                cfg.get("site_id"), detected["site_id"], df.columns
            ),
            "session": _best_override_or_detect(
                cfg.get("session"), detected["session"], df.columns
            ),
            "series": _best_override_or_detect(
                cfg.get("series"), detected["series"], df.columns
            ),
            "unique_id": _best_override_or_detect(
                cfg.get("unique_id"), detected["unique_id"], df.columns
            ),
        }
        if not qap_mapping[name]["subject_id"]:
            raise ValueError(f"Could not infer subject ID column for {name}.")

    log(
        "Schema inferred: "
        f"phen(subject={phen_subject}, site={phen_site}, diagnosis={phen_diagnosis}, age={phen_age}, sex={phen_sex})"
    )

    return SchemaMapping(
        phen_subject=phen_subject,
        phen_site=phen_site,
        phen_diagnosis=phen_diagnosis,
        phen_age=phen_age,
        phen_sex=phen_sex,
        qap_mapping=qap_mapping,
    )


def _first_non_null(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.iloc[0]


def _collapse_duplicate_subjects(df: pd.DataFrame, subject_col: str) -> pd.DataFrame:
    if subject_col not in df.columns:
        return df.copy()
    dups = int(df[subject_col].duplicated().sum())
    if dups == 0:
        return df.copy()
    log(
        f"Phenotypic duplicate subject rows detected: {dups}. Collapsing with first-non-null/median."
    )
    agg_map = {}
    for c in df.columns:
        if c == subject_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg_map[c] = "median"
        else:
            agg_map[c] = _first_non_null
    out = df.groupby(subject_col, dropna=False, as_index=False).agg(agg_map)
    return out


def _map_diagnosis(
    series: pd.Series, diagnosis_cfg: Dict
) -> Tuple[pd.Series, pd.Series]:
    pos_labels = {
        str(x).strip().upper()
        for x in diagnosis_cfg.get("positive_labels", ["1", "ASD", "AUTISM"])
    }
    neg_labels = {
        str(x).strip().upper()
        for x in diagnosis_cfg.get("negative_labels", ["2", "CTRL", "CONTROL"])
    }
    pos_name = diagnosis_cfg.get("positive_name", "ASD")
    neg_name = diagnosis_cfg.get("negative_name", "CTRL")

    binary: List[Optional[int]] = []
    labels: List[Optional[str]] = []
    for v in series:
        if pd.isna(v):
            binary.append(pd.NA)
            labels.append(pd.NA)
            continue
        s = str(v).strip().upper()
        n = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if s in pos_labels or (pd.notna(n) and int(n) == 1):
            binary.append(1)
            labels.append(pos_name)
        elif s in neg_labels or (pd.notna(n) and int(n) == 2):
            binary.append(0)
            labels.append(neg_name)
        elif "ASD" in s or "AUT" in s:
            binary.append(1)
            labels.append(pos_name)
        elif "CTRL" in s or "CONTROL" in s or "TDC" in s or s == "TD":
            binary.append(0)
            labels.append(neg_name)
        else:
            binary.append(pd.NA)
            labels.append("UNKNOWN")
    return pd.Series(binary, dtype="Int64"), pd.Series(labels, dtype="string")


def build_phenotypic_base(
    df: pd.DataFrame, schema: SchemaMapping, config: Dict
) -> pd.DataFrame:
    phen = df.copy()
    phen = _collapse_duplicate_subjects(phen, schema.phen_subject)

    phen["subject_id"] = normalize_subject_id_series(phen[schema.phen_subject])
    phen["site_id"] = (
        phen[schema.phen_site].astype("string").str.strip()
        if schema.phen_site and schema.phen_site in phen.columns
        else pd.Series(pd.NA, index=phen.index, dtype="string")
    )
    phen["age"] = (
        pd.to_numeric(phen[schema.phen_age], errors="coerce")
        if schema.phen_age and schema.phen_age in phen.columns
        else np.nan
    )
    phen["sex_raw"] = (
        phen[schema.phen_sex].astype("string")
        if schema.phen_sex and schema.phen_sex in phen.columns
        else pd.Series(pd.NA, index=phen.index, dtype="string")
    )
    if schema.phen_diagnosis and schema.phen_diagnosis in phen.columns:
        phen["diagnosis_raw"] = phen[schema.phen_diagnosis].astype("string")
        d_bin, d_lbl = _map_diagnosis(
            phen[schema.phen_diagnosis], config.get("schema", {}).get("diagnosis", {})
        )
        phen["diagnosis"] = d_bin
        phen["diagnosis_label"] = d_lbl
    else:
        phen["diagnosis_raw"] = pd.Series(pd.NA, index=phen.index, dtype="string")
        phen["diagnosis"] = pd.Series(pd.NA, index=phen.index, dtype="Int64")
        phen["diagnosis_label"] = pd.Series(pd.NA, index=phen.index, dtype="string")

    # Simplify sex values.
    sex_map = {"1": "M", "2": "F", "M": "M", "MALE": "M", "F": "F", "FEMALE": "F"}
    phen["sex"] = (
        phen["sex_raw"]
        .astype("string")
        .str.upper()
        .str.strip()
        .map(sex_map)
        .fillna("UNK")
    )

    return phen


def _find_directional_metrics(
    columns: Iterable[str], modality_key: str
) -> Tuple[List[str], List[int]]:
    dirs = QAP_QUALITY_DIRECTIONS.get(
        modality_key, {"higher_better": [], "lower_better": []}
    )
    cols = list(columns)
    metric_cols: List[str] = []
    signs: List[int] = []
    upper_map = {c.upper(): c for c in cols}
    for metric in dirs["higher_better"]:
        if metric.upper() in upper_map:
            metric_cols.append(upper_map[metric.upper()])
            signs.append(+1)
    for metric in dirs["lower_better"]:
        if metric.upper() in upper_map:
            metric_cols.append(upper_map[metric.upper()])
            signs.append(-1)
    return metric_cols, signs


def _compute_quality_score(
    df: pd.DataFrame, modality_key: str
) -> Tuple[pd.Series, List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metric_cols, signs = _find_directional_metrics(numeric_cols, modality_key)
    if not metric_cols:
        return pd.Series(np.zeros(len(df)), index=df.index), []

    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    n_used = 0
    for col, sign in zip(metric_cols, signs):
        x = pd.to_numeric(df[col], errors="coerce")
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            continue
        z = (x - mu) / sd
        z = z.fillna(0.0)
        score = score + sign * z
        n_used += 1
    if n_used > 0:
        score = score / n_used
    return score, metric_cols


def _numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    ex = set(exclude)
    return [
        c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])
    ]


def deduplicate_qap(
    qap_df: pd.DataFrame,
    modality_key: str,
    modality_prefix: str,
    mapping: Dict[str, Optional[str]],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = qap_df.copy()
    subject_col = mapping["subject_id"]
    site_col = mapping.get("site_id")
    session_col = mapping.get("session")
    series_col = mapping.get("series")
    unique_col = mapping.get("unique_id")

    df["subject_id"] = normalize_subject_id_series(df[subject_col])
    df["__site_id"] = (
        df[site_col].astype("string").str.strip()
        if site_col
        else pd.Series(pd.NA, index=df.index)
    )
    df["__session"] = (
        df[session_col].astype("string").str.strip()
        if session_col
        else pd.Series(pd.NA, index=df.index)
    )
    df["__series"] = (
        df[series_col].astype("string").str.strip()
        if series_col
        else pd.Series(pd.NA, index=df.index)
    )
    if unique_col and unique_col in df.columns:
        df["__unique_id"] = df[unique_col].astype("string").str.strip()
    else:
        df["__unique_id"] = pd.Series(
            [f"{modality_prefix}_row_{i}" for i in range(len(df))], dtype="string"
        )

    score, metrics_used = _compute_quality_score(df, modality_key=modality_key)
    df["__quality_score"] = score

    exclude_cols = {
        "subject_id",
        "__site_id",
        "__session",
        "__series",
        "__unique_id",
        "__quality_score",
        subject_col,
    }
    if site_col:
        exclude_cols.add(site_col)
    if session_col:
        exclude_cols.add(session_col)
    if series_col:
        exclude_cols.add(series_col)
    if unique_col:
        exclude_cols.add(unique_col)
    metric_cols = _numeric_feature_columns(df, exclude=exclude_cols)

    audit_rows = []
    dedup_rows = []

    for sid, grp in df.groupby("subject_id", dropna=True):
        grp_sorted = grp.copy()
        n_scans = int(len(grp_sorted))
        selected = None
        reason = ""

        if mode == "aggregate":
            selected = None
            reason = "aggregate_median"
            agg_numeric = grp_sorted[metric_cols].median(numeric_only=True)
            row = {"subject_id": sid}
            for c in metric_cols:
                row[f"{modality_prefix}_{c}"] = agg_numeric.get(c, np.nan)
            row[f"{modality_prefix}_site_id"] = _first_non_null(grp_sorted["__site_id"])
            row[f"{modality_prefix}_session"] = "AGGREGATED"
            row[f"{modality_prefix}_series"] = "AGGREGATED"
            row[f"{modality_prefix}_unique_id"] = "AGGREGATED"
            row[f"{modality_prefix}_quality_score"] = float(
                grp_sorted["__quality_score"].median()
            )
            row[f"{modality_prefix}_n_scans_subject"] = n_scans
            dedup_rows.append(row)
        else:
            grp_sorted = grp_sorted.sort_values(
                ["__quality_score", "__unique_id"], ascending=[False, True]
            )
            selected = grp_sorted.iloc[0]
            reason = "max_quality_score"
            row = {"subject_id": sid}
            for c in metric_cols:
                row[f"{modality_prefix}_{c}"] = selected[c]
            row[f"{modality_prefix}_site_id"] = selected["__site_id"]
            row[f"{modality_prefix}_session"] = selected["__session"]
            row[f"{modality_prefix}_series"] = selected["__series"]
            row[f"{modality_prefix}_unique_id"] = selected["__unique_id"]
            row[f"{modality_prefix}_quality_score"] = float(selected["__quality_score"])
            row[f"{modality_prefix}_n_scans_subject"] = n_scans
            dedup_rows.append(row)

        audit_rows.append(
            {
                "subject_id": sid,
                "modality": modality_prefix,
                "n_scans": n_scans,
                "mode": mode,
                "selected_unique_id": selected["__unique_id"]
                if selected is not None
                else "AGGREGATED",
                "selected_session": selected["__session"]
                if selected is not None
                else "AGGREGATED",
                "selected_series": selected["__series"]
                if selected is not None
                else "AGGREGATED",
                "selected_site_id": selected["__site_id"]
                if selected is not None
                else _first_non_null(grp_sorted["__site_id"]),
                "selected_quality_score": float(selected["__quality_score"])
                if selected is not None
                else float(grp_sorted["__quality_score"].median()),
                "selection_reason": reason,
                "metrics_used_for_score": ",".join(metrics_used),
                "candidate_unique_ids": "|".join(
                    grp_sorted["__unique_id"].astype(str).tolist()
                ),
            }
        )

    dedup_df = pd.DataFrame(dedup_rows)
    audit_df = pd.DataFrame(audit_rows)
    return dedup_df, audit_df


def build_subject_level_dataset(
    phen_base: pd.DataFrame, dedup_tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    out = phen_base.copy()
    keep_meta = ["subject_id", "site_id", "diagnosis", "diagnosis_label", "age", "sex"]
    keep_cols = [c for c in keep_meta if c in out.columns] + [
        c for c in out.columns if c not in keep_meta
    ]
    out = out[keep_cols]

    for _, ddf in dedup_tables.items():
        out = out.merge(ddf, on="subject_id", how="left")

    # If primary site is missing, fill from modality-specific site columns.
    if "site_id" in out.columns:
        site_sources = [c for c in out.columns if c.endswith("_site_id")]
        for c in site_sources:
            out["site_id"] = out["site_id"].fillna(out[c].astype("string"))
    return out


def compute_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rows.append(
            {
                "column": c,
                "non_missing_n": int(n - miss),
                "missing_n": miss,
                "missing_pct": float(100.0 * miss / n if n else np.nan),
                "dtype": str(df[c].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def select_model_features(
    df: pd.DataFrame, config: Dict
) -> Tuple[List[str], pd.DataFrame]:
    fe_cfg = config.get("feature_engineering", {})
    covariates = fe_cfg.get("covariates", ["age", "sex", "diagnosis", "site_id"])
    coverage_thr = float(fe_cfg.get("missing_coverage_threshold", 0.8))
    qap_prefixes = fe_cfg.get("qap_prefixes", ["anat_", "func_", "dti_"])

    qap_cols = [c for c in df.columns if any(c.startswith(p) for p in qap_prefixes)]
    qap_cols = [c for c in qap_cols if pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    keep_cols: List[str] = []
    for c in qap_cols:
        coverage = float(df[c].notna().mean())
        keep = coverage >= coverage_thr
        rows.append(
            {
                "feature": c,
                "feature_group": "qap",
                "coverage": coverage,
                "kept": keep,
                "reason": "coverage_ok" if keep else "dropped_low_coverage",
            }
        )
        if keep:
            keep_cols.append(c)

    for c in covariates:
        if c not in df.columns:
            rows.append(
                {
                    "feature": c,
                    "feature_group": "covariate",
                    "coverage": 0.0,
                    "kept": False,
                    "reason": "missing_column",
                }
            )
            continue
        coverage = float(df[c].notna().mean())
        # Mandatory columns kept if present.
        mandatory = c in {"site_id", "diagnosis", "subject_id"}
        keep = mandatory or coverage >= coverage_thr
        rows.append(
            {
                "feature": c,
                "feature_group": "covariate",
                "coverage": coverage,
                "kept": keep,
                "reason": "mandatory"
                if mandatory
                else ("coverage_ok" if keep else "dropped_low_coverage"),
            }
        )
        if keep and c not in keep_cols:
            keep_cols.append(c)

    # Ensure subject_id is always retained.
    if "subject_id" in df.columns and "subject_id" not in keep_cols:
        keep_cols = ["subject_id"] + keep_cols
    else:
        keep_cols = ["subject_id"] + [c for c in keep_cols if c != "subject_id"]

    # Ensure domain/target retained when present.
    for c in ["site_id", "diagnosis", "diagnosis_label"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    feature_selection_df = pd.DataFrame(rows).sort_values(
        ["feature_group", "kept", "coverage"], ascending=[True, False, False]
    )
    return keep_cols, feature_selection_df
