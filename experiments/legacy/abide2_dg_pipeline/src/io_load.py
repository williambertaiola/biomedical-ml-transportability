"""Transportability experiment: io load."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import log


PHENOTYPIC_SUBJECT_PATTERNS = [
    r"^SUB_ID$",
    r"^SUBID$",
    r"^subject_id$",
    r"^participant_id$",
]
PHENOTYPIC_SITE_PATTERNS = [r"^SITE_ID$", r"^SITE$", r".*SITE.*", r".*CENTER.*"]
PHENOTYPIC_DIAG_PATTERNS = [r"^DX_GROUP$", r".*DIAG.*", r".*DX.*", r".*GROUP.*"]
PHENOTYPIC_AGE_PATTERNS = [
    r"^AGE_AT_SCAN$",
    r"^AGE_AT_SCAN\s*$",
    r"^AGE$",
    r".*AGE.*SCAN.*",
]
PHENOTYPIC_SEX_PATTERNS = [r"^SEX$", r"^GENDER$", r".*SEX.*"]

QAP_SUBJECT_PATTERNS = [r"^Sub_ID$", r"^SUB_ID$", r"^subject_id$"]
QAP_SITE_PATTERNS = [r"^Site_ID$", r"^SITE_ID$", r"^SITE$"]
QAP_SESSION_PATTERNS = [r"^Session$", r"^SESSION$", r".*session.*"]
QAP_SERIES_PATTERNS = [r"^Series$", r"^SERIES$", r".*series.*"]
QAP_UNIQUE_PATTERNS = [
    r"^Unique_ID$",
    r"^Unique_ID\.1$",
    r".*unique.*id.*",
    r".*scan.*id.*",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _first_match(columns: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    col_list = list(columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in col_list:
            if rx.match(str(c)):
                return c
    return None


def find_column_candidates(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "subject_id": _first_match(cols, PHENOTYPIC_SUBJECT_PATTERNS),
        "site_id": _first_match(cols, PHENOTYPIC_SITE_PATTERNS),
        "diagnosis": _first_match(cols, PHENOTYPIC_DIAG_PATTERNS),
        "age": _first_match(cols, PHENOTYPIC_AGE_PATTERNS),
        "sex": _first_match(cols, PHENOTYPIC_SEX_PATTERNS),
    }


def find_qap_column_candidates(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "subject_id": _first_match(cols, QAP_SUBJECT_PATTERNS),
        "site_id": _first_match(cols, QAP_SITE_PATTERNS),
        "session": _first_match(cols, QAP_SESSION_PATTERNS),
        "series": _first_match(cols, QAP_SERIES_PATTERNS),
        "unique_id": _first_match(cols, QAP_UNIQUE_PATTERNS),
    }


def _resolve_optional_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        return p
    p2 = root / value
    if p2.exists():
        return p2
    return None


def _discover_csv_files(data_root: Path) -> List[Path]:
    blocked_tokens = {
        "analysis_output",
        "outputs",
        "report",
        "figures",
        "splits",
        "__pycache__",
    }
    files = []
    for p in data_root.rglob("*.csv"):
        parts = {x.lower() for x in p.parts}
        if any(tok in part for part in parts for tok in blocked_tokens):
            continue
        files.append(p)
    return sorted(files)


def _pick_by_name(
    candidates: List[Path], include_keywords: List[str]
) -> Optional[Path]:
    for p in candidates:
        name = p.name.lower()
        if all(k in name for k in include_keywords):
            return p
    return None


def _pick_phenotypic(candidates: List[Path]) -> Optional[Path]:
    # Heuristic: prefer files not containing qap and with many columns.
    best_path = None
    best_cols = -1
    for p in candidates:
        lname = p.name.lower()
        if "qap" in lname:
            continue
        if "merged" in lname or "cleaned" in lname or "split" in lname:
            continue
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        n_cols = len(df.columns)
        col_names = [str(c).upper() for c in df.columns]
        has_sub = any(c in {"SUB_ID", "SUBID", "PARTICIPANT_ID"} for c in col_names)
        has_diag = any("DX" in c or "DIAG" in c for c in col_names)
        if has_sub and has_diag and n_cols > best_cols:
            best_cols = n_cols
            best_path = p
    return best_path


def discover_input_paths(config_paths: Dict[str, Optional[str]]) -> Dict[str, Path]:
    data_root = Path(config_paths["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    explicit = {
        "phenotypic": _resolve_optional_path(
            data_root, config_paths.get("phenotypic_file")
        ),
        "anat_qap": _resolve_optional_path(
            data_root, config_paths.get("anat_qap_file")
        ),
        "dti_qap": _resolve_optional_path(data_root, config_paths.get("dti_qap_file")),
        "functional_qap": _resolve_optional_path(
            data_root, config_paths.get("functional_qap_file")
        ),
    }

    csv_candidates = _discover_csv_files(data_root)
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found in data root: {data_root}")

    picked = dict(explicit)
    if picked["phenotypic"] is None:
        picked["phenotypic"] = _pick_phenotypic(csv_candidates)
    if picked["anat_qap"] is None:
        picked["anat_qap"] = _pick_by_name(csv_candidates, ["anat", "qap"])
    if picked["dti_qap"] is None:
        picked["dti_qap"] = _pick_by_name(csv_candidates, ["dti", "qap"])
    if picked["functional_qap"] is None:
        picked["functional_qap"] = _pick_by_name(csv_candidates, ["functional", "qap"])
        if picked["functional_qap"] is None:
            picked["functional_qap"] = _pick_by_name(csv_candidates, ["func", "qap"])

    missing = [k for k, v in picked.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"Could not discover required files for keys: {missing}. "
            f"Set them explicitly in config paths."
        )

    for key, path in picked.items():
        log(f"Using {key}: {path}")

    return {k: Path(v) for k, v in picked.items()}


def load_tables(input_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key, path in input_paths.items():
        log(f"Loading {key} from {path}")
        df = pd.read_csv(path, low_memory=False)
        df = normalize_columns(df)
        out[key] = df
    return out


def summarize_tables(
    input_paths: Dict[str, Path], tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for key, df in tables.items():
        rows.append(
            {
                "table": key,
                "path": str(input_paths[key]),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "missing_pct": float(df.isna().mean().mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)
