#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABIDE II - Baseline vs Transductive Learning (ASD vs Control)
==============================================================

Questo script implementa un'analisi di transportability multi-site su ABIDE II
con task reale ASD vs Control, confrontando:
- baseline source-only
- transductive learning con pseudo-labeling iterativo

Perche questo setup e metodologicamente rilevante:
- ABIDE II e un dataset multi-site reale (domini reali), non simulato.
- Il task ASD vs Control e clinicamente/interpretabile (non artificiale).
- Il domain shift e plausibile per differenze tra site/scanner/protocolli.

Vincoli anti-leakage implementati:
1) Leave-One-Domain-Out sui site validi.
2) In ogni fold il dominio target e separato in:
   - target_adaptation (senza usare label vere nel training)
   - target_test (solo valutazione finale, completamente unseen)
3) Imputer e scaler sono fittati SOLO sui source.
4) target_test non entra mai nel training.
"""

from __future__ import annotations

# Locate the repository without depending on the current working directory.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys

_REPOSITORY_ROOT = next(
    parent
    for parent in _RepositoryPath(__file__).resolve().parents
    if (parent / ".thesis-root").is_file()
)
_repository_sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
from thesis_tools.paths import configured_path as _configured_path


import argparse
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Config ambiente grafico headless
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Config
# =============================================================================

RANDOM_STATE = 42

DEFAULT_OUTPUT_DIR = Path(_configured_path("legacy_abide2_transductive_output"))

# Se l'utente non passa --input-file, lo script prova in quest'ordine.
DEFAULT_INPUT_CANDIDATES = [
    Path(_configured_path("abide_raw_abideii_composite_phenotypic.csv")),
    Path(_configured_path("abide_raw_abideii_long_composite_phenotypic.csv")),
    Path(_configured_path("legacy_input_cleaned_dataset.csv")),
    Path(_configured_path("legacy_input_subject_level_merged.csv")),
    Path(_configured_path("legacy_input_abide2_features.csv")),
]

# File fenotipico usato per recuperare DX_GROUP quando l'input e una cartella
# ABIDEII_MRI_Quality_Metrics (che contiene solo metriche MRI).
DEFAULT_PHENO_CANDIDATES = [
    Path(_configured_path("abide_raw_abideii_composite_phenotypic.csv")),
    Path(_configured_path("abide_raw_abideii_long_composite_phenotypic.csv")),
]

# Candidati colonne target/site
TARGET_CANDIDATES = [
    "dx_group",
    "diagnosis",
    "diagnosis_label",
    "label",
    "class",
    "y",
]
SITE_CANDIDATES = [
    "site_id",
    "site",
    "center",
    "domain_site",
]

# Soglie domini validi
MIN_SITE_SAMPLES = 20
MIN_VALID_SITES = 3
TARGET_TEST_SIZE = 0.50

# Pseudo-labeling
PSEUDO_THRESHOLD = 0.95
PSEUDO_MAX_ITERS = 5

# Metriche richieste
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class Config:
    input_file: Path
    output_dir: Path
    random_state: int = RANDOM_STATE
    min_site_samples: int = MIN_SITE_SAMPLES
    min_valid_sites: int = MIN_VALID_SITES
    target_test_size: float = TARGET_TEST_SIZE
    pseudo_threshold: float = PSEUDO_THRESHOLD
    pseudo_max_iters: int = PSEUDO_MAX_ITERS


class Logger:
    """Logger minimale: stampa + salvataggio file."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# =============================================================================
# Utility
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(v: object) -> str:
    s = str(v).replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    return s[:120]


def can_stratify(y: pd.Series) -> bool:
    vc = y.value_counts(dropna=True)
    return y.nunique(dropna=True) >= 2 and len(vc) > 0 and int(vc.min()) >= 2


def assert_disjoint(ids_a: pd.Series, ids_b: pd.Series, where: str) -> None:
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] Overlap in {where}: {len(overlap)} samples"
        )


def class_dist_text(y: pd.Series) -> str:
    vc = y.value_counts(dropna=False).sort_index()
    return ", ".join([f"class_{k}={int(v)}" for k, v in vc.items()])


def build_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=6000,
        solver="lbfgs",
        random_state=seed,
    )


def find_column_case_insensitive(
    columns: List[str], candidates: List[str]
) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in columns}

    # Prima prova: match esatto case-insensitive
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # Seconda prova: match "contains"
    for cand in candidates:
        cand_l = cand.lower()
        for c in columns:
            if cand_l in str(c).lower():
                return str(c)

    return None


def discover_default_input() -> Optional[Path]:
    for p in DEFAULT_INPUT_CANDIDATES:
        if p.exists() and p.is_file():
            return p

    # Fallback: cerca file plausibili in cartella progetto
    root = Path(_configured_path("legacy_new_project"))
    patterns = ["*abide*clean*.csv", "*abide*merged*.csv", "*abide*.csv"]
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]

    return None


def discover_default_pheno_file() -> Optional[Path]:
    for p in DEFAULT_PHENO_CANDIDATES:
        if p.exists() and p.is_file():
            return p
    return None


def read_table_with_encoding_fallback(path: Path, sep: str) -> pd.DataFrame:
    """
    Lettura robusta del CSV/TSV con fallback encoding.

    Alcuni file ABIDE II (es. Composite Phenotypic) sono in latin1.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_exc: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Impossibile leggere {path} con gli encoding provati: {encodings}. Last error: {last_exc}"
    )


def load_qap_directory_with_pheno(
    qap_dir: Path,
    pheno_file: Path,
    logger: Logger,
) -> pd.DataFrame:
    """
    Carica cartella ABIDEII_MRI_Quality_Metrics e costruisce una tabella campione-level.

    Input attesi nella cartella:
    - anat_qap.csv
    - dti_qap.csv
    - functional_qap.csv

    Poi unisce DX_GROUP dal file fenotipico usando chiavi:
    - SITE_ID
    - SUB_ID
    """
    anat_path = qap_dir / "anat_qap.csv"
    dti_path = qap_dir / "dti_qap.csv"
    func_path = qap_dir / "functional_qap.csv"

    missing = [str(p) for p in [anat_path, dti_path, func_path] if not p.exists()]
    if missing:
        raise RuntimeError(f"Cartella QAP incompleta, file mancanti: {missing}")

    a = read_table_with_encoding_fallback(anat_path, sep=",")
    d = read_table_with_encoding_fallback(dti_path, sep=",")
    f = read_table_with_encoding_fallback(func_path, sep=",")

    p = read_table_with_encoding_fallback(pheno_file, sep=",")
    need_cols = {"SITE_ID", "SUB_ID", "DX_GROUP"}
    if not need_cols.issubset(set(p.columns)):
        raise RuntimeError(
            f"File fenotipico senza colonne richieste {need_cols}: {pheno_file}"
        )

    p = p[["SITE_ID", "SUB_ID", "DX_GROUP"]].copy()

    def _normalize_keys(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        if "Site_ID" in df.columns:
            df = df.rename(columns={"Site_ID": "SITE_ID"})
        if "Sub_ID" in df.columns:
            df = df.rename(columns={"Sub_ID": "SUB_ID"})
        return df

    a = _normalize_keys(a)
    d = _normalize_keys(d)
    f = _normalize_keys(f)

    for df_name, df_ref in [("anat", a), ("dti", d), ("func", f)]:
        if "SITE_ID" not in df_ref.columns or "SUB_ID" not in df_ref.columns:
            raise RuntimeError(f"File {df_name}_qap.csv senza SITE_ID/SUB_ID")

    def _prep_modality(df_mod: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep = ["SITE_ID", "SUB_ID"]
        out = df_mod[keep].copy()

        for c in df_mod.columns:
            if c in keep:
                continue
            c_l = str(c).lower()
            if c_l.startswith("unnamed"):
                continue
            if c in ["Session", "Series", "Unique_ID", "Unique_ID.1"]:
                continue

            s = pd.to_numeric(df_mod[c], errors="coerce")
            if s.notna().any():
                out[f"{prefix}_{c}"] = s

        # Se esistono scans multiple per soggetto, facciamo media numerica
        out = out.groupby(["SITE_ID", "SUB_ID"], as_index=False).mean(numeric_only=True)
        return out

    a2 = _prep_modality(a, "anat")
    d2 = _prep_modality(d, "dti")
    f2 = _prep_modality(f, "func")

    merged = a2.merge(d2, on=["SITE_ID", "SUB_ID"], how="outer")
    merged = merged.merge(f2, on=["SITE_ID", "SUB_ID"], how="outer")
    merged = merged.merge(p, on=["SITE_ID", "SUB_ID"], how="left")

    logger.log(f"[QAP] merge qap shape={merged.shape}")
    logger.log(
        f"[QAP] campioni senza DX_GROUP dopo join={int(merged['DX_GROUP'].isna().sum())}"
    )

    return merged


# =============================================================================
# Target ASD vs Control
# =============================================================================


def map_target_asd_control(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Mappa target in binario:
    - ASD = 1
    - Control = 0

    Gestisce casi comuni ABIDE:
    - DX_GROUP: 1=ASD, 2=Control
    - diagnosis: 1/0 oppure stringhe ASD/Control
    - diagnosis_label: ASD/CONTROL
    """
    s = series.copy()

    # prova numerica
    s_num = pd.to_numeric(s, errors="coerce")
    mapped_num = pd.Series(np.nan, index=s.index, dtype="float")

    # Convenzioni frequenti
    mapped_num.loc[s_num == 1] = 1.0  # spesso ASD
    mapped_num.loc[s_num == 2] = 0.0  # spesso Control in DX_GROUP
    mapped_num.loc[s_num == 0] = 0.0  # spesso Control in label binarie 0/1

    # mappa da testo se disponibile
    s_txt = s.astype("string").str.strip().str.lower()
    mapped_txt = pd.Series(np.nan, index=s.index, dtype="float")

    asd_tokens = {"asd", "autism", "autism spectrum disorder", "patient", "case"}
    ctrl_tokens = {"control", "td", "typical", "healthy", "hc"}

    mapped_txt.loc[s_txt.isin(asd_tokens)] = 1.0
    mapped_txt.loc[s_txt.isin(ctrl_tokens)] = 0.0

    # merge: priorita testo quando definito, altrimenti numerico
    out = mapped_num.copy()
    out.loc[mapped_txt.notna()] = mapped_txt.loc[mapped_txt.notna()]

    # fallback stringhe numeriche "1"/"2"/"0"
    out.loc[(out.isna()) & (s_txt == "1")] = 1.0
    out.loc[(out.isna()) & (s_txt == "2")] = 0.0
    out.loc[(out.isna()) & (s_txt == "0")] = 0.0

    out = out.astype("Int64")

    mapping_info = {
        "n_total": int(len(series)),
        "n_mapped": int(out.notna().sum()),
        "n_unmapped": int(out.isna().sum()),
    }
    return out, mapping_info


# =============================================================================
# Feature selection (solo colonne numeriche, escludendo ID/target/site)
# =============================================================================


def identify_id_like_columns(
    columns: List[str], site_col: str, target_col: str
) -> List[str]:
    """
    Identifica colonne ID-like da escludere dalle feature.

    Esempi: subject_id, sub_id, NDAR_GUID, id, guid, unique_id, session_id.
    """
    id_like = []
    id_tokens = ["id", "guid", "ndar", "subject", "sub_id", "unique", "session"]

    for c in columns:
        c_l = str(c).lower()

        if c == site_col or c == target_col:
            continue

        if any(tok in c_l for tok in id_tokens):
            id_like.append(c)

    return sorted(set(id_like))


# =============================================================================
# Preprocessing leakage-safe (fit SOLO source)
# =============================================================================


def fit_source_preprocessor(
    X_source: pd.DataFrame, fold_id: str, logger: Logger
) -> Dict[str, object]:
    """
    Fit preprocessore SOLO su source (anti-leakage):
    - imputer median
    - scaler standard
    """
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    if not usable_cols:
        raise RuntimeError(f"[{fold_id}] Nessuna feature numerica utilizzabile")

    X_num = X_num[usable_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    scaler = StandardScaler()
    X_out = scaler.fit_transform(X_imp)

    logger.log(
        f"[PREP][{fold_id}] features_in={X_source.shape[1]} usable_numeric={len(usable_cols)}"
    )

    return {
        "usable_cols": usable_cols,
        "imputer": imputer,
        "scaler": scaler,
        "X_source_final": X_out,
        "n_features": int(len(usable_cols)),
    }


def transform_with_source_preprocessor(
    X_df: pd.DataFrame, prep: Dict[str, object]
) -> np.ndarray:
    X_num = X_df[prep["usable_cols"]].apply(pd.to_numeric, errors="coerce")
    X_imp = prep["imputer"].transform(X_num)
    X_out = prep["scaler"].transform(X_imp)
    return X_out


# =============================================================================
# Metriche
# =============================================================================


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_classes: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    out["recall"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = np.nan
        return out

    try:
        cls_to_idx = {c: i for i, c in enumerate(model_classes)}
        if 1 in cls_to_idx:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, cls_to_idx[1]]))
        else:
            out["roc_auc"] = np.nan
    except Exception:
        out["roc_auc"] = np.nan

    return out


# =============================================================================
# Pseudo-labeling
# =============================================================================


def run_pseudo_labeling(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_adapt_unlabeled: np.ndarray,
    threshold: float,
    max_iters: int,
    seed: int,
    fold_id: str,
    logger: Logger,
) -> Tuple[LogisticRegression, List[Dict[str, object]], pd.DataFrame, int, int]:
    """
    Transductive learning con pseudo-labeling iterativo.

    Regola: aggiunge al training TUTTI i campioni adaptation con confidenza >= threshold.
    """
    X_train = np.asarray(X_source)
    y_train = np.asarray(y_source, dtype=int)
    X_pool = np.asarray(X_adapt_unlabeled)

    history: List[Dict[str, object]] = []
    confidence_rows: List[Dict[str, object]] = []
    total_added = 0

    for it in range(1, max_iters + 1):
        model = build_lr(seed + it)
        model.fit(X_train, y_train)

        pool_n = int(X_pool.shape[0])
        if pool_n == 0:
            history.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "n_pool_before": 0,
                    "n_candidates_high_conf": 0,
                    "n_added": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": np.nan,
                    "conf_min": np.nan,
                    "conf_max": np.nan,
                }
            )
            logger.log(f"[PSEUDO][{fold_id}] Iter {it}: pool vuoto, stop")
            break

        proba = model.predict_proba(X_pool)
        pred = model.predict(X_pool)
        conf = np.max(proba, axis=1)

        high_conf_idx = np.where(conf >= threshold)[0]
        n_candidates = int(len(high_conf_idx))

        for c, is_high in zip(conf.tolist(), (conf >= threshold).tolist()):
            confidence_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "confidence": float(c),
                    "selected_high_conf": int(bool(is_high)),
                }
            )

        conf_mean = float(np.mean(conf))
        conf_min = float(np.min(conf))
        conf_max = float(np.max(conf))

        if n_candidates == 0:
            history.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "n_pool_before": pool_n,
                    "n_candidates_high_conf": 0,
                    "n_added": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": conf_mean,
                    "conf_min": conf_min,
                    "conf_max": conf_max,
                }
            )
            logger.log(
                f"[PSEUDO][{fold_id}] Iter {it}: nessun campione >= {threshold:.2f}, stop"
            )
            break

        chosen_idx = np.sort(high_conf_idx)

        X_sel = X_pool[chosen_idx]
        y_sel = pred[chosen_idx].astype(int)

        X_train = np.vstack([X_train, X_sel])
        y_train = np.concatenate([y_train, y_sel], axis=0)

        keep = np.ones(pool_n, dtype=bool)
        keep[chosen_idx] = False
        X_pool = X_pool[keep]

        cls_counts = pd.Series(y_sel).value_counts().sort_index().to_dict()
        cls_str = "{" + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()]) + "}"

        history.append(
            {
                "fold_id": fold_id,
                "iteration": it,
                "n_pool_before": pool_n,
                "n_candidates_high_conf": n_candidates,
                "n_added": int(len(chosen_idx)),
                "pseudo_class_distribution": cls_str,
                "conf_mean": conf_mean,
                "conf_min": conf_min,
                "conf_max": conf_max,
            }
        )

        total_added += int(len(chosen_idx))
        logger.log(
            f"[PSEUDO][{fold_id}] Iter {it}: high_conf={n_candidates}, added={len(chosen_idx)}, class_dist={cls_str}, pool_restante={int(X_pool.shape[0])}"
        )

    final_model = build_lr(seed + 777)
    final_model.fit(X_train, y_train)

    return (
        final_model,
        history,
        pd.DataFrame(confidence_rows),
        int(total_added),
        int(len(y_train)),
    )


# =============================================================================
# Fold runner
# =============================================================================


def run_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    source_idx: pd.Index,
    adapt_idx: pd.Index,
    test_idx: pd.Index,
    fold_id: str,
    target_site: str,
    cfg: Config,
    logger: Logger,
) -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame, Dict[str, object]
]:
    source_df = df.loc[source_idx].copy()
    adapt_df = df.loc[adapt_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Controlli anti-leakage sugli indici campione
    assert_disjoint(
        source_df["row_id"], adapt_df["row_id"], f"{fold_id}: source vs adaptation"
    )
    assert_disjoint(
        source_df["row_id"], test_df["row_id"], f"{fold_id}: source vs test"
    )
    assert_disjoint(
        adapt_df["row_id"], test_df["row_id"], f"{fold_id}: adaptation vs test"
    )

    logger.log("\n" + "=" * 100)
    logger.log(f"[FOLD] {fold_id} | target_site={target_site}")
    logger.log("=" * 100)
    logger.log(
        f"[SIZE][{fold_id}] source={len(source_df)} adaptation={len(adapt_df)} test={len(test_df)}"
    )
    logger.log(
        f"[DIST][{fold_id}] source: {class_dist_text(source_df['target_binary'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] adaptation (solo diagnostica): {class_dist_text(adapt_df['target_binary'])}"
    )
    logger.log(f"[DIST][{fold_id}] test: {class_dist_text(test_df['target_binary'])}")

    if source_df["target_binary"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] source mono-classe")
    if test_df["target_binary"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] target_test mono-classe")

    X_source = source_df[feature_cols]
    y_source = source_df["target_binary"].to_numpy(dtype=int)

    X_adapt = adapt_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["target_binary"].to_numpy(dtype=int)

    # Preprocess fit SOLO su source
    prep = fit_source_preprocessor(X_source, fold_id, logger)
    logger.log(f"[CHECK][{fold_id}] preprocess fit SOLO su source")

    Xs = prep["X_source_final"]
    Xa = transform_with_source_preprocessor(X_adapt, prep)
    Xt = transform_with_source_preprocessor(X_test, prep)

    # Baseline
    baseline = build_lr(cfg.random_state)
    baseline.fit(Xs, y_source)

    ypred_b = baseline.predict(Xt)
    yprob_b = baseline.predict_proba(Xt)
    m_b = compute_binary_metrics(y_test, ypred_b, yprob_b, baseline.classes_)

    # Transductive
    logger.log(f"[CHECK][{fold_id}] y vere adaptation NON usate nel fit transductive")
    trans_model, pseudo_hist, conf_df, pseudo_total, final_train_size = (
        run_pseudo_labeling(
            X_source=Xs,
            y_source=y_source,
            X_adapt_unlabeled=Xa,
            threshold=cfg.pseudo_threshold,
            max_iters=cfg.pseudo_max_iters,
            seed=cfg.random_state,
            fold_id=fold_id,
            logger=logger,
        )
    )

    expected_final = int(len(y_source) + pseudo_total)
    if final_train_size != expected_final:
        raise RuntimeError(f"[LEAKAGE ERROR][{fold_id}] train_size incoerente")

    ypred_t = trans_model.predict(Xt)
    yprob_t = trans_model.predict_proba(Xt)
    m_t = compute_binary_metrics(y_test, ypred_t, yprob_t, trans_model.classes_)

    rows = [
        {
            "fold_id": fold_id,
            "target_site": target_site,
            "method": "baseline",
            "source_n": len(source_df),
            "target_adaptation_n": len(adapt_df),
            "target_test_n": len(test_df),
            "n_features": prep["n_features"],
            "pseudo_total_added": 0,
            "pseudo_iterations": 0,
            **m_b,
        },
        {
            "fold_id": fold_id,
            "target_site": target_site,
            "method": "transductive",
            "source_n": len(source_df),
            "target_adaptation_n": len(adapt_df),
            "target_test_n": len(test_df),
            "n_features": prep["n_features"],
            "pseudo_total_added": int(pseudo_total),
            "pseudo_iterations": int(len(pseudo_hist)),
            **m_t,
        },
    ]

    pseudo_rows = [
        {"fold_id": fold_id, "target_site": target_site, **r} for r in pseudo_hist
    ]

    if not conf_df.empty:
        conf_df["target_site"] = target_site

    diag = {
        "fold_id": fold_id,
        "target_site": target_site,
        "source_n": len(source_df),
        "adapt_n": len(adapt_df),
        "test_n": len(test_df),
        "n_features": prep["n_features"],
        "train_test_disjoint_ok": 1,
        "used_true_adapt_labels_in_train": 0,
        "preprocess_fit_source_only": 1,
        "pseudo_total_added": int(pseudo_total),
        "baseline_accuracy": m_b.get("accuracy", np.nan),
        "baseline_roc_auc": m_b.get("roc_auc", np.nan),
    }

    return rows, pseudo_rows, conf_df, diag


# =============================================================================
# Aggregazioni e plot
# =============================================================================


def build_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["baseline", "transductive"]:
        sub = fold_results.loc[fold_results["method"] == method]
        if sub.empty:
            continue

        for m in METRICS:
            vals = pd.to_numeric(sub[m], errors="coerce")
            rows.append(
                {
                    "method": method,
                    "metric": m,
                    "mean": float(vals.mean(skipna=True)),
                    "std": float(vals.std(skipna=True, ddof=1)),
                    "n_valid": int(vals.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def build_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = fold_results.groupby(["fold_id", "target_site"], dropna=False)

    for (fold_id, site), grp in grouped:
        b = grp.loc[grp["method"] == "baseline"]
        t = grp.loc[grp["method"] == "transductive"]
        if b.empty or t.empty:
            continue

        b = b.iloc[0]
        t = t.iloc[0]

        for m in METRICS:
            bv = b[m]
            tv = t[m]
            dv = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
            rows.append(
                {
                    "fold_id": fold_id,
                    "target_site": site,
                    "metric": m,
                    "baseline": bv,
                    "transductive": tv,
                    "delta_transductive_minus_baseline": dv,
                }
            )

    return pd.DataFrame(rows)


def plot_metrics_per_site(fold_results: pd.DataFrame, output_path: Path) -> None:
    if fold_results.empty:
        return

    sites = sorted(fold_results["target_site"].astype(str).unique().tolist())
    x = np.arange(len(sites), dtype=float)
    width = 0.36

    fig_w = max(16, 8 + 0.45 * len(sites))
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        b_vals, t_vals = [], []
        for s in sites:
            sb = fold_results[
                (fold_results["target_site"].astype(str) == s)
                & (fold_results["method"] == "baseline")
            ]
            st = fold_results[
                (fold_results["target_site"].astype(str) == s)
                & (fold_results["method"] == "transductive")
            ]
            b_vals.append(
                float(pd.to_numeric(sb[metric], errors="coerce").mean())
                if not sb.empty
                else np.nan
            )
            t_vals.append(
                float(pd.to_numeric(st[metric], errors="coerce").mean())
                if not st.empty
                else np.nan
            )

        ax.bar(x - width / 2, b_vals, width=width, label="baseline")
        ax.bar(x + width / 2, t_vals, width=width, label="transductive")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=80)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2)

    fig.suptitle("ABIDE II - Baseline vs Transductive per site target")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mean_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    pivot = summary_df.pivot(index="metric", columns="method", values="mean").reindex(
        METRICS
    )
    x = np.arange(len(METRICS), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, pivot.get("baseline", np.nan), width=width, label="baseline")
    ax.bar(
        x + width / 2,
        pivot.get("transductive", np.nan),
        width=width,
        label="transductive",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean score")
    ax.set_title("ABIDE II - Metriche medie finali")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_delta_metrics(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    sites = sorted(comparison_df["target_site"].astype(str).unique().tolist())
    x = np.arange(len(sites), dtype=float)

    fig_w = max(16, 8 + 0.45 * len(sites))
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        vals = []
        for s in sites:
            sub = comparison_df[
                (comparison_df["target_site"].astype(str) == s)
                & (comparison_df["metric"] == metric)
            ]
            vals.append(
                float(
                    pd.to_numeric(
                        sub["delta_transductive_minus_baseline"], errors="coerce"
                    ).mean()
                )
                if not sub.empty
                else np.nan
            )

        colors = ["#2ca02c" if (pd.notna(v) and v >= 0) else "#d62728" for v in vals]
        ax.bar(x, vals, color=colors)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"delta {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=80)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ABIDE II - Delta metriche (transductive - baseline)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confidence_distribution(
    conf_df: pd.DataFrame, threshold: float, output_path: Path
) -> None:
    if conf_df.empty:
        return

    selected = conf_df.loc[conf_df["selected_high_conf"] == 1, "confidence"].dropna()
    not_selected = conf_df.loc[
        conf_df["selected_high_conf"] == 0, "confidence"
    ].dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(not_selected, bins=40, alpha=0.55, density=True, label="not_selected")
    ax.hist(selected, bins=40, alpha=0.55, density=True, label="selected_high_conf")
    ax.axvline(
        threshold, color="black", linestyle="--", linewidth=1.2, label="threshold"
    )
    ax.set_xlabel("confidence")
    ax.set_ylabel("density")
    ax.set_title("ABIDE II - Distribuzione confidence su target_adaptation")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pseudo_iterations(pseudo_df: pd.DataFrame, output_path: Path) -> None:
    if pseudo_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for fold_id, sub in pseudo_df.groupby("fold_id", dropna=False):
        sub = sub.sort_values("iteration")
        ax.plot(
            sub["iteration"], sub["n_added"], marker="o", alpha=0.75, label=str(fold_id)
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("n pseudo-label aggiunte")
    ax.set_title("ABIDE II - Pseudo-label aggiunte per iterazione")
    ax.grid(alpha=0.3)

    if pseudo_df["fold_id"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Report diagnostico/interpretativo
# =============================================================================


def detect_suspicious_performance_warning(
    fold_results: pd.DataFrame,
) -> Tuple[bool, str]:
    if fold_results.empty:
        return False, "nessun fold"

    base = fold_results.loc[fold_results["method"] == "baseline"].copy()
    if base.empty:
        return False, "nessuna baseline"

    acc = pd.to_numeric(base["accuracy"], errors="coerce")
    roc = pd.to_numeric(base["roc_auc"], errors="coerce")

    acc_frac = float((acc > 0.95).mean()) if acc.notna().any() else 0.0
    roc_valid = roc.dropna()
    roc_frac = float((roc_valid > 0.95).mean()) if len(roc_valid) > 0 else 0.0

    warning = (acc_frac >= 0.80) or (roc_frac >= 0.80 and len(roc_valid) > 0)
    msg = f"baseline_acc>0.95_fraction={acc_frac:.2f}, baseline_roc_auc>0.95_fraction={roc_frac:.2f}"
    return warning, msg


def write_interpretation(
    output_path: Path,
    input_file: Path,
    target_col: str,
    site_col: str,
    n_samples_total: int,
    n_samples_used: int,
    valid_sites: List[str],
    site_stats: pd.DataFrame,
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    warnings_list: List[str],
) -> None:
    lines: List[str] = []

    lines.append("ABIDE II - Baseline vs Transductive (ASD vs Control)")
    lines.append("=" * 76)
    lines.append("")

    lines.append("Nota di contesto:")
    lines.append("- ABIDE II e multi-site reale (domini reali)")
    lines.append("- task clinico reale: ASD(1) vs Control(0)")
    lines.append("- il domain shift deriva da site/scanner/protocolli diversi")
    lines.append(
        "- il transductive learning puo aiutare ad adattarsi al target domain, ma puo anche peggiorare"
    )
    lines.append("")

    lines.append("Anti-leakage:")
    lines.append("- target_adaptation separato da target_test")
    lines.append("- label vere di target_adaptation NON usate nel training")
    lines.append("- preprocess fit solo su source")
    lines.append("- target_test usato solo per valutazione finale")
    lines.append("")

    lines.append(f"Input file: {input_file}")
    lines.append(f"Colonna target usata: {target_col}")
    lines.append(f"Colonna site usata: {site_col}")
    lines.append(f"Campioni totali letti: {n_samples_total}")
    lines.append(f"Campioni usati dopo filtri: {n_samples_used}")
    lines.append("")

    lines.append(f"Site validi: {len(valid_sites)}")
    lines.append(f"Elenco site validi: {valid_sites}")
    lines.append("")

    if not site_stats.empty:
        lines.append("Distribuzione per site (prime 20 righe):")
        lines.append(site_stats.head(20).to_string(index=False))
        lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valutabile.")
        if warnings_list:
            lines.append("")
            lines.append("Warning:")
            for w in warnings_list:
                lines.append(f"- {w}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(f"Folds valutati: {fold_results['fold_id'].nunique()}")
    lines.append(f"Folds saltati: {len(skipped_df)}")
    lines.append("")

    lines.append("Metriche medie:")
    for method in ["baseline", "transductive"]:
        sub = summary_df.loc[summary_df["method"] == method]
        if sub.empty:
            continue
        lines.append(f"- {method}")
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['metric']}: mean={r['mean']:.4f}, std={r['std']:.4f}, n_valid={int(r['n_valid'])}"
            )
    lines.append("")

    lines.append("Delta medio (transductive - baseline):")
    for metric in METRICS:
        sub = comparison_df.loc[comparison_df["metric"] == metric]
        if sub.empty:
            continue
        delta_mean = float(
            pd.to_numeric(
                sub["delta_transductive_minus_baseline"], errors="coerce"
            ).mean(skipna=True)
        )
        lines.append(f"- {metric}: {delta_mean:+.4f}")
    lines.append("")

    if warnings_list:
        lines.append("Warning:")
        for w in warnings_list:
            lines.append(f"- {w}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABIDE II baseline vs transductive (ASD vs Control)"
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-site-samples", type=int, default=MIN_SITE_SAMPLES)
    parser.add_argument("--min-valid-sites", type=int, default=MIN_VALID_SITES)
    parser.add_argument("--target-test-size", type=float, default=TARGET_TEST_SIZE)
    parser.add_argument("--pseudo-threshold", type=float, default=PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-max-iters", type=int, default=PSEUDO_MAX_ITERS)
    return parser.parse_args()


def copy_script_to_output(output_dir: Path, logger: Logger) -> None:
    try:
        src = Path(__file__).resolve()
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        logger.log(f"[SAVE] Script copiato in: {dst}")
    except Exception as exc:
        logger.log(f"[WARN] Impossibile copiare script: {exc}")


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    input_path = (
        args.input_file if args.input_file is not None else discover_default_input()
    )
    if input_path is None:
        raise FileNotFoundError(
            "Nessun input ABIDE II trovato automaticamente. "
            "Passa --input-file /path/to/abide_features.csv"
        )

    cfg = Config(
        input_file=input_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        min_site_samples=args.min_site_samples,
        min_valid_sites=args.min_valid_sites,
        target_test_size=args.target_test_size,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
    )

    ensure_dir(cfg.output_dir)
    logger = Logger()

    logger.log("[START] ABIDE II baseline vs transductive (ASD vs Control)")
    logger.log(f"[CONFIG] input_file={cfg.input_file}")
    logger.log(f"[CONFIG] output_dir={cfg.output_dir}")

    # ------------------------------------------------------------------
    # 1) Caricamento dati
    # ------------------------------------------------------------------
    if not cfg.input_file.exists():
        raise FileNotFoundError(f"Input path non trovato: {cfg.input_file}")

    if cfg.input_file.is_file():
        sep = "\t" if cfg.input_file.suffix.lower() == ".tsv" else ","
        df = read_table_with_encoding_fallback(cfg.input_file, sep=sep)
    elif cfg.input_file.is_dir():
        pheno_file = discover_default_pheno_file()
        if pheno_file is None:
            raise FileNotFoundError(
                "Input e una cartella QAP, ma non trovo un file fenotipico ABIDE II "
                "con DX_GROUP (es. ABIDEII_Composite_Phenotypic.csv)."
            )
        logger.log(f"[QAP] input directory rilevata: {cfg.input_file}")
        logger.log(f"[QAP] pheno file usato per DX_GROUP: {pheno_file}")
        df = load_qap_directory_with_pheno(cfg.input_file, pheno_file, logger)
        df.to_csv(cfg.output_dir / "abide2_qap_merged_with_dx.csv", index=False)
    else:
        raise RuntimeError(f"Input path non supportato: {cfg.input_file}")

    logger.log(f"[DATA] shape={df.shape}")
    logger.log(f"[DATA] n_columns={len(df.columns)}")
    logger.log(f"[DATA] columns (prime 50)={list(df.columns)[:50]}")

    # Salviamo anche una preview in output
    df.head(10).to_csv(cfg.output_dir / "data_head_preview.csv", index=False)

    # Identificazione colonne target/site
    target_col = find_column_case_insensitive(list(df.columns), TARGET_CANDIDATES)
    site_col = find_column_case_insensitive(list(df.columns), SITE_CANDIDATES)

    if target_col is None:
        raise RuntimeError(
            "Impossibile identificare colonna target (DX_GROUP/diagnosis/label)"
        )
    if site_col is None:
        raise RuntimeError(
            "Impossibile identificare colonna site (SITE_ID/site/center)"
        )

    logger.log(f"[MAP] target_col={target_col}")
    logger.log(f"[MAP] site_col={site_col}")

    # ------------------------------------------------------------------
    # 2) Creazione target binario ASD vs Control
    # ------------------------------------------------------------------
    target_binary, target_map_info = map_target_asd_control(df[target_col])

    logger.log(
        f"[TARGET] mapped={target_map_info['n_mapped']} | unmapped={target_map_info['n_unmapped']} | total={target_map_info['n_total']}"
    )

    work = df.copy()
    work["target_binary"] = target_binary
    work[site_col] = work[site_col].astype("string")

    # Teniamo solo campioni mappati 0/1 e site non null
    before_filter = len(work)
    work = work.loc[work["target_binary"].notna()].copy()
    work = work.loc[work[site_col].notna()].copy()
    work["target_binary"] = work["target_binary"].astype(int)
    after_filter = len(work)

    logger.log(
        f"[TARGET] campioni iniziali={before_filter} | dopo mapping+site validi={after_filter}"
    )

    global_dist = work["target_binary"].value_counts().sort_index()
    logger.log(
        f"[TARGET] distribuzione globale={global_dist.to_dict()} (0=Control,1=ASD)"
    )

    site_target_dist = (
        work.groupby(site_col, dropna=False)["target_binary"]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
    )
    site_target_dist.to_csv(
        cfg.output_dir / "site_target_distribution.csv", index=False
    )

    # ------------------------------------------------------------------
    # 3) Definizione feature
    # ------------------------------------------------------------------
    id_like_cols = identify_id_like_columns(
        list(work.columns), site_col=site_col, target_col=target_col
    )
    always_exclude = set(id_like_cols + [target_col, site_col, "target_binary"])

    candidate_cols = [c for c in work.columns if c not in always_exclude]

    # Solo feature numeriche
    numeric_cols = (
        work[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    # Tentativo conversione robusta di eventuali colonne stringa numeriche
    residual_cols = [c for c in candidate_cols if c not in numeric_cols]
    dropped_non_numeric: List[str] = []

    for col in residual_cols:
        conv = pd.to_numeric(work[col], errors="coerce")
        if float(conv.notna().mean()) >= 0.95:
            work[col] = conv
            numeric_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    numeric_cols = sorted(set(numeric_cols))

    if not numeric_cols:
        raise RuntimeError("Nessuna feature numerica disponibile dopo il filtering")

    logger.log(f"[FEATURE] id_like_excluded={len(id_like_cols)}")
    logger.log(
        f"[FEATURE] candidate_cols={len(candidate_cols)} | numeric_cols_final={len(numeric_cols)}"
    )
    if dropped_non_numeric:
        logger.log(
            f"[FEATURE] non numeriche escluse (prime 20)={dropped_non_numeric[:20]}"
        )

    pd.DataFrame({"id_like_excluded_columns": id_like_cols}).to_csv(
        cfg.output_dir / "id_like_excluded_columns.csv", index=False
    )

    # ------------------------------------------------------------------
    # 4) Filtraggio site validi
    # ------------------------------------------------------------------
    site_rows = []
    for site, grp in work.groupby(site_col, dropna=False):
        n_total = int(len(grp))
        n_ctrl = int((grp["target_binary"] == 0).sum())
        n_asd = int((grp["target_binary"] == 1).sum())
        n_classes = int(grp["target_binary"].nunique())

        valid = pd.notna(site) and n_total >= cfg.min_site_samples and n_classes == 2

        site_rows.append(
            {
                "site": site,
                "n_samples": n_total,
                "n_control_0": n_ctrl,
                "n_asd_1": n_asd,
                "n_classes": n_classes,
                "is_valid_site": int(valid),
            }
        )

    site_stats = (
        pd.DataFrame(site_rows)
        .sort_values("n_samples", ascending=False)
        .reset_index(drop=True)
    )
    site_stats.to_csv(cfg.output_dir / "site_stats_validity.csv", index=False)

    valid_sites = (
        site_stats.loc[site_stats["is_valid_site"] == 1, "site"].astype(str).tolist()
    )

    logger.log(f"[SITE] site_totali={work[site_col].nunique(dropna=True)}")
    logger.log(f"[SITE] site_validi={len(valid_sites)}")
    logger.log(f"[SITE] elenco_site_validi={valid_sites}")

    warnings_list: List[str] = []

    # Warning sbilanciamento globale
    minority_ratio = float(min(global_dist.values) / max(1, sum(global_dist.values)))
    if minority_ratio < 0.20:
        msg = f"Class imbalance globale forte (minority_ratio={minority_ratio:.3f})"
        warnings_list.append(msg)
        logger.log(f"[WARNING] {msg}")

    if len(valid_sites) < cfg.min_valid_sites:
        msg = (
            f"Site validi insufficienti per valutazione multi-site: "
            f"{len(valid_sites)} < {cfg.min_valid_sites}"
        )
        warnings_list.append(msg)
        logger.log(f"[STOP] {msg}")

        write_interpretation(
            output_path=cfg.output_dir / "interpretation.txt",
            input_file=cfg.input_file,
            target_col=target_col,
            site_col=site_col,
            n_samples_total=before_filter,
            n_samples_used=after_filter,
            valid_sites=valid_sites,
            site_stats=site_stats,
            fold_results=pd.DataFrame(),
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            skipped_df=pd.DataFrame(),
            warnings_list=warnings_list,
        )

        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # Manteniamo solo site validi
    work = work.loc[work[site_col].astype(str).isin(valid_sites)].copy()
    work = work.reset_index(drop=True)
    work["row_id"] = np.arange(len(work), dtype=int)

    logger.log(f"[DATA] shape dopo filtro site validi={work.shape}")

    # ------------------------------------------------------------------
    # 5-9) LODO + split target + baseline + transductive
    # ------------------------------------------------------------------
    fold_rows_all: List[Dict[str, object]] = []
    pseudo_rows_all: List[Dict[str, object]] = []
    diag_rows_all: List[Dict[str, object]] = []
    conf_frames: List[pd.DataFrame] = []
    skipped_rows: List[Dict[str, object]] = []

    fold_count = 0

    for target_site in sorted(valid_sites):
        target_df = work.loc[work[site_col].astype(str) == target_site].copy()
        source_df = work.loc[work[site_col].astype(str) != target_site].copy()

        reason = None
        if len(target_df) < 2:
            reason = "target_too_small"
        elif source_df["target_binary"].nunique() < 2:
            reason = "source_single_class"
        elif target_df["target_binary"].nunique() < 2:
            reason = "target_single_class"

        if reason is not None:
            skipped_rows.append(
                {
                    "target_site": target_site,
                    "reason": reason,
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{target_site}] {reason}")
            continue

        # 6) Split target domain in adaptation/test
        strat = (
            target_df["target_binary"]
            if can_stratify(target_df["target_binary"])
            else None
        )
        if strat is None:
            logger.log(
                f"[WARN][{target_site}] split NON stratificato (classi troppo piccole)"
            )

        try:
            adapt_df, test_df = train_test_split(
                target_df,
                test_size=cfg.target_test_size,
                random_state=cfg.random_state,
                stratify=strat,
            )
        except ValueError as exc:
            logger.log(
                f"[WARN][{target_site}] split stratificato fallito, fallback non stratificato: {exc}"
            )
            try:
                adapt_df, test_df = train_test_split(
                    target_df,
                    test_size=cfg.target_test_size,
                    random_state=cfg.random_state,
                    stratify=None,
                )
            except Exception as exc2:
                skipped_rows.append(
                    {
                        "target_site": target_site,
                        "reason": f"split_failed:{exc2}",
                        "target_n": len(target_df),
                    }
                )
                logger.log(f"[SKIP][{target_site}] split fallito: {exc2}")
                continue

        if test_df["target_binary"].nunique() < 2:
            skipped_rows.append(
                {
                    "target_site": target_site,
                    "reason": "target_test_single_class",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{target_site}] target_test mono-classe")
            continue

        fold_count += 1
        fold_id = f"fold_{fold_count}_{safe_name(target_site)}"

        try:
            rows, pseudo_rows, conf_df, diag = run_fold(
                df=work,
                feature_cols=numeric_cols,
                source_idx=source_df.index,
                adapt_idx=adapt_df.index,
                test_idx=test_df.index,
                fold_id=fold_id,
                target_site=target_site,
                cfg=cfg,
                logger=logger,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "target_site": target_site,
                    "reason": f"fold_error:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{target_site}] fold error: {exc}")
            continue

        fold_rows_all.extend(rows)
        pseudo_rows_all.extend(pseudo_rows)
        diag_rows_all.append(diag)
        if not conf_df.empty:
            conf_frames.append(conf_df)

    fold_results = pd.DataFrame(fold_rows_all)
    pseudo_log = pd.DataFrame(pseudo_rows_all)
    diagnostics_df = pd.DataFrame(diag_rows_all)
    confidence_log = (
        pd.concat(conf_frames, ignore_index=True) if conf_frames else pd.DataFrame()
    )
    skipped_df = pd.DataFrame(skipped_rows)

    skipped_df.to_csv(cfg.output_dir / "fold_skipped.csv", index=False)

    n_total_folds_attempted = len(valid_sites)
    n_folds_evaluated = (
        fold_results["fold_id"].nunique() if not fold_results.empty else 0
    )
    n_folds_skipped = len(skipped_df)

    if n_folds_skipped > 0:
        frac_skipped = n_folds_skipped / max(1, n_total_folds_attempted)
        if frac_skipped >= 0.50:
            msg = f"Molti fold saltati ({n_folds_skipped}/{n_total_folds_attempted})"
            warnings_list.append(msg)
            logger.log(f"[WARNING] {msg}")

    # Warning: pochi pseudo-label selezionati
    if not pseudo_log.empty:
        avg_added = float(pd.to_numeric(pseudo_log["n_added"], errors="coerce").mean())
        if avg_added < 2.0:
            msg = f"Pochi pseudo-label selezionati in media per iterazione (avg_added={avg_added:.2f})"
            warnings_list.append(msg)
            logger.log(f"[WARNING] {msg}")

    if n_folds_evaluated == 0:
        msg = "Nessun fold valutabile dopo i controlli"
        warnings_list.append(msg)
        logger.log(f"[STOP] {msg}")

        diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)

        write_interpretation(
            output_path=cfg.output_dir / "interpretation.txt",
            input_file=cfg.input_file,
            target_col=target_col,
            site_col=site_col,
            n_samples_total=before_filter,
            n_samples_used=len(work),
            valid_sites=valid_sites,
            site_stats=site_stats,
            fold_results=fold_results,
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            skipped_df=skipped_df,
            warnings_list=warnings_list,
        )

        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # ------------------------------------------------------------------
    # 11) Valutazione finale
    # ------------------------------------------------------------------
    summary_df = build_summary(fold_results)
    comparison_df = build_comparison(fold_results)

    suspicious, suspicious_msg = detect_suspicious_performance_warning(fold_results)
    if suspicious:
        warnings_list.append(f"Performance sospette: {suspicious_msg}")
        logger.log(f"[WARNING] {warnings_list[-1]}")

    # ------------------------------------------------------------------
    # 12-13) Salvataggio output
    # ------------------------------------------------------------------
    fold_results.to_csv(cfg.output_dir / "results_fold_by_fold.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "summary_metrics.csv", index=False)
    comparison_df.to_csv(
        cfg.output_dir / "comparison_baseline_vs_transductive.csv", index=False
    )
    pseudo_log.to_csv(cfg.output_dir / "pseudo_labeling_iterations.csv", index=False)
    confidence_log.to_csv(cfg.output_dir / "confidence_log.csv", index=False)
    diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)

    plot_metrics_per_site(fold_results, cfg.output_dir / "plot_metrics_per_site.png")
    plot_mean_metrics(summary_df, cfg.output_dir / "plot_mean_metrics.png")
    plot_delta_metrics(comparison_df, cfg.output_dir / "plot_delta_metrics.png")
    plot_confidence_distribution(
        confidence_log,
        cfg.pseudo_threshold,
        cfg.output_dir / "plot_confidence_distribution.png",
    )
    plot_pseudo_iterations(pseudo_log, cfg.output_dir / "plot_pseudo_per_iteration.png")

    write_interpretation(
        output_path=cfg.output_dir / "interpretation.txt",
        input_file=cfg.input_file,
        target_col=target_col,
        site_col=site_col,
        n_samples_total=before_filter,
        n_samples_used=len(work),
        valid_sites=valid_sites,
        site_stats=site_stats,
        fold_results=fold_results,
        summary_df=summary_df,
        comparison_df=comparison_df,
        skipped_df=skipped_df,
        warnings_list=warnings_list,
    )

    copy_script_to_output(cfg.output_dir, logger)
    logger.save(cfg.output_dir / "log.txt")

    logger.log("[END] Analisi completata")
    logger.log(
        f"[SAVE] folds valutati={n_folds_evaluated}, folds saltati={n_folds_skipped}"
    )
    logger.log(f"[SAVE] {cfg.output_dir / 'results_fold_by_fold.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'summary_metrics.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'comparison_baseline_vs_transductive.csv'}")


if __name__ == "__main__":
    main()
