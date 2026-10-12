#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA-BRCA per-site: Baseline vs Transductive Learning (leakage-safe)
=====================================================================

Questo script lavora sui database TCGA-BRCA gia separati per site (domini reali)
e implementa un confronto metodologicamente corretto tra:
- baseline source-only
- transductive learning con pseudo-labeling iterativo

Punti metodologici chiave (fondamentali in tesi):
1) Ogni file site-specific e un dominio reale.
2) Il target binario e costruito da sample_type_code:
   - normal = 0 se sample_type_code == "11"
   - tumor  = 1 se sample_type_code == "01"
3) I campioni con sample_type_code diversi da 01/11 (es. 06) sono esclusi.
4) In ogni fold Leave-One-Domain-Out il target e diviso in:
   - target_adaptation (usato senza label nel transductive)
   - target_test (usato solo per valutazione finale)
5) Tutte le trasformazioni che apprendono dai dati (imputer, scaler,
   feature selection per varianza) sono fittate SOLO sui source.

Nota su TCGA/genomica:
- Il leakage nei dati ad alta dimensionalita puo gonfiare fortemente le metriche.
- Per questo motivo lo script include guardrail espliciti e warning automatici
  se il task appare "troppo facile" (accuracy/ROC-AUC quasi perfette in quasi tutti i fold).
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
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configurazione environment prima di import matplotlib (utile in ambienti sandbox)
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


# ============================================================================
# Config
# ============================================================================

RANDOM_STATE = 42

# Filtri domini
MIN_SITE_SAMPLES = 20
MIN_VALID_SITES = 3

# Split target domain (anti-leakage)
TARGET_TEST_SIZE = 0.50

# Feature selection: top-N varianza calcolata SOLO sui source
TOP_N_FEATURES = 200

# Pseudo-labeling transductive
PSEUDO_THRESHOLD = 0.99
PSEUDO_MAX_ITERS = 3
PSEUDO_MAX_PER_ITER = 20
PSEUDO_MAX_POOL_FRACTION = 0.10

# Metriche richieste
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]

# Colonne obbligatorie in ogni file site
REQUIRED_COLUMNS = {"sample_barcode", "site", "sample_type_code"}


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    random_state: int = RANDOM_STATE
    min_site_samples: int = MIN_SITE_SAMPLES
    min_valid_sites: int = MIN_VALID_SITES
    target_test_size: float = TARGET_TEST_SIZE
    top_n_features: int = TOP_N_FEATURES
    pseudo_threshold: float = PSEUDO_THRESHOLD
    pseudo_max_iters: int = PSEUDO_MAX_ITERS
    pseudo_max_per_iter: int = PSEUDO_MAX_PER_ITER
    pseudo_max_pool_fraction: float = PSEUDO_MAX_POOL_FRACTION


class Logger:
    """Logger semplice: salva sia a schermo sia in file testuale."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ============================================================================
# Utility
# ============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    s = str(value).replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    return s[:120]


def can_stratify(y: pd.Series) -> bool:
    vc = y.value_counts(dropna=True)
    return y.nunique(dropna=True) >= 2 and len(vc) > 0 and int(vc.min()) >= 2


def assert_disjoint(ids_a: pd.Series, ids_b: pd.Series, where: str) -> None:
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] overlap in {where}: {len(overlap)} samples"
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


# ============================================================================
# Parsing sample_type_code e target
# ============================================================================


def normalize_sample_type_code(value: object) -> Optional[str]:
    """
    Normalizza sample_type_code in stringa a due cifre.

    Esempi:
    - 1, "1", "01", "01A" -> "01"
    - 11, "11", "11A"     -> "11"
    - valore non parseabile  -> None
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().upper()
    if text == "" or text == "NAN":
        return None

    # Caso numerico pulito (es. 1, 11, 1.0)
    try:
        f = float(text)
        if float(int(f)) == f:
            code = int(f)
            if 0 <= code <= 99:
                return f"{code:02d}"
    except Exception:
        pass

    # Caso stringa con prefisso numerico (es. 01A)
    import re

    m2 = re.match(r"^(\d{2})", text)
    if m2:
        return m2.group(1)

    m1 = re.match(r"^(\d)$", text)
    if m1:
        return f"0{m1.group(1)}"

    return None


# ============================================================================
# Caricamento file site
# ============================================================================


def load_site_csvs(
    input_dir: Path, logger: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Legge tutti i CSV nella cartella input e valida le colonne obbligatorie.

    Restituisce:
    - dataframe concatenato dei file validi
    - tabella file_validi
    - tabella file_invalidi
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Nessun CSV trovato in: {input_dir}")

    valid_frames: List[pd.DataFrame] = []
    valid_rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []

    logger.log(f"[LOAD] CSV trovati: {len(csv_files)}")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            invalid_rows.append(
                {
                    "file_name": file_path.name,
                    "status": "read_error",
                    "details": str(exc),
                }
            )
            logger.log(f"[WARN] Impossibile leggere {file_path.name}: {exc}")
            continue

        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            invalid_rows.append(
                {
                    "file_name": file_path.name,
                    "status": "missing_required_columns",
                    "details": ",".join(missing),
                }
            )
            logger.log(f"[WARN] {file_path.name} senza colonne obbligatorie: {missing}")
            continue

        # Metadata utile per tracing
        df["source_file"] = file_path.name

        # Normalizziamo il tipo site in stringa
        df["site"] = df["site"].astype("string")

        # Check coerenza: idealmente ogni file site dovrebbe avere un solo site
        n_site_in_file = int(df["site"].nunique(dropna=True))
        if n_site_in_file != 1:
            logger.log(
                f"[WARN] {file_path.name} contiene {n_site_in_file} valori site distinti (atteso ~1)."
            )

        valid_rows.append(
            {
                "file_name": file_path.name,
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "n_unique_site_in_file": n_site_in_file,
                "status": "ok",
            }
        )
        valid_frames.append(df)

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    if not valid_frames:
        raise RuntimeError("Nessun file CSV valido con colonne richieste.")

    merged = pd.concat(valid_frames, axis=0, ignore_index=True)
    return merged, valid_df, invalid_df


# ============================================================================
# Feature preprocessing leakage-safe
# ============================================================================


def fit_source_preprocessor(
    X_source: pd.DataFrame,
    top_n_features: int,
    fold_id: str,
    logger: Logger,
) -> Dict[str, object]:
    """
    Fit del preprocessing SOLO su source_labeled.

    Questo e il punto critico anti-leakage:
    - imputazione, selezione feature e scaling NON devono vedere adaptation/test.

    Pipeline:
    1) conversione numerica + imputazione mediana (fit su source)
    2) varianza calcolata su source imputato
    3) top-N feature piu variabili (source-only)
    4) standardizzazione fit su source selezionato
    """
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    if not usable_cols:
        raise RuntimeError(
            f"[{fold_id}] Nessuna feature numerica utilizzabile nei source"
        )

    X_num = X_num[usable_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    variances = np.var(X_imp, axis=0)
    n_keep = int(min(max(1, top_n_features), X_imp.shape[1]))
    top_idx = np.argsort(variances)[::-1][:n_keep]
    top_idx = np.sort(top_idx)

    selected_features = [usable_cols[i] for i in top_idx]
    X_sel = X_imp[:, top_idx]

    scaler = StandardScaler()
    X_source_final = scaler.fit_transform(X_sel)

    logger.log(
        f"[PREP][{fold_id}] features_before={X_source.shape[1]} | usable_numeric={len(usable_cols)} | after_topN={len(selected_features)}"
    )

    return {
        "usable_cols": usable_cols,
        "imputer": imputer,
        "top_idx": top_idx,
        "scaler": scaler,
        "selected_features": selected_features,
        "X_source_final": X_source_final,
        "n_before": int(X_source.shape[1]),
        "n_after": int(len(selected_features)),
    }


def transform_with_source_preprocessor(
    X_df: pd.DataFrame, prep: Dict[str, object]
) -> np.ndarray:
    """Applica su adaptation/test trasformazioni gia fittate sui source."""
    X_num = X_df[prep["usable_cols"]].apply(pd.to_numeric, errors="coerce")
    X_imp = prep["imputer"].transform(X_num)
    X_sel = X_imp[:, prep["top_idx"]]
    X_out = prep["scaler"].transform(X_sel)
    return X_out


# ============================================================================
# Metriche
# ============================================================================


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


# ============================================================================
# Transductive pseudo-labeling
# ============================================================================


def run_pseudo_labeling(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_adapt_unlabeled: np.ndarray,
    threshold: float,
    max_iters: int,
    max_per_iter: int,
    max_pool_fraction: float,
    seed: int,
    fold_id: str,
    logger: Logger,
) -> Tuple[LogisticRegression, List[Dict[str, object]], pd.DataFrame, int, int]:
    """
    Pseudo-labeling iterativo manuale.

    Vincoli anti-leakage rispettati:
    - input adaptation senza vere label
    - target_test non entra mai qui
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

        # Limite pseudo-label per iterazione: massimo 20 o massimo 10% del pool
        cap_by_fraction = max(1, int(math.ceil(max_pool_fraction * pool_n)))
        cap = min(max_per_iter, cap_by_fraction, n_candidates)

        sorted_candidates = high_conf_idx[np.argsort(conf[high_conf_idx])[::-1]]
        chosen_idx = np.sort(sorted_candidates[:cap])

        X_sel = X_pool[chosen_idx]
        y_sel = pred[chosen_idx].astype(int)

        # Update training set
        X_train = np.vstack([X_train, X_sel])
        y_train = np.concatenate([y_train, y_sel], axis=0)

        # Rimozione selezionati dal pool
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

    # Fit finale sul training aggiornato
    final_model = build_lr(seed + 777)
    final_model.fit(X_train, y_train)

    return (
        final_model,
        history,
        pd.DataFrame(confidence_rows),
        int(total_added),
        int(len(y_train)),
    )


# ============================================================================
# Fold runner
# ============================================================================


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

    # Check anti-leakage sugli ID
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

    # PREPROCESS FIT SOLO SOURCE (fondamentale anti-leakage)
    prep = fit_source_preprocessor(X_source, cfg.top_n_features, fold_id, logger)
    logger.log(f"[CHECK][{fold_id}] preprocess fit SOLO su source")

    Xs = prep["X_source_final"]
    Xa = transform_with_source_preprocessor(X_adapt, prep)
    Xt = transform_with_source_preprocessor(X_test, prep)

    # ------------------------------------------------------------------
    # Baseline source-only
    # ------------------------------------------------------------------
    baseline = build_lr(cfg.random_state)
    baseline.fit(Xs, y_source)

    ypred_b = baseline.predict(Xt)
    yprob_b = baseline.predict_proba(Xt)
    m_b = compute_binary_metrics(y_test, ypred_b, yprob_b, baseline.classes_)

    # ------------------------------------------------------------------
    # Transductive (pseudo-labeling)
    # ------------------------------------------------------------------
    logger.log(f"[CHECK][{fold_id}] y vere adaptation NON usate nel fit transductive")

    trans_model, pseudo_hist, conf_df, pseudo_total, final_train_size = (
        run_pseudo_labeling(
            X_source=Xs,
            y_source=y_source,
            X_adapt_unlabeled=Xa,
            threshold=cfg.pseudo_threshold,
            max_iters=cfg.pseudo_max_iters,
            max_per_iter=cfg.pseudo_max_per_iter,
            max_pool_fraction=cfg.pseudo_max_pool_fraction,
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
            "features_before": prep["n_before"],
            "features_after": prep["n_after"],
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
            "features_before": prep["n_before"],
            "features_after": prep["n_after"],
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
        "features_before": prep["n_before"],
        "features_after": prep["n_after"],
        "train_test_disjoint_ok": 1,
        "used_true_adapt_labels_in_train": 0,
        "preprocess_fit_source_only": 1,
        "pseudo_total_added": int(pseudo_total),
        "baseline_accuracy": m_b.get("accuracy", np.nan),
        "baseline_roc_auc": m_b.get("roc_auc", np.nan),
    }

    return rows, pseudo_rows, conf_df, diag


# ============================================================================
# Aggregazioni
# ============================================================================


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


# ============================================================================
# Plot
# ============================================================================


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

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("Baseline vs Transductive per site target")
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
    ax.set_title("Metriche medie finali")
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

    fig.suptitle("Delta metriche (transductive - baseline)")
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
    ax.set_title("Distribuzione confidence su target_adaptation")
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
    ax.set_title("Pseudo-label aggiunte per iterazione")
    ax.grid(alpha=0.3)

    if pseudo_df["fold_id"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================================
# Report testuale
# ============================================================================


def detect_too_easy_warning(fold_results: pd.DataFrame) -> Tuple[bool, str]:
    """Warning se baseline quasi perfetta in quasi tutti i fold."""
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
    n_files_valid: int,
    n_files_invalid: int,
    n_samples_before: int,
    n_samples_after: int,
    n_samples_excluded: int,
    valid_sites: List[str],
    site_stats: pd.DataFrame,
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    task_warning: bool,
    task_warning_msg: str,
) -> None:
    lines: List[str] = []

    lines.append("TCGA separati per site - Baseline vs Transductive")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Setup dominio/target:")
    lines.append("- ogni file site-specific e trattato come dominio reale")
    lines.append("- target binario costruito da sample_type_code")
    lines.append("  normal=0 (11), tumor=1 (01)")
    lines.append("- campioni con codice diverso da 01/11 esclusi")
    lines.append("")

    lines.append(f"File validi letti: {n_files_valid}")
    lines.append(f"File invalidi/skippati: {n_files_invalid}")
    lines.append(f"Campioni prima filtro 01/11: {n_samples_before}")
    lines.append(f"Campioni dopo filtro 01/11: {n_samples_after}")
    lines.append(f"Campioni esclusi: {n_samples_excluded}")
    lines.append("")

    lines.append(f"Site validi (>=20 campioni + entrambe classi): {len(valid_sites)}")
    lines.append(f"Elenco site validi: {valid_sites}")
    lines.append("")

    if not site_stats.empty:
        lines.append("Distribuzione per site (prime 20 righe):")
        lines.append(site_stats.head(20).to_string(index=False))
        lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valutabile.")
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

    if task_warning:
        lines.append(
            "WARNING: task potenzialmente troppo facile per valutare la transportability"
        )
        lines.append(f"Dettaglio: {task_warning_msg}")
        lines.append(
            "Se baseline e quasi perfetta in quasi tutti i fold, il contributo del transductive "
            "puo essere poco informativo o nullo."
        )
        lines.append("")

    lines.append("Nota anti-leakage:")
    lines.append("- split target_adaptation/target_test prima del training")
    lines.append("- preprocessing e feature selection fit solo sui source")
    lines.append("- y vere di target_adaptation mai usate nel fit")
    lines.append("- target_test completamente unseen fino alla valutazione")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCGA per-site baseline vs transductive learning"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(_configured_path("legacy_tcga_brca_by_site")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_tcga_separati_output_trans")),
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--min-site-samples", type=int, default=MIN_SITE_SAMPLES)
    parser.add_argument("--min-valid-sites", type=int, default=MIN_VALID_SITES)
    parser.add_argument("--target-test-size", type=float, default=TARGET_TEST_SIZE)
    parser.add_argument("--pseudo-threshold", type=float, default=PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-max-iters", type=int, default=PSEUDO_MAX_ITERS)
    parser.add_argument("--pseudo-max-per-iter", type=int, default=PSEUDO_MAX_PER_ITER)
    parser.add_argument(
        "--pseudo-max-pool-fraction", type=float, default=PSEUDO_MAX_POOL_FRACTION
    )
    return parser.parse_args()


def copy_script_to_output(output_dir: Path, logger: Logger) -> None:
    try:
        src = Path(__file__).resolve()
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        logger.log(f"[SAVE] Script copiato in: {dst}")
    except Exception as exc:
        logger.log(f"[WARN] Impossibile copiare script in output: {exc}")


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        min_site_samples=args.min_site_samples,
        min_valid_sites=args.min_valid_sites,
        target_test_size=args.target_test_size,
        top_n_features=args.top_n_features,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
        pseudo_max_per_iter=args.pseudo_max_per_iter,
        pseudo_max_pool_fraction=args.pseudo_max_pool_fraction,
    )

    ensure_dir(cfg.output_dir)
    logger = Logger()

    logger.log("[START] TCGA per-site baseline vs transductive")
    logger.log(f"[CONFIG] input_dir={cfg.input_dir}")
    logger.log(f"[CONFIG] output_dir={cfg.output_dir}")

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Input directory non trovata: {cfg.input_dir}")

    # ------------------------------------------------------------------
    # 1) Caricamento file per site
    # ------------------------------------------------------------------
    merged, valid_files_df, invalid_files_df = load_site_csvs(cfg.input_dir, logger)

    n_files_valid = len(valid_files_df)
    n_files_invalid = len(invalid_files_df)

    valid_files_df.to_csv(cfg.output_dir / "loaded_files_valid.csv", index=False)
    invalid_files_df.to_csv(cfg.output_dir / "loaded_files_invalid.csv", index=False)

    logger.log(f"[LOAD] File validi={n_files_valid} | invalidi={n_files_invalid}")
    logger.log(f"[DATA] Shape merge iniziale: {merged.shape}")

    # ------------------------------------------------------------------
    # 2) Costruzione target binario da sample_type_code
    # ------------------------------------------------------------------
    merged["sample_type_code_norm"] = merged["sample_type_code"].apply(
        normalize_sample_type_code
    )

    n_samples_before = len(merged)

    # Manteniamo solo 01/11 per target binario pulito
    keep_mask = merged["sample_type_code_norm"].isin(["01", "11"])
    filtered = merged.loc[keep_mask].copy()

    n_samples_after = len(filtered)
    n_samples_excluded = n_samples_before - n_samples_after

    filtered["target_binary"] = np.where(
        filtered["sample_type_code_norm"] == "01", 1, 0
    )

    # Log esclusioni per site
    per_site_before = merged.groupby("site", dropna=False).size().rename("n_before")
    per_site_after = filtered.groupby("site", dropna=False).size().rename("n_after")
    exclusion_table = (
        pd.concat([per_site_before, per_site_after], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    exclusion_table["n_excluded"] = (
        exclusion_table["n_before"] - exclusion_table["n_after"]
    )
    exclusion_table = exclusion_table.sort_values(
        "n_after", ascending=False
    ).reset_index(drop=True)
    exclusion_table.to_csv(cfg.output_dir / "site_exclusion_counts.csv", index=False)

    logger.log(f"[TARGET] Campioni prima filtro 01/11: {n_samples_before}")
    logger.log(f"[TARGET] Campioni dopo filtro 01/11: {n_samples_after}")
    logger.log(f"[TARGET] Campioni esclusi: {n_samples_excluded}")

    global_target_dist = filtered["target_binary"].value_counts().sort_index()
    logger.log(f"[TARGET] Distribuzione globale target: {global_target_dist.to_dict()}")

    # Distribuzione per site
    site_target_dist = (
        filtered.groupby("site", dropna=False)["target_binary"]
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
    excluded_feature_cols = {
        "sample_barcode",
        "site",
        "sample_type_code",
        "sample_type_code_norm",
        "target_binary",
        "source_file",
    }

    candidate_cols = [c for c in filtered.columns if c not in excluded_feature_cols]

    # Prima passata: feature numeriche gia riconosciute da pandas
    numeric_cols = (
        filtered[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    # Seconda passata: tentativo robusto su eventuali residue non numeriche
    residual_cols = [c for c in candidate_cols if c not in numeric_cols]
    converted_cols: List[str] = []
    dropped_non_numeric: List[str] = []

    for col in residual_cols:
        conv = pd.to_numeric(filtered[col], errors="coerce")
        # Se almeno 95% valori convertibili, consideriamo la colonna numerica
        if float(conv.notna().mean()) >= 0.95:
            filtered[col] = conv
            numeric_cols.append(col)
            converted_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    numeric_cols = sorted(set(numeric_cols))

    if not numeric_cols:
        raise RuntimeError("Nessuna feature numerica disponibile dopo il filtering")

    logger.log(f"[FEATURE] candidate_cols={len(candidate_cols)}")
    logger.log(f"[FEATURE] numeric_cols_final={len(numeric_cols)}")
    if converted_cols:
        logger.log(
            f"[FEATURE] colonne convertite a numerico (prime 20): {converted_cols[:20]}"
        )
    if dropped_non_numeric:
        logger.log(
            f"[FEATURE] colonne non numeriche escluse (prime 20): {dropped_non_numeric[:20]}"
        )

    # ------------------------------------------------------------------
    # 4) Filtraggio site validi
    # ------------------------------------------------------------------
    site_rows = []
    for site, grp in filtered.groupby("site", dropna=False):
        n_total = int(len(grp))
        n_tumor = int((grp["target_binary"] == 1).sum())
        n_normal = int((grp["target_binary"] == 0).sum())
        n_classes = int(grp["target_binary"].nunique())

        valid = pd.notna(site) and n_total >= cfg.min_site_samples and n_classes == 2

        site_rows.append(
            {
                "site": site,
                "n_samples": n_total,
                "n_tumor": n_tumor,
                "n_normal": n_normal,
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

    logger.log(
        f"[SITE] Site validi={len(valid_sites)} (min_samples={cfg.min_site_samples}, due classi richieste)"
    )
    logger.log(f"[SITE] Elenco site validi: {valid_sites}")

    # Se pochi domini, stop pulito con log
    if len(valid_sites) < cfg.min_valid_sites:
        msg = f"Site validi insufficienti per analisi multi-site sensata: {len(valid_sites)} < {cfg.min_valid_sites}."
        logger.log(f"[STOP] {msg}")

        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # Manteniamo solo site validi
    filtered = filtered.loc[filtered["site"].astype(str).isin(valid_sites)].copy()
    filtered = filtered.reset_index(drop=True)
    filtered["row_id"] = np.arange(len(filtered), dtype=int)

    logger.log(f"[DATA] Shape dopo filtro site validi: {filtered.shape}")

    # ------------------------------------------------------------------
    # 5-9) Esperimento LODO con baseline + transductive
    # ------------------------------------------------------------------
    fold_rows_all: List[Dict[str, object]] = []
    pseudo_rows_all: List[Dict[str, object]] = []
    diag_rows_all: List[Dict[str, object]] = []
    conf_frames: List[pd.DataFrame] = []
    skipped_rows: List[Dict[str, object]] = []

    fold_count = 0

    for target_site in sorted(valid_sites):
        target_df = filtered.loc[filtered["site"].astype(str) == target_site].copy()
        source_df = filtered.loc[filtered["site"].astype(str) != target_site].copy()

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

        strat = (
            target_df["target_binary"]
            if can_stratify(target_df["target_binary"])
            else None
        )
        if strat is None:
            logger.log(
                f"[WARN][{target_site}] Stratify non applicabile nel target split"
            )

        try:
            adapt_df, test_df = train_test_split(
                target_df,
                test_size=cfg.target_test_size,
                random_state=cfg.random_state,
                stratify=strat,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "target_site": target_site,
                    "reason": f"split_failed:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{target_site}] split fallito: {exc}")
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
                df=filtered,
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

    # Se nessun fold valido
    if fold_results.empty:
        msg = "Nessun fold valutabile dopo i controlli di validita."
        logger.log(f"[STOP] {msg}")

        diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)
        write_interpretation(
            output_path=cfg.output_dir / "interpretation.txt",
            n_files_valid=n_files_valid,
            n_files_invalid=n_files_invalid,
            n_samples_before=n_samples_before,
            n_samples_after=n_samples_after,
            n_samples_excluded=n_samples_excluded,
            valid_sites=valid_sites,
            site_stats=site_stats,
            fold_results=fold_results,
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            skipped_df=skipped_df,
            task_warning=False,
            task_warning_msg="nessun fold",
        )

        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # ------------------------------------------------------------------
    # 10-11) Aggregazioni finali
    # ------------------------------------------------------------------
    summary_df = build_summary(fold_results)
    comparison_df = build_comparison(fold_results)

    # Warning task troppo facile
    task_warning, task_warning_msg = detect_too_easy_warning(fold_results)
    if task_warning:
        logger.log(
            "[WARNING] Task probabilmente troppo facile per transportability informativa"
        )
        logger.log(f"[WARNING] {task_warning_msg}")

    # ------------------------------------------------------------------
    # 12-13) Salvataggio risultati
    # ------------------------------------------------------------------
    fold_results.to_csv(cfg.output_dir / "results_fold_by_fold.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "summary_metrics.csv", index=False)
    comparison_df.to_csv(
        cfg.output_dir / "comparison_baseline_vs_transductive.csv", index=False
    )
    pseudo_log.to_csv(cfg.output_dir / "pseudo_labeling_iterations.csv", index=False)
    confidence_log.to_csv(cfg.output_dir / "confidence_log.csv", index=False)
    diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)

    # Plot richiesti
    plot_metrics_per_site(fold_results, cfg.output_dir / "plot_metrics_per_site.png")
    plot_mean_metrics(summary_df, cfg.output_dir / "plot_mean_metrics.png")
    plot_delta_metrics(comparison_df, cfg.output_dir / "plot_delta_metrics.png")
    plot_confidence_distribution(
        confidence_log,
        cfg.pseudo_threshold,
        cfg.output_dir / "plot_confidence_distribution.png",
    )
    plot_pseudo_iterations(pseudo_log, cfg.output_dir / "plot_pseudo_per_iteration.png")

    # Report testuale
    write_interpretation(
        output_path=cfg.output_dir / "interpretation.txt",
        n_files_valid=n_files_valid,
        n_files_invalid=n_files_invalid,
        n_samples_before=n_samples_before,
        n_samples_after=n_samples_after,
        n_samples_excluded=n_samples_excluded,
        valid_sites=valid_sites,
        site_stats=site_stats,
        fold_results=fold_results,
        summary_df=summary_df,
        comparison_df=comparison_df,
        skipped_df=skipped_df,
        task_warning=task_warning,
        task_warning_msg=task_warning_msg,
    )

    copy_script_to_output(cfg.output_dir, logger)
    logger.save(cfg.output_dir / "log.txt")

    logger.log("[END] Analisi completata")
    logger.log(f"[SAVE] {cfg.output_dir / 'results_fold_by_fold.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'summary_metrics.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'comparison_baseline_vs_transductive.csv'}")


if __name__ == "__main__":
    main()
