#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA-BRCA baseline vs transductive learning (LEAKAGE-SAFE, versione FIXED).

Questo script e pensato per un uso tesi e privilegia correttezza metodologica
sul puro valore numerico delle performance.

Perche TCGA e ad alto rischio leakage:
- dataset ad alta dimensionalita (migliaia di feature) con relativamente pochi campioni;
- anche piccoli errori in preprocessing (scaler/feature selection fit globali) possono
  trasferire informazione dal test al training;
- questo porta facilmente a performance artificialmente alte (anche ~1.0), non realistiche.

Principio chiave anti-leakage (obbligatorio):
- TUTTE le trasformazioni che apprendono dai dati (imputer, scaler, feature selection)
  vengono fittate SOLO su source_labeled del fold corrente.
- target_adaptation e target_test sono solo trasformati con oggetti gia fittati sui source.
- target_test non entra mai nel training.
- le label vere di target_adaptation non entrano mai nel fit transductive.

Lo script gestisce due scenari:
A) Domain presente -> Leave-One-Domain-Out (LODO)
B) Domain assente -> simulazione train/test + adaptation (esplicitamente indicata)
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
import inspect
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Config env PRIMA di import matplotlib per evitare cache non scrivibili.
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.feature_selection import VarianceThreshold
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
TARGET_TEST_SIZE = 0.50
SIM_TEST_SIZE = 0.30
SIM_ADAPT_SIZE = 0.30
PSEUDO_CONF_THRESHOLD = 0.95
PSEUDO_MAX_ITERS = 10
VARIANCE_THRESHOLD = 1e-8
TOP_N_FEATURES = 1000
MIN_DOMAIN_SAMPLES = 20

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]

TARGET_EXACT = [
    "target",
    "label",
    "class",
    "y",
    "subtype",
    "diagnosis",
    "outcome",
    "phenotype",
    "sample_type_code",
    "sample_type",
]
TARGET_PARTIAL = [
    "target",
    "label",
    "class",
    "subtype",
    "diagnos",
    "outcome",
    "phenotype",
    "sample_type",
]

DOMAIN_EXACT = [
    "site",
    "domain",
    "batch",
    "source",
    "center",
    "centre",
    "dataset",
    "cohort",
    "tss_code",
    "tss",
]
DOMAIN_PARTIAL = [
    "site",
    "domain",
    "batch",
    "source",
    "center",
    "centre",
    "dataset",
    "cohort",
    "tss",
]

ID_HINTS = [
    "sample_id",
    "patient_id",
    "barcode",
    "submitter",
    "case_id",
    "id",
    "uuid",
    "aliquot",
]


@dataclass
class Config:
    input_csv: Path
    output_dir: Path
    random_state: int = RANDOM_STATE
    target_test_size: float = TARGET_TEST_SIZE
    sim_test_size: float = SIM_TEST_SIZE
    sim_adapt_size: float = SIM_ADAPT_SIZE
    pseudo_threshold: float = PSEUDO_CONF_THRESHOLD
    pseudo_max_iters: int = PSEUDO_MAX_ITERS
    variance_threshold: float = VARIANCE_THRESHOLD
    top_n_features: int = TOP_N_FEATURES
    min_domain_samples: int = MIN_DOMAIN_SAMPLES


class Logger:
    """Logger semplice per terminale + file."""

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


def safe_name(v: object) -> str:
    s = str(v).replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    return s[:120]


def class_dist_text(y: pd.Series) -> str:
    vc = y.value_counts(dropna=False).sort_index()
    return ", ".join([f"class_{k}={int(v)}" for k, v in vc.items()])


def can_stratify(y: pd.Series) -> bool:
    vc = y.value_counts(dropna=True)
    return y.nunique(dropna=True) >= 2 and (len(vc) > 0 and vc.min() >= 2)


def assert_disjoint(ids_a: pd.Series, ids_b: pd.Series, where: str) -> None:
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] overlap in {where}: {len(overlap)} samples"
        )


def build_lr(seed: int) -> LogisticRegression:
    params = {
        "class_weight": "balanced",
        "max_iter": 6000,
        "solver": "lbfgs",
        "random_state": seed,
    }
    sig = inspect.signature(LogisticRegression.__init__)
    if "multi_class" in sig.parameters:
        params["multi_class"] = "auto"
    return LogisticRegression(**params)


def stack_rows(Xa, Xb):
    if sparse.issparse(Xa) or sparse.issparse(Xb):
        return sparse.vstack([sparse.csr_matrix(Xa), sparse.csr_matrix(Xb)])
    return np.vstack([Xa, Xb])


# ============================================================================
# Detection colonne
# ============================================================================


def _score_target_candidate(col: str, series: pd.Series, n_rows: int) -> float:
    low = col.lower()
    score = 0.0

    # score nome
    if low in TARGET_EXACT:
        score += 12.0
    if any(k in low for k in TARGET_PARTIAL):
        score += 6.0

    # score cardinalita (target supervisionato dovrebbe avere poche classi)
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(1, n_rows)

    if 2 <= nunique <= 10:
        score += 8.0
    elif 2 <= nunique <= 30:
        score += 3.0
    elif nunique > 200:
        score -= 10.0

    if ratio < 0.15:
        score += 4.0
    elif ratio < 0.5:
        score += 1.0
    else:
        score -= 6.0

    return score


def detect_target_column(df: pd.DataFrame, logger: Logger) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        score = _score_target_candidate(col, s, n)
        if score > 0:
            candidates.append((score, col))

    if not candidates:
        logger.log("[DETECT] Nessuna target plausibile trovata.")
        return None

    candidates.sort(reverse=True)
    best_score, best_col = candidates[0]
    logger.log(
        f"[DETECT] Target candidata migliore: {best_col} (score={best_score:.2f})"
    )
    return best_col


def _score_domain_candidate(col: str, series: pd.Series, n_rows: int) -> float:
    low = col.lower()
    score = 0.0

    if low in DOMAIN_EXACT:
        score += 12.0
    if any(k in low for k in DOMAIN_PARTIAL):
        score += 6.0

    nunique = series.nunique(dropna=True)
    ratio = nunique / max(1, n_rows)

    # Domain utile: >1 categoria, ma non quasi unico per campione.
    if nunique <= 1:
        score -= 100.0
    elif 2 <= nunique <= 200:
        score += 4.0

    if ratio < 0.5:
        score += 2.0
    else:
        score -= 4.0

    return score


def detect_domain_column(
    df: pd.DataFrame, target_col: Optional[str], logger: Logger
) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    n = len(df)

    for col in df.columns:
        if col == target_col:
            continue
        score = _score_domain_candidate(col, df[col], n)
        if score > 0:
            candidates.append((score, col))

    if not candidates:
        logger.log("[DETECT] Nessuna domain plausibile trovata.")
        return None

    candidates.sort(reverse=True)
    best_score, best_col = candidates[0]

    # Guardrail: evitiamo di scegliere colonne quasi uniche (tipici ID).
    nunique = df[best_col].nunique(dropna=True)
    if nunique / max(1, len(df)) > 0.85:
        logger.log(
            f"[DETECT] Domain candidata {best_col} troppo unica (ratio alto), ignorata."
        )
        return None

    logger.log(
        f"[DETECT] Domain candidata migliore: {best_col} (score={best_score:.2f})"
    )
    return best_col


def detect_id_columns(
    df: pd.DataFrame,
    target_col: Optional[str],
    domain_col: Optional[str],
    logger: Logger,
) -> List[str]:
    id_cols: List[str] = []

    for col in df.columns:
        if col in {target_col, domain_col}:
            continue

        low = col.lower()
        if any(h in low for h in ID_HINTS):
            id_cols.append(col)
            continue

        # colonne stringa quasi uniche: quasi sicuramente identificatori.
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / max(1, len(df))
            if ratio >= 0.90:
                id_cols.append(col)

    id_cols = sorted(set(id_cols))
    logger.log(f"[DETECT] ID columns escluse: {id_cols if id_cols else 'nessuna'}")
    return id_cols


# ============================================================================
# Target encoding
# ============================================================================


def encode_target(y_raw: pd.Series) -> Tuple[pd.Series, Dict[object, int]]:
    # Numeric se possibile.
    y_num = pd.to_numeric(y_raw, errors="coerce")

    if y_num.notna().sum() > 0 and (y_num.notna().mean() >= 0.8):
        uniq = sorted(y_num.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(uniq)}
        y_enc = y_num.map(mapping).astype("Int64")
        return y_enc, mapping

    # fallback string.
    y_str = y_raw.astype("string")
    uniq = sorted(y_str.dropna().unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    y_enc = y_str.map(mapping).astype("Int64")
    return y_enc, mapping


# ============================================================================
# Preprocessing source-only
# ============================================================================


def fit_source_preprocessor(
    X_source: pd.DataFrame,
    variance_threshold: float,
    top_n: int,
    fold_id: str,
    logger: Logger,
) -> Dict[str, object]:
    """
    Fit trasformazioni SOLO su source_labeled (anti-leakage).

    Sequenza corretta:
    1) conversione numerica su source
    2) imputazione median fit su source
    3) VarianceThreshold fit su source
    4) top-N per varianza calcolata su source
    5) StandardScaler fit su source selezionato

    ERRORE da evitare (leakage):
    - scegliere feature o fit scaler su dataset completo.
    """
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    # teniamo solo colonne con almeno un valore valido nei source.
    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    dropped_non_numeric = sorted(set(X_source.columns) - set(usable_cols))

    if not usable_cols:
        raise RuntimeError(
            f"[{fold_id}] Nessuna feature numerica disponibile nei source."
        )

    X_num = X_num[usable_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    vt = VarianceThreshold(threshold=variance_threshold)
    try:
        X_vt = vt.fit_transform(X_imp)
    except ValueError:
        vt = VarianceThreshold(threshold=0.0)
        X_vt = vt.fit_transform(X_imp)

    if X_vt.shape[1] == 0:
        raise RuntimeError(f"[{fold_id}] Nessuna feature dopo VarianceThreshold.")

    vt_idx = vt.get_support(indices=True)
    vt_feature_names = [usable_cols[i] for i in vt_idx]

    # top-N per varianza calcolata SOLO sui source (post VT).
    var_source = np.var(X_vt, axis=0)
    n_keep = int(min(max(1, top_n), X_vt.shape[1]))
    top_rel = np.argsort(var_source)[::-1][:n_keep]
    top_rel = np.sort(top_rel)

    selected_features = [vt_feature_names[i] for i in top_rel]

    scaler = StandardScaler()
    X_sel = X_vt[:, top_rel]
    X_source_final = scaler.fit_transform(X_sel)

    logger.log(
        f"[PREP][{fold_id}] features_before={X_source.shape[1]} | usable_numeric={len(usable_cols)} | "
        f"after_VT={len(vt_feature_names)} | after_topN={len(selected_features)}"
    )

    if dropped_non_numeric:
        logger.log(
            f"[PREP][{fold_id}] colonne non numeriche scartate (prime 20): {dropped_non_numeric[:20]}"
            + (" ..." if len(dropped_non_numeric) > 20 else "")
        )

    return {
        "usable_cols": usable_cols,
        "imputer": imputer,
        "vt": vt,
        "top_rel": top_rel,
        "scaler": scaler,
        "selected_features": selected_features,
        "n_before": int(X_source.shape[1]),
        "n_after": int(len(selected_features)),
        "X_source_final": X_source_final,
    }


def transform_with_source_preprocessor(X_df: pd.DataFrame, prep: Dict[str, object]):
    usable_cols = prep["usable_cols"]
    imputer = prep["imputer"]
    vt = prep["vt"]
    top_rel = prep["top_rel"]
    scaler = prep["scaler"]

    X_num = X_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    X_imp = imputer.transform(X_num)
    X_vt = vt.transform(X_imp)
    X_sel = X_vt[:, top_rel]
    X_final = scaler.transform(X_sel)
    return X_final


# ============================================================================
# Metriche
# ============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_classes: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    uniq_true = np.unique(y_true)
    uniq_pred = np.unique(y_pred)
    uniq_union = np.unique(np.concatenate([uniq_true, uniq_pred]))

    if len(uniq_true) <= 1:
        out["precision"] = np.nan
        out["recall"] = np.nan
        out["f1"] = np.nan
        out["roc_auc"] = np.nan
        return out

    # binario solo se davvero binario osservato (true+pred).
    if len(uniq_union) == 2:
        pos_label = sorted(uniq_union.tolist())[-1]
        out["precision"] = float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )
        out["recall"] = float(
            recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )
        out["f1"] = float(
            f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )

        try:
            cls_to_idx = {c: i for i, c in enumerate(model_classes)}
            pos_idx = cls_to_idx[pos_label]
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, pos_idx]))
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out["recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        try:
            out["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=model_classes,
                )
            )
        except Exception:
            out["roc_auc"] = np.nan

    return out


# ============================================================================
# Pseudo-labeling transductive
# ============================================================================


def run_pseudo_labeling(
    X_source,
    y_source: np.ndarray,
    X_adapt_unlabeled,
    threshold: float,
    max_iters: int,
    seed: int,
    fold_id: str,
    logger: Logger,
) -> Tuple[LogisticRegression, List[Dict[str, object]], pd.DataFrame, int, int]:
    """
    Pseudo-labeling iterativo manuale.

    Vincoli anti-leakage:
    - riceve SOLO X_adapt_unlabeled (nessuna y vera adaptation)
    - target_test non compare in alcun input della funzione
    """
    X_train = X_source
    y_train = np.asarray(y_source, dtype=int)
    X_pool = X_adapt_unlabeled

    history: List[Dict[str, object]] = []
    conf_rows: List[Dict[str, object]] = []
    total_added = 0

    for it in range(1, max_iters + 1):
        model = build_lr(seed + it)
        model.fit(X_train, y_train)

        n_pool = int(X_pool.shape[0])
        if n_pool == 0:
            history.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "n_pool_before": 0,
                    "n_selected": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": np.nan,
                    "conf_min": np.nan,
                    "conf_max": np.nan,
                }
            )
            logger.log(f"[PSEUDO][{fold_id}] Iter {it}: pool vuoto, stop.")
            break

        proba = model.predict_proba(X_pool)
        pred = model.predict(X_pool)
        conf = np.max(proba, axis=1)
        sel = conf >= threshold
        n_sel = int(sel.sum())

        for c, s in zip(conf.tolist(), sel.tolist()):
            conf_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "confidence": float(c),
                    "selected": int(bool(s)),
                }
            )

        conf_mean = float(np.mean(conf))
        conf_min = float(np.min(conf))
        conf_max = float(np.max(conf))

        if n_sel == 0:
            history.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "n_pool_before": n_pool,
                    "n_selected": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": conf_mean,
                    "conf_min": conf_min,
                    "conf_max": conf_max,
                }
            )
            logger.log(
                f"[PSEUDO][{fold_id}] Iter {it}: nessun campione >= {threshold:.2f}, stop."
            )
            break

        X_sel = X_pool[sel]
        y_sel = pred[sel].astype(int)

        X_train = stack_rows(X_train, X_sel)
        y_train = np.concatenate([y_train, y_sel], axis=0)

        X_pool = X_pool[~sel]

        cls_counts = pd.Series(y_sel).value_counts().sort_index().to_dict()
        cls_str = "{" + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()]) + "}"

        history.append(
            {
                "fold_id": fold_id,
                "iteration": it,
                "n_pool_before": n_pool,
                "n_selected": n_sel,
                "pseudo_class_distribution": cls_str,
                "conf_mean": conf_mean,
                "conf_min": conf_min,
                "conf_max": conf_max,
            }
        )

        total_added += n_sel
        logger.log(
            f"[PSEUDO][{fold_id}] Iter {it}: selected={n_sel} | class_dist={cls_str} | pool_restante={int(X_pool.shape[0])}"
        )

    final_model = build_lr(seed + 777)
    final_model.fit(X_train, y_train)

    return (
        final_model,
        history,
        pd.DataFrame(conf_rows),
        int(total_added),
        int(y_train.shape[0]),
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
    target_domain_label: str,
    cfg: Config,
    logger: Logger,
) -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame, Dict[str, object]
]:
    source_df = df.loc[source_idx].copy()
    adapt_df = df.loc[adapt_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Controlli anti-overlap (fondamentali).
    assert_disjoint(
        source_df["row_id"], test_df["row_id"], f"{fold_id}: source vs test"
    )
    assert_disjoint(
        adapt_df["row_id"], test_df["row_id"], f"{fold_id}: adaptation vs test"
    )

    logger.log("\n" + "=" * 100)
    logger.log(f"[FOLD] {fold_id} | target_domain={target_domain_label}")
    logger.log("=" * 100)
    logger.log(
        f"[SIZE][{fold_id}] source={len(source_df)} adaptation={len(adapt_df)} test={len(test_df)}"
    )
    logger.log(f"[DIST][{fold_id}] source: {class_dist_text(source_df['target'])}")
    logger.log(
        f"[DIST][{fold_id}] adaptation (diagnostico): {class_dist_text(adapt_df['target'])}"
    )
    logger.log(f"[DIST][{fold_id}] test: {class_dist_text(test_df['target'])}")

    # Per valutazione classificazione robusta chiediamo almeno 2 classi nel test.
    if test_df["target"].nunique() < 2:
        raise RuntimeError(
            f"[{fold_id}] test con una sola classe: fold non informativo."
        )

    if source_df["target"].nunique() < 2:
        raise RuntimeError(
            f"[{fold_id}] source con una sola classe: fold non addestrabile."
        )

    X_source = source_df[feature_cols]
    y_source = source_df["target"].to_numpy(dtype=int)

    X_adapt = adapt_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["target"].to_numpy(dtype=int)

    # PREPROCESSING FIT SOLO SU SOURCE.
    prep = fit_source_preprocessor(
        X_source=X_source,
        variance_threshold=cfg.variance_threshold,
        top_n=cfg.top_n_features,
        fold_id=fold_id,
        logger=logger,
    )

    logger.log(
        f"[CHECK][{fold_id}] preprocess fit SOLO su source ({len(source_df)} campioni)."
    )

    Xs = prep["X_source_final"]
    Xa = transform_with_source_preprocessor(X_adapt, prep)
    Xt = transform_with_source_preprocessor(X_test, prep)

    # ------------------------
    # Baseline source-only
    # ------------------------
    baseline = build_lr(cfg.random_state)
    baseline.fit(Xs, y_source)

    yp_b = baseline.predict(Xt)
    yprob_b = baseline.predict_proba(Xt)
    m_b = compute_metrics(y_test, yp_b, yprob_b, baseline.classes_)

    # ------------------------
    # Transductive pseudo-labeling
    # ------------------------
    logger.log(f"[CHECK][{fold_id}] y vere adaptation NON usate nel fit transductive.")

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
        raise RuntimeError(
            f"[LEAKAGE ERROR][{fold_id}] final_train_size={final_train_size} != source+pseudo={expected_final}"
        )

    logger.log(
        f"[CHECK][{fold_id}] train finale OK: source({len(y_source)}) + pseudo({pseudo_total})"
    )

    yp_t = trans_model.predict(Xt)
    yprob_t = trans_model.predict_proba(Xt)
    m_t = compute_metrics(y_test, yp_t, yprob_t, trans_model.classes_)

    # ------------------------
    # Sanity check con label random
    # ------------------------
    rng = np.random.default_rng(cfg.random_state)
    y_source_rand = rng.permutation(y_source)

    rnd_model = build_lr(cfg.random_state + 9999)
    rnd_model.fit(Xs, y_source_rand)
    yp_r = rnd_model.predict(Xt)

    rnd_acc = float(accuracy_score(y_test, yp_r))
    rnd_bal = float(balanced_accuracy_score(y_test, yp_r))

    random_warning = bool(rnd_bal >= 0.60)
    if random_warning:
        logger.log(
            f"[WARNING][{fold_id}] Random-label sanity check alto (balanced_accuracy={rnd_bal:.3f}). Possible leakage detected"
        )

    # Warning performance quasi perfette.
    high_perf_warning = bool(
        (pd.notna(m_b.get("accuracy")) and m_b["accuracy"] >= 0.98)
        or (pd.notna(m_t.get("accuracy")) and m_t["accuracy"] >= 0.98)
    )
    if high_perf_warning:
        logger.log(
            f"[WARNING][{fold_id}] Accuracy quasi perfetta. Possible leakage detected"
        )

    rows = [
        {
            "fold_id": fold_id,
            "target_domain": target_domain_label,
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
            "target_domain": target_domain_label,
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

    pseudo_rows = []
    for r in pseudo_hist:
        pseudo_rows.append(
            {"fold_id": fold_id, "target_domain": target_domain_label, **r}
        )

    if not conf_df.empty:
        conf_df["target_domain"] = target_domain_label

    diag = {
        "fold_id": fold_id,
        "target_domain": target_domain_label,
        "source_n": len(source_df),
        "adapt_n": len(adapt_df),
        "test_n": len(test_df),
        "features_before": prep["n_before"],
        "features_after": prep["n_after"],
        "train_test_overlap_ok": 1,
        "used_true_adapt_labels_in_train": 0,
        "preprocess_fit_source_only": 1,
        "pseudo_total_added": int(pseudo_total),
        "random_label_accuracy": rnd_acc,
        "random_label_balanced_accuracy": rnd_bal,
        "random_label_warning": int(random_warning),
        "high_accuracy_warning": int(high_perf_warning),
    }

    return rows, pseudo_rows, conf_df, diag


# ============================================================================
# Scenario A / B
# ============================================================================


def run_case_a_lodo(
    df: pd.DataFrame, feature_cols: List[str], cfg: Config, logger: Logger
):
    rows_all: List[Dict[str, object]] = []
    pseudo_all: List[Dict[str, object]] = []
    diag_all: List[Dict[str, object]] = []
    conf_frames: List[pd.DataFrame] = []
    skipped: List[Dict[str, object]] = []

    domains = sorted(df["domain"].dropna().astype(str).unique().tolist())

    fold_num = 0
    for d in domains:
        target_df = df.loc[df["domain"].astype(str) == d].copy()
        source_df = df.loc[df["domain"].astype(str) != d].copy()

        reason = None
        if len(target_df) < cfg.min_domain_samples:
            reason = f"domain_too_small_n<{cfg.min_domain_samples}"
        elif target_df["target"].nunique() < 2:
            reason = "single_class_target_domain"
        elif source_df["target"].nunique() < 2:
            reason = "single_class_source"

        if reason is not None:
            skipped.append(
                {"target_domain": d, "reason": reason, "target_n": len(target_df)}
            )
            logger.log(f"[SKIP][{d}] {reason} | n={len(target_df)}")
            continue

        strat = target_df["target"] if can_stratify(target_df["target"]) else None
        if strat is None:
            logger.log(
                f"[WARN][{d}] Stratify non applicabile, split target non stratificato."
            )

        try:
            adapt_df, test_df = train_test_split(
                target_df,
                test_size=cfg.target_test_size,
                random_state=cfg.random_state,
                stratify=strat,
            )
        except ValueError as exc:
            skipped.append(
                {
                    "target_domain": d,
                    "reason": f"target_split_failed:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{d}] split fallito: {exc}")
            continue

        if test_df["target"].nunique() < 2:
            skipped.append(
                {
                    "target_domain": d,
                    "reason": "single_class_target_test",
                    "target_n": len(target_df),
                }
            )
            logger.log(
                f"[SKIP][{d}] target_test una sola classe -> fold non informativo"
            )
            continue

        fold_num += 1
        fold_id = f"fold_{fold_num}_{safe_name(d)}"

        try:
            rows, pseudo_rows, conf_df, diag = run_fold(
                df=df,
                feature_cols=feature_cols,
                source_idx=source_df.index,
                adapt_idx=adapt_df.index,
                test_idx=test_df.index,
                fold_id=fold_id,
                target_domain_label=str(d),
                cfg=cfg,
                logger=logger,
            )
        except Exception as exc:
            skipped.append(
                {
                    "target_domain": d,
                    "reason": f"fold_error:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{d}] errore fold: {exc}")
            continue

        rows_all.extend(rows)
        pseudo_all.extend(pseudo_rows)
        diag_all.append(diag)
        if not conf_df.empty:
            conf_frames.append(conf_df)

    return {
        "fold_results": pd.DataFrame(rows_all),
        "pseudo_log": pd.DataFrame(pseudo_all),
        "confidence_log": pd.concat(conf_frames, ignore_index=True)
        if conf_frames
        else pd.DataFrame(),
        "diagnostics": pd.DataFrame(diag_all),
        "skipped": pd.DataFrame(skipped),
    }


def run_case_b_simulation(
    df: pd.DataFrame, feature_cols: List[str], cfg: Config, logger: Logger
):
    """
    Simulazione quando manca domain.

    Nota metodologica:
    - e meno fedele di un vero scenario multi-domain;
    - utile come fallback tecnico quando la struttura domain non e disponibile.
    """
    logger.log(
        "[INFO] CASE B (simulazione): nessuna domain reale. "
        "Creo source/adaptation/test con split stratificati quando possibile."
    )

    strat_main = df["target"] if can_stratify(df["target"]) else None
    if strat_main is None:
        logger.log("[WARN][SIM] Stratify non disponibile nello split principale.")

    train_pool, test_df = train_test_split(
        df,
        test_size=cfg.sim_test_size,
        random_state=cfg.random_state,
        stratify=strat_main,
    )

    strat_sub = train_pool["target"] if can_stratify(train_pool["target"]) else None
    if strat_sub is None:
        logger.log(
            "[WARN][SIM] Stratify non disponibile nello split adaptation/source."
        )

    source_df, adapt_df = train_test_split(
        train_pool,
        test_size=cfg.sim_adapt_size,
        random_state=cfg.random_state,
        stratify=strat_sub,
    )

    if test_df["target"].nunique() < 2:
        raise RuntimeError(
            "[SIM] target_test con una sola classe, simulazione non informativa."
        )

    rows, pseudo_rows, conf_df, diag = run_fold(
        df=df,
        feature_cols=feature_cols,
        source_idx=source_df.index,
        adapt_idx=adapt_df.index,
        test_idx=test_df.index,
        fold_id="fold_simulated_1",
        target_domain_label="simulated_no_domain",
        cfg=cfg,
        logger=logger,
    )

    return {
        "fold_results": pd.DataFrame(rows),
        "pseudo_log": pd.DataFrame(pseudo_rows),
        "confidence_log": conf_df if not conf_df.empty else pd.DataFrame(),
        "diagnostics": pd.DataFrame([diag]),
        "skipped": pd.DataFrame(),
    }


# ============================================================================
# Tabelle aggregate
# ============================================================================


def build_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method in ["baseline", "transductive"]:
        sub = fold_results[fold_results["method"] == method]
        if sub.empty:
            continue
        for m in METRICS:
            vals = pd.to_numeric(sub[m], errors="coerce")
            out.append(
                {
                    "method": method,
                    "metric": m,
                    "mean": float(vals.mean(skipna=True)),
                    "std": float(vals.std(skipna=True, ddof=1)),
                    "n_valid": int(vals.notna().sum()),
                }
            )
    return pd.DataFrame(out)


def build_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold_id, dom), grp in fold_results.groupby(
        ["fold_id", "target_domain"], dropna=False
    ):
        b = grp[grp["method"] == "baseline"]
        t = grp[grp["method"] == "transductive"]
        if b.empty or t.empty:
            continue
        b = b.iloc[0]
        t = t.iloc[0]
        for m in METRICS:
            bv = b[m]
            tv = t[m]
            dv = tv - bv if pd.notna(bv) and pd.notna(tv) else np.nan
            rows.append(
                {
                    "fold_id": fold_id,
                    "target_domain": dom,
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


def plot_metrics_per_domain(fold_results: pd.DataFrame, output_path: Path) -> None:
    if fold_results.empty:
        return

    domains = sorted(fold_results["target_domain"].astype(str).unique().tolist())
    x = np.arange(len(domains), dtype=float)
    width = 0.36

    fig_w = max(16, 8 + 0.45 * len(domains))
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        b_vals = []
        t_vals = []
        for d in domains:
            sb = fold_results[
                (fold_results["target_domain"].astype(str) == d)
                & (fold_results["method"] == "baseline")
            ]
            st = fold_results[
                (fold_results["target_domain"].astype(str) == d)
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
        ax.set_xticklabels(domains, rotation=80)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2)

    fig.suptitle("Baseline vs Transductive per target domain/fold")
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

    domains = sorted(comparison_df["target_domain"].astype(str).unique().tolist())
    x = np.arange(len(domains), dtype=float)

    fig_w = max(16, 8 + 0.45 * len(domains))
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        vals = []
        for d in domains:
            sub = comparison_df[
                (comparison_df["target_domain"].astype(str) == d)
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
        ax.set_xticklabels(domains, rotation=80)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Delta metriche (transductive - baseline)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pseudo_iterations(pseudo_df: pd.DataFrame, output_path: Path) -> None:
    if pseudo_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for fold_id, sub in pseudo_df.groupby("fold_id", dropna=False):
        sub = sub.sort_values("iteration")
        ax.plot(
            sub["iteration"],
            sub["n_selected"],
            marker="o",
            alpha=0.7,
            label=str(fold_id),
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("n_selected pseudo-labels")
    ax.set_title("Pseudo-label aggiunte per iterazione")
    ax.grid(alpha=0.3)
    if pseudo_df["fold_id"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confidence_distribution(conf_df: pd.DataFrame, output_path: Path) -> None:
    if conf_df.empty:
        return

    sel = conf_df.loc[conf_df["selected"] == 1, "confidence"].dropna()
    nsel = conf_df.loc[conf_df["selected"] == 0, "confidence"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(nsel, bins=40, alpha=0.55, density=True, label="not_selected")
    ax.hist(sel, bins=40, alpha=0.55, density=True, label="selected")
    ax.axvline(
        PSEUDO_CONF_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="threshold",
    )
    ax.set_xlabel("confidence")
    ax.set_ylabel("density")
    ax.set_title("Distribuzione confidence nel target_adaptation")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================================
# Reporting testuale
# ============================================================================


def write_interpretation_txt(
    output_path: Path,
    scenario: str,
    target_col: Optional[str],
    domain_col: Optional[str],
    id_cols: List[str],
    target_mapping: Dict[object, int],
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("TCGA-BRCA Baseline vs Transductive - Interpretazione")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Scenario: {scenario}")
    lines.append(f"Target col: {target_col}")
    lines.append(f"Domain col: {domain_col if domain_col else 'ASSENTE (simulazione)'}")
    lines.append(f"ID columns escluse: {id_cols if id_cols else 'nessuna'}")
    lines.append(f"Mapping target: {target_mapping}")
    lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valutabile. Vedi log e fold_saltati.csv")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    n_folds = fold_results["fold_id"].nunique()
    lines.append(f"Folds valutati: {n_folds}")
    lines.append(f"Folds saltati: {len(skipped_df)}")
    lines.append("")

    lines.append("Metriche medie:")
    for method in ["baseline", "transductive"]:
        sub = summary_df[summary_df["method"] == method]
        if sub.empty:
            continue
        lines.append(f"- {method}")
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['metric']}: mean={r['mean']:.4f}, std={r['std']:.4f}, n_valid={int(r['n_valid'])}"
            )
    lines.append("")

    lines.append("Delta per fold (transductive - baseline):")
    for metric in METRICS:
        sub = comparison_df[comparison_df["metric"] == metric]
        if sub.empty:
            continue
        mean_delta = float(
            pd.to_numeric(
                sub["delta_transductive_minus_baseline"], errors="coerce"
            ).mean(skipna=True)
        )
        lines.append(f"- {metric}: delta medio={mean_delta:+.4f}")
    lines.append("")

    # Warning sospetto leakage.
    leakage_flags = []
    if not diagnostics_df.empty:
        if diagnostics_df["high_accuracy_warning"].sum() > 0:
            leakage_flags.append("accuracy quasi perfetta in almeno un fold")
        if diagnostics_df["random_label_warning"].sum() > 0:
            leakage_flags.append("random-label sanity check insolitamente alto")

    if leakage_flags:
        lines.append("WARNING: Possible leakage detected")
        for f in leakage_flags:
            lines.append(f"- {f}")
    else:
        lines.append("Nessun warning forte di leakage dai controlli automatici.")

    lines.append("")
    lines.append("Nota metodologica anti-leakage:")
    lines.append(
        "- split train/adaptation/test eseguito prima di qualsiasi trasformazione"
    )
    lines.append("- imputer, scaler e feature selection fittati solo sui source")
    lines.append("- target_test mai visto in training")
    lines.append("- y vere di target_adaptation mai usate nel fit transductive")
    lines.append("")
    lines.append("Perche risultati perfetti sono sospetti:")
    lines.append(
        "- in dati genomici ad alta dimensionalita, leakage minimo puo gonfiare le metriche"
    )
    lines.append(
        "- anche split non informativi (test quasi mono-classe) possono dare accuracy elevata"
    )
    lines.append("- per questo i fold non informativi vengono saltati e tracciati")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCGA-BRCA baseline vs transductive learning - leakage-safe fixed"
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path(_configured_path("legacy_input_tcga_brca_preprocessed.csv")),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_report_tcga_transd_learning_fixed")),
    )
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    p.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
    p.add_argument("--pseudo-threshold", type=float, default=PSEUDO_CONF_THRESHOLD)
    p.add_argument("--pseudo-max-iters", type=int, default=PSEUDO_MAX_ITERS)
    return p.parse_args()


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

    cfg = Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        top_n_features=args.top_n_features,
        variance_threshold=args.variance_threshold,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
    )

    ensure_dir(cfg.output_dir)
    logger = Logger()

    logger.log("[START] TCGA-BRCA FIXED leakage-safe analysis")
    logger.log(f"[CONFIG] input={cfg.input_csv}")
    logger.log(f"[CONFIG] output={cfg.output_dir}")
    logger.log(f"[CONFIG] random_state={cfg.random_state}")

    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input file non trovato: {cfg.input_csv}")

    # 1) Load
    df_raw = pd.read_csv(cfg.input_csv)
    logger.log(f"[DATA] shape={df_raw.shape}")
    logger.log(f"[DATA] n_columns={len(df_raw.columns)}")
    logger.log(f"[DATA] columns prime 30: {df_raw.columns[:30].tolist()}")
    preview_cols = df_raw.columns[:20].tolist()
    logger.log(
        f"[DATA] head(3) prime 20 colonne:\n{df_raw[preview_cols].head(3).to_string(index=False)}"
    )

    target_col = detect_target_column(df_raw, logger)
    domain_col = detect_domain_column(df_raw, target_col, logger)

    # target mancante -> stop pulito con spiegazione.
    if target_col is None:
        msg = (
            "Target/label non trovata. "
            "Il transductive learning supervisionato non e applicabile senza una colonna target valida."
        )
        logger.log(f"[STOP] {msg}")
        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    id_cols = detect_id_columns(df_raw, target_col, domain_col, logger)

    # 2) Define target/domain/features
    df = df_raw.copy()
    df["target"], target_mapping = encode_target(df[target_col])

    n_missing_target = int(df["target"].isna().sum())
    if n_missing_target > 0:
        logger.log(f"[CLEAN] Righe con target mancante rimosse: {n_missing_target}")
        df = df.loc[df["target"].notna()].copy()

    if df["target"].nunique() < 2:
        msg = "Target con meno di 2 classi utili dopo cleaning. Classificazione non applicabile."
        logger.log(f"[STOP] {msg}")
        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    df["target"] = df["target"].astype(int)

    if domain_col is not None:
        df["domain"] = df[domain_col].astype("string")
        n_missing_domain = int(df["domain"].isna().sum())
        if n_missing_domain > 0:
            logger.log(f"[CLEAN] Righe con domain mancante rimosse: {n_missing_domain}")
            df = df.loc[df["domain"].notna()].copy()

    exclude = {target_col, "target"}
    if domain_col is not None:
        exclude.add(domain_col)
        exclude.add("domain")
    for c in id_cols:
        exclude.add(c)

    feature_cols = [c for c in df.columns if c not in exclude]
    if not feature_cols:
        raise RuntimeError("Nessuna feature rimasta dopo esclusioni target/domain/id.")

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    logger.log(
        f"[DATA] Final n={len(df)} | target_dist={class_dist_text(df['target'])}"
    )
    logger.log(f"[DATA] features candidate={len(feature_cols)}")

    # 3) Scenario
    use_case_a = domain_col is not None and df["domain"].nunique() >= 2
    if use_case_a:
        scenario = "CASE_A_LODO"
        outputs = run_case_a_lodo(df, feature_cols, cfg, logger)
    else:
        scenario = "CASE_B_SIMULATION"
        logger.log(
            "[INFO] Domain assente (o non valida): uso simulazione stratificata."
        )
        outputs = run_case_b_simulation(df, feature_cols, cfg, logger)

    fold_results = outputs["fold_results"]
    pseudo_log = outputs["pseudo_log"]
    conf_log = outputs["confidence_log"]
    diag_df = outputs["diagnostics"]
    skipped_df = outputs["skipped"]

    # 4) Save tables
    if fold_results.empty:
        logger.log("[STOP] Nessun fold valutabile.")
        skipped_df.to_csv(cfg.output_dir / "fold_saltati.csv", index=False)
        write_interpretation_txt(
            output_path=cfg.output_dir / "interpretation.txt",
            scenario=scenario,
            target_col=target_col,
            domain_col=domain_col,
            id_cols=id_cols,
            target_mapping=target_mapping,
            fold_results=fold_results,
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            diagnostics_df=diag_df,
            skipped_df=skipped_df,
        )
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    summary_df = build_summary(fold_results)
    comparison_df = build_comparison(fold_results)

    fold_results.to_csv(cfg.output_dir / "results_fold_by_fold.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "summary_metrics.csv", index=False)
    comparison_df.to_csv(
        cfg.output_dir / "comparison_baseline_vs_transductive.csv", index=False
    )
    pseudo_log.to_csv(cfg.output_dir / "pseudo_labeling_iterations.csv", index=False)
    conf_log.to_csv(cfg.output_dir / "confidence_log.csv", index=False)
    diag_df.to_csv(cfg.output_dir / "leakage_diagnostics.csv", index=False)
    skipped_df.to_csv(cfg.output_dir / "fold_saltati.csv", index=False)

    # 5) Plots
    plot_metrics_per_domain(
        fold_results, cfg.output_dir / "plot_metrics_per_domain.png"
    )
    plot_mean_metrics(summary_df, cfg.output_dir / "plot_mean_metrics.png")
    plot_delta_metrics(comparison_df, cfg.output_dir / "plot_delta_metrics.png")
    plot_pseudo_iterations(pseudo_log, cfg.output_dir / "plot_pseudo_per_iteration.png")
    plot_confidence_distribution(
        conf_log, cfg.output_dir / "plot_confidence_distribution.png"
    )

    # 6) Interpretation
    write_interpretation_txt(
        output_path=cfg.output_dir / "interpretation.txt",
        scenario=scenario,
        target_col=target_col,
        domain_col=domain_col,
        id_cols=id_cols,
        target_mapping=target_mapping,
        fold_results=fold_results,
        summary_df=summary_df,
        comparison_df=comparison_df,
        diagnostics_df=diag_df,
        skipped_df=skipped_df,
    )

    copy_script_to_output(cfg.output_dir, logger)
    logger.save(cfg.output_dir / "log.txt")

    # warning globale se accuracy sospetta o random-label test sospetto
    global_warning = False
    if not diag_df.empty:
        if (
            diag_df["high_accuracy_warning"].sum() > 0
            or diag_df["random_label_warning"].sum() > 0
        ):
            global_warning = True

    if global_warning:
        logger.log("[WARNING] Possible leakage detected")

    logger.log("[END] Pipeline completata.")
    logger.log(f"[SAVE] {cfg.output_dir / 'results_fold_by_fold.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'summary_metrics.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'comparison_baseline_vs_transductive.csv'}")


if __name__ == "__main__":
    main()
