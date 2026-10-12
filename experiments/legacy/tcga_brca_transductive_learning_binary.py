#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA-BRCA baseline vs transductive learning (binary tumor vs normal)
===========================================================================

Questo script implementa una pipeline completa, commentata e leakage-safe per:
1) estrarre il sample type code direttamente dal barcode TCGA;
2) costruire un target binario biologicamente piu interpretabile:
   - tumor  = 1 se sample type code == 01
   - normal = 0 se sample type code == 11
3) confrontare baseline source-only vs transductive learning (pseudo-labeling);
4) salvare risultati, plot e report testuale.

Nota metodologica importante:
- Il task qui e "tumor vs normal" in senso di tipo di campione (01 vs 11),
  non "malato vs sano" in senso clinico generale.
- Nei dati genomici ad alta dimensionalita il leakage e molto facile:
  per questo split, preprocessing e feature selection sono costruiti in modo
  rigorosamente separato.

Vincolo anti-leakage (critico):
- TUTTE le trasformazioni che apprendono dai dati (imputer, scaler,
  feature selection per varianza) sono fittate SOLO su source_labeled.
- target_adaptation e target_test sono trasformati con oggetti gia fittati.
- target_test non entra mai nel training.
- le label vere di target_adaptation non vengono mai usate nel fit.
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
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configurazione ambiente prima di importare matplotlib
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

# Split
TARGET_TEST_SIZE = 0.50
SIM_TEST_SIZE = 0.30
SIM_ADAPT_SIZE = 0.30

# Feature selection (moderata: 200 feature) per ridurre overfitting
TOP_N_FEATURES = 200

# Pseudo-labeling transductive
PSEUDO_THRESHOLD = 0.99
PSEUDO_MAX_ITERS = 3
PSEUDO_MAX_PER_ITER = 20
PSEUDO_MAX_POOL_FRACTION = 0.10

# Domain quality
MIN_DOMAIN_SAMPLES = 20
MIN_VALID_DOMAINS = 5
MIN_DOMAIN_CLASS_COUNT = 2

# Metriche principali
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]

# Sample type usati per il task binario
TUMOR_CODE = 1  # 01
NORMAL_CODE = 11  # 11

# Candidati colonne barcode
BARCODE_CANDIDATE_NAMES = [
    "sample_id",
    "barcode",
    "tcga_barcode",
    "aliquot_barcode",
    "patient_id",
    "id",
]

# Candidati colonne domain
DOMAIN_EXACT_NAMES = [
    "site",
    "domain",
    "batch",
    "source",
    "center",
    "centre",
    "tss_code",
    "cohort",
]
DOMAIN_PARTIAL_NAMES = [
    "site",
    "domain",
    "batch",
    "source",
    "center",
    "centre",
    "tss",
    "cohort",
]

# Colonne da trattare come ID (da escludere dalle feature)
ID_HINTS = [
    "sample_id",
    "patient_id",
    "barcode",
    "id",
    "uuid",
    "aliquot",
    "submitter",
    "case_id",
]

# Possibili target multiclass da escludere dalle feature
POSSIBLE_TARGET_COLS = [
    "target",
    "label",
    "class",
    "y",
    "sample_type",
    "sample_type_code",
    "outcome",
    "diagnosis",
    "phenotype",
    "subtype",
]


@dataclass
class Config:
    input_csv: Path
    output_dir: Path
    random_state: int = RANDOM_STATE
    target_test_size: float = TARGET_TEST_SIZE
    sim_test_size: float = SIM_TEST_SIZE
    sim_adapt_size: float = SIM_ADAPT_SIZE
    top_n_features: int = TOP_N_FEATURES
    pseudo_threshold: float = PSEUDO_THRESHOLD
    pseudo_max_iters: int = PSEUDO_MAX_ITERS
    pseudo_max_per_iter: int = PSEUDO_MAX_PER_ITER
    pseudo_max_pool_fraction: float = PSEUDO_MAX_POOL_FRACTION
    min_domain_samples: int = MIN_DOMAIN_SAMPLES
    min_valid_domains: int = MIN_VALID_DOMAINS
    min_domain_class_count: int = MIN_DOMAIN_CLASS_COUNT


class Logger:
    """Logger semplice: stampa a terminale e salva su file."""

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


def class_distribution_text(y: pd.Series) -> str:
    vc = y.value_counts(dropna=False).sort_index()
    return ", ".join([f"class_{k}={int(v)}" for k, v in vc.items()])


def assert_disjoint(ids_a: pd.Series, ids_b: pd.Series, where: str) -> None:
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] overlap in {where}: {len(overlap)} samples"
        )


def build_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=6000,
        solver="lbfgs",
        random_state=seed,
    )


# ============================================================================
# Barcode parsing e target binario
# ============================================================================


def parse_tcga_sample_type_code(barcode: object) -> Optional[int]:
    """
    Estrae il sample type code (2 cifre) dal barcode TCGA.

    Esempio tipico barcode: TCGA-ZZ-0000-01A
    - il 4o blocco contiene il sample type: 01A -> codice 01

    La funzione usa una strategia robusta:
    1) parsing diretto per blocchi separati da '-', '_', '.'
    2) fallback regex se il formato non e perfettamente standard
    """
    if barcode is None or (isinstance(barcode, float) and math.isnan(barcode)):
        return None

    text = str(barcode).strip()
    if text == "" or text.lower() == "nan":
        return None

    normalized = text.replace("_", "-").replace(".", "-")
    parts = normalized.split("-")

    # Caso standard: TCGA-XX-YYYY-01A
    if len(parts) >= 4 and parts[0].upper() == "TCGA":
        token = parts[3]
        m = re.match(r"(\d{2})", token)
        if m:
            return int(m.group(1))

    # Fallback regex su stringa completa
    m = re.search(
        r"TCGA[-_][A-Za-z0-9]+[-_][A-Za-z0-9]+[-_](\d{2})", text, flags=re.IGNORECASE
    )
    if m:
        return int(m.group(1))

    return None


def extract_sample_type_from_series(series: pd.Series) -> pd.Series:
    codes = series.apply(parse_tcga_sample_type_code)
    return pd.Series(codes, index=series.index, dtype="Int64")


def detect_barcode_column(
    df: pd.DataFrame, logger: Logger
) -> Tuple[Optional[str], Optional[pd.Series], pd.DataFrame]:
    """
    Identifica una colonna barcode plausibile e prova a parseare i sample type code.

    Strategia:
    - priorita ai nomi espliciti (sample_id, barcode, ...)
    - scoring basato su nome + percentuale di parsing valido
    - richiede almeno 50% di parsing valido per essere considerata affidabile
    """
    candidates: List[Dict[str, object]] = []

    for col in df.columns:
        low = col.lower()

        # Consideriamo soprattutto colonne testuali; se un nome e esplicito le proviamo comunque.
        likely_name = low in BARCODE_CANDIDATE_NAMES
        text_like = pd.api.types.is_object_dtype(
            df[col]
        ) or pd.api.types.is_string_dtype(df[col])
        if not likely_name and not text_like:
            continue

        codes = extract_sample_type_from_series(df[col].astype("string"))
        valid = int(codes.notna().sum())
        ratio = valid / max(1, len(df))

        # Bonus per nomi altamente plausibili
        name_bonus = 0.0
        if low in BARCODE_CANDIDATE_NAMES:
            # priorita decrescente nell'ordine della lista
            name_bonus = 5.0 - float(BARCODE_CANDIDATE_NAMES.index(low)) * 0.25
        elif any(k in low for k in ["barcode", "sample", "aliquot", "tcga"]):
            name_bonus = 1.0

        score = ratio * 10.0 + name_bonus
        if ratio > 0.0:
            candidates.append(
                {
                    "column": col,
                    "valid_ratio": ratio,
                    "n_valid": valid,
                    "score": score,
                    "codes": codes,
                }
            )

    if not candidates:
        logger.log("[DETECT] Nessuna colonna barcode candidata trovata.")
        return None, None, pd.DataFrame()

    cand_df = pd.DataFrame(
        [{k: v for k, v in c.items() if k != "codes"} for c in candidates]
    )
    cand_df = cand_df.sort_values(
        ["score", "valid_ratio", "n_valid"], ascending=False
    ).reset_index(drop=True)

    logger.log("[DETECT] Top colonne barcode candidate:")
    logger.log(cand_df.head(10).to_string(index=False))

    best_col = str(cand_df.iloc[0]["column"])
    best_ratio = float(cand_df.iloc[0]["valid_ratio"])

    # Guardrail: se il parsing valido e troppo basso, non possiamo usare il barcode in modo affidabile.
    if best_ratio < 0.50:
        logger.log(
            f"[DETECT] Miglior colonna barcode ({best_col}) con ratio={best_ratio:.3f} < 0.50: non affidabile."
        )
        return None, None, cand_df

    best_codes = next(c["codes"] for c in candidates if c["column"] == best_col)
    logger.log(
        f"[DETECT] Colonna barcode selezionata: {best_col} (valid_ratio={best_ratio:.3f})"
    )

    return best_col, best_codes, cand_df


# ============================================================================
# Domain detection e qualita domini
# ============================================================================


def detect_domain_column(
    df: pd.DataFrame, excluded_cols: List[str], logger: Logger
) -> Optional[str]:
    """
    Cerca una colonna domain/site/batch plausibile.

    Nota:
    - La sola presenza della colonna non basta.
    - Dopo il rilevamento, la qualita del dominio viene verificata separatamente.
    """
    excluded = set(excluded_cols)

    # 1) Match esatto
    for name in DOMAIN_EXACT_NAMES:
        for col in df.columns:
            if col in excluded:
                continue
            if col.lower() == name:
                logger.log(f"[DETECT] Domain candidata (exact): {col}")
                return col

    # 2) Match parziale
    partial_candidates: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        low = col.lower()
        if any(k in low for k in DOMAIN_PARTIAL_NAMES):
            partial_candidates.append(col)

    if partial_candidates:
        # preferiamo cardinalita non estrema
        scored = []
        for col in partial_candidates:
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / max(1, len(df))
            score = 0.0
            if nunique >= 2:
                score += 2.0
            if ratio < 0.50:
                score += 1.0
            if ratio > 0.90:
                score -= 5.0
            scored.append((score, col))
        scored.sort(reverse=True)
        best = scored[0][1]
        logger.log(f"[DETECT] Domain candidata (partial): {best}")
        return best

    logger.log("[DETECT] Nessuna colonna domain plausibile trovata.")
    return None


def evaluate_domain_quality(
    df: pd.DataFrame,
    domain_col: str,
    min_samples: int,
    min_class_count: int,
    logger: Logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Valuta la qualita dei domini per un setup Leave-One-Domain-Out realistico.

    Un dominio e considerato valido se:
    - ha almeno min_samples campioni
    - contiene entrambe le classi (tumor e normal)
    - ogni classe ha almeno min_class_count campioni

    Questo evita fold inutili (domini minuscoli o mono-classe).
    """
    tmp = df.copy()
    tmp["domain"] = tmp[domain_col].astype("string")

    rows = []
    for dom, sub in tmp.groupby("domain", dropna=True):
        n_total = int(len(sub))
        n_tumor = int((sub["target_binary"] == 1).sum())
        n_normal = int((sub["target_binary"] == 0).sum())
        n_classes = int(sub["target_binary"].nunique())
        min_class = int(min(n_tumor, n_normal))

        valid = (
            n_total >= min_samples and n_classes == 2 and min_class >= min_class_count
        )

        rows.append(
            {
                "domain": str(dom),
                "n_total": n_total,
                "n_tumor": n_tumor,
                "n_normal": n_normal,
                "n_classes": n_classes,
                "min_class_count": min_class,
                "valid_for_lodo": int(valid),
            }
        )

    domain_stats = (
        pd.DataFrame(rows)
        .sort_values("n_total", ascending=False)
        .reset_index(drop=True)
    )
    valid_domains = (
        domain_stats.loc[domain_stats["valid_for_lodo"] == 1, "domain"]
        .astype(str)
        .tolist()
    )

    logger.log(
        f"[DOMAIN] Domain totali={len(domain_stats)} | validi per LODO={len(valid_domains)} "
        f"(min_samples={min_samples}, min_class_count={min_class_count})"
    )

    return domain_stats, valid_domains


# ============================================================================
# Feature engineering leakage-safe
# ============================================================================


def detect_id_columns(
    df: pd.DataFrame, protected_cols: List[str], logger: Logger
) -> List[str]:
    """Rileva colonne identificative da escludere dalle feature."""
    protected = set(protected_cols)
    id_cols: List[str] = []

    for col in df.columns:
        if col in protected:
            continue
        low = col.lower()

        if any(h in low for h in ID_HINTS):
            id_cols.append(col)
            continue

        # Stringhe quasi uniche: tipico comportamento da identificatore
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / max(1, len(df))
            if ratio >= 0.90:
                id_cols.append(col)

    id_cols = sorted(set(id_cols))
    logger.log(f"[DETECT] Colonne ID escluse: {id_cols if id_cols else 'nessuna'}")
    return id_cols


def select_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    domain_col: Optional[str],
    barcode_col: str,
    sample_type_col: str,
    id_cols: List[str],
    logger: Logger,
) -> List[str]:
    """
    Costruisce la lista finale di feature:
    - esclude target, barcode, domain, id, target multiclass potenziali
    - mantiene solo colonne numeriche
    """
    exclude = {target_col, sample_type_col, barcode_col}

    if domain_col is not None:
        exclude.add(domain_col)
        exclude.add("domain")

    for c in id_cols:
        exclude.add(c)

    for c in POSSIBLE_TARGET_COLS:
        if c in df.columns:
            exclude.add(c)

    candidates = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]

    logger.log(
        f"[FEATURE] candidate_columns={len(candidates)} | numeric_features={len(numeric_cols)}"
    )

    if len(numeric_cols) == 0:
        raise RuntimeError("Nessuna feature numerica disponibile dopo le esclusioni.")

    return numeric_cols


def fit_source_preprocessor(
    X_source: pd.DataFrame,
    top_n_features: int,
    fold_id: str,
    logger: Logger,
) -> Dict[str, object]:
    """
    Fit del preprocessing SOLO sui source_labeled (anti-leakage).

    Passi:
    1) conversione numerica e rimozione colonne all-NaN nei source
    2) imputazione mediana fit su source
    3) feature selection top-N per varianza calcolata SOLO su source
    4) standardizzazione fit su source selezionato

    Errore da evitare:
    - usare tutto il dataset per scegliere le feature (leakage classico in omics)
    """
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    # Tenere solo colonne con almeno un valore osservato nei source
    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    if not usable_cols:
        raise RuntimeError(
            f"[{fold_id}] Nessuna colonna numerica utilizzabile nei source."
        )

    X_num = X_num[usable_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    # Selezione top-N per varianza sui source
    variances = np.var(X_imp, axis=0)
    n_keep = int(min(max(1, top_n_features), X_imp.shape[1]))
    top_idx = np.argsort(variances)[::-1][:n_keep]
    top_idx = np.sort(top_idx)

    selected_cols = [usable_cols[i] for i in top_idx]
    X_sel = X_imp[:, top_idx]

    scaler = StandardScaler()
    X_source_final = scaler.fit_transform(X_sel)

    logger.log(
        f"[PREP][{fold_id}] features_before={X_source.shape[1]} | usable_after_numeric={len(usable_cols)} "
        f"| after_topN={len(selected_cols)}"
    )

    return {
        "usable_cols": usable_cols,
        "imputer": imputer,
        "top_idx": top_idx,
        "scaler": scaler,
        "selected_cols": selected_cols,
        "X_source_final": X_source_final,
        "n_before": int(X_source.shape[1]),
        "n_after": int(len(selected_cols)),
    }


def transform_with_source_preprocessor(
    X_df: pd.DataFrame, prep: Dict[str, object]
) -> np.ndarray:
    usable_cols = prep["usable_cols"]
    imputer = prep["imputer"]
    top_idx = prep["top_idx"]
    scaler = prep["scaler"]

    X_num = X_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    X_imp = imputer.transform(X_num)
    X_sel = X_imp[:, top_idx]
    X_out = scaler.transform(X_sel)
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

    # ROC-AUC richiede entrambe le classi nel test
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = np.nan
        return out

    try:
        cls_to_idx = {c: i for i, c in enumerate(model_classes)}
        if 1 not in cls_to_idx:
            out["roc_auc"] = np.nan
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, cls_to_idx[1]]))
    except Exception:
        out["roc_auc"] = np.nan

    return out


# ============================================================================
# Transductive pseudo-labeling
# ============================================================================


def run_pseudo_labeling(
    X_source: np.ndarray,
    y_source: np.ndarray,
    source_row_ids: np.ndarray,
    X_adapt_unlabeled: np.ndarray,
    adapt_row_ids: np.ndarray,
    test_row_ids: np.ndarray,
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

    Input volutamente leakage-safe:
    - riceve X_adapt senza label vere
    - non riceve y_target_adaptation
    - target_test non viene passato come feature
    """
    X_train = np.asarray(X_source)
    y_train = np.asarray(y_source, dtype=int)
    train_ids = np.asarray(source_row_ids, dtype=int)

    X_pool = np.asarray(X_adapt_unlabeled)
    pool_ids = np.asarray(adapt_row_ids, dtype=int)

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
            logger.log(f"[PSEUDO][{fold_id}] Iter {it}: pool vuoto, stop.")
            break

        proba = model.predict_proba(X_pool)
        pred = model.predict(X_pool)
        conf = np.max(proba, axis=1)

        high_conf_idx = np.where(conf >= threshold)[0]
        n_candidates = int(len(high_conf_idx))

        for c, sel in zip(conf.tolist(), (conf >= threshold).tolist()):
            confidence_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": it,
                    "confidence": float(c),
                    "selected_high_conf": int(bool(sel)),
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
                f"[PSEUDO][{fold_id}] Iter {it}: nessun campione con conf >= {threshold:.2f}, stop."
            )
            break

        # Cap per iterazione: massimo 20 o 10% del pool (quello piu stringente)
        cap_by_fraction = max(1, int(math.ceil(max_pool_fraction * pool_n)))
        cap = min(max_per_iter, cap_by_fraction, n_candidates)

        # Seleziona i piu confidenti tra quelli sopra soglia
        sorted_candidates = high_conf_idx[np.argsort(conf[high_conf_idx])[::-1]]
        chosen_idx = np.sort(sorted_candidates[:cap])
        n_added = int(len(chosen_idx))

        X_sel = X_pool[chosen_idx]
        y_sel = pred[chosen_idx].astype(int)
        ids_sel = pool_ids[chosen_idx]

        # Guardrail anti-leakage: pseudo-label non deve mai includere sample di test
        if set(ids_sel.tolist()).intersection(set(test_row_ids.tolist())):
            raise RuntimeError(
                f"[LEAKAGE ERROR][{fold_id}] target_test trovato nei pseudo-label"
            )

        # Aggiorna train con pseudo-label
        X_train = np.vstack([X_train, X_sel])
        y_train = np.concatenate([y_train, y_sel], axis=0)
        train_ids = np.concatenate([train_ids, ids_sel], axis=0)

        # Rimuove i selezionati dal pool unlabeled
        keep_mask = np.ones(pool_n, dtype=bool)
        keep_mask[chosen_idx] = False
        X_pool = X_pool[keep_mask]
        pool_ids = pool_ids[keep_mask]

        cls_counts = pd.Series(y_sel).value_counts().sort_index().to_dict()
        cls_str = "{" + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()]) + "}"

        history.append(
            {
                "fold_id": fold_id,
                "iteration": it,
                "n_pool_before": pool_n,
                "n_candidates_high_conf": n_candidates,
                "n_added": n_added,
                "pseudo_class_distribution": cls_str,
                "conf_mean": conf_mean,
                "conf_min": conf_min,
                "conf_max": conf_max,
            }
        )

        total_added += n_added
        logger.log(
            f"[PSEUDO][{fold_id}] Iter {it}: high_conf={n_candidates}, added={n_added}, "
            f"class_dist={cls_str}, pool_restante={int(X_pool.shape[0])}"
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

    # Controlli anti-leakage sugli indici
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
    logger.log(f"[FOLD] {fold_id} | target_domain={target_domain_label}")
    logger.log("=" * 100)
    logger.log(
        f"[SIZE][{fold_id}] source={len(source_df)} adaptation={len(adapt_df)} test={len(test_df)}"
    )
    logger.log(
        f"[DIST][{fold_id}] source: {class_distribution_text(source_df['target_binary'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] adaptation (solo diagnostica): {class_distribution_text(adapt_df['target_binary'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] test: {class_distribution_text(test_df['target_binary'])}"
    )

    if source_df["target_binary"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] source mono-classe: fold non addestrabile")
    if test_df["target_binary"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] test mono-classe: fold non valutabile")

    X_source = source_df[feature_cols]
    y_source = source_df["target_binary"].to_numpy(dtype=int)

    X_adapt = adapt_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["target_binary"].to_numpy(dtype=int)

    # PREPROCESSING fit solo sui source
    prep = fit_source_preprocessor(
        X_source=X_source,
        top_n_features=cfg.top_n_features,
        fold_id=fold_id,
        logger=logger,
    )

    logger.log(f"[CHECK][{fold_id}] preprocess fittato SOLO su source_labeled")

    Xs = prep["X_source_final"]
    Xa = transform_with_source_preprocessor(X_adapt, prep)
    Xt = transform_with_source_preprocessor(X_test, prep)

    # Baseline source-only
    baseline = build_lr(cfg.random_state)
    baseline.fit(Xs, y_source)

    ypred_b = baseline.predict(Xt)
    yprob_b = baseline.predict_proba(Xt)
    m_baseline = compute_binary_metrics(y_test, ypred_b, yprob_b, baseline.classes_)

    # Transductive pseudo-labeling (senza y vere adaptation)
    logger.log(
        f"[CHECK][{fold_id}] y vere di target_adaptation NON usate nel fit transductive"
    )

    trans_model, pseudo_hist, conf_df, pseudo_total, final_train_size = (
        run_pseudo_labeling(
            X_source=Xs,
            y_source=y_source,
            source_row_ids=source_df["row_id"].to_numpy(dtype=int),
            X_adapt_unlabeled=Xa,
            adapt_row_ids=adapt_df["row_id"].to_numpy(dtype=int),
            test_row_ids=test_df["row_id"].to_numpy(dtype=int),
            threshold=cfg.pseudo_threshold,
            max_iters=cfg.pseudo_max_iters,
            max_per_iter=cfg.pseudo_max_per_iter,
            max_pool_fraction=cfg.pseudo_max_pool_fraction,
            seed=cfg.random_state,
            fold_id=fold_id,
            logger=logger,
        )
    )

    expected_final_train = int(len(y_source) + pseudo_total)
    if final_train_size != expected_final_train:
        raise RuntimeError(
            f"[LEAKAGE ERROR][{fold_id}] train_size finale incoerente: {final_train_size} != {expected_final_train}"
        )

    ypred_t = trans_model.predict(Xt)
    yprob_t = trans_model.predict_proba(Xt)
    m_trans = compute_binary_metrics(y_test, ypred_t, yprob_t, trans_model.classes_)

    fold_rows = [
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
            **m_baseline,
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
            **m_trans,
        },
    ]

    pseudo_rows = []
    for r in pseudo_hist:
        pseudo_rows.append(
            {"fold_id": fold_id, "target_domain": target_domain_label, **r}
        )

    if not conf_df.empty:
        conf_df["target_domain"] = target_domain_label

    diag_row = {
        "fold_id": fold_id,
        "target_domain": target_domain_label,
        "source_n": len(source_df),
        "adapt_n": len(adapt_df),
        "test_n": len(test_df),
        "features_before": prep["n_before"],
        "features_after": prep["n_after"],
        "train_test_disjoint_ok": 1,
        "adapt_test_disjoint_ok": 1,
        "used_true_adapt_labels_in_train": 0,
        "preprocess_fit_source_only": 1,
        "pseudo_total_added": int(pseudo_total),
        "baseline_accuracy": m_baseline.get("accuracy", np.nan),
        "baseline_roc_auc": m_baseline.get("roc_auc", np.nan),
    }

    return fold_rows, pseudo_rows, conf_df, diag_row


# ============================================================================
# Scenario A: LODO
# ============================================================================


def run_case_a_lodo(
    df: pd.DataFrame,
    feature_cols: List[str],
    valid_domains: List[str],
    cfg: Config,
    logger: Logger,
) -> Dict[str, pd.DataFrame]:
    rows_all: List[Dict[str, object]] = []
    pseudo_all: List[Dict[str, object]] = []
    diag_all: List[Dict[str, object]] = []
    conf_frames: List[pd.DataFrame] = []
    skipped: List[Dict[str, object]] = []

    fold_num = 0

    for dom in sorted(valid_domains):
        target_df = df.loc[df["domain"].astype(str) == str(dom)].copy()
        source_df = df.loc[df["domain"].astype(str) != str(dom)].copy()

        # Split target -> adaptation + test
        strat = (
            target_df["target_binary"]
            if can_stratify(target_df["target_binary"])
            else None
        )
        if strat is None:
            logger.log(f"[WARN][{dom}] Stratify non applicabile per target split")

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
                    "target_domain": dom,
                    "reason": f"split_failed:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{dom}] split target fallito: {exc}")
            continue

        if source_df["target_binary"].nunique() < 2:
            skipped.append(
                {
                    "target_domain": dom,
                    "reason": "source_single_class",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{dom}] source mono-classe")
            continue

        if test_df["target_binary"].nunique() < 2:
            skipped.append(
                {
                    "target_domain": dom,
                    "reason": "test_single_class",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{dom}] target_test mono-classe")
            continue

        fold_num += 1
        fold_id = f"fold_{fold_num}_{safe_name(dom)}"

        try:
            rows, pseudo_rows, conf_df, diag_row = run_fold(
                df=df,
                feature_cols=feature_cols,
                source_idx=source_df.index,
                adapt_idx=adapt_df.index,
                test_idx=test_df.index,
                fold_id=fold_id,
                target_domain_label=str(dom),
                cfg=cfg,
                logger=logger,
            )
        except Exception as exc:
            skipped.append(
                {
                    "target_domain": dom,
                    "reason": f"fold_error:{exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{dom}] errore fold: {exc}")
            continue

        rows_all.extend(rows)
        pseudo_all.extend(pseudo_rows)
        diag_all.append(diag_row)
        if not conf_df.empty:
            conf_frames.append(conf_df)

    return {
        "fold_results": pd.DataFrame(rows_all),
        "pseudo_log": pd.DataFrame(pseudo_all),
        "confidence_log": pd.concat(conf_frames, ignore_index=True)
        if conf_frames
        else pd.DataFrame(),
        "diagnostics": pd.DataFrame(diag_all),
        "fold_skipped": pd.DataFrame(skipped),
    }


# ============================================================================
# Scenario B: simulazione (no domain robusto)
# ============================================================================


def run_case_b_simulation(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
    logger: Logger,
) -> Dict[str, pd.DataFrame]:
    """
    Simulazione quando non esiste una struttura domain robusta.

    Nota metodologica:
    - questo non e un vero scenario multi-domain;
    - e un fallback utile ma meno forte del caso LODO reale.
    """
    logger.log(
        "[INFO] CASE B (simulazione): domain assente/non adeguata. "
        "Uso split stratificato train/test + adaptation simulato."
    )

    strat_main = df["target_binary"] if can_stratify(df["target_binary"]) else None
    if strat_main is None:
        logger.log("[WARN][SIM] Stratify non disponibile nello split principale")

    train_pool, test_df = train_test_split(
        df,
        test_size=cfg.sim_test_size,
        random_state=cfg.random_state,
        stratify=strat_main,
    )

    strat_sub = (
        train_pool["target_binary"]
        if can_stratify(train_pool["target_binary"])
        else None
    )
    if strat_sub is None:
        logger.log("[WARN][SIM] Stratify non disponibile nello split source/adaptation")

    source_df, adapt_df = train_test_split(
        train_pool,
        test_size=cfg.sim_adapt_size,
        random_state=cfg.random_state,
        stratify=strat_sub,
    )

    rows, pseudo_rows, conf_df, diag_row = run_fold(
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
        "diagnostics": pd.DataFrame([diag_row]),
        "fold_skipped": pd.DataFrame(),
    }


# ============================================================================
# Aggregazione risultati
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
    grouped = fold_results.groupby(["fold_id", "target_domain"], dropna=False)
    for (fold_id, target_domain), grp in grouped:
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
                    "target_domain": target_domain,
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
            sub_b = fold_results[
                (fold_results["target_domain"].astype(str) == d)
                & (fold_results["method"] == "baseline")
            ]
            sub_t = fold_results[
                (fold_results["target_domain"].astype(str) == d)
                & (fold_results["method"] == "transductive")
            ]

            b_vals.append(
                float(pd.to_numeric(sub_b[metric], errors="coerce").mean())
                if not sub_b.empty
                else np.nan
            )
            t_vals.append(
                float(pd.to_numeric(sub_t[metric], errors="coerce").mean())
                if not sub_t.empty
                else np.nan
            )

        ax.bar(x - width / 2, b_vals, width=width, label="baseline")
        ax.bar(x + width / 2, t_vals, width=width, label="transductive")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=80)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("Baseline vs Transductive per dominio/fold (task binario 01 vs 11)")
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
    ax.hist(selected, bins=40, alpha=0.55, density=True, label="selected")
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
            sub["iteration"], sub["n_added"], marker="o", alpha=0.7, label=str(fold_id)
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
# Reporting testuale
# ============================================================================


def detect_too_easy_task_warning(fold_results: pd.DataFrame) -> Tuple[bool, str]:
    """
    Warning automatico se il task appare troppo facile in quasi tutti i fold.
    """
    if fold_results.empty:
        return False, "nessun fold disponibile"

    base = fold_results.loc[fold_results["method"] == "baseline"].copy()
    if base.empty:
        return False, "nessuna riga baseline"

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
    scenario: str,
    barcode_col: Optional[str],
    domain_col: Optional[str],
    n_total_raw: int,
    n_after_filter: int,
    n_excluded_non_01_11: int,
    sample_type_dist: pd.Series,
    target_dist: pd.Series,
    feature_n: int,
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    task_warning: bool,
    task_warning_msg: str,
) -> None:
    lines: List[str] = []

    lines.append("TCGA-BRCA Binary (01 vs 11) - Baseline vs Transductive")
    lines.append("=" * 76)
    lines.append("")
    lines.append("Impostazione biologica del target:")
    lines.append("- sample type code estratto dal barcode TCGA")
    lines.append("- target_binary = 1 (tumor) per codice 01")
    lines.append("- target_binary = 0 (normal) per codice 11")
    lines.append("- campioni con codici diversi da 01/11 esclusi")
    lines.append("")

    lines.append(f"Scenario sperimentale: {scenario}")
    lines.append(f"Barcode column usata: {barcode_col}")
    lines.append(
        f"Domain column usata: {domain_col if domain_col else 'ASSENTE/NO_DOMAIN_VALIDO -> simulazione'}"
    )
    lines.append("")

    lines.append(f"Campioni raw: {n_total_raw}")
    lines.append(f"Campioni dopo filtro 01/11: {n_after_filter}")
    lines.append(
        f"Campioni esclusi (codice != 01/11 o non parseabile): {n_excluded_non_01_11}"
    )
    lines.append("Distribuzione sample_type_code_extracted:")
    lines.append(sample_type_dist.to_string())
    lines.append("")
    lines.append("Distribuzione target_binary:")
    lines.append(target_dist.to_string())
    lines.append("")
    lines.append(f"Numero feature finali: {feature_n}")
    lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valutabile. Vedere log e CSV diagnostici.")
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
            "WARNING: task potenzialmente troppo facile per valutare bene la transportability"
        )
        lines.append(f"Dettaglio: {task_warning_msg}")
        lines.append(
            "Interpretazione: se baseline e gia quasi perfetta in quasi tutti i fold, "
            "il vantaggio del transductive puo risultare poco informativo."
        )
        lines.append("")

    lines.append("Nota anti-leakage:")
    lines.append("- split source/adaptation/test fatto prima di ogni trasformazione")
    lines.append("- imputer/scaler/feature selection fittati solo su source_labeled")
    lines.append("- target_adaptation usato senza label vere")
    lines.append("- target_test usato solo alla fine per valutazione")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCGA-BRCA binary transductive learning (target da barcode, leakage-safe)"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(_configured_path("legacy_input_tcga_brca_preprocessed.csv")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_report_tcga_transd_learning_binary")),
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--pseudo-threshold", type=float, default=PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-max-iters", type=int, default=PSEUDO_MAX_ITERS)
    parser.add_argument("--pseudo-max-per-iter", type=int, default=PSEUDO_MAX_PER_ITER)
    parser.add_argument(
        "--pseudo-max-pool-fraction", type=float, default=PSEUDO_MAX_POOL_FRACTION
    )
    parser.add_argument("--min-domain-samples", type=int, default=MIN_DOMAIN_SAMPLES)
    parser.add_argument("--min-valid-domains", type=int, default=MIN_VALID_DOMAINS)
    parser.add_argument(
        "--min-domain-class-count", type=int, default=MIN_DOMAIN_CLASS_COUNT
    )
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

    cfg = Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        top_n_features=args.top_n_features,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
        pseudo_max_per_iter=args.pseudo_max_per_iter,
        pseudo_max_pool_fraction=args.pseudo_max_pool_fraction,
        min_domain_samples=args.min_domain_samples,
        min_valid_domains=args.min_valid_domains,
        min_domain_class_count=args.min_domain_class_count,
    )

    ensure_dir(cfg.output_dir)
    logger = Logger()

    logger.log("[START] TCGA-BRCA binary transductive analysis")
    logger.log(f"[CONFIG] input={cfg.input_csv}")
    logger.log(f"[CONFIG] output={cfg.output_dir}")
    logger.log(f"[CONFIG] random_state={cfg.random_state}")

    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input file non trovato: {cfg.input_csv}")

    # ---------------------------------------------------------------------
    # 1) Load dataset
    # ---------------------------------------------------------------------
    df_raw = pd.read_csv(cfg.input_csv)
    n_total_raw = len(df_raw)

    logger.log(f"[DATA] shape={df_raw.shape}")
    logger.log(f"[DATA] n_columns={len(df_raw.columns)}")
    logger.log(f"[DATA] columns prime 30: {df_raw.columns[:30].tolist()}")

    preview_cols = df_raw.columns[:20].tolist()
    logger.log(
        f"[DATA] head(3) prime 20 colonne:\n{df_raw[preview_cols].head(3).to_string(index=False)}"
    )

    # ---------------------------------------------------------------------
    # 2) Detect barcode e estrai sample type code
    # ---------------------------------------------------------------------
    barcode_col, extracted_codes, barcode_candidates_df = detect_barcode_column(
        df_raw, logger
    )

    if barcode_col is None or extracted_codes is None:
        msg = (
            "Impossibile identificare una colonna barcode affidabile. "
            "Senza barcode non e possibile estrarre il sample type code e costruire il target binario."
        )
        logger.log(f"[STOP] {msg}")

        if not barcode_candidates_df.empty:
            barcode_candidates_df.to_csv(
                cfg.output_dir / "barcode_candidates.csv", index=False
            )

        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    df = df_raw.copy()
    df["sample_type_code_extracted"] = extracted_codes

    sample_type_dist = (
        df["sample_type_code_extracted"].value_counts(dropna=False).sort_index()
    )
    logger.log("[DATA] Distribuzione sample_type_code_extracted (incl. NaN):")
    logger.log(sample_type_dist.to_string())

    # ---------------------------------------------------------------------
    # 3) Crea target binario: usa solo 01 (tumor) e 11 (normal)
    # ---------------------------------------------------------------------
    keep_mask = df["sample_type_code_extracted"].isin([TUMOR_CODE, NORMAL_CODE])
    n_after_filter = int(keep_mask.sum())
    n_excluded_non_01_11 = int(len(df) - n_after_filter)

    logger.log(
        f"[DATA] Campioni totali={len(df)} | tenuti (01/11)={n_after_filter} | esclusi={n_excluded_non_01_11}"
    )

    df = df.loc[keep_mask].copy()
    if df.empty:
        msg = "Nessun campione con sample type code 01/11: analisi binaria non eseguibile."
        logger.log(f"[STOP] {msg}")
        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # target_binary: tumor=1 (01), normal=0 (11)
    df["target_binary"] = (df["sample_type_code_extracted"] == TUMOR_CODE).astype(int)
    target_dist = df["target_binary"].value_counts(dropna=False).sort_index()

    logger.log("[DATA] Distribuzione target_binary (0=normal, 1=tumor):")
    logger.log(target_dist.to_string())

    if df["target_binary"].nunique() < 2:
        msg = "Target binario mono-classe dopo il filtro 01/11: classificazione non applicabile."
        logger.log(f"[STOP] {msg}")
        (cfg.output_dir / "interpretation.txt").write_text(msg + "\n", encoding="utf-8")
        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    # ---------------------------------------------------------------------
    # 4) Domain detection e scelta scenario
    # ---------------------------------------------------------------------
    protected_for_domain = [barcode_col, "sample_type_code_extracted", "target_binary"]
    domain_col = detect_domain_column(df, protected_for_domain, logger)

    scenario = "CASE_B_SIMULATION"
    valid_domains: List[str] = []

    if domain_col is not None:
        domain_stats_df, valid_domains = evaluate_domain_quality(
            df=df,
            domain_col=domain_col,
            min_samples=cfg.min_domain_samples,
            min_class_count=cfg.min_domain_class_count,
            logger=logger,
        )
        domain_stats_df.to_csv(cfg.output_dir / "domain_quality.csv", index=False)

        if len(valid_domains) >= cfg.min_valid_domains:
            scenario = "CASE_A_LODO"
            logger.log(
                f"[SCENARIO] Uso LODO con {len(valid_domains)} domini validi: {valid_domains}"
            )

            # Manteniamo solo domini robusti per evitare fold non informativi
            before = len(df)
            df = df.loc[df[domain_col].astype("string").isin(valid_domains)].copy()
            after = len(df)
            logger.log(
                f"[DOMAIN] Campioni dopo filtro domini validi: {after} (rimossi {before - after})"
            )

            # Colonna standardizzata domain
            df["domain"] = df[domain_col].astype("string")
        else:
            logger.log(
                f"[SCENARIO] Domain trovata ({domain_col}) ma non adeguata per LODO: "
                f"domini validi={len(valid_domains)} < {cfg.min_valid_domains}. Uso simulazione."
            )
    else:
        logger.log("[SCENARIO] Nessuna domain trovata: uso simulazione stratificata.")

    # ---------------------------------------------------------------------
    # 5) Definizione feature
    # ---------------------------------------------------------------------
    protected_cols_for_id = ["target_binary", "sample_type_code_extracted", barcode_col]
    if domain_col is not None:
        protected_cols_for_id.append(domain_col)
    id_cols = detect_id_columns(df, protected_cols_for_id, logger)

    feature_cols = select_numeric_feature_columns(
        df=df,
        target_col="target_binary",
        domain_col=domain_col if scenario == "CASE_B_SIMULATION" else "domain",
        barcode_col=barcode_col,
        sample_type_col="sample_type_code_extracted",
        id_cols=id_cols,
        logger=logger,
    )

    # Indice stabile per controlli anti-overlap
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    logger.log(f"[DATA] Campioni finali={len(df)} | feature finali={len(feature_cols)}")

    # ---------------------------------------------------------------------
    # 6) Esecuzione scenario sperimentale
    # ---------------------------------------------------------------------
    if scenario == "CASE_A_LODO":
        outputs = run_case_a_lodo(
            df=df,
            feature_cols=feature_cols,
            valid_domains=valid_domains,
            cfg=cfg,
            logger=logger,
        )
    else:
        outputs = run_case_b_simulation(
            df=df,
            feature_cols=feature_cols,
            cfg=cfg,
            logger=logger,
        )

    fold_results = outputs["fold_results"]
    pseudo_log = outputs["pseudo_log"]
    confidence_log = outputs["confidence_log"]
    diagnostics_df = outputs["diagnostics"]
    fold_skipped_df = outputs["fold_skipped"]

    # Salvataggi diagnostici sempre utili
    barcode_candidates_df.to_csv(cfg.output_dir / "barcode_candidates.csv", index=False)
    pd.DataFrame(
        {
            "sample_type_code_extracted": sample_type_dist.index.astype("string"),
            "count": sample_type_dist.values,
        }
    ).to_csv(cfg.output_dir / "sample_type_distribution.csv", index=False)

    # ---------------------------------------------------------------------
    # 7) Aggregazione risultati
    # ---------------------------------------------------------------------
    if fold_results.empty:
        logger.log("[STOP] Nessun fold valutabile dopo i controlli.")

        fold_skipped_df.to_csv(cfg.output_dir / "fold_skipped.csv", index=False)
        diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)

        write_interpretation(
            output_path=cfg.output_dir / "interpretation.txt",
            scenario=scenario,
            barcode_col=barcode_col,
            domain_col=domain_col,
            n_total_raw=n_total_raw,
            n_after_filter=n_after_filter,
            n_excluded_non_01_11=n_excluded_non_01_11,
            sample_type_dist=sample_type_dist,
            target_dist=target_dist,
            feature_n=len(feature_cols),
            fold_results=fold_results,
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            diagnostics_df=diagnostics_df,
            skipped_df=fold_skipped_df,
            task_warning=False,
            task_warning_msg="nessun fold",
        )

        copy_script_to_output(cfg.output_dir, logger)
        logger.save(cfg.output_dir / "log.txt")
        return

    summary_df = build_summary(fold_results)
    comparison_df = build_comparison(fold_results)

    # Warning automatico se task troppo facile
    task_warning, task_warning_msg = detect_too_easy_task_warning(fold_results)
    if task_warning:
        logger.log(
            "[WARNING] Task probabilmente troppo facile: interpretare con cautela il transductive"
        )
        logger.log(f"[WARNING] {task_warning_msg}")

    # ---------------------------------------------------------------------
    # 8) Salva CSV principali
    # ---------------------------------------------------------------------
    fold_results.to_csv(cfg.output_dir / "results_fold_by_fold.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "summary_metrics.csv", index=False)
    comparison_df.to_csv(
        cfg.output_dir / "comparison_baseline_vs_transductive.csv", index=False
    )
    pseudo_log.to_csv(cfg.output_dir / "pseudo_labeling_iterations.csv", index=False)
    confidence_log.to_csv(cfg.output_dir / "confidence_log.csv", index=False)
    diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)
    fold_skipped_df.to_csv(cfg.output_dir / "fold_skipped.csv", index=False)

    # ---------------------------------------------------------------------
    # 9) Plot
    # ---------------------------------------------------------------------
    plot_metrics_per_domain(
        fold_results, cfg.output_dir / "plot_metrics_per_domain.png"
    )
    plot_mean_metrics(summary_df, cfg.output_dir / "plot_mean_metrics.png")
    plot_delta_metrics(comparison_df, cfg.output_dir / "plot_delta_metrics.png")
    plot_confidence_distribution(
        confidence_log,
        cfg.pseudo_threshold,
        cfg.output_dir / "plot_confidence_distribution.png",
    )
    plot_pseudo_iterations(pseudo_log, cfg.output_dir / "plot_pseudo_per_iteration.png")

    # ---------------------------------------------------------------------
    # 10) Interpretazione finale
    # ---------------------------------------------------------------------
    write_interpretation(
        output_path=cfg.output_dir / "interpretation.txt",
        scenario=scenario,
        barcode_col=barcode_col,
        domain_col=domain_col,
        n_total_raw=n_total_raw,
        n_after_filter=n_after_filter,
        n_excluded_non_01_11=n_excluded_non_01_11,
        sample_type_dist=sample_type_dist,
        target_dist=target_dist,
        feature_n=len(feature_cols),
        fold_results=fold_results,
        summary_df=summary_df,
        comparison_df=comparison_df,
        diagnostics_df=diagnostics_df,
        skipped_df=fold_skipped_df,
        task_warning=task_warning,
        task_warning_msg=task_warning_msg,
    )

    copy_script_to_output(cfg.output_dir, logger)
    logger.save(cfg.output_dir / "log.txt")

    logger.log("[END] Pipeline completata")
    logger.log(f"[SAVE] {cfg.output_dir / 'results_fold_by_fold.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'summary_metrics.csv'}")
    logger.log(f"[SAVE] {cfg.output_dir / 'comparison_baseline_vs_transductive.csv'}")


if __name__ == "__main__":
    main()
