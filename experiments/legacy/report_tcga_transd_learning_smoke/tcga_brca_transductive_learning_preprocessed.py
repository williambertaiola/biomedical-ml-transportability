#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA-BRCA preprocessed - Baseline vs Transductive learning leakage-safe.

Questo script implementa la stessa logica metodologica corretta usata su UCI,
adattata a un dataset genomico ad alta dimensionalita.

Punti chiave del protocollo:
1) Rilevazione automatica di target e domain (se presenti).
2) Split anti-leakage con separazione netta tra:
   - source_labeled
   - target_adaptation (solo unlabeled nel transductive)
   - target_test (solo valutazione finale)
3) Preprocessing e feature selection fittati SOLO sui source_labeled.
4) Baseline source-only vs transductive pseudo-labeling iterativo.
5) Logging e controlli espliciti per evitare leakage.

Caso speciale gestito in modo robusto:
- Se la colonna target non esiste: la pipeline supervisionata NON e applicabile.
  Lo script termina pulitamente dopo aver scritto una spiegazione in output.
- Se non esiste una colonna domain: viene eseguita una simulazione train/test
  stratificata (meno fedele del vero scenario multi-domain).
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

import matplotlib

# Impostazioni utili in ambienti sandbox/CI headless.
# Usiamo assegnazione diretta (non setdefault) per evitare cache non scrivibili.
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
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
# Configurazione
# ============================================================================

RANDOM_STATE = 42
TARGET_TEST_SIZE = 0.50
SIM_TEST_SIZE = 0.30
SIM_ADAPT_SIZE = 0.30
PSEUDO_CONF_THRESHOLD = 0.95
PSEUDO_MAX_ITERS = 10
VARIANCE_THRESHOLD = 1e-8
TOP_N_FEATURES = 1000

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]

TARGET_CANDIDATES_EXACT = [
    "target",
    "label",
    "y",
    "class",
    "subtype",
    "diagnosis",
    "outcome",
    "phenotype",
    "sample_type_code",
    "sample_type",
]
TARGET_CANDIDATES_PARTIAL = [
    "target",
    "label",
    "class",
    "subtype",
    "diagnos",
    "outcome",
    "phenotype",
    "sample_type",
]

DOMAIN_CANDIDATES_EXACT = [
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
DOMAIN_CANDIDATES_PARTIAL = [
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
]


@dataclass
class RunConfig:
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


class RunLogger:
    """Logger semplice: stampa a terminale e salva tutto su file finale."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ============================================================================
# Utility
# ============================================================================


def ensure_output_dir(path: Path) -> None:
    """Crea la directory output se non esiste."""
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    """Converte un testo arbitrario in un nome file sicuro."""
    text = str(value).replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return text[:120]


def class_distribution_text(y: pd.Series) -> str:
    """Rende leggibile la distribuzione classi per logging diagnostico."""
    counts = y.value_counts(dropna=False).sort_index()
    return ", ".join([f"class_{k}={int(v)}" for k, v in counts.items()])


def can_stratify(y: pd.Series) -> bool:
    """True se possiamo usare stratify in modo stabile."""
    vc = y.value_counts(dropna=False)
    return y.nunique(dropna=True) >= 2 and (len(vc) > 0 and vc.min() >= 2)


def assert_no_overlap(ids_a: pd.Series, ids_b: pd.Series, context: str) -> None:
    """Controllo anti-leakage: due set non devono condividere campioni."""
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] Overlap trovato in {context}. n_overlap={len(overlap)}"
        )


def build_logistic_model(random_state: int) -> LogisticRegression:
    """Logistic Regression robusta per scenario sbilanciato."""
    params = {
        "max_iter": 6000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": random_state,
    }
    sig = inspect.signature(LogisticRegression.__init__)
    if "multi_class" in sig.parameters:
        params["multi_class"] = "auto"
    return LogisticRegression(**params)


def stack_rows(X_a, X_b):
    """Stack robusto per matrici dense/sparse."""
    if sparse.issparse(X_a) or sparse.issparse(X_b):
        return sparse.vstack([sparse.csr_matrix(X_a), sparse.csr_matrix(X_b)])
    return np.vstack([X_a, X_b])


def encode_target_for_classification(
    y_raw: pd.Series,
) -> Tuple[pd.Series, Dict[object, int]]:
    """
    Converte il target in etichette intere stabili.

    Nota: se il target e numerico ma con codici non consecutivi (es. 1, 6, 11),
    facciamo mapping esplicito in 0..K-1 per semplificare metriche e fit.
    """
    y_no_na = y_raw.dropna()

    if pd.api.types.is_numeric_dtype(y_no_na):
        y_num = pd.to_numeric(y_raw, errors="coerce")
        uniq = sorted(y_num.dropna().unique().tolist())
        mapping = {val: idx for idx, val in enumerate(uniq)}
        return y_num.map(mapping).astype("Int64"), mapping

    y_str = y_raw.astype("string")
    uniq = sorted(y_str.dropna().unique().tolist())
    mapping = {val: idx for idx, val in enumerate(uniq)}
    return y_str.map(mapping).astype("Int64"), mapping


# ============================================================================
# Auto-detection colonne
# ============================================================================


def detect_target_column(df: pd.DataFrame, logger: RunLogger) -> Optional[str]:
    """Rileva la colonna target usando candidati plausibili."""
    norm = {c: c.strip().lower() for c in df.columns}

    for key in TARGET_CANDIDATES_EXACT:
        for col, low in norm.items():
            if low == key:
                logger.log(f"[DETECT] Target trovata (exact): {col}")
                return col

    for key in TARGET_CANDIDATES_PARTIAL:
        for col, low in norm.items():
            if key in low:
                logger.log(f"[DETECT] Target trovata (partial '{key}'): {col}")
                return col

    logger.log("[DETECT] Nessuna colonna target plausibile trovata.")
    return None


def detect_domain_column(
    df: pd.DataFrame,
    target_col: Optional[str],
    logger: RunLogger,
) -> Optional[str]:
    """Rileva la colonna domain/site/batch/source, se presente."""
    norm = {c: c.strip().lower() for c in df.columns if c != target_col}

    for key in DOMAIN_CANDIDATES_EXACT:
        for col, low in norm.items():
            if low == key:
                logger.log(f"[DETECT] Domain trovata (exact): {col}")
                return col

    for key in DOMAIN_CANDIDATES_PARTIAL:
        for col, low in norm.items():
            if key in low:
                logger.log(f"[DETECT] Domain trovata (partial '{key}'): {col}")
                return col

    logger.log("[DETECT] Nessuna colonna domain plausibile trovata.")
    return None


def detect_identifier_columns(
    df: pd.DataFrame,
    target_col: Optional[str],
    domain_col: Optional[str],
    logger: RunLogger,
) -> List[str]:
    """
    Identifica colonne ID pure da escludere dalle feature.

    Regole:
    - nome colonna con pattern tipici ID/barcode/patient/sample
    - colonne stringa ad altissima unicita (quasi un valore per riga)
    """
    id_cols: List[str] = []

    for col in df.columns:
        if col in {target_col, domain_col}:
            continue

        low = col.lower()
        hint_match = any(h in low for h in ID_HINTS)

        if hint_match:
            id_cols.append(col)
            continue

        # Heuristica su colonne stringa quasi uniche.
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / max(1, len(df))
            if ratio >= 0.95:
                id_cols.append(col)

    id_cols = sorted(set(id_cols))
    if id_cols:
        logger.log(f"[DETECT] Colonne ID escluse dalle feature: {id_cols}")
    else:
        logger.log("[DETECT] Nessuna colonna ID esplicita da escludere.")
    return id_cols


# ============================================================================
# Preprocessing source-only per alta dimensionalita
# ============================================================================


def fit_source_only_preprocessor(
    X_source: pd.DataFrame,
    variance_threshold: float,
    top_n_features: int,
    logger: RunLogger,
    fold_id: str,
) -> Dict[str, object]:
    """
    Fit del preprocessing SOLO sui source_labeled.

    Pipeline source-only (anti-leakage):
    1) conversione numerica robusta
    2) imputazione median
    3) VarianceThreshold (rimuove feature quasi costanti)
    4) selezione top-N feature piu variabili (sempre calcolata solo sui source)
    5) StandardScaler fit solo sui source selezionati

    Nota metodologica cruciale:
    Nei dati genomici, fittare feature selection/scaler su tutto il dataset
    (inclusi target_adaptation/target_test) produce performance artificialmente alte.
    Qui lo evitiamo esplicitamente.
    """
    # 1) Conversione numerica: eventuali colonne non numeriche diventano NaN.
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    # Teniamo solo colonne con almeno un valore numerico osservato nei source.
    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    dropped_non_numeric = sorted(set(X_source.columns) - set(usable_cols))

    if not usable_cols:
        raise RuntimeError(
            f"[{fold_id}] Nessuna feature numerica utilizzabile dopo conversione."
        )

    X_num = X_num[usable_cols]

    # 2) Imputazione median fit SOLO su source.
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    # 3) Variance threshold fit SOLO su source.
    vt = VarianceThreshold(threshold=variance_threshold)
    try:
        X_vt = vt.fit_transform(X_imp)
    except ValueError:
        # Fallback robusto: se la soglia e troppo aggressiva, usiamo threshold=0.
        vt = VarianceThreshold(threshold=0.0)
        X_vt = vt.fit_transform(X_imp)

    if X_vt.shape[1] == 0:
        raise RuntimeError(
            f"[{fold_id}] Nessuna feature rimasta dopo VarianceThreshold."
        )

    vt_indices = vt.get_support(indices=True)
    vt_feature_names = [usable_cols[i] for i in vt_indices]

    # 4) Top-N variabili sui source (sempre source-only).
    variances = np.var(X_vt, axis=0)
    n_keep = int(min(max(1, top_n_features), X_vt.shape[1]))
    top_idx_rel = np.argsort(variances)[::-1][:n_keep]
    top_idx_rel = np.sort(top_idx_rel)
    selected_feature_names = [vt_feature_names[i] for i in top_idx_rel]

    # 5) Scaling finale fit SOLO su source selezionati.
    scaler = StandardScaler()
    X_sel = X_vt[:, top_idx_rel]
    X_scaled = scaler.fit_transform(X_sel)

    logger.log(
        f"[PREP][{fold_id}] source_n={X_source.shape[0]} | input_features={X_source.shape[1]} | "
        f"usable_numeric={len(usable_cols)} | vt_kept={len(vt_feature_names)} | "
        f"topN_selected={len(selected_feature_names)}"
    )
    if dropped_non_numeric:
        logger.log(
            f"[PREP][{fold_id}] Colonne non numeriche escluse dal preprocessing: {dropped_non_numeric[:20]}"
            + (" ..." if len(dropped_non_numeric) > 20 else "")
        )

    return {
        "usable_cols": usable_cols,
        "dropped_non_numeric": dropped_non_numeric,
        "imputer": imputer,
        "vt": vt,
        "top_idx_rel": top_idx_rel,
        "scaler": scaler,
        "selected_feature_names": selected_feature_names,
        "n_input_features": int(X_source.shape[1]),
        "n_selected_features": int(len(selected_feature_names)),
        "X_source_scaled": X_scaled,
    }


def transform_with_preprocessor(X_df: pd.DataFrame, prep: Dict[str, object]):
    """Applica il preprocessing gia fittato sui source ad altri set (adapt/test)."""
    usable_cols = prep["usable_cols"]
    imputer = prep["imputer"]
    vt = prep["vt"]
    top_idx_rel = prep["top_idx_rel"]
    scaler = prep["scaler"]

    X_num = X_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    X_imp = imputer.transform(X_num)
    X_vt = vt.transform(X_imp)
    X_sel = X_vt[:, top_idx_rel]
    X_scaled = scaler.transform(X_sel)
    return X_scaled


# ============================================================================
# Metriche
# ============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_classes: np.ndarray,
) -> Dict[str, float]:
    """
    Calcola metriche binarie/multiclasse in modo robusto.

    - precision/recall/f1: binary se 2 classi, altrimenti macro.
    - roc_auc: binario standard o multiclass ovr macro.
      Se non calcolabile, ritorna NaN senza interrompere il run.
    """
    out: Dict[str, float] = {}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_union = np.unique(np.concatenate([unique_true, unique_pred]))
    n_classes_true = len(unique_true)
    n_classes_union = len(unique_union)

    if n_classes_true <= 1:
        # Se nel test c'e una sola classe, alcune metriche non sono informative.
        out["precision"] = np.nan
        out["recall"] = np.nan
        out["f1"] = np.nan
        out["roc_auc"] = np.nan
        return out

    # Usiamo metriche binarie SOLO quando il problema osservato e davvero binario
    # (sia in y_true che in y_pred). Se compare una terza classe in predizione,
    # passiamo a macro per evitare errori e bias.
    if n_classes_union == 2:
        pos_label = sorted(unique_union.tolist())[-1]
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
            # Prendiamo la probabilita della classe positiva.
            class_to_idx = {c: i for i, c in enumerate(model_classes)}
            pos_idx = class_to_idx[pos_label]
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
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
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=model_classes,
                )
            )
        except Exception:
            out["roc_auc"] = np.nan

    return out


# ============================================================================
# Transductive pseudo-labeling
# ============================================================================


def run_transductive_pseudo_labeling(
    X_source,
    y_source: np.ndarray,
    X_target_adapt_unlabeled,
    pseudo_threshold: float,
    pseudo_max_iters: int,
    random_state: int,
    fold_id: str,
    logger: RunLogger,
) -> Tuple[LogisticRegression, List[Dict[str, object]], pd.DataFrame, int, int]:
    """
    Pseudo-labeling iterativo leakage-safe.

    IMPORTANTISSIMO:
    - questa funzione riceve SOLO X_target_adapt_unlabeled (senza y vere)
    - quindi non puo usare accidentalmente le label vere del target_adaptation

    Ritorna:
    - modello finale transductive
    - storico iterazioni pseudo-labeling
    - dataframe confidence diagnostico
    - totale pseudo-label aggiunte
    - dimensione finale del training
    """
    X_train = X_source
    y_train = np.asarray(y_source, dtype=int)
    X_unlabeled = X_target_adapt_unlabeled

    history_rows: List[Dict[str, object]] = []
    confidence_rows: List[Dict[str, object]] = []
    total_added = 0

    for iter_idx in range(1, pseudo_max_iters + 1):
        model = build_logistic_model(random_state + iter_idx)
        model.fit(X_train, y_train)

        n_pool_before = int(X_unlabeled.shape[0])
        if n_pool_before == 0:
            history_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": iter_idx,
                    "n_pool_before": 0,
                    "n_selected": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": np.nan,
                    "conf_median": np.nan,
                    "conf_min": np.nan,
                    "conf_max": np.nan,
                }
            )
            logger.log(f"[PSEUDO][{fold_id}] Iter {iter_idx}: pool vuoto, stop.")
            break

        proba = model.predict_proba(X_unlabeled)
        pred = model.predict(X_unlabeled)
        max_conf = np.max(proba, axis=1)
        selected_mask = max_conf >= pseudo_threshold

        # Salviamo tutte le confidence per plot diagnostico.
        for c, sel in zip(max_conf.tolist(), selected_mask.tolist()):
            confidence_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": iter_idx,
                    "confidence": float(c),
                    "selected": int(bool(sel)),
                }
            )

        n_selected = int(selected_mask.sum())
        conf_mean = float(np.mean(max_conf))
        conf_median = float(np.median(max_conf))
        conf_min = float(np.min(max_conf))
        conf_max = float(np.max(max_conf))

        if n_selected == 0:
            history_rows.append(
                {
                    "fold_id": fold_id,
                    "iteration": iter_idx,
                    "n_pool_before": n_pool_before,
                    "n_selected": 0,
                    "pseudo_class_distribution": "{}",
                    "conf_mean": conf_mean,
                    "conf_median": conf_median,
                    "conf_min": conf_min,
                    "conf_max": conf_max,
                }
            )
            logger.log(
                f"[PSEUDO][{fold_id}] Iter {iter_idx}: nessun campione sopra soglia "
                f"{pseudo_threshold:.2f}, stop."
            )
            break

        X_selected = X_unlabeled[selected_mask]
        y_selected = pred[selected_mask].astype(int)

        # Aggiorna training con pseudo-label selezionate.
        X_train = stack_rows(X_train, X_selected)
        y_train = np.concatenate([y_train, y_selected], axis=0)

        # Rimuove campioni selezionati dal pool unlabeled.
        X_unlabeled = X_unlabeled[~selected_mask]

        cls_counts = pd.Series(y_selected).value_counts().sort_index().to_dict()
        cls_dist_str = (
            "{" + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()]) + "}"
        )

        history_rows.append(
            {
                "fold_id": fold_id,
                "iteration": iter_idx,
                "n_pool_before": n_pool_before,
                "n_selected": n_selected,
                "pseudo_class_distribution": cls_dist_str,
                "conf_mean": conf_mean,
                "conf_median": conf_median,
                "conf_min": conf_min,
                "conf_max": conf_max,
            }
        )

        total_added += n_selected
        logger.log(
            f"[PSEUDO][{fold_id}] Iter {iter_idx}: selected={n_selected} | "
            f"class_dist={cls_dist_str} | pool_restante={int(X_unlabeled.shape[0])}"
        )

    # Fit finale sul training arricchito.
    final_model = build_logistic_model(random_state + 777)
    final_model.fit(X_train, y_train)

    final_train_size = int(y_train.shape[0])

    return (
        final_model,
        history_rows,
        pd.DataFrame(confidence_rows),
        int(total_added),
        final_train_size,
    )


# ============================================================================
# Esperimento fold singolo (riusato da Caso A e Caso B)
# ============================================================================


def run_single_fold(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    source_idx: pd.Index,
    target_adapt_idx: pd.Index,
    target_test_idx: pd.Index,
    fold_id: str,
    target_domain_label: str,
    config: RunConfig,
    logger: RunLogger,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame]:
    """
    Esegue baseline + transductive su un fold gia definito.

    Anti-leakage esplicito:
    - target_test mai nel training
    - y_target_adaptation vere non usate nel fit transductive
    - preprocessing/feature selection fit SOLO su source
    """
    source_df = df_all.loc[source_idx].copy()
    target_adapt_df = df_all.loc[target_adapt_idx].copy()
    target_test_df = df_all.loc[target_test_idx].copy()

    # Check anti-leakage: disgiunzione set.
    assert_no_overlap(
        source_df["row_id"],
        target_test_df["row_id"],
        f"{fold_id}: source vs target_test",
    )
    assert_no_overlap(
        target_adapt_df["row_id"],
        target_test_df["row_id"],
        f"{fold_id}: target_adaptation vs target_test",
    )

    # Log diagnostico.
    logger.log("\n" + "=" * 100)
    logger.log(f"[FOLD] {fold_id} | target_domain={target_domain_label}")
    logger.log("=" * 100)
    logger.log(f"[SIZE][{fold_id}] source_n={len(source_df)}")
    logger.log(f"[SIZE][{fold_id}] target_adaptation_n={len(target_adapt_df)}")
    logger.log(f"[SIZE][{fold_id}] target_test_n={len(target_test_df)}")
    logger.log(
        f"[DIST][{fold_id}] source target: {class_distribution_text(source_df['target'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] target_adaptation target (solo diagnostico): "
        f"{class_distribution_text(target_adapt_df['target'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] target_test target: {class_distribution_text(target_test_df['target'])}"
    )

    # Se i source hanno una sola classe, logistic non e addestrabile.
    if source_df["target"].nunique() < 2:
        logger.log(f"[WARN][{fold_id}] Source con una sola classe. Fold saltato.")
        return [], [], pd.DataFrame()

    X_source = source_df[feature_cols]
    y_source = source_df["target"].to_numpy(dtype=int)

    X_adapt = target_adapt_df[feature_cols]
    X_test = target_test_df[feature_cols]
    y_test = target_test_df["target"].to_numpy(dtype=int)

    # ---------------------------------------------------------------------
    # PREPROCESSING SOURCE-ONLY (imputer + feature selection + scaler)
    # ---------------------------------------------------------------------
    prep = fit_source_only_preprocessor(
        X_source,
        variance_threshold=config.variance_threshold,
        top_n_features=config.top_n_features,
        logger=logger,
        fold_id=fold_id,
    )

    # Check esplicito: preprocessor fit solo sui source.
    logger.log(
        f"[CHECK][{fold_id}] Preprocessing fittato SOLO sui source: n_source_fit={len(source_df)}"
    )

    X_source_tr = prep["X_source_scaled"]
    X_adapt_tr = transform_with_preprocessor(X_adapt, prep)
    X_test_tr = transform_with_preprocessor(X_test, prep)

    # ---------------------------------------------------------------------
    # Baseline source-only
    # ---------------------------------------------------------------------
    baseline_model = build_logistic_model(config.random_state)
    baseline_model.fit(X_source_tr, y_source)

    y_pred_baseline = baseline_model.predict(X_test_tr)
    y_prob_baseline = baseline_model.predict_proba(X_test_tr)

    baseline_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred_baseline,
        y_proba=y_prob_baseline,
        model_classes=baseline_model.classes_,
    )

    # ---------------------------------------------------------------------
    # Transductive pseudo-labeling leakage-safe
    # ---------------------------------------------------------------------
    logger.log(
        f"[CHECK][{fold_id}] Le vere label di target_adaptation NON vengono passate al fit transductive."
    )

    n_source_labels = int(y_source.shape[0])

    (
        trans_model,
        pseudo_history,
        confidence_df,
        pseudo_total_added,
        final_train_size,
    ) = run_transductive_pseudo_labeling(
        X_source=X_source_tr,
        y_source=y_source,
        X_target_adapt_unlabeled=X_adapt_tr,
        pseudo_threshold=config.pseudo_threshold,
        pseudo_max_iters=config.pseudo_max_iters,
        random_state=config.random_state,
        fold_id=fold_id,
        logger=logger,
    )

    # Check anti-leakage forte: train finale = source + pseudo (mai y vere adaptation).
    expected_final_train = n_source_labels + int(pseudo_total_added)
    if final_train_size != expected_final_train:
        raise RuntimeError(
            f"[LEAKAGE ERROR][{fold_id}] final_train_size inattesa: {final_train_size} "
            f"!= source_plus_pseudo {expected_final_train}"
        )

    logger.log(
        f"[CHECK][{fold_id}] OK train_transductive_size={final_train_size} "
        f"(source={n_source_labels} + pseudo={pseudo_total_added})"
    )

    # Anche nel transductive, target_test resta unseen.
    assert_no_overlap(
        pd.Series(
            np.concatenate(
                [source_df["row_id"].to_numpy(), target_adapt_df["row_id"].to_numpy()]
            )
        ),
        target_test_df["row_id"],
        f"{fold_id}: transductive_train vs target_test",
    )

    y_pred_trans = trans_model.predict(X_test_tr)
    y_prob_trans = trans_model.predict_proba(X_test_tr)

    trans_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred_trans,
        y_proba=y_prob_trans,
        model_classes=trans_model.classes_,
    )

    # Warning se transductive peggiora rispetto baseline su precision.
    if (
        pd.notna(trans_metrics.get("precision"))
        and pd.notna(baseline_metrics.get("precision"))
        and trans_metrics["precision"] < baseline_metrics["precision"]
    ):
        logger.log(
            f"[WARN][{fold_id}] Precision transductive peggiora: "
            f"baseline={baseline_metrics['precision']:.4f}, "
            f"transductive={trans_metrics['precision']:.4f}"
        )

    # Log roc_auc non calcolabile.
    if pd.isna(baseline_metrics.get("roc_auc")):
        logger.log(f"[WARN][{fold_id}] ROC-AUC baseline non calcolabile.")
    if pd.isna(trans_metrics.get("roc_auc")):
        logger.log(f"[WARN][{fold_id}] ROC-AUC transductive non calcolabile.")

    fold_rows = [
        {
            "fold_id": fold_id,
            "target_domain": target_domain_label,
            "method": "baseline",
            "source_n": len(source_df),
            "target_adaptation_n": len(target_adapt_df),
            "target_test_n": len(target_test_df),
            "n_input_features": prep["n_input_features"],
            "n_selected_features": prep["n_selected_features"],
            "pseudo_total_added": 0,
            "pseudo_iterations": 0,
            **baseline_metrics,
        },
        {
            "fold_id": fold_id,
            "target_domain": target_domain_label,
            "method": "transductive",
            "source_n": len(source_df),
            "target_adaptation_n": len(target_adapt_df),
            "target_test_n": len(target_test_df),
            "n_input_features": prep["n_input_features"],
            "n_selected_features": prep["n_selected_features"],
            "pseudo_total_added": int(pseudo_total_added),
            "pseudo_iterations": int(len(pseudo_history)),
            **trans_metrics,
        },
    ]

    pseudo_rows: List[Dict[str, object]] = []
    for row in pseudo_history:
        pseudo_rows.append(
            {
                "fold_id": fold_id,
                "target_domain": target_domain_label,
                **row,
            }
        )

    if not confidence_df.empty:
        confidence_df["target_domain"] = target_domain_label

    return fold_rows, pseudo_rows, confidence_df


# ============================================================================
# Caso A: Leave-One-Domain-Out reale
# ============================================================================


def run_case_a_lodo(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    domain_col: str,
    config: RunConfig,
    logger: RunLogger,
) -> Dict[str, pd.DataFrame]:
    """Esegue Leave-One-Domain-Out quando esiste una vera colonna domain."""
    fold_rows_all: List[Dict[str, object]] = []
    pseudo_rows_all: List[Dict[str, object]] = []
    conf_frames: List[pd.DataFrame] = []
    skipped_rows: List[Dict[str, object]] = []

    domain_values = sorted(df_all[domain_col].dropna().astype(str).unique().tolist())

    for fold_idx, domain_value in enumerate(domain_values, start=1):
        fold_id = f"lodo_{fold_idx}_{safe_name(domain_value)}"

        target_df = df_all.loc[df_all[domain_col].astype(str) == domain_value].copy()
        source_df = df_all.loc[df_all[domain_col].astype(str) != domain_value].copy()

        if len(target_df) < 2:
            skipped_rows.append(
                {
                    "fold_id": fold_id,
                    "target_domain": domain_value,
                    "reason": "target_domain_too_small",
                    "target_n": len(target_df),
                }
            )
            logger.log(
                f"[SKIP][{fold_id}] Target domain troppo piccolo: n={len(target_df)}"
            )
            continue

        # Split target -> adaptation + test (anti-leakage).
        strat = target_df["target"] if can_stratify(target_df["target"]) else None
        if strat is None:
            logger.log(
                f"[WARN][{fold_id}] Stratify non applicabile sul target domain, uso split non stratificato."
            )

        try:
            target_adapt_df, target_test_df = train_test_split(
                target_df,
                test_size=config.target_test_size,
                random_state=config.random_state,
                stratify=strat,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "fold_id": fold_id,
                    "target_domain": domain_value,
                    "reason": f"target_split_failed: {exc}",
                    "target_n": len(target_df),
                }
            )
            logger.log(f"[SKIP][{fold_id}] Split target fallito: {exc}")
            continue

        # Esecuzione fold.
        fold_rows, pseudo_rows, conf_df = run_single_fold(
            df_all=df_all,
            feature_cols=feature_cols,
            source_idx=source_df.index,
            target_adapt_idx=target_adapt_df.index,
            target_test_idx=target_test_df.index,
            fold_id=fold_id,
            target_domain_label=str(domain_value),
            config=config,
            logger=logger,
        )

        if not fold_rows:
            skipped_rows.append(
                {
                    "fold_id": fold_id,
                    "target_domain": domain_value,
                    "reason": "fold_not_trainable",
                    "target_n": len(target_df),
                }
            )
            continue

        fold_rows_all.extend(fold_rows)
        pseudo_rows_all.extend(pseudo_rows)
        if not conf_df.empty:
            conf_frames.append(conf_df)

    return {
        "fold_results": pd.DataFrame(fold_rows_all),
        "pseudo_history": pd.DataFrame(pseudo_rows_all),
        "confidence": pd.concat(conf_frames, ignore_index=True)
        if conf_frames
        else pd.DataFrame(),
        "skipped_folds": pd.DataFrame(skipped_rows),
    }


# ============================================================================
# Caso B: Simulazione senza domain
# ============================================================================


def run_case_b_simulation(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    config: RunConfig,
    logger: RunLogger,
) -> Dict[str, pd.DataFrame]:
    """
    Simulazione quando manca una vera colonna domain.

    Nota metodologica (esplicitata nel report):
    - NON e un vero scenario multi-domain come UCI.
    - E una proxy meno fedele, utile solo se manca la struttura domain reale.
    """
    logger.log(
        "[INFO] Caso B attivo: nessuna colonna domain trovata. "
        "Uso simulazione stratificata train/test + adaptation (meno fedele del multi-domain reale)."
    )

    # 1) split principale in pool train e target_test.
    strat_main = df_all["target"] if can_stratify(df_all["target"]) else None
    if strat_main is None:
        logger.log(
            "[WARN][SIM] Stratify non applicabile sul dataset completo, uso split non stratificato."
        )

    train_pool_df, target_test_df = train_test_split(
        df_all,
        test_size=config.sim_test_size,
        random_state=config.random_state,
        stratify=strat_main,
    )

    # 2) split del train pool in source_labeled e target_adaptation simulato.
    strat_sub = (
        train_pool_df["target"] if can_stratify(train_pool_df["target"]) else None
    )
    if strat_sub is None:
        logger.log(
            "[WARN][SIM] Stratify non applicabile nel train_pool, uso split non stratificato."
        )

    source_df, target_adapt_df = train_test_split(
        train_pool_df,
        test_size=config.sim_adapt_size,
        random_state=config.random_state,
        stratify=strat_sub,
    )

    fold_rows, pseudo_rows, conf_df = run_single_fold(
        df_all=df_all,
        feature_cols=feature_cols,
        source_idx=source_df.index,
        target_adapt_idx=target_adapt_df.index,
        target_test_idx=target_test_df.index,
        fold_id="simulated_fold_1",
        target_domain_label="simulated_no_domain",
        config=config,
        logger=logger,
    )

    return {
        "fold_results": pd.DataFrame(fold_rows),
        "pseudo_history": pd.DataFrame(pseudo_rows),
        "confidence": conf_df if not conf_df.empty else pd.DataFrame(),
        "skipped_folds": pd.DataFrame(),
    }


# ============================================================================
# Tabelle aggregate
# ============================================================================


def build_summary_metrics(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Media/std metriche per metodo."""
    rows: List[Dict[str, object]] = []
    for method in ["baseline", "transductive"]:
        sub = fold_results.loc[fold_results["method"] == method].copy()
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


def build_comparison_table(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Confronto baseline vs transductive e delta per fold/target_domain/metrica."""
    rows: List[Dict[str, object]] = []

    if fold_results.empty:
        return pd.DataFrame(rows)

    key_cols = ["fold_id", "target_domain"]
    grouped = fold_results.groupby(key_cols, dropna=False)

    for (fold_id, target_domain), grp in grouped:
        base = grp.loc[grp["method"] == "baseline"]
        trans = grp.loc[grp["method"] == "transductive"]
        if base.empty or trans.empty:
            continue

        base_row = base.iloc[0]
        trans_row = trans.iloc[0]

        for m in METRICS:
            b = base_row.get(m, np.nan)
            t = trans_row.get(m, np.nan)
            delta = t - b if pd.notna(b) and pd.notna(t) else np.nan
            rows.append(
                {
                    "fold_id": fold_id,
                    "target_domain": target_domain,
                    "metric": m,
                    "baseline": b,
                    "transductive": t,
                    "delta_transductive_minus_baseline": delta,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# Plot
# ============================================================================


def plot_metrics_by_domain(fold_results: pd.DataFrame, output_path: Path) -> None:
    """Barplot baseline vs transductive per target_domain su 6 metriche."""
    if fold_results.empty:
        return

    domains = sorted(fold_results["target_domain"].astype(str).unique().tolist())
    fig_w = max(16, 8 + 0.45 * len(domains))

    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    x = np.arange(len(domains), dtype=float)
    width = 0.36

    for ax, metric in zip(axes, METRICS):
        baseline_vals = []
        trans_vals = []

        for d in domains:
            base = fold_results[
                (fold_results["target_domain"].astype(str) == d)
                & (fold_results["method"] == "baseline")
            ]
            trans = fold_results[
                (fold_results["target_domain"].astype(str) == d)
                & (fold_results["method"] == "transductive")
            ]

            baseline_vals.append(
                float(pd.to_numeric(base[metric], errors="coerce").mean())
                if not base.empty
                else np.nan
            )
            trans_vals.append(
                float(pd.to_numeric(trans[metric], errors="coerce").mean())
                if not trans.empty
                else np.nan
            )

        ax.bar(x - width / 2, baseline_vals, width=width, label="baseline")
        ax.bar(x + width / 2, trans_vals, width=width, label="transductive")

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=80)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("Baseline vs Transductive per target domain/fold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mean_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot riassuntivo metriche medie baseline vs transductive."""
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


def plot_delta_by_domain(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Plot delta (transductive - baseline) per target domain e metrica."""
    if comparison_df.empty:
        return

    domains = sorted(comparison_df["target_domain"].astype(str).unique().tolist())
    fig_w = max(16, 8 + 0.45 * len(domains))

    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 10))
    axes = axes.flatten()

    x = np.arange(len(domains), dtype=float)

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

    fig.suptitle("Delta metriche: transductive - baseline")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pseudo_labels_per_iteration(
    pseudo_df: pd.DataFrame, output_path: Path
) -> None:
    """Plot diagnostico: numero pseudo-label aggiunte a ogni iterazione."""
    if pseudo_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for fold_id, sub in pseudo_df.groupby("fold_id", dropna=False):
        sub = sub.sort_values("iteration")
        ax.plot(
            sub["iteration"],
            sub["n_selected"],
            marker="o",
            linewidth=1.5,
            alpha=0.7,
            label=str(fold_id),
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("n pseudo-label selected")
    ax.set_title("Pseudo-label aggiunte per iterazione")
    ax.grid(alpha=0.3)

    # In caso di molti fold, non mostriamo legenda completa per non saturare il grafico.
    if pseudo_df["fold_id"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confidence_distribution(conf_df: pd.DataFrame, output_path: Path) -> None:
    """Plot diagnostico: distribuzione confidence con evidenza selected/non-selected."""
    if conf_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sel = conf_df.loc[conf_df["selected"] == 1, "confidence"].dropna()
    nosel = conf_df.loc[conf_df["selected"] == 0, "confidence"].dropna()

    ax.hist(nosel, bins=40, alpha=0.55, label="not_selected", density=True)
    ax.hist(sel, bins=40, alpha=0.55, label="selected", density=True)

    ax.axvline(
        PSEUDO_CONF_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="threshold",
    )

    ax.set_xlabel("confidence (max predicted probability)")
    ax.set_ylabel("density")
    ax.set_title("Distribuzione confidence nel target_adaptation")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================================
# Report testuale
# ============================================================================


def write_not_applicable_report(
    output_path: Path,
    input_csv: Path,
    reason: str,
) -> None:
    """Report quando la pipeline supervisionata non e applicabile."""
    lines = [
        "# Transductive Learning Non Applicabile",
        "",
        f"Input file: `{input_csv}`",
        "",
        "## Motivo",
        reason,
        "",
        "## Nota metodologica",
        "Per fare transductive learning supervisionato servono label target valide.",
        "In assenza di una colonna target, il training baseline/transductive di classificazione non e definibile.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    output_path: Path,
    scenario: str,
    target_col: str,
    domain_col: Optional[str],
    id_cols: List[str],
    target_mapping: Dict[object, int],
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    skipped_folds: pd.DataFrame,
) -> None:
    """Sintesi automatica finale orientata alla discussione in tesi."""
    lines: List[str] = []
    lines.append("# Sintesi automatica risultati")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Scenario: **{scenario}**")
    lines.append(f"- Target identificata: **{target_col}**")
    lines.append(
        f"- Domain identificata: **{domain_col if domain_col is not None else 'ASSENTE (simulazione)'}**"
    )
    lines.append(
        f"- Colonne ID escluse dalle feature: {id_cols if id_cols else 'nessuna'}"
    )
    lines.append(f"- Mapping target -> codici interi: {target_mapping}")
    lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valido e stato addestrato.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Risultati globali")
    lines.append(f"- Numero fold valutati: {fold_results['fold_id'].nunique()}")
    if not skipped_folds.empty:
        lines.append(f"- Fold saltati: {len(skipped_folds)}")
    lines.append("")

    lines.append("## Metriche medie")
    for method in ["baseline", "transductive"]:
        sub = summary_df[summary_df["method"] == method]
        if sub.empty:
            continue
        lines.append(f"### {method}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['metric']}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={int(row['n_valid'])}"
            )
    lines.append("")

    # Focus miglioramento/peggioramento transductive.
    precision_cmp = comparison_df[comparison_df["metric"] == "precision"].copy()
    lines.append("## Focus su precision (transductive - baseline)")
    if precision_cmp.empty:
        lines.append(
            "- Precision non disponibile (es. classi insufficienti nei test fold)."
        )
    else:
        precision_cmp = precision_cmp.sort_values(
            "delta_transductive_minus_baseline", ascending=False
        )
        for _, row in precision_cmp.iterrows():
            delta = row["delta_transductive_minus_baseline"]
            tag = (
                "migliora"
                if pd.notna(delta) and delta > 0
                else ("peggiora" if pd.notna(delta) and delta < 0 else "invariata")
            )
            lines.append(
                f"- {row['fold_id']} ({row['target_domain']}): baseline={row['baseline']:.4f}, "
                f"transductive={row['transductive']:.4f}, delta={delta:+.4f} -> {tag}"
            )

        mean_base = float(precision_cmp["baseline"].mean(skipna=True))
        mean_trans = float(precision_cmp["transductive"].mean(skipna=True))
        mean_delta = float(
            precision_cmp["delta_transductive_minus_baseline"].mean(skipna=True)
        )
        lines.append("")
        lines.append(f"- Precision media baseline: **{mean_base:.4f}**")
        lines.append(f"- Precision media transductive: **{mean_trans:.4f}**")
        lines.append(f"- Delta medio precision: **{mean_delta:+.4f}**")

    lines.append("")
    lines.append("## Nota anti-leakage")
    lines.append("- target_test non entra mai nel training.")
    lines.append(
        "- le label vere di target_adaptation non vengono mai usate nel fit transductive."
    )
    lines.append("- scaler e feature selection sono fittati solo sui source_labeled.")
    lines.append(
        "- nessuna informazione di target_test entra in preprocessing o selezione feature."
    )
    lines.append(
        "- nei dati genomici, violare questi vincoli produce metriche artificialmente alte e non realistiche."
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Argomenti CLI con default richiesti."""
    parser = argparse.ArgumentParser(
        description="Baseline vs transductive learning leakage-safe su TCGA-BRCA preprocessato"
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(_configured_path("legacy_input_tcga_brca_preprocessed.csv")),
        help="Input CSV preprocessato",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_report_tcga_transd_learning")),
        help="Directory output",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--target-test-size", type=float, default=TARGET_TEST_SIZE)
    parser.add_argument("--pseudo-threshold", type=float, default=PSEUDO_CONF_THRESHOLD)
    parser.add_argument("--pseudo-max-iters", type=int, default=PSEUDO_MAX_ITERS)
    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)

    return parser.parse_args()


def copy_script_to_output(output_dir: Path, logger: RunLogger) -> None:
    """Copia lo script corrente nella directory output."""
    try:
        src = Path(__file__).resolve()
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        logger.log(f"[SAVE] Copia script: {dst}")
    except Exception as exc:
        logger.log(f"[WARN] Copia script fallita: {exc}")


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    config = RunConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        target_test_size=args.target_test_size,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
        top_n_features=args.top_n_features,
        variance_threshold=args.variance_threshold,
    )

    ensure_output_dir(config.output_dir)
    logger = RunLogger()

    logger.log(
        "[START] Avvio analisi TCGA-BRCA baseline vs transductive (leakage-safe)."
    )
    logger.log(f"[CONFIG] input_csv={config.input_csv}")
    logger.log(f"[CONFIG] output_dir={config.output_dir}")
    logger.log(f"[CONFIG] random_state={config.random_state}")
    logger.log(f"[CONFIG] pseudo_threshold={config.pseudo_threshold}")
    logger.log(f"[CONFIG] pseudo_max_iters={config.pseudo_max_iters}")
    logger.log(f"[CONFIG] top_n_features={config.top_n_features}")
    logger.log(f"[CONFIG] variance_threshold={config.variance_threshold}")

    if not config.input_csv.exists():
        raise FileNotFoundError(f"Input file non trovato: {config.input_csv}")

    # ---------------------------------------------------------------------
    # 1) Caricamento + ispezione iniziale
    # ---------------------------------------------------------------------
    df_raw = pd.read_csv(config.input_csv)
    logger.log(f"[DATA] shape={df_raw.shape}")
    logger.log(f"[DATA] n_columns={len(df_raw.columns)}")
    logger.log(f"[DATA] columns (prime 30): {df_raw.columns[:30].tolist()}")
    # Per leggibilita del log mostriamo solo un sottoinsieme di colonne.
    preview_cols = df_raw.columns[:20].tolist()
    logger.log(
        f"[DATA] head(3) su prime 20 colonne:\n"
        f"{df_raw[preview_cols].head(3).to_string(index=False)}"
    )

    target_col = detect_target_column(df_raw, logger)
    domain_col = detect_domain_column(df_raw, target_col=target_col, logger=logger)

    # Se manca target, pipeline supervisionata non applicabile.
    if target_col is None:
        reason = (
            "Nessuna colonna target/label plausibile trovata nel file. "
            "Senza una label supervisionata, non e possibile fare baseline di classificazione "
            "ne transductive learning pseudo-labeling supervisionato."
        )
        logger.log(f"[STOP] {reason}")

        write_not_applicable_report(
            output_path=config.output_dir / "pipeline_non_applicabile.md",
            input_csv=config.input_csv,
            reason=reason,
        )
        copy_script_to_output(config.output_dir, logger)
        logger.save(config.output_dir / "log_esperimento.txt")
        return

    id_cols = detect_identifier_columns(
        df_raw, target_col=target_col, domain_col=domain_col, logger=logger
    )

    # ---------------------------------------------------------------------
    # 2) Definizione target/domain/feature
    # ---------------------------------------------------------------------
    df = df_raw.copy()

    # Target encoding robusto in classi intere.
    df["target"], target_mapping = encode_target_for_classification(df[target_col])

    # Rimuoviamo righe senza target.
    n_missing_target = int(df["target"].isna().sum())
    if n_missing_target > 0:
        logger.log(f"[CLEAN] Righe con target mancante rimosse: {n_missing_target}")
        df = df.loc[df["target"].notna()].copy()

    # Se il target ha meno di 2 classi, non e classificazione supervisionata valida.
    n_classes = int(df["target"].nunique())
    if n_classes < 2:
        reason = (
            f"Target '{target_col}' con meno di 2 classi utili dopo cleaning. "
            "La classificazione supervisionata non e applicabile."
        )
        logger.log(f"[STOP] {reason}")
        write_not_applicable_report(
            output_path=config.output_dir / "pipeline_non_applicabile.md",
            input_csv=config.input_csv,
            reason=reason,
        )
        copy_script_to_output(config.output_dir, logger)
        logger.save(config.output_dir / "log_esperimento.txt")
        return

    df["target"] = df["target"].astype(int)

    if domain_col is not None:
        # Manteniamo domain esplicito per LODO.
        df["domain"] = df[domain_col].astype("string")
        n_missing_domain = int(df["domain"].isna().sum())
        if n_missing_domain > 0:
            logger.log(f"[CLEAN] Righe con domain mancante rimosse: {n_missing_domain}")
            df = df.loc[df["domain"].notna()].copy()

    # Costruiamo lista feature escludendo target/domain/id.
    exclude_cols = set([target_col, "target"])
    if domain_col is not None:
        exclude_cols.add(domain_col)
        exclude_cols.add("domain")
    for c in id_cols:
        exclude_cols.add(c)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        raise RuntimeError(
            "Nessuna feature disponibile dopo esclusione target/domain/id."
        )

    # row_id per check anti-leakage.
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(df.shape[0], dtype=int)

    logger.log(f"[DATA] Campioni finali dopo cleaning: {len(df)}")
    logger.log(
        f"[DATA] Numero classi target: {df['target'].nunique()} | distribuzione: {class_distribution_text(df['target'])}"
    )
    if domain_col is not None:
        logger.log(
            f"[DATA] Domain unique: {df['domain'].nunique()} | domain_col={domain_col}"
        )
    logger.log(f"[DATA] Feature candidate count: {len(feature_cols)}")

    # ---------------------------------------------------------------------
    # 3) Scelta scenario sperimentale (Case A vs Case B)
    # ---------------------------------------------------------------------
    # Se domain rilevata ma con <2 domini utili, fallback al Caso B.
    use_case_a = domain_col is not None and df["domain"].nunique() >= 2

    if use_case_a:
        scenario = "CASE_A_MULTI_DOMAIN_LODO"
        outputs = run_case_a_lodo(
            df_all=df,
            feature_cols=feature_cols,
            domain_col="domain",
            config=config,
            logger=logger,
        )
    else:
        scenario = "CASE_B_SIMULATED_NO_DOMAIN"
        if domain_col is not None and df["domain"].nunique() < 2:
            logger.log(
                "[WARN] Domain rilevata ma con <2 valori utili. Passo a simulazione Case B."
            )
        outputs = run_case_b_simulation(
            df_all=df,
            feature_cols=feature_cols,
            config=config,
            logger=logger,
        )

    fold_results = outputs["fold_results"]
    pseudo_history = outputs["pseudo_history"]
    confidence_df = outputs["confidence"]
    skipped_folds = outputs["skipped_folds"]

    if fold_results.empty:
        logger.log("[STOP] Nessun fold valido addestrabile.")
        # Salviamo comunque diagnostica minima.
        skipped_folds.to_csv(config.output_dir / "fold_saltati.csv", index=False)
        write_summary_markdown(
            output_path=config.output_dir / "sintesi_risultati.md",
            scenario=scenario,
            target_col=target_col,
            domain_col=domain_col,
            id_cols=id_cols,
            target_mapping=target_mapping,
            fold_results=fold_results,
            summary_df=pd.DataFrame(),
            comparison_df=pd.DataFrame(),
            skipped_folds=skipped_folds,
        )
        copy_script_to_output(config.output_dir, logger)
        logger.save(config.output_dir / "log_esperimento.txt")
        return

    # ---------------------------------------------------------------------
    # 4) Tabelle finali
    # ---------------------------------------------------------------------
    summary_df = build_summary_metrics(fold_results)
    comparison_df = build_comparison_table(fold_results)

    fold_results.to_csv(config.output_dir / "risultati_fold_by_fold.csv", index=False)
    summary_df.to_csv(config.output_dir / "summary_metriche_finali.csv", index=False)
    comparison_df.to_csv(
        config.output_dir / "confronto_baseline_vs_transductive.csv", index=False
    )
    pseudo_history.to_csv(
        config.output_dir / "pseudo_labeling_iterazioni.csv", index=False
    )
    confidence_df.to_csv(config.output_dir / "confidence_diagnostica.csv", index=False)
    skipped_folds.to_csv(config.output_dir / "fold_saltati.csv", index=False)

    # ---------------------------------------------------------------------
    # 5) Plot
    # ---------------------------------------------------------------------
    plot_metrics_by_domain(
        fold_results=fold_results,
        output_path=config.output_dir
        / "barplot_metriche_baseline_vs_transductive_per_domain.png",
    )
    plot_mean_metrics(
        summary_df=summary_df,
        output_path=config.output_dir / "plot_metriche_medie_finali.png",
    )
    plot_delta_by_domain(
        comparison_df=comparison_df,
        output_path=config.output_dir / "plot_delta_metriche_per_domain.png",
    )
    plot_pseudo_labels_per_iteration(
        pseudo_df=pseudo_history,
        output_path=config.output_dir / "plot_pseudo_labels_per_iterazione.png",
    )
    plot_confidence_distribution(
        conf_df=confidence_df,
        output_path=config.output_dir / "plot_distribuzione_confidence.png",
    )

    # ---------------------------------------------------------------------
    # 6) Sintesi testuale
    # ---------------------------------------------------------------------
    write_summary_markdown(
        output_path=config.output_dir / "sintesi_risultati.md",
        scenario=scenario,
        target_col=target_col,
        domain_col=domain_col,
        id_cols=id_cols,
        target_mapping=target_mapping,
        fold_results=fold_results,
        summary_df=summary_df,
        comparison_df=comparison_df,
        skipped_folds=skipped_folds,
    )

    # Copia script + log.
    copy_script_to_output(config.output_dir, logger)
    logger.save(config.output_dir / "log_esperimento.txt")

    logger.log("[END] Pipeline completata con successo.")
    logger.log(
        f"[SAVE] risultati_fold_by_fold.csv -> {config.output_dir / 'risultati_fold_by_fold.csv'}"
    )
    logger.log(
        f"[SAVE] summary_metriche_finali.csv -> {config.output_dir / 'summary_metriche_finali.csv'}"
    )
    logger.log(
        f"[SAVE] confronto_baseline_vs_transductive.csv -> "
        f"{config.output_dir / 'confronto_baseline_vs_transductive.csv'}"
    )


if __name__ == "__main__":
    main()
