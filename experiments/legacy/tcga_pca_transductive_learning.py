#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA-BRCA: Baseline vs Transductive Learning con target artificiale PCA+KMeans
===============================================================================

Obiettivo:
- costruire un task di classificazione NON banale usando clustering su PCA
- confrontare baseline source-only vs transductive learning (pseudo-labeling)
- mantenere una pipeline metodologicamente corretta e leakage-safe nei fold

Punto chiave metodologico:
- Per creare il target artificiale (target_cluster) usiamo tutto il dataset
  in modo unsupervised (StandardScaler + PCA + KMeans), come richiesto.
- Dopo questa fase di etichettatura, TUTTE le trasformazioni che apprendono dati
  nel training supervisionato (imputer, feature selection, scaler) sono fittate
  SOLO sui source di ogni fold.

Perche usare PCA + clustering:
- Evita task banali predefiniti (es. 01 vs 11) che possono risultare troppo facili.
- Evita anche setup con classe rara estrema (es. 01 vs 06 con pochissimi campioni).
- Crea un compito che riflette struttura latente dei profili trascrittomici,
  spesso piu realistico per stressare la transportability tra domini/site.
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

# Config ambiente per matplotlib (utile in ambienti headless/sandbox)
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Config
# =============================================================================

RANDOM_STATE = 42

# Input/Output di default richiesti dall'utente
DEFAULT_INPUT = Path(_configured_path("tcga_expression"))
DEFAULT_OUTPUT = Path(_configured_path("legacy_tcga_pca_transductive_output"))

# Parametri target artificiale
PCA_COMPONENTS_FOR_TARGET = 20
KMEANS_K = 2
KMEANS_RESTARTS_FOR_BALANCE = 50
MIN_TARGET_MINORITY_RATIO = 0.35

# Parametri domini
MIN_SITE_SAMPLES = 20
MIN_VALID_SITES = 3
TARGET_TEST_SIZE = 0.50

# Preprocessing leakage-safe nel training supervisionato
TOP_N_FEATURES = 100

# Pseudo-labeling
PSEUDO_THRESHOLD = 0.95
PSEUDO_MAX_ITERS = 5
PSEUDO_MAX_PER_ITER = 20
PSEUDO_MAX_POOL_FRACTION = 0.10

# Metriche richieste
METRICS = ["accuracy", "precision", "recall", "f1"]

# Colonne che non devono entrare nelle feature
NON_FEATURE_COLS = {
    "sample_barcode",
    "site",
    "target_cluster",
    "row_id",
    "source_file",
}


@dataclass
class Config:
    input_path: Path
    output_dir: Path
    random_state: int = RANDOM_STATE

    pca_components_for_target: int = PCA_COMPONENTS_FOR_TARGET
    kmeans_k: int = KMEANS_K
    kmeans_restarts_for_balance: int = KMEANS_RESTARTS_FOR_BALANCE
    min_target_minority_ratio: float = MIN_TARGET_MINORITY_RATIO

    min_site_samples: int = MIN_SITE_SAMPLES
    min_valid_sites: int = MIN_VALID_SITES
    target_test_size: float = TARGET_TEST_SIZE

    top_n_features: int = TOP_N_FEATURES

    pseudo_threshold: float = PSEUDO_THRESHOLD
    pseudo_max_iters: int = PSEUDO_MAX_ITERS
    pseudo_max_per_iter: int = PSEUDO_MAX_PER_ITER
    pseudo_max_pool_fraction: float = PSEUDO_MAX_POOL_FRACTION


class Logger:
    """Logger semplice: stampa e accumula per salvataggio su file."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def save(self, output_path: Path) -> None:
        output_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# =============================================================================
# Utility generali
# =============================================================================


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


def make_unique_columns(cols: List[str]) -> List[str]:
    """Rende univoci i nomi colonna preservando l'ordine."""
    seen: Dict[str, int] = {}
    out: List[str] = []

    for c in cols:
        c0 = str(c)
        if c0 not in seen:
            seen[c0] = 0
            out.append(c0)
        else:
            seen[c0] += 1
            out.append(f"{c0}__dup{seen[c0]}")

    return out


def build_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=6000,
        solver="lbfgs",
        random_state=seed,
    )


# =============================================================================
# Barcode TCGA e parsing site
# =============================================================================


def is_tcga_like_barcode(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    txt = str(value).strip().upper()
    return bool(re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-", txt))


def extract_site_from_barcode(value: object) -> Optional[str]:
    """
    Estrae il TSS code (site) dal barcode TCGA.

    Esempio:
    TCGA-ZZ-0000-01A-11R-A115-07 -> site = BH
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    txt = str(value).strip().upper().replace("_", "-").replace(".", "-")
    parts = txt.split("-")

    if len(parts) < 2:
        return None
    if parts[0] != "TCGA":
        return None

    site = parts[1]
    if not re.match(r"^[A-Z0-9]{2}$", site):
        return None

    return site


# =============================================================================
# Load dati: supporta file star_counts.tsv o directory di file per-site
# =============================================================================


def detect_orientation(df_raw: pd.DataFrame) -> str:
    """
    Euristica orientamento:
    - genes_x_samples: molte colonne simili a barcode e prima colonna non-barcode
    - samples_x_features: altrimenti
    """
    col_names = [str(c) for c in df_raw.columns]
    if not col_names:
        return "samples_x_features"

    first_col = col_names[0]
    n_barcode_like_cols = sum(is_tcga_like_barcode(c) for c in col_names)

    if n_barcode_like_cols >= 10 and not is_tcga_like_barcode(first_col):
        return "genes_x_samples"

    return "samples_x_features"


def find_barcode_column(samples_df: pd.DataFrame) -> Optional[str]:
    """Trova una colonna barcode in un df campioni x feature."""
    candidate_names = [
        "sample_barcode",
        "sample_id",
        "barcode",
        "tcga_barcode",
        "aliquot_barcode",
        "id",
    ]

    lower_map = {str(c).lower(): str(c) for c in samples_df.columns}
    for cand in candidate_names:
        if cand in lower_map:
            return lower_map[cand]

    best_col = None
    best_ratio = 0.0

    for c in samples_df.columns:
        s = samples_df[c]
        if not (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            continue

        ratio = float(s.astype("string").map(is_tcga_like_barcode).mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = str(c)

    if best_col is not None and best_ratio >= 0.50:
        return best_col

    return None


def convert_starcounts_to_samples(
    df_raw: pd.DataFrame, logger: Logger
) -> Tuple[pd.DataFrame, str]:
    """
    Converte il file in formato con una riga per campione.

    Ritorna:
    - samples_df con colonna esplicita sample_barcode
    - orientamento rilevato
    """
    orientation = detect_orientation(df_raw)
    logger.log(f"[LOAD] orientamento rilevato: {orientation}")

    if orientation == "genes_x_samples":
        gene_col = str(df_raw.columns[0])
        expr = df_raw.set_index(gene_col)
        expr.columns = make_unique_columns([str(c) for c in expr.columns])

        # Trasposizione: da geni x campioni a campioni x geni
        samples_df = expr.transpose().copy()
        samples_df.index.name = "sample_barcode"
        samples_df = samples_df.reset_index()
        return samples_df, orientation

    # Se gia campioni x feature
    samples_df = df_raw.copy()

    # Caso indice con barcode
    idx_ratio = float(
        pd.Series(samples_df.index.astype(str)).map(is_tcga_like_barcode).mean()
    )
    if idx_ratio >= 0.50:
        samples_df = samples_df.reset_index().rename(
            columns={"index": "sample_barcode"}
        )
        return samples_df, orientation

    # Cerca colonna barcode
    barcode_col = find_barcode_column(samples_df)
    if barcode_col is None:
        raise RuntimeError(
            "Impossibile individuare barcode TCGA nel dataset. "
            "Servono barcode per estrarre il dominio site."
        )

    if barcode_col != "sample_barcode":
        samples_df = samples_df.rename(columns={barcode_col: "sample_barcode"})

    return samples_df, orientation


def load_from_single_starcounts_tsv(
    input_path: Path, logger: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carica TCGA-BRCA.star_counts.tsv e produce dataframe con colonne:
    - sample_barcode
    - site
    - feature numeriche

    Ritorna anche due tabelle inventario (valid/invalid) per coerenza con output.
    """
    logger.log(f"[LOAD] lettura TSV: {input_path}")
    df_raw = pd.read_csv(input_path, sep="\t", low_memory=False)
    logger.log(f"[LOAD] shape raw: {df_raw.shape}")

    samples_df, orientation = convert_starcounts_to_samples(df_raw, logger)
    logger.log(f"[LOAD] shape campioni x feature: {samples_df.shape}")

    if "sample_barcode" not in samples_df.columns:
        raise RuntimeError("Colonna sample_barcode assente dopo conversione")

    samples_df["sample_barcode"] = samples_df["sample_barcode"].astype("string")
    samples_df["site"] = (
        samples_df["sample_barcode"].apply(extract_site_from_barcode).astype("string")
    )

    bad_barcode_mask = samples_df["site"].isna()
    n_bad = int(bad_barcode_mask.sum())
    if n_bad > 0:
        logger.log(f"[WARN] barcode non parseabili per site: {n_bad}")

    samples_df = samples_df.loc[~bad_barcode_mask].copy()
    samples_df["source_file"] = input_path.name

    # Conversione feature a numerico (lasciamo fuori colonne meta)
    meta_cols = {"sample_barcode", "site", "source_file"}
    candidate_cols = [c for c in samples_df.columns if c not in meta_cols]

    for c in candidate_cols:
        samples_df[c] = pd.to_numeric(samples_df[c], errors="coerce")

    # Rimuove feature completamente NaN
    feature_cols = [c for c in candidate_cols if samples_df[c].notna().any()]
    dropped_all_nan = len(candidate_cols) - len(feature_cols)
    if dropped_all_nan > 0:
        logger.log(f"[WARN] feature rimosse (tutte NaN): {dropped_all_nan}")

    out_df = samples_df[["sample_barcode", "site", "source_file", *feature_cols]].copy()

    valid_df = pd.DataFrame(
        [
            {
                "file_name": input_path.name,
                "status": "ok",
                "n_rows": len(out_df),
                "n_cols": len(out_df.columns),
                "orientation": orientation,
            }
        ]
    )

    invalid_df = pd.DataFrame(
        [
            {
                "file_name": input_path.name,
                "status": "invalid_barcodes_removed",
                "details": f"n_removed={n_bad}",
            }
        ]
        if n_bad > 0
        else []
    )

    return out_df, valid_df, invalid_df


def load_from_directory(
    input_dir: Path, logger: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fallback robusto: se l'input e una directory, legge CSV/TSV e unisce i file.
    Si aspetta almeno sample_barcode e site (o barcode da cui estrarre site).
    """
    files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.tsv")))
    if not files:
        raise RuntimeError(f"Nessun file CSV/TSV trovato in: {input_dir}")

    valid_frames: List[pd.DataFrame] = []
    valid_rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []

    logger.log(f"[LOAD] file trovati in directory: {len(files)}")

    for fp in files:
        try:
            sep = "\t" if fp.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(fp, sep=sep, low_memory=False)
        except Exception as exc:
            invalid_rows.append(
                {"file_name": fp.name, "status": "read_error", "details": str(exc)}
            )
            continue

        # prova ad allineare nome barcode
        if "sample_barcode" not in df.columns:
            bcol = find_barcode_column(df)
            if bcol is not None:
                df = df.rename(columns={bcol: "sample_barcode"})

        if "sample_barcode" not in df.columns:
            invalid_rows.append(
                {
                    "file_name": fp.name,
                    "status": "missing_sample_barcode",
                    "details": "",
                }
            )
            continue

        if "site" not in df.columns:
            df["site"] = df["sample_barcode"].apply(extract_site_from_barcode)

        df["sample_barcode"] = df["sample_barcode"].astype("string")
        df["site"] = df["site"].astype("string")
        df["source_file"] = fp.name

        # Conversione candidate feature a numerico
        meta_cols = {"sample_barcode", "site", "source_file"}
        candidate_cols = [c for c in df.columns if c not in meta_cols]
        for c in candidate_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        feature_cols = [c for c in candidate_cols if df[c].notna().any()]
        if not feature_cols:
            invalid_rows.append(
                {"file_name": fp.name, "status": "no_numeric_features", "details": ""}
            )
            continue

        df = df[["sample_barcode", "site", "source_file", *feature_cols]].copy()

        valid_frames.append(df)
        valid_rows.append(
            {
                "file_name": fp.name,
                "status": "ok",
                "n_rows": len(df),
                "n_cols": len(df.columns),
            }
        )

    if not valid_frames:
        raise RuntimeError("Nessun file valido in input directory")

    merged = pd.concat(valid_frames, axis=0, ignore_index=True)
    return merged, pd.DataFrame(valid_rows), pd.DataFrame(invalid_rows)


def load_input_data(
    input_path: Path, logger: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Entry point caricamento:
    - se input_path e file -> parser star_counts TSV
    - se input_path e directory -> merge file multipli
    """
    if input_path.is_file():
        return load_from_single_starcounts_tsv(input_path, logger)

    if input_path.is_dir():
        return load_from_directory(input_path, logger)

    raise FileNotFoundError(f"Input non trovato: {input_path}")


# =============================================================================
# Creazione target artificiale (uso globale consentito SOLO qui)
# =============================================================================


def create_target_by_pca_kmeans(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
    logger: Logger,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Costruisce target_cluster usando tutto il dataset in modo UNSUPERVISED.

    Nota importante:
    - Questa e una fase di label engineering artificiale richiesta dal protocollo.
    - Non e training supervisionato del classificatore baseline/transductive.

    Passi:
    1) imputazione mediana globale (per poter fare scaler/PCA senza errori)
    2) StandardScaler globale
    3) PCA globale (n componenti configurabili)
    4) KMeans(k=2) sulle componenti PCA

    Per rispettare la richiesta di classi bilanciate:
    - eseguiamo piu restart di KMeans e scegliamo la soluzione piu bilanciata
    - se il miglior KMeans resta troppo sbilanciato, applichiamo un fallback
      nello stesso spazio PCA usando split mediano della signed-distance tra
      centroidi KMeans (target artificiale 50/50 circa, non banale).
    """
    X = df[feature_cols].copy()

    # Imputiamo solo per consentire PCA/KMeans robusti in presenza di NaN
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_samples, n_features = X_scaled.shape
    n_components = int(min(cfg.pca_components_for_target, n_samples - 1, n_features))
    if n_components < 2:
        raise RuntimeError("Troppi pochi campioni/feature per PCA")

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Ricerca multi-restart: scegliamo il KMeans con miglior bilanciamento.
    candidate_rows: List[Dict[str, object]] = []
    best = None
    best_score = None

    for i in range(cfg.kmeans_restarts_for_balance):
        seed = cfg.random_state + i
        km = KMeans(n_clusters=cfg.kmeans_k, random_state=seed, n_init=20)
        lbl = km.fit_predict(X_pca)

        counts = pd.Series(lbl).value_counts().sort_index()
        if len(counts) < cfg.kmeans_k:
            minority_ratio = 0.0
        else:
            minority_ratio = float(counts.min() / max(1, counts.sum()))

        # score lessicografico: prima bilanciamento, poi inertia
        balance_gap = abs(0.5 - minority_ratio)
        score = (balance_gap, float(km.inertia_))

        candidate_rows.append(
            {
                "restart_id": i,
                "seed": seed,
                "minority_ratio": minority_ratio,
                "inertia": float(km.inertia_),
                "balance_gap_from_0_5": balance_gap,
                "n_cluster_0": int((lbl == 0).sum()),
                "n_cluster_1": int((lbl == 1).sum()),
            }
        )

        if best is None or score < best_score:
            best = (km, lbl)
            best_score = score

    assert best is not None
    kmeans, labels = best

    labels_series = pd.Series(labels)
    counts_best = labels_series.value_counts().sort_index()
    minority_ratio_best = float(counts_best.min() / max(1, counts_best.sum()))

    target_build_method = "kmeans_best_of_restarts"
    balance_adjustment_used = 0

    # Fallback bilanciamento:
    # Se KMeans resta troppo sbilanciato, trasformiamo la signed-distance
    # ai due centroidi in un punteggio e tagliamo a mediana.
    if minority_ratio_best < cfg.min_target_minority_ratio:
        c0 = kmeans.cluster_centers_[0]
        c1 = kmeans.cluster_centers_[1]

        d0 = np.linalg.norm(X_pca - c0, axis=1)
        d1 = np.linalg.norm(X_pca - c1, axis=1)

        signed_distance = d1 - d0
        median_thr = float(np.median(signed_distance))
        labels_balanced = (signed_distance > median_thr).astype(int)

        # Safety fallback: se mediana produce accidentalmente mono-classe.
        if len(np.unique(labels_balanced)) < 2:
            order = np.argsort(signed_distance)
            half = len(order) // 2
            labels_balanced = np.zeros(len(order), dtype=int)
            labels_balanced[order[half:]] = 1

        labels = labels_balanced
        labels_series = pd.Series(labels)
        counts_best = labels_series.value_counts().sort_index()
        minority_ratio_best = float(counts_best.min() / max(1, counts_best.sum()))
        target_build_method = "kmeans_signed_distance_median_split"
        balance_adjustment_used = 1

    # Tabelle diagnostiche per output
    pca_var_df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    pca_scores_df = pd.DataFrame(
        X_pca[:, : min(4, n_components)],
        columns=[f"PC{i + 1}" for i in range(min(4, n_components))],
    )
    pca_scores_df.insert(0, "sample_barcode", df["sample_barcode"].astype(str).values)

    logger.log(
        f"[TARGET] PCA globale per target_cluster: n_components={n_components}, "
        f"var_cumulativa_PC{min(4, n_components)}="
        f"{float(pca_var_df['cumulative_explained_variance'].iloc[min(3, n_components - 1)]):.4f}"
    )
    logger.log(
        f"[TARGET] Metodo target: {target_build_method} | minority_ratio={minority_ratio_best:.3f}"
    )

    kmeans_candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["balance_gap_from_0_5", "inertia"], ascending=[True, True]
    )
    target_info_df = pd.DataFrame(
        [
            {
                "target_build_method": target_build_method,
                "balance_adjustment_used": balance_adjustment_used,
                "minority_ratio_final": minority_ratio_best,
                "min_target_minority_ratio": cfg.min_target_minority_ratio,
                "kmeans_restarts_for_balance": cfg.kmeans_restarts_for_balance,
                "pca_components_used": n_components,
            }
        ]
    )

    return (
        pd.Series(labels, index=df.index, name="target_cluster"),
        pca_var_df,
        pca_scores_df,
        kmeans_candidates_df,
        target_info_df,
    )


# =============================================================================
# Preprocessing leakage-safe nel fold (fit SOLO su source)
# =============================================================================


def fit_source_preprocessor(
    X_source: pd.DataFrame,
    top_n_features: int,
    fold_id: str,
    logger: Logger,
) -> Dict[str, object]:
    """
    Fit del preprocessore SOLO sui source.

    Questo e il punto anti-leakage piu importante del training supervisionato:
    - feature selection per varianza e scaling non devono mai vedere adaptation/test.
    """
    X_num = X_source.apply(pd.to_numeric, errors="coerce")

    # Teniamo solo colonne con almeno un valore non-NaN sui source
    usable_cols = [c for c in X_num.columns if X_num[c].notna().any()]
    if not usable_cols:
        raise RuntimeError(f"[{fold_id}] Nessuna feature numerica utilizzabile")

    X_num = X_num[usable_cols]

    # Imputer fit SOLO su source
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)

    # Top-N per varianza calcolata SOLO su source
    variances = np.var(X_imp, axis=0)
    n_keep = int(min(max(1, top_n_features), X_imp.shape[1]))
    top_idx = np.argsort(variances)[::-1][:n_keep]
    top_idx = np.sort(top_idx)

    selected_features = [usable_cols[i] for i in top_idx]
    X_sel = X_imp[:, top_idx]

    # Scaler fit SOLO su source
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
    X_num = X_df[prep["usable_cols"]].apply(pd.to_numeric, errors="coerce")
    X_imp = prep["imputer"].transform(X_num)
    X_sel = X_imp[:, prep["top_idx"]]
    X_out = prep["scaler"].transform(X_sel)
    return X_out


# =============================================================================
# Metriche
# =============================================================================


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcola le metriche richieste nel task."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


# =============================================================================
# Pseudo-labeling transductive
# =============================================================================


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

    Vincoli rispettati:
    - adaptation senza vere label
    - target_test mai usato nel training
    - selezione solo campioni ad alta confidenza
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

        # Limite massimo aggiunte per iterazione
        cap_by_fraction = max(1, int(math.ceil(max_pool_fraction * pool_n)))
        cap = min(max_per_iter, cap_by_fraction, n_candidates)

        sorted_candidates = high_conf_idx[np.argsort(conf[high_conf_idx])[::-1]]
        chosen_idx = np.sort(sorted_candidates[:cap])

        X_sel = X_pool[chosen_idx]
        y_sel = pred[chosen_idx].astype(int)

        # Aggiornamento training con pseudo-label
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
# Fold runner (LODO)
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

    # Controlli anti-leakage sugli ID dei campioni
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
        f"[DIST][{fold_id}] source: {class_dist_text(source_df['target_cluster'])}"
    )
    logger.log(
        f"[DIST][{fold_id}] adaptation (solo diagnostica): {class_dist_text(adapt_df['target_cluster'])}"
    )
    logger.log(f"[DIST][{fold_id}] test: {class_dist_text(test_df['target_cluster'])}")

    if source_df["target_cluster"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] source mono-classe")
    if test_df["target_cluster"].nunique() < 2:
        raise RuntimeError(f"[{fold_id}] target_test mono-classe")

    X_source = source_df[feature_cols]
    y_source = source_df["target_cluster"].to_numpy(dtype=int)

    X_adapt = adapt_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["target_cluster"].to_numpy(dtype=int)

    # PREPROCESS FIT SOLO SU SOURCE (cruciale anti-leakage)
    prep = fit_source_preprocessor(X_source, cfg.top_n_features, fold_id, logger)
    logger.log(f"[CHECK][{fold_id}] preprocess fit SOLO su source")

    Xs = prep["X_source_final"]
    Xa = transform_with_source_preprocessor(X_adapt, prep)
    Xt = transform_with_source_preprocessor(X_test, prep)

    # Baseline source-only
    baseline = build_lr(cfg.random_state)
    baseline.fit(Xs, y_source)
    ypred_b = baseline.predict(Xt)
    m_b = compute_binary_metrics(y_test, ypred_b)

    # Transductive con pseudo-labeling
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

    # Sanity check: la dimensione train finale deve essere source + pseudo aggiunti
    expected_final = int(len(y_source) + pseudo_total)
    if final_train_size != expected_final:
        raise RuntimeError(f"[LEAKAGE ERROR][{fold_id}] train_size incoerente")

    ypred_t = trans_model.predict(Xt)
    m_t = compute_binary_metrics(y_test, ypred_t)

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
    }

    return rows, pseudo_rows, conf_df, diag


# =============================================================================
# Aggregazioni e plot
# =============================================================================


def build_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for method in ["baseline", "transductive"]:
        sub = fold_results.loc[fold_results["method"] == method]
        if sub.empty:
            continue

        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(vals.mean(skipna=True)),
                    "std": float(vals.std(skipna=True, ddof=1)),
                    "n_valid": int(vals.notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def build_comparison(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouped = fold_results.groupby(["fold_id", "target_site"], dropna=False)
    for (fold_id, site), grp in grouped:
        b = grp.loc[grp["method"] == "baseline"]
        t = grp.loc[grp["method"] == "transductive"]

        if b.empty or t.empty:
            continue

        b0 = b.iloc[0]
        t0 = t.iloc[0]

        for metric in METRICS:
            bv = b0[metric]
            tv = t0[metric]
            dv = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
            rows.append(
                {
                    "fold_id": fold_id,
                    "target_site": site,
                    "metric": metric,
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

    n_metrics = len(METRICS)
    ncols = 2
    nrows = int(math.ceil(n_metrics / ncols))

    fig_w = max(14, 8 + 0.4 * len(sites))
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
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
        ax.grid(axis="y", alpha=0.30)

    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2)

    fig.suptitle(
        "Baseline vs Transductive per site target (target_cluster da PCA+KMeans)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
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

    fig, ax = plt.subplots(figsize=(10, 5.2))
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

    n_metrics = len(METRICS)
    ncols = 2
    nrows = int(math.ceil(n_metrics / ncols))

    fig_w = max(14, 8 + 0.4 * len(sites))
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
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

    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Delta metriche (transductive - baseline)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
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


def plot_target_cluster_balance(target_counts: pd.Series, output_path: Path) -> None:
    if target_counts.empty:
        return

    classes = target_counts.index.astype(str).tolist()
    values = target_counts.values.astype(float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(classes, values, color=["#1f77b4", "#ff7f0e"])
    ax.set_xlabel("target_cluster")
    ax.set_ylabel("n campioni")
    ax.set_title("Distribuzione classi target_cluster")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(
            i, v + max(values) * 0.01, f"{int(v)}", ha="center", va="bottom", fontsize=9
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Report interpretativo
# =============================================================================


def detect_easy_task_warning(fold_results: pd.DataFrame) -> Tuple[bool, str]:
    if fold_results.empty:
        return False, "nessun fold"

    base = fold_results.loc[fold_results["method"] == "baseline"].copy()
    if base.empty:
        return False, "nessuna baseline"

    acc = pd.to_numeric(base["accuracy"], errors="coerce")
    frac = float((acc > 0.95).mean()) if acc.notna().any() else 0.0

    warning = frac >= 0.80
    msg = f"baseline_accuracy>0.95_fraction={frac:.2f}"
    return warning, msg


def write_interpretation(
    output_path: Path,
    n_files_valid: int,
    n_files_invalid: int,
    n_samples_total: int,
    n_samples_after_site_filter: int,
    valid_sites: List[str],
    site_stats: pd.DataFrame,
    target_counts: pd.Series,
    fold_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    warnings_list: List[str],
) -> None:
    lines: List[str] = []

    lines.append("TCGA PCA+KMeans - Baseline vs Transductive")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Perche clustering + PCA:")
    lines.append("- evita task predefiniti troppo facili (es. 01 vs 11)")
    lines.append(
        "- usa struttura latente dell'espressione genica per definire un task piu informativo"
    )
    lines.append("- permette di testare la transportability su un target non banale")
    lines.append("")

    lines.append("Nota metodologica cruciale:")
    lines.append(
        "- StandardScaler + PCA + KMeans globali usati SOLO per costruire target_cluster"
    )
    lines.append(
        "- nei fold supervisionati, preprocess/feature selection fit solo sui source"
    )
    lines.append("- target_adaptation senza label vere nel training transductive")
    lines.append("- target_test completamente unseen fino alla valutazione finale")
    lines.append("")

    lines.append(f"File validi letti: {n_files_valid}")
    lines.append(f"File invalidi/skippati: {n_files_invalid}")
    lines.append(f"Campioni totali iniziali: {n_samples_total}")
    lines.append(f"Campioni dopo filtro site validi: {n_samples_after_site_filter}")
    lines.append("")

    lines.append(f"Site validi: {len(valid_sites)}")
    lines.append(f"Elenco site validi: {valid_sites}")
    lines.append("")

    if not target_counts.empty:
        lines.append("Distribuzione target_cluster globale:")
        for k, v in target_counts.sort_index().items():
            lines.append(f"- cluster {k}: {int(v)}")
        lines.append("")

    if not site_stats.empty:
        lines.append("Distribuzione per site (prime 20 righe):")
        lines.append(site_stats.head(20).to_string(index=False))
        lines.append("")

    if fold_results.empty:
        lines.append("Nessun fold valutabile.")
        if warnings_list:
            lines.append("")
            lines.append("Warning diagnostici:")
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
        lines.append("Warning diagnostici:")
        for w in warnings_list:
            lines.append(f"- {w}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCGA PCA+KMeans transductive learning (LODO)"
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)

    parser.add_argument(
        "--pca-components-for-target", type=int, default=PCA_COMPONENTS_FOR_TARGET
    )
    parser.add_argument("--kmeans-k", type=int, default=KMEANS_K)
    parser.add_argument(
        "--kmeans-restarts-for-balance", type=int, default=KMEANS_RESTARTS_FOR_BALANCE
    )
    parser.add_argument(
        "--min-target-minority-ratio", type=float, default=MIN_TARGET_MINORITY_RATIO
    )

    parser.add_argument("--min-site-samples", type=int, default=MIN_SITE_SAMPLES)
    parser.add_argument("--min-valid-sites", type=int, default=MIN_VALID_SITES)
    parser.add_argument("--target-test-size", type=float, default=TARGET_TEST_SIZE)

    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)

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
        logger.log(f"[WARN] Impossibile copiare script: {exc}")


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    cfg = Config(
        input_path=args.input_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        pca_components_for_target=args.pca_components_for_target,
        kmeans_k=args.kmeans_k,
        kmeans_restarts_for_balance=args.kmeans_restarts_for_balance,
        min_target_minority_ratio=args.min_target_minority_ratio,
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

    logger.log("[START] TCGA PCA+KMeans baseline vs transductive")
    logger.log(f"[CONFIG] input_path={cfg.input_path}")
    logger.log(f"[CONFIG] output_dir={cfg.output_dir}")

    # ------------------------------------------------------------------
    # 1) Caricamento dati
    # ------------------------------------------------------------------
    merged, valid_files_df, invalid_files_df = load_input_data(cfg.input_path, logger)

    n_files_valid = len(valid_files_df)
    n_files_invalid = len(invalid_files_df)

    valid_files_df.to_csv(cfg.output_dir / "loaded_files_valid.csv", index=False)
    invalid_files_df.to_csv(cfg.output_dir / "loaded_files_invalid.csv", index=False)

    logger.log(f"[LOAD] File validi={n_files_valid} | invalidi={n_files_invalid}")
    logger.log(f"[DATA] Shape iniziale: {merged.shape}")

    # Definizione feature numeriche (geni)
    feature_cols = [
        c for c in merged.columns if c not in {"sample_barcode", "site", "source_file"}
    ]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]

    if not feature_cols:
        raise RuntimeError("Nessuna feature numerica disponibile")

    logger.log(f"[FEATURE] Numero feature numeriche: {len(feature_cols)}")

    n_samples_total = len(merged)

    # ------------------------------------------------------------------
    # 2-3) Preprocessing globale SOLO per creare target artificiale
    # ------------------------------------------------------------------
    logger.log(
        "[TARGET] Costruzione target_cluster via StandardScaler + PCA + KMeans (uso globale consentito)"
    )

    (
        target_cluster,
        pca_var_df,
        pca_scores_df,
        kmeans_candidates_df,
        target_info_df,
    ) = create_target_by_pca_kmeans(
        df=merged,
        feature_cols=feature_cols,
        cfg=cfg,
        logger=logger,
    )

    merged = merged.copy()
    merged["target_cluster"] = target_cluster.astype(int)

    target_counts = merged["target_cluster"].value_counts().sort_index()
    merged["target_cluster"].value_counts().sort_index().rename("count").to_csv(
        cfg.output_dir / "target_cluster_distribution.csv", header=True
    )
    plot_target_cluster_balance(
        target_counts, cfg.output_dir / "plot_target_cluster_balance.png"
    )

    pca_var_df.to_csv(cfg.output_dir / "pca_explained_variance.csv", index=False)
    pca_scores_df.to_csv(cfg.output_dir / "pca_scores_preview.csv", index=False)
    kmeans_candidates_df.to_csv(
        cfg.output_dir / "kmeans_balance_candidates.csv", index=False
    )
    target_info_df.to_csv(cfg.output_dir / "target_cluster_build_info.csv", index=False)

    logger.log(f"[TARGET] Distribuzione target_cluster: {target_counts.to_dict()}")

    # Bilanciamento target (diagnostica)
    warnings_list: List[str] = []
    if len(target_counts) == 2:
        minority_ratio = float(
            min(target_counts.values) / max(1, sum(target_counts.values))
        )
        if minority_ratio < 0.30:
            msg = f"Target cluster sbilanciato (minority_ratio={minority_ratio:.3f})"
            warnings_list.append(msg)
            logger.log(f"[WARNING] {msg}")

    # Distribuzione cluster per site
    site_cluster_dist = (
        merged.groupby("site", dropna=False)["target_cluster"]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
    )
    site_cluster_dist.to_csv(
        cfg.output_dir / "site_target_cluster_distribution.csv", index=False
    )

    # ------------------------------------------------------------------
    # 4) Filtraggio site validi
    # ------------------------------------------------------------------
    site_rows: List[Dict[str, object]] = []
    for site, grp in merged.groupby("site", dropna=False):
        n_total = int(len(grp))
        n_c0 = int((grp["target_cluster"] == 0).sum())
        n_c1 = int((grp["target_cluster"] == 1).sum())
        n_classes = int(grp["target_cluster"].nunique())

        valid = pd.notna(site) and n_total >= cfg.min_site_samples and n_classes == 2

        site_rows.append(
            {
                "site": site,
                "n_samples": n_total,
                "n_cluster0": n_c0,
                "n_cluster1": n_c1,
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

    logger.log(f"[SITE] Site validi={len(valid_sites)}")
    logger.log(f"[SITE] Elenco site validi: {valid_sites}")

    if len(valid_sites) < cfg.min_valid_sites:
        msg = (
            f"Site validi insufficienti per LODO informativo: "
            f"{len(valid_sites)} < {cfg.min_valid_sites}"
        )
        warnings_list.append(msg)
        logger.log(f"[STOP] {msg}")

        write_interpretation(
            output_path=cfg.output_dir / "interpretation.md",
            n_files_valid=n_files_valid,
            n_files_invalid=n_files_invalid,
            n_samples_total=n_samples_total,
            n_samples_after_site_filter=0,
            valid_sites=valid_sites,
            site_stats=site_stats,
            target_counts=target_counts,
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
    filtered = merged.loc[merged["site"].astype(str).isin(valid_sites)].copy()
    filtered = filtered.reset_index(drop=True)
    filtered["row_id"] = np.arange(len(filtered), dtype=int)

    n_samples_after_site_filter = len(filtered)
    logger.log(f"[DATA] Shape dopo filtro site validi: {filtered.shape}")

    # ------------------------------------------------------------------
    # 5) LODO + split target (adaptation/test)
    # 6-8) Preprocessing leakage-safe + baseline + transductive
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
        elif source_df["target_cluster"].nunique() < 2:
            reason = "source_single_class"
        elif target_df["target_cluster"].nunique() < 2:
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

        # Split target -> adaptation + test
        strat = (
            target_df["target_cluster"]
            if can_stratify(target_df["target_cluster"])
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

        if test_df["target_cluster"].nunique() < 2:
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
                feature_cols=feature_cols,
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

    if n_folds_evaluated == 0:
        msg = "Nessun fold valutabile dopo i controlli"
        warnings_list.append(msg)
        logger.log(f"[STOP] {msg}")

        diagnostics_df.to_csv(cfg.output_dir / "diagnostics.csv", index=False)

        write_interpretation(
            output_path=cfg.output_dir / "interpretation.md",
            n_files_valid=n_files_valid,
            n_files_invalid=n_files_invalid,
            n_samples_total=n_samples_total,
            n_samples_after_site_filter=n_samples_after_site_filter,
            valid_sites=valid_sites,
            site_stats=site_stats,
            target_counts=target_counts,
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
    # 9) Metriche finali
    # ------------------------------------------------------------------
    summary_df = build_summary(fold_results)
    comparison_df = build_comparison(fold_results)

    easy_warning, easy_msg = detect_easy_task_warning(fold_results)
    if easy_warning:
        msg = f"Task ancora potenzialmente facile: {easy_msg}"
        warnings_list.append(msg)
        logger.log(f"[WARNING] {msg}")

    # ------------------------------------------------------------------
    # 10) Salvataggi output richiesti
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
        output_path=cfg.output_dir / "interpretation.md",
        n_files_valid=n_files_valid,
        n_files_invalid=n_files_invalid,
        n_samples_total=n_samples_total,
        n_samples_after_site_filter=n_samples_after_site_filter,
        valid_sites=valid_sites,
        site_stats=site_stats,
        target_counts=target_counts,
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
