#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCI Heart Disease - Transductive learning leakage-safe sui 4 site originali separati.

Questo script e stato scritto per una pipeline metodologicamente pulita in scenario multi-domain.

Perche partire dai 4 file separati (site) e meglio di partire da un file gia unito:
- Manteniamo l'informazione di dominio alla fonte, senza ricostruzioni post-hoc.
- Possiamo implementare Leave-One-Domain-Out in modo trasparente e riproducibile.
- Evitiamo semplificazioni che possono nascondere shift tra center reali.

Concetti chiave usati nel protocollo:
- source: tutti i site tranne quello scelto come target nel fold corrente.
- target_adaptation: parte del site target usata solo come unlabeled nel transductive learning.
- target_test: parte del site target usata solo per valutazione finale, totalmente unseen.

Nota anti-leakage fondamentale:
- Se usassimo tutto il target sia per adattamento sia per test, introdurremmo leakage.
- Qui, target_test resta sempre separato e non entra mai nel training.
- Nel transductive learning non si usano mai le label vere di target_adaptation durante il fit.
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
import io
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Backend non interattivo per eseguire lo script anche in ambienti headless.
# Cache matplotlib in una directory tipicamente scrivibile anche in sandbox/CI.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# Configurazione globale
# ============================================================================

# Nomi canonici delle 14 colonne classiche UCI Heart Disease.
HEART_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Mapping 1-based dagli attributi raw UCI (formato multi-line) alle 14 colonne classiche.
# Questo mapping e ampiamente usato nella letteratura per ricondurre il raw format al subset classico.
RAW_TO_CLASSIC_INDEX_1BASED: List[int] = [
    3,
    4,
    9,
    10,
    12,
    16,
    19,
    32,
    38,
    40,
    41,
    44,
    51,
    58,
]

# Definizione esplicita delle feature numeriche e categoriche.
# Nota: non includiamo target e site nelle feature predittive.
# Richiesta metodologica specifica:
# - `ca` viene trattata come categorica (codifica discreta), non come numerica continua.
NUMERIC_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

METRIC_COLUMNS: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


@dataclass
class ExperimentConfig:
    """Parametri principali dell'esperimento."""

    input_files: Dict[str, Path]
    output_dir: Path
    random_state: int = 42
    target_test_size: float = 0.50
    pseudo_threshold: float = 0.95
    pseudo_max_iters: int = 10


class RunLogger:
    """
    Logger semplice: salva ogni messaggio sia a schermo sia in memoria,
    cosi possiamo scrivere un log completo a fine esecuzione.
    """

    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def save(self, output_path: Path) -> None:
        output_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ============================================================================
# Utility generiche
# ============================================================================


def safe_filename(value: str) -> str:
    """Converte una stringa in formato safe per nome file."""
    value = value.strip().replace(" ", "_")
    value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return value[:120]


def class_distribution_text(y: pd.Series) -> str:
    """Rende leggibile la distribuzione classi per logging diagnostico."""
    counts = y.value_counts(dropna=False).sort_index()
    parts = [f"class_{idx}={int(val)}" for idx, val in counts.items()]
    return ", ".join(parts)


def assert_no_overlap(ids_a: pd.Series, ids_b: pd.Series, context: str) -> None:
    """
    Check anti-leakage: nessun id deve comparire in entrambi i set.
    Se compare, fermiamo l'esecuzione per evitare risultati non affidabili.
    """
    overlap = set(ids_a.tolist()).intersection(set(ids_b.tolist()))
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE ERROR] Overlap trovato in '{context}'. "
            f"Numero campioni in overlap: {len(overlap)}"
        )


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Compatibilita tra versioni sklearn:
    - versioni nuove: sparse_output
    - versioni vecchie: sparse
    """
    params = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in signature.parameters:
        params["sparse_output"] = True
    else:
        params["sparse"] = True
    return OneHotEncoder(**params)


def build_preprocessor() -> ColumnTransformer:
    """
    Preprocessing leakage-safe:
    - imputazione mediana sulle numeriche
    - imputazione moda sulle categoriche
    - one-hot encoding delle categoriche
    - standardizzazione numeriche

    Importante: questo preprocessore verra fittato SOLO sui source in ogni fold.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_model(random_state: int) -> LogisticRegression:
    """
    Modello baseline richiesto: Logistic Regression con class_weight='balanced'.
    """
    return LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def stack_rows(X_a, X_b):
    """Stack robusto per matrici dense o sparse."""
    if sparse.issparse(X_a) or sparse.issparse(X_b):
        return sparse.vstack([sparse.csr_matrix(X_a), sparse.csr_matrix(X_b)])
    return np.vstack([X_a, X_b])


def ensure_output_dir(output_dir: Path) -> None:
    """Crea la cartella di output se non esiste."""
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Parsing file UCI per site
# ============================================================================


def decode_text_with_fallback(file_path: Path) -> Tuple[str, str]:
    """
    Decodifica robusta (utile per file raw con caratteri sporchi).
    """
    raw_bytes = file_path.read_bytes()
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # Ultimo fallback permissivo: non blocchiamo l'analisi, ma segnaliamo nel log.
    return raw_bytes.decode("latin1", errors="ignore"), "latin1-ignore"


def try_parse_as_14col_csv(text: str) -> Tuple[pd.DataFrame | None, Dict[str, object]]:
    """
    Prova a leggere il file come CSV a 14 colonne (caso processed classico).
    Se non e quel formato, ritorna None.
    """
    info: Dict[str, object] = {"parser": "csv_14"}
    try:
        df = pd.read_csv(
            io.StringIO(text), header=None, dtype=str, sep=",", engine="python"
        )
    except Exception as exc:
        info["status"] = "failed"
        info["reason"] = str(exc)
        return None, info

    if df.shape[1] < len(HEART_COLUMNS):
        info["status"] = "not_14col"
        info["n_columns"] = int(df.shape[1])
        return None, info

    if df.shape[1] > len(HEART_COLUMNS):
        info["status"] = "trimmed_extra_columns"
        info["n_columns_original"] = int(df.shape[1])
        df = df.iloc[:, : len(HEART_COLUMNS)].copy()
    else:
        info["status"] = "ok"

    df.columns = HEART_COLUMNS
    return df, info


def parse_raw_uci_to_classic_14(text: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Parsing del formato raw UCI multi-line:
    - i record sono separati dal token 'name'
    - ogni record raw contiene molte piu di 14 colonne
    - estraiamo le 14 colonne classiche via mapping raw->classic

    Robustezza:
    - record corti: non li scartiamo in modo arbitrario; estraiamo quello che possiamo
      e mettiamo NaN dove mancano colonne.
    - token non numerici: vengono ignorati e conteggiati nel log.
    """
    numeric_token_re = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

    records: List[List[str]] = []
    current_record: List[str] = []
    invalid_tokens = 0

    for raw_line in text.splitlines():
        # Normalizzazione prudente: teniamo ASCII stampabile e spazi/tab.
        clean_line = "".join(
            ch if ch in {" ", "\t"} or 32 <= ord(ch) <= 126 else " " for ch in raw_line
        )

        # Rende robusto il delimitatore record e i segni '-' incollati a numeri.
        clean_line = re.sub(r"(?i)name", " name ", clean_line)
        clean_line = re.sub(r"(?<=\d)-(?=\d)", " -", clean_line)

        for token in clean_line.split():
            tok = token.strip().strip(",")
            if not tok:
                continue
            if tok.lower() == "name":
                if current_record:
                    records.append(current_record)
                    current_record = []
                continue
            if numeric_token_re.match(tok):
                current_record.append(tok)
            else:
                invalid_tokens += 1

    if current_record:
        records.append(current_record)

    max_idx = max(RAW_TO_CLASSIC_INDEX_1BASED)
    extracted_rows: List[List[object]] = []
    short_records = 0

    for rec in records:
        if len(rec) < max_idx:
            short_records += 1

        # Estrazione safe: se manca un indice, mettiamo NaN.
        extracted = []
        for idx_1based in RAW_TO_CLASSIC_INDEX_1BASED:
            idx_0based = idx_1based - 1
            if idx_0based < len(rec):
                extracted.append(rec[idx_0based])
            else:
                extracted.append(np.nan)
        extracted_rows.append(extracted)

    df = pd.DataFrame(extracted_rows, columns=HEART_COLUMNS)
    info = {
        "parser": "raw_multiline",
        "records_found": len(records),
        "rows_extracted": len(extracted_rows),
        "short_records_padded_with_nan": short_records,
        "invalid_tokens_ignored": invalid_tokens,
    }
    return df, info


def load_single_site(
    file_path: Path, site_name: str, logger: RunLogger
) -> pd.DataFrame:
    """
    Carica un singolo file site con parser robusto:
    1) tenta parse come 14-col CSV
    2) se non valido, parse raw multi-line
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato per site '{site_name}': {file_path}")

    text, encoding = decode_text_with_fallback(file_path)

    df_14, csv_info = try_parse_as_14col_csv(text)
    if df_14 is not None:
        df = df_14
        parse_info = csv_info
        parse_mode = "csv_14"
    else:
        df, raw_info = parse_raw_uci_to_classic_14(text)
        parse_info = raw_info
        parse_info["csv_attempt"] = csv_info
        parse_mode = "raw_to_classic_14"

    df["site"] = site_name

    logger.log(
        f"[LOAD] site={site_name} rows={len(df)} cols={df.shape[1]} "
        f"mode={parse_mode} encoding={encoding} path={file_path}"
    )
    logger.log(f"[LOAD-DETAILS] {site_name}: {parse_info}")
    return df


def load_all_sites(input_files: Dict[str, Path], logger: RunLogger) -> pd.DataFrame:
    """
    Legge i 4 file separati e costruisce un unico dataframe interno mantenendo `site`.
    """
    frames: List[pd.DataFrame] = []

    for site_name, site_path in input_files.items():
        frames.append(load_single_site(site_path, site_name, logger))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.log(f"[LOAD] Dataset unificato interno creato: shape={combined.shape}")
    return combined


# ============================================================================
# Cleaning e preparazione target
# ============================================================================


def clean_and_prepare_dataset(df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    """
    Pulizia iniziale conforme ai vincoli:
    - '?' -> NaN
    - conversione numerica robusta delle colonne canoniche
    - target binario da num (0->0, >0->1)
    - mai imputare target/site

    Nota: nei file raw UCI possono comparire anche codifiche missing tipo -9.
    Le trattiamo come NaN per evitare distorsioni nelle statistiche.
    """
    clean_df = df.copy()

    # Strip su eventuali colonne stringa.
    for col in clean_df.columns:
        if clean_df[col].dtype == object:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    # Missing tokens espliciti.
    clean_df = clean_df.replace(
        {
            "?": np.nan,
            "": np.nan,
            "-9": np.nan,
            "-9.0": np.nan,
            "-9.00": np.nan,
        }
    )

    # Conversione delle 14 colonne in numerico (errori -> NaN).
    for col in HEART_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Costruzione target binario senza imputazione della variabile di output.
    clean_df["target"] = np.where(
        clean_df["num"] > 0,
        1,
        np.where(clean_df["num"] == 0, 0, np.nan),
    )

    missing_target = int(clean_df["target"].isna().sum())
    if missing_target > 0:
        logger.log(
            f"[CLEAN] Righe con target mancante (num non disponibile): {missing_target}. "
            "Queste righe vengono escluse solo dalla modellazione perche la label non esiste."
        )

    # Rimuoviamo solo dove target e assente (non possiamo supervisionare senza label).
    clean_df = clean_df.loc[clean_df["target"].notna()].copy()
    clean_df["target"] = clean_df["target"].astype(int)

    # `site` resta un'informazione di dominio, non viene imputata.
    clean_df["site"] = clean_df["site"].astype(str)

    # ID univoco per controlli anti-leakage tra split.
    clean_df = clean_df.reset_index(drop=True)
    clean_df["row_id"] = np.arange(clean_df.shape[0], dtype=int)

    logger.log(f"[CLEAN] Dataset pronto per esperimento: shape={clean_df.shape}")
    logger.log(
        "[CLEAN] Distribuzione target globale: "
        + class_distribution_text(clean_df["target"])
    )
    return clean_df


# ============================================================================
# Metriche e plotting
# ============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Calcolo metriche richieste.
    ROC-AUC puo non essere calcolabile in fold con una sola classe nel test.
    """
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


def save_confusion_matrix_png(cm: np.ndarray, title: str, output_path: Path) -> None:
    """Salva confusion matrix in PNG."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_baseline_vs_transductive_by_site(
    fold_results_df: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
) -> None:
    """
    Barplot baseline vs transductive per site target.
    Per leggibilita mostriamo tutte le metriche in sottografi.
    """
    sites = sorted(fold_results_df["target_site"].unique().tolist())
    methods = ["baseline", "transductive"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(sites), dtype=float)
        width = 0.36

        for method_idx, method_name in enumerate(methods):
            values = []
            for site in sites:
                match = fold_results_df[
                    (fold_results_df["target_site"] == site)
                    & (fold_results_df["method"] == method_name)
                ]
                if match.empty:
                    values.append(np.nan)
                else:
                    values.append(float(match.iloc[0][metric]))

            offset = (-0.5 * width) if method_idx == 0 else (0.5 * width)
            label = method_name if metric == metrics[0] else None
            ax.bar(x + offset, values, width=width, label=label, alpha=0.9)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("Baseline vs Transductive per site target", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mean_metrics(
    mean_df: pd.DataFrame, metrics: List[str], output_path: Path
) -> None:
    """Plot finale delle metriche medie baseline vs transductive."""
    pivot_df = mean_df.pivot(index="metric", columns="method", values="mean").reindex(
        metrics
    )

    x = np.arange(len(metrics), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(
        x - width / 2, pivot_df.get("baseline", np.nan), width=width, label="baseline"
    )
    ax.bar(
        x + width / 2,
        pivot_df.get("transductive", np.nan),
        width=width,
        label="transductive",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean score")
    ax.set_title("Metriche medie finali")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _extract_precision_by_site(fold_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae una tabella compatta per il focus principale richiesto:
    precision baseline vs precision transductive per ogni site target.
    """
    rows: List[Dict[str, float | str]] = []
    for target_site in sorted(fold_results_df["target_site"].unique().tolist()):
        base = fold_results_df[
            (fold_results_df["target_site"] == target_site)
            & (fold_results_df["method"] == "baseline")
        ]
        trans = fold_results_df[
            (fold_results_df["target_site"] == target_site)
            & (fold_results_df["method"] == "transductive")
        ]
        if base.empty or trans.empty:
            continue

        precision_baseline = float(base.iloc[0]["precision"])
        precision_transductive = float(trans.iloc[0]["precision"])
        rows.append(
            {
                "target_site": target_site,
                "precision_baseline": precision_baseline,
                "precision_transductive": precision_transductive,
                "delta_precision": precision_transductive - precision_baseline,
            }
        )

    return pd.DataFrame(rows)


def plot_precision_by_site(precision_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot principale richiesto:
    - asse x: site target
    - due barre: baseline precision vs transductive precision
    """
    sites = precision_df["target_site"].tolist()
    x = np.arange(len(sites), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_vals = precision_df["precision_baseline"].to_numpy(dtype=float)
    trans_vals = precision_df["precision_transductive"].to_numpy(dtype=float)

    bars_baseline = ax.bar(x - width / 2, baseline_vals, width=width, label="baseline")
    bars_trans = ax.bar(x + width / 2, trans_vals, width=width, label="transductive")

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("precision")
    ax.set_title("Precision per site target: baseline vs transductive")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Etichette valore sopra le barre per leggibilita immediata.
    for bar in list(bars_baseline) + list(bars_trans):
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_mean_aggregate(
    precision_df: pd.DataFrame, output_path: Path
) -> None:
    """
    Plot aggregato richiesto:
    - precision media baseline
    - precision media transductive
    """
    mean_baseline = float(precision_df["precision_baseline"].mean())
    mean_transductive = float(precision_df["precision_transductive"].mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["baseline_mean", "transductive_mean"]
    values = [mean_baseline, mean_transductive]
    bars = ax.bar(labels, values, width=0.55)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("precision")
    ax.set_title("Precision media: baseline vs transductive")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_delta_by_site(precision_df: pd.DataFrame, output_path: Path) -> None:
    """
    Terzo plot richiesto:
    delta precision per site = precision_transductive - precision_baseline.
    """
    sites = precision_df["target_site"].tolist()
    delta = precision_df["delta_precision"].to_numpy(dtype=float)
    x = np.arange(len(sites), dtype=float)
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(x, delta, color=colors, width=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20)
    ax.set_ylabel("delta precision (transductive - baseline)")
    ax.set_title("Delta precision per site target")
    ax.grid(axis="y", alpha=0.3)

    # Limiti dinamici ma sempre simmetrici rispetto allo zero.
    max_abs = max(0.02, float(np.max(np.abs(delta))) if len(delta) else 0.02)
    ax.set_ylim(-max_abs * 1.35, max_abs * 1.35)

    for bar, value in zip(bars, delta):
        offset = 0.005 if value >= 0 else -0.005
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================================
# Transductive learning (pseudo-labeling iterativo)
# ============================================================================


def run_transductive_pseudo_labeling(
    X_source,
    y_source: np.ndarray,
    X_target_unlabeled,
    pseudo_threshold: float,
    pseudo_max_iters: int,
    random_state: int,
    logger: RunLogger,
    fold_name: str,
) -> Tuple[LogisticRegression, List[Dict[str, int]], int, int]:
    """
    Procedura transductive richiesta (manuale, iterativa):
    1) train iniziale su source labeled
    2) predizione su target_adaptation unlabeled
    3) selezione campioni confidenti
    4) aggiunta pseudo-label al training
    5) iterazione fino a max iter o stop

    Vincolo rispettato: le label vere di target_adaptation NON entrano mai nel fit.
    """
    # Set di training iniziale = solo source labeled.
    X_train = X_source
    y_train = np.asarray(y_source, dtype=int)

    # Pool unlabeled = target_adaptation trasformato con preprocessore source-fitted.
    X_unlabeled = X_target_unlabeled

    pseudo_history: List[Dict[str, int]] = []
    total_pseudo_added = 0

    for iteration in range(1, pseudo_max_iters + 1):
        model = build_logistic_model(random_state + iteration)
        model.fit(X_train, y_train)

        n_pool_before = int(X_unlabeled.shape[0])
        if n_pool_before == 0:
            pseudo_history.append(
                {
                    "iteration": iteration,
                    "n_pool_before": 0,
                    "n_added": 0,
                    "class_0_added": 0,
                    "class_1_added": 0,
                }
            )
            logger.log(f"[PSEUDO][{fold_name}] Iter {iteration}: pool vuoto, stop.")
            break

        # Predizione probabilistica sui campioni unlabeled.
        proba = model.predict_proba(X_unlabeled)
        max_conf = proba.max(axis=1)
        pseudo_label = proba.argmax(axis=1).astype(int)

        selected_mask = max_conf >= pseudo_threshold
        n_added = int(selected_mask.sum())

        if n_added == 0:
            pseudo_history.append(
                {
                    "iteration": iteration,
                    "n_pool_before": n_pool_before,
                    "n_added": 0,
                    "class_0_added": 0,
                    "class_1_added": 0,
                }
            )
            logger.log(
                f"[PSEUDO][{fold_name}] Iter {iteration}: nessun campione sopra soglia "
                f"{pseudo_threshold:.2f}, stop."
            )
            break

        X_selected = X_unlabeled[selected_mask]
        y_selected = pseudo_label[selected_mask]

        # Aggiorna training set con pseudo-label confidenti.
        X_train = stack_rows(X_train, X_selected)
        y_train = np.concatenate([y_train, y_selected], axis=0)

        # Rimuove i campioni appena pseudo-labelati dal pool unlabeled.
        X_unlabeled = X_unlabeled[~selected_mask]

        class_counts = pd.Series(y_selected).value_counts().to_dict()
        class_0_added = int(class_counts.get(0, 0))
        class_1_added = int(class_counts.get(1, 0))

        total_pseudo_added += n_added
        pseudo_history.append(
            {
                "iteration": iteration,
                "n_pool_before": n_pool_before,
                "n_added": n_added,
                "class_0_added": class_0_added,
                "class_1_added": class_1_added,
            }
        )

        logger.log(
            f"[PSEUDO][{fold_name}] Iter {iteration}: added={n_added}, "
            f"class_0={class_0_added}, class_1={class_1_added}, "
            f"pool_restante={int(X_unlabeled.shape[0])}"
        )

    # Refit finale sul training arricchito (source + pseudo-label accumulati).
    final_model = build_logistic_model(random_state + 999)
    final_model.fit(X_train, y_train)

    final_train_size = int(y_train.shape[0])
    return final_model, pseudo_history, total_pseudo_added, final_train_size


# ============================================================================
# Esperimento Leave-One-Domain-Out
# ============================================================================


def run_lodo_experiment(
    df: pd.DataFrame, config: ExperimentConfig, logger: RunLogger
) -> Dict[str, pd.DataFrame]:
    """
    Esegue il protocollo Leave-One-Domain-Out corretto.

    Per ogni site target:
    - source = tutti gli altri site
    - split target in target_adaptation + target_test
    - baseline source-only e transductive pseudo-labeling
    - valutazione solo su target_test
    """
    fold_rows: List[Dict[str, object]] = []
    pseudo_rows: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []

    domain_order = list(config.input_files.keys())

    for fold_idx, target_site in enumerate(domain_order, start=1):
        fold_name = f"fold_{fold_idx}_{target_site}"
        logger.log("\n" + "=" * 95)
        logger.log(f"[FOLD] {fold_name}")
        logger.log("=" * 95)

        source_df = df.loc[df["site"] != target_site].copy()
        target_df = df.loc[df["site"] == target_site].copy()

        if target_df.empty:
            logger.log(
                f"[WARN][{fold_name}] Nessun campione nel target site. Fold saltato."
            )
            continue

        # Split del SOLO target domain (anti-leakage).
        stratify_target = None
        class_counts_target = target_df["target"].value_counts()
        if target_df["target"].nunique() >= 2 and class_counts_target.min() >= 2:
            stratify_target = target_df["target"]
        else:
            logger.log(
                f"[WARN][{fold_name}] Stratify non applicabile (classi insufficienti). "
                "Uso split non stratificato."
            )

        target_adaptation_df, target_test_df = train_test_split(
            target_df,
            test_size=config.target_test_size,
            random_state=config.random_state,
            stratify=stratify_target,
        )

        # Check anti-leakage sugli ID.
        assert_no_overlap(
            source_df["row_id"],
            target_test_df["row_id"],
            f"{fold_name}: source vs target_test",
        )
        assert_no_overlap(
            target_adaptation_df["row_id"],
            target_test_df["row_id"],
            f"{fold_name}: target_adaptation vs target_test",
        )

        # Logging diagnostico richiesto.
        logger.log(f"[SIZE][{fold_name}] source_n={len(source_df)}")
        logger.log(
            f"[SIZE][{fold_name}] target_adaptation_n={len(target_adaptation_df)}"
        )
        logger.log(f"[SIZE][{fold_name}] target_test_n={len(target_test_df)}")
        logger.log(
            f"[DIST][{fold_name}] source target distribution: "
            f"{class_distribution_text(source_df['target'])}"
        )
        logger.log(
            f"[DIST][{fold_name}] target_adaptation distribution (diagnostico): "
            f"{class_distribution_text(target_adaptation_df['target'])}"
        )
        logger.log(
            f"[DIST][{fold_name}] target_test distribution: "
            f"{class_distribution_text(target_test_df['target'])}"
        )

        # Definizione matrici X/y.
        X_source = source_df[FEATURE_COLUMNS]
        y_source = source_df["target"].to_numpy(dtype=int)

        X_target_adapt = target_adaptation_df[FEATURE_COLUMNS]
        X_target_test = target_test_df[FEATURE_COLUMNS]
        y_target_test = target_test_df["target"].to_numpy(dtype=int)

        # ---------------------------------------------------------
        # Preprocessing fit SOLO su source (vincolo anti-leakage).
        # ---------------------------------------------------------
        preprocessor = build_preprocessor()
        preprocessor.fit(X_source)
        logger.log(
            f"[CHECK][{fold_name}] Preprocessore fittato solo sui source: "
            f"n_source_fit={X_source.shape[0]}"
        )

        X_source_tr = preprocessor.transform(X_source)
        X_target_adapt_tr = preprocessor.transform(X_target_adapt)
        X_target_test_tr = preprocessor.transform(X_target_test)

        # ---------------------------------------------------------
        # Baseline source-only
        # ---------------------------------------------------------
        baseline_model = build_logistic_model(config.random_state)
        baseline_model.fit(X_source_tr, y_source)

        baseline_pred = baseline_model.predict(X_target_test_tr)
        baseline_prob = baseline_model.predict_proba(X_target_test_tr)[:, 1]
        baseline_metrics = compute_metrics(y_target_test, baseline_pred, baseline_prob)
        baseline_cm = confusion_matrix(y_target_test, baseline_pred, labels=[0, 1])

        baseline_cm_path = (
            config.output_dir
            / f"confusion_matrix_baseline_{safe_filename(target_site)}.png"
        )
        save_confusion_matrix_png(
            baseline_cm,
            title=f"Baseline confusion matrix - target={target_site}",
            output_path=baseline_cm_path,
        )

        fold_rows.append(
            {
                "target_site": target_site,
                "method": "baseline",
                "source_n": len(source_df),
                "target_adaptation_n": len(target_adaptation_df),
                "target_test_n": len(target_test_df),
                "pseudo_total_added": 0,
                "pseudo_iterations": 0,
                **baseline_metrics,
            }
        )

        # ---------------------------------------------------------
        # Transductive learning corretto
        # ---------------------------------------------------------
        # NOTA critica: le label vere di target_adaptation NON vengono passate
        # alla funzione di training transductive. Si passa solo X unlabeled.
        logger.log(
            f"[CHECK][{fold_name}] Le label vere di target_adaptation sono isolate e non usate nel fit transductive."
        )

        n_source_labels = int(y_source.shape[0])
        trans_model, pseudo_history, pseudo_total_added, final_train_size = (
            run_transductive_pseudo_labeling(
                X_source=X_source_tr,
                y_source=y_source,
                X_target_unlabeled=X_target_adapt_tr,
                pseudo_threshold=config.pseudo_threshold,
                pseudo_max_iters=config.pseudo_max_iters,
                random_state=config.random_state,
                logger=logger,
                fold_name=fold_name,
            )
        )

        # Check anti-leakage esplicito:
        # se avessimo concatenato per errore le label vere di target_adaptation,
        # la dimensione del training non sarebbe source + pseudo-label aggiunte.
        expected_final_train_size = n_source_labels + int(pseudo_total_added)
        if final_train_size != expected_final_train_size:
            raise RuntimeError(
                f"[LEAKAGE ERROR][{fold_name}] training size inattesa nel transductive: "
                f"final_train_size={final_train_size}, "
                f"expected_source_plus_pseudo={expected_final_train_size}"
            )
        logger.log(
            f"[CHECK][{fold_name}] OK: train_transductive_size={final_train_size} "
            f"(source={n_source_labels} + pseudo={pseudo_total_added})"
        )

        # Anche nel transductive il target_test resta completamente separato.
        assert_no_overlap(
            pd.Series(
                np.concatenate(
                    [
                        source_df["row_id"].to_numpy(),
                        target_adaptation_df["row_id"].to_numpy(),
                    ]
                )
            ),
            target_test_df["row_id"],
            f"{fold_name}: train_transductive vs target_test",
        )

        trans_pred = trans_model.predict(X_target_test_tr)
        trans_prob = trans_model.predict_proba(X_target_test_tr)[:, 1]
        trans_metrics = compute_metrics(y_target_test, trans_pred, trans_prob)
        trans_cm = confusion_matrix(y_target_test, trans_pred, labels=[0, 1])

        trans_cm_path = (
            config.output_dir
            / f"confusion_matrix_transductive_{safe_filename(target_site)}.png"
        )
        save_confusion_matrix_png(
            trans_cm,
            title=f"Transductive confusion matrix - target={target_site}",
            output_path=trans_cm_path,
        )

        fold_rows.append(
            {
                "target_site": target_site,
                "method": "transductive",
                "source_n": len(source_df),
                "target_adaptation_n": len(target_adaptation_df),
                "target_test_n": len(target_test_df),
                "pseudo_total_added": int(pseudo_total_added),
                "pseudo_iterations": len(pseudo_history),
                **trans_metrics,
            }
        )

        for hist in pseudo_history:
            pseudo_rows.append(
                {
                    "target_site": target_site,
                    "fold_name": fold_name,
                    **hist,
                }
            )

        # Warning richiesto se transductive peggiora rispetto baseline.
        if trans_metrics["balanced_accuracy"] < baseline_metrics["balanced_accuracy"]:
            warning_message = (
                f"[WARN][{fold_name}] Transductive peggiora su balanced_accuracy: "
                f"baseline={baseline_metrics['balanced_accuracy']:.4f}, "
                f"transductive={trans_metrics['balanced_accuracy']:.4f}"
            )
            logger.log(warning_message)
            warning_rows.append(
                {
                    "target_site": target_site,
                    "metric": "balanced_accuracy",
                    "baseline": baseline_metrics["balanced_accuracy"],
                    "transductive": trans_metrics["balanced_accuracy"],
                    "warning": warning_message,
                }
            )

        # Se ROC-AUC non e calcolabile, logghiamo ma non fermiamo il run.
        if np.isnan(baseline_metrics["roc_auc"]):
            logger.log(
                f"[WARN][{fold_name}] ROC-AUC baseline non calcolabile (classe unica nel target_test)."
            )
        if np.isnan(trans_metrics["roc_auc"]):
            logger.log(
                f"[WARN][{fold_name}] ROC-AUC transductive non calcolabile (classe unica nel target_test)."
            )

    fold_results_df = pd.DataFrame(fold_rows)
    pseudo_history_df = pd.DataFrame(pseudo_rows)
    warnings_df = pd.DataFrame(warning_rows)

    return {
        "fold_results": fold_results_df,
        "pseudo_history": pseudo_history_df,
        "warnings": warnings_df,
    }


# ============================================================================
# Tabelle finali e report
# ============================================================================


def build_mean_metrics_table(fold_results_df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce tabella delle metriche medie finali (mean/std) per metodo."""
    rows: List[Dict[str, object]] = []

    for method_name in ["baseline", "transductive"]:
        sub = fold_results_df.loc[fold_results_df["method"] == method_name].copy()
        if sub.empty:
            continue

        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(sub[metric], errors="coerce")
            rows.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "mean": float(values.mean(skipna=True)),
                    "std": float(values.std(skipna=True, ddof=1)),
                    "n_valid_folds": int(values.notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def build_baseline_vs_transductive_comparison(
    fold_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tabella confronto baseline vs transductive per site target e metrica,
    con delta = transductive - baseline.
    """
    rows: List[Dict[str, object]] = []

    for target_site in sorted(fold_results_df["target_site"].unique().tolist()):
        base_row = fold_results_df[
            (fold_results_df["target_site"] == target_site)
            & (fold_results_df["method"] == "baseline")
        ]
        trans_row = fold_results_df[
            (fold_results_df["target_site"] == target_site)
            & (fold_results_df["method"] == "transductive")
        ]

        if base_row.empty or trans_row.empty:
            continue

        base_row = base_row.iloc[0]
        trans_row = trans_row.iloc[0]

        for metric in METRIC_COLUMNS:
            base_value = (
                float(base_row[metric]) if pd.notna(base_row[metric]) else np.nan
            )
            trans_value = (
                float(trans_row[metric]) if pd.notna(trans_row[metric]) else np.nan
            )
            delta = (
                trans_value - base_value
                if pd.notna(base_value) and pd.notna(trans_value)
                else np.nan
            )

            rows.append(
                {
                    "target_site": target_site,
                    "metric": metric,
                    "baseline": base_value,
                    "transductive": trans_value,
                    "delta_transductive_minus_baseline": delta,
                }
            )

    return pd.DataFrame(rows)


def write_summary_markdown(
    output_path: Path,
    clean_df: pd.DataFrame,
    fold_results_df: pd.DataFrame,
    mean_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    precision_df: pd.DataFrame,
) -> None:
    """
    Scrive una sintesi automatica orientata alla discussione in tesi.

    Come interpretare baseline vs transductive:
    - se il delta e positivo, il transductive ha migliorato sul target_test unseen;
    - se negativo, il pseudo-labeling puo avere introdotto pseudo-label rumorose.
    """
    lines: List[str] = []
    lines.append("# Sintesi automatica risultati")
    lines.append("")
    lines.append("## Dataset e protocollo")
    lines.append(
        "L'analisi e stata eseguita partendo dai 4 file originali separati per site "
        "(cleveland, hungarian, long_beach_va, switzerland), mantenendo esplicitamente la variabile di dominio."
    )
    lines.append(
        "Il protocollo usa Leave-One-Domain-Out con split del target in target_adaptation (unlabeled) "
        "e target_test (solo valutazione), per evitare leakage."
    )
    lines.append("")

    lines.append("## Numerosita finale")
    lines.append(f"Campioni totali usati in modellazione: **{len(clean_df)}**")
    site_counts = clean_df["site"].value_counts().sort_index()
    for site_name, n_samples in site_counts.items():
        lines.append(f"- {site_name}: {int(n_samples)} campioni")
    lines.append("")

    # Focus principale richiesto: precision baseline vs transductive.
    lines.append("## Focus principale: precision")
    if precision_df.empty:
        lines.append("Nessun confronto precision disponibile.")
    else:
        mean_precision_baseline = float(precision_df["precision_baseline"].mean())
        mean_precision_transductive = float(
            precision_df["precision_transductive"].mean()
        )
        mean_delta = float(precision_df["delta_precision"].mean())

        lines.append(f"- Precision media baseline: **{mean_precision_baseline:.4f}**")
        lines.append(
            f"- Precision media transductive: **{mean_precision_transductive:.4f}**"
        )
        lines.append(f"- Delta medio (transductive - baseline): **{mean_delta:+.4f}**")
        lines.append("")

        increased = precision_df.loc[
            precision_df["delta_precision"] > 0, "target_site"
        ].tolist()
        unchanged = precision_df.loc[
            precision_df["delta_precision"] == 0, "target_site"
        ].tolist()
        decreased = precision_df.loc[
            precision_df["delta_precision"] < 0, "target_site"
        ].tolist()

        lines.append("### Site con precision aumentata")
        if increased:
            for site in increased:
                row = precision_df.loc[precision_df["target_site"] == site].iloc[0]
                lines.append(
                    f"- {site}: {row['precision_baseline']:.4f} -> {row['precision_transductive']:.4f} "
                    f"(delta {row['delta_precision']:+.4f})"
                )
        else:
            lines.append("- Nessuno")

        lines.append("### Site con precision invariata")
        if unchanged:
            for site in unchanged:
                row = precision_df.loc[precision_df["target_site"] == site].iloc[0]
                lines.append(
                    f"- {site}: {row['precision_baseline']:.4f} -> {row['precision_transductive']:.4f} "
                    f"(delta {row['delta_precision']:+.4f})"
                )
        else:
            lines.append("- Nessuno")

        lines.append("### Site con precision diminuita")
        if decreased:
            for site in decreased:
                row = precision_df.loc[precision_df["target_site"] == site].iloc[0]
                lines.append(
                    f"- {site}: {row['precision_baseline']:.4f} -> {row['precision_transductive']:.4f} "
                    f"(delta {row['delta_precision']:+.4f})"
                )
        else:
            lines.append("- Nessuno")
    lines.append("")

    lines.append("## Metriche medie (across folds)")
    for method_name in ["baseline", "transductive"]:
        sub = mean_metrics_df[mean_metrics_df["method"] == method_name]
        if sub.empty:
            continue
        lines.append(f"### {method_name}")
        for _, row in sub.iterrows():
            mean_value = row["mean"]
            std_value = row["std"]
            lines.append(
                f"- {row['metric']}: mean={mean_value:.4f} | std={std_value:.4f} "
                f"(n={int(row['n_valid_folds'])})"
            )
    lines.append("")

    lines.append("## Confronto baseline vs transductive per site")
    for target_site in sorted(comparison_df["target_site"].unique().tolist()):
        lines.append(f"### target site: {target_site}")
        site_sub = comparison_df[comparison_df["target_site"] == target_site]
        for metric in METRIC_COLUMNS:
            row = site_sub[site_sub["metric"] == metric]
            if row.empty:
                continue
            row = row.iloc[0]
            b = row["baseline"]
            t = row["transductive"]
            d = row["delta_transductive_minus_baseline"]
            lines.append(
                f"- {metric}: baseline={b:.4f}, transductive={t:.4f}, delta={d:+.4f}"
            )
    lines.append("")

    lines.append("## Interpretazione")
    lines.append(
        "Un miglioramento transductive indica che il modello ha beneficiato di struttura non etichettata "
        "del dominio target (target_adaptation) senza usare le label vere durante il fit."
    )
    lines.append(
        "Un peggioramento indica che i pseudo-label possono essere rumorosi in presenza di forte domain shift "
        "o sbilanciamento classi nel target."
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_script_to_output(output_dir: Path, logger: RunLogger) -> None:
    """Copia lo script corrente nella cartella output (richiesta esplicita)."""
    try:
        script_path = Path(__file__).resolve()
        destination = output_dir / script_path.name
        shutil.copy2(script_path, destination)
        logger.log(f"[SAVE] Copia script salvata in: {destination}")
    except Exception as exc:
        logger.log(f"[WARN] Impossibile copiare lo script in output: {exc}")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Argomenti CLI con default esatti richiesti."""
    parser = argparse.ArgumentParser(
        description=(
            "Transductive learning leakage-safe su UCI Heart Disease usando i 4 site originali separati."
        )
    )

    parser.add_argument(
        "--cleveland-file",
        type=Path,
        default=Path(_configured_path("uci_raw_cleveland.data")),
        help="Path al file cleveland.data",
    )
    parser.add_argument(
        "--hungarian-file",
        type=Path,
        default=Path(_configured_path("uci_raw_hungarian.data")),
        help="Path al file hungarian.data",
    )
    parser.add_argument(
        "--long-beach-file",
        type=Path,
        default=Path(_configured_path("uci_raw_long_beach_va.data")),
        help="Path al file long-beach-va.data",
    )
    parser.add_argument(
        "--switzerland-file",
        type=Path,
        default=Path(_configured_path("uci_raw_switzerland.data")),
        help="Path al file switzerland.data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_configured_path("legacy_report_uci_transd_learning_sites")),
        help="Directory di output dove salvare script, CSV, PNG e report",
    )

    parser.add_argument(
        "--target-test-size",
        type=float,
        default=0.50,
        help="Quota del dominio target destinata a target_test (default: 0.50)",
    )
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.95,
        help="Soglia confidenza pseudo-labeling (default: 0.95)",
    )
    parser.add_argument(
        "--pseudo-max-iters",
        type=int,
        default=10,
        help="Numero massimo iterazioni pseudo-labeling (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed random",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    input_files = {
        "cleveland": args.cleveland_file,
        "hungarian": args.hungarian_file,
        "long_beach_va": args.long_beach_file,
        "switzerland": args.switzerland_file,
    }

    config = ExperimentConfig(
        input_files=input_files,
        output_dir=args.output_dir,
        random_state=args.random_state,
        target_test_size=args.target_test_size,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_max_iters=args.pseudo_max_iters,
    )

    ensure_output_dir(config.output_dir)
    logger = RunLogger()

    logger.log(
        "[START] Avvio pipeline transductive learning leakage-safe su UCI Heart Disease."
    )
    logger.log(f"[CONFIG] output_dir={config.output_dir}")
    logger.log(f"[CONFIG] target_test_size={config.target_test_size}")
    logger.log(f"[CONFIG] pseudo_threshold={config.pseudo_threshold}")
    logger.log(f"[CONFIG] pseudo_max_iters={config.pseudo_max_iters}")
    logger.log(f"[CONFIG] random_state={config.random_state}")

    # 1) Caricamento 4 site separati e unione interna.
    raw_df = load_all_sites(config.input_files, logger)

    # 2) Pulizia + target binario.
    clean_df = clean_and_prepare_dataset(raw_df, logger)

    # Salvataggi diagnostici del dataset usato.
    clean_df.to_csv(
        config.output_dir / "dataset_unificato_interno_clean.csv", index=False
    )
    (
        clean_df.groupby("site")["target"].value_counts().rename("count").reset_index()
    ).to_csv(config.output_dir / "distribuzioni_classi_per_site.csv", index=False)

    # 3) Esperimento LODO con baseline + transductive.
    experiment_outputs = run_lodo_experiment(clean_df, config, logger)
    fold_results_df = experiment_outputs["fold_results"]
    pseudo_history_df = experiment_outputs["pseudo_history"]
    warnings_df = experiment_outputs["warnings"]

    if fold_results_df.empty:
        raise RuntimeError("Nessun fold valido generato. Controllare parsing e split.")

    # 4) Tabelle finali richieste.
    mean_metrics_df = build_mean_metrics_table(fold_results_df)
    comparison_df = build_baseline_vs_transductive_comparison(fold_results_df)
    precision_df = _extract_precision_by_site(fold_results_df)

    fold_results_path = config.output_dir / "risultati_fold_by_fold.csv"
    mean_metrics_path = config.output_dir / "metriche_medie_finali.csv"
    comparison_path = config.output_dir / "confronto_baseline_vs_transductive.csv"
    precision_site_path = config.output_dir / "precision_confronto_per_site.csv"
    pseudo_path = config.output_dir / "pseudo_labeling_iterazioni.csv"
    warnings_path = config.output_dir / "warning_transductive_vs_baseline.csv"

    fold_results_df.to_csv(fold_results_path, index=False)
    mean_metrics_df.to_csv(mean_metrics_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    precision_df.to_csv(precision_site_path, index=False)
    pseudo_history_df.to_csv(pseudo_path, index=False)
    warnings_df.to_csv(warnings_path, index=False)

    # 5) Plot richiesti (con focus principale sulla precision).
    # Manteniamo anche i plot completi su tutte le metriche, utili come supporto diagnostico.
    plot_baseline_vs_transductive_by_site(
        fold_results_df=fold_results_df,
        metrics=METRIC_COLUMNS,
        output_path=config.output_dir / "barplot_baseline_vs_transductive_per_site.png",
    )
    plot_mean_metrics(
        mean_df=mean_metrics_df,
        metrics=METRIC_COLUMNS,
        output_path=config.output_dir / "plot_metriche_medie_finali.png",
    )
    plot_precision_by_site(
        precision_df=precision_df,
        output_path=config.output_dir / "precision_barplot_per_site.png",
    )
    plot_precision_mean_aggregate(
        precision_df=precision_df,
        output_path=config.output_dir / "precision_media_aggregata.png",
    )
    plot_precision_delta_by_site(
        precision_df=precision_df,
        output_path=config.output_dir / "precision_delta_per_site.png",
    )

    # 6) Sintesi automatica.
    write_summary_markdown(
        output_path=config.output_dir / "sintesi_risultati.md",
        clean_df=clean_df,
        fold_results_df=fold_results_df,
        mean_metrics_df=mean_metrics_df,
        comparison_df=comparison_df,
        precision_df=precision_df,
    )

    # 7) Copia script e log finale.
    copy_script_to_output(config.output_dir, logger)
    logger.save(config.output_dir / "log_esperimento.txt")

    logger.log("[END] Pipeline completata con successo.")
    logger.log(f"[SAVE] Fold-by-fold CSV: {fold_results_path}")
    logger.log(f"[SAVE] Mean metrics CSV: {mean_metrics_path}")
    logger.log(f"[SAVE] Comparison CSV: {comparison_path}")
    logger.log(f"[SAVE] Precision by-site CSV: {precision_site_path}")
    logger.log(f"[SAVE] Pseudo-label history CSV: {pseudo_path}")
    logger.log(f"[SAVE] Warnings CSV: {warnings_path}")


if __name__ == "__main__":
    main()
