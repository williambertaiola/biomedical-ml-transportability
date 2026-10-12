#!/usr/bin/env python3
"""
UCI Heart Disease - Workflow completo per analisi multi-dominio (transportability/domain shift)

Cosa fa questo script:
1) Carica i 4 file UCI Heart Disease come 4 domini distinti.
2) Mantiene esplicitamente la colonna `domain` in tutte le fasi.
3) Esegue data cleaning trasparente (`?` -> NaN, conversione numerica sicura).
4) Crea `target_bin` mantenendo anche `target` originale.
5) Produce EDA comparativa tra domini (tabelle + figure).
6) Analizza qualità dei dati e missingness per dominio.
7) Prepara matrici X/y per uso successivo con DASHI (raw + preprocessata).
8) Esegue analisi proxy di domain shift (feature-level e dominio-vs-dominio).
9) Esegue baseline di transportability con schema leave-one-domain-out.
10) Salva output (CSV + figure) in una cartella dedicata nella stessa cartella UCI.

Assunzioni esplicitate:
- I file sorgente hanno 14 colonne classiche UCI Heart Disease.
- Le colonne sono trattate come numeriche dove previsto; conversioni non valide diventano NaN.
- La soglia per escludere feature in preparazione DASHI e` esplicita in config.
- Qualsiasi standardizzazione usata per modellazione e` fatta solo dentro i fold
  (pipeline sklearn) per evitare leakage.
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


from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple
import os
import random
import re
import warnings

# Evita crash in ambienti senza cache grafica scrivibile (es. sandbox/CI).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except ModuleNotFoundError:
    sns = None
    HAS_SEABORN = False

try:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ModuleNotFoundError:
    clone = None
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    accuracy_score = None
    balanced_accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None
    StratifiedKFold = None
    Pipeline = None
    StandardScaler = None
    HAS_SKLEARN = False


@dataclass
class WorkflowConfig:
    random_seed: int = 42
    expected_num_columns: int = 14
    min_valid_fraction_for_raw_parse: float = 0.95
    missing_threshold_for_feature_drop: float = 0.40
    histogram_bins: int = 20
    pooled_cv_folds: int = 5

    column_names: List[str] = field(
        default_factory=lambda: [
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
            "target",
        ]
    )

    numeric_columns: List[str] = field(
        default_factory=lambda: [
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
            "target",
        ]
    )

    predictor_columns: List[str] = field(
        default_factory=lambda: [
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
        ]
    )

    key_numeric_for_eda: List[str] = field(
        default_factory=lambda: ["age", "trestbps", "chol", "thalach", "oldpeak"]
    )

    domain_order: List[str] = field(
        default_factory=lambda: [
            "cleveland",
            "long_beach_va",
            "switzerland",
            "hungarian",
        ]
    )

    domain_paths: Dict[str, Path] = field(
        default_factory=lambda: {
            "cleveland": Path(_configured_path("uci_raw_cleveland.data")),
            "long_beach_va": Path(_configured_path("uci_raw_long_beach_va.data")),
            "switzerland": Path(_configured_path("uci_raw_switzerland.data")),
            "hungarian": Path(_configured_path("uci_raw_hungarian.data")),
        }
    )

    expected_rows_by_domain: Dict[str, int] = field(
        default_factory=lambda: {
            "cleveland": 303,
            "long_beach_va": 200,
            "switzerland": 123,
            "hungarian": 294,
        }
    )

    processed_fallback_paths: Dict[str, Path] = field(
        default_factory=lambda: {
            "cleveland": Path(_configured_path("uci_raw_processed.cleveland.data")),
            "long_beach_va": Path(_configured_path("uci_raw_processed.va.data")),
            "switzerland": Path(_configured_path("uci_raw_processed.switzerland.data")),
            "hungarian": Path(_configured_path("uci_raw_processed.hungarian.data")),
        }
    )

    # Mapping 1-based dagli attributi raw UCI (75) alle 14 feature classiche.
    raw_to_classic_index_1based: List[int] = field(
        default_factory=lambda: [3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51, 58]
    )

    output_root: Path = Path(_configured_path("legacy_output"))


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    print(f"[INFO] Seed random impostato a {seed}.")


def set_plot_theme() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
        print("[INFO] Plot theme: seaborn whitegrid.")
    else:
        plt.style.use("ggplot")
        print(
            "[INFO] Plot theme fallback: matplotlib ggplot. "
            "Se vuoi stile seaborn, installa pacchetto `seaborn`."
        )


def create_output_dirs(output_root: Path) -> Dict[str, Path]:
    def _build_dirs(root: Path) -> Dict[str, Path]:
        return {
            "root": root,
            "data": root / "data",
            "tables": root / "tables",
            "figures": root / "figures",
            "fig_eda": root / "figures" / "eda",
            "fig_quality": root / "figures" / "quality",
            "fig_shift": root / "figures" / "domain_shift",
            "fig_modeling": root / "figures" / "modeling",
        }

    def _materialize_and_test(dirs: Dict[str, Path]) -> None:
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        test_file = dirs["root"] / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)

    dirs = _build_dirs(output_root)
    try:
        _materialize_and_test(dirs)
        print(f"[INFO] Cartella output pronta: {output_root}")
        return dirs
    except PermissionError:
        fallback_root = Path.cwd() / "output_uci_heart_fallback"
        print(
            f"[WARN] Output root non scrivibile: {output_root}. "
            f"Uso fallback locale: {fallback_root}"
        )
        dirs = _build_dirs(fallback_root)
        _materialize_and_test(dirs)
        print(f"[INFO] Cartella output fallback pronta: {fallback_root}")
        return dirs


def decode_text_with_fallback(file_path: Path) -> Tuple[str, str]:
    raw_bytes = file_path.read_bytes()
    encoding_candidates = ["utf-8", "latin1", "cp1252"]
    decode_errors: List[str] = []

    for encoding in encoding_candidates:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")

    error_details = "\n".join(decode_errors)
    raise UnicodeError(
        f"[ERRORE] Impossibile decodificare il file: {file_path}\n"
        f"Encoding tentati: {encoding_candidates}\nDettagli:\n{error_details}"
    )


def load_processed_14col_csv(
    file_path: Path, column_names: List[str], expected_num_columns: int, encoding: str
) -> pd.DataFrame:
    df = pd.read_csv(
        file_path, header=None, dtype=str, sep=",", encoding=encoding, engine="python"
    )
    if df.shape[1] != expected_num_columns:
        raise ValueError(
            f"[ERRORE] File {file_path}: attese {expected_num_columns} colonne, trovate {df.shape[1]}"
        )
    df.columns = column_names
    return df


def parse_raw_uci_to_classic_14(
    text: str, column_names: List[str], raw_to_classic_index_1based: List[int]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    numeric_token_re = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
    records: List[List[str]] = []
    current: List[str] = []
    invalid_tokens = 0
    required_idx = max(raw_to_classic_index_1based)

    for raw_line in text.splitlines():
        clean_line = "".join(
            ch if ch in {" ", "\t"} or 32 <= ord(ch) <= 126 else " " for ch in raw_line
        )
        clean_line = re.sub(r"(?i)name", " name ", clean_line)
        clean_line = re.sub(r"(?<=\d)-(?=\d)", " -", clean_line)

        for token in clean_line.split():
            tok = token.strip().strip(",")
            if not tok:
                continue
            if tok.lower() == "name":
                if current:
                    records.append(current)
                    current = []
                continue
            if numeric_token_re.match(tok):
                current.append(tok)
            else:
                invalid_tokens += 1

    if current:
        records.append(current)

    extracted_rows: List[List[str]] = []
    dropped_short = 0
    for rec in records:
        if len(rec) < required_idx:
            dropped_short += 1
            continue
        extracted_rows.append([rec[idx - 1] for idx in raw_to_classic_index_1based])

    df = pd.DataFrame(extracted_rows, columns=column_names)
    parse_info = {
        "raw_records_found": len(records),
        "rows_extracted": len(extracted_rows),
        "rows_dropped_short": dropped_short,
        "invalid_tokens_ignored": invalid_tokens,
    }
    return df, parse_info


def load_single_domain(
    file_path: Path, domain: str, config: WorkflowConfig
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"[ERRORE] File non trovato per il dominio '{domain}': {file_path}"
        )

    text, encoding = decode_text_with_fallback(file_path)
    lower_text = text.lower()

    parsed_from = str(file_path)
    parse_mode = "csv_14"
    parse_info = {}

    if "name" in lower_text:
        parse_mode = "raw_75_to_classic_14"
        df, parse_info = parse_raw_uci_to_classic_14(
            text=text,
            column_names=config.column_names,
            raw_to_classic_index_1based=config.raw_to_classic_index_1based,
        )

        expected_rows = config.expected_rows_by_domain.get(domain)
        if expected_rows is not None and expected_rows > 0:
            raw_rows_extracted = len(df)
            valid_fraction = len(df) / expected_rows
            if valid_fraction < config.min_valid_fraction_for_raw_parse:
                fallback_path = config.processed_fallback_paths.get(domain)
                if fallback_path is not None and fallback_path.exists():
                    _, fallback_encoding = decode_text_with_fallback(fallback_path)
                    df = load_processed_14col_csv(
                        file_path=fallback_path,
                        column_names=config.column_names,
                        expected_num_columns=config.expected_num_columns,
                        encoding=fallback_encoding,
                    )
                    encoding = fallback_encoding
                    parsed_from = str(fallback_path)
                    parse_mode = "processed_fallback_csv_14"
                    parse_info["fallback_reason"] = (
                        f"raw parse sotto soglia: {raw_rows_extracted}/{expected_rows} "
                        f"(soglia={config.min_valid_fraction_for_raw_parse:.0%})"
                    )
                else:
                    print(
                        f"[WARN] Dominio '{domain}': parse raw sotto soglia ma fallback non disponibile. "
                        "Continuo con righe valide raw."
                    )
    else:
        df = load_processed_14col_csv(
            file_path=file_path,
            column_names=config.column_names,
            expected_num_columns=config.expected_num_columns,
            encoding=encoding,
        )

    if df.shape[1] != config.expected_num_columns:
        raise ValueError(
            f"[ERRORE] Dominio '{domain}': attese {config.expected_num_columns} colonne, trovate {df.shape[1]}"
        )

    df["domain"] = domain
    print(
        f"[INFO] Caricato dominio '{domain}' - shape: {df.shape} "
        f"(mode={parse_mode}, encoding={encoding}, source={parsed_from})"
    )
    if parse_info:
        print(f"[INFO] Dettagli parse '{domain}': {parse_info}")
    return df


def load_all_domains(
    config: WorkflowConfig,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    print_section("1-3) Import, schema, caricamento domini e unione tecnica")
    domain_dfs: Dict[str, pd.DataFrame] = {}

    for domain in config.domain_order:
        domain_dfs[domain] = load_single_domain(
            file_path=config.domain_paths[domain],
            domain=domain,
            config=config,
        )

    combined_df = pd.concat(
        [domain_dfs[d] for d in config.domain_order], ignore_index=True, sort=False
    )
    print(f"[INFO] DataFrame concatenato creato - shape: {combined_df.shape}")
    print("[INFO] La colonna `domain` e` mantenuta esplicitamente.")
    return domain_dfs, combined_df


def clean_dataframe(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    clean_df = df.copy()

    for col in clean_df.select_dtypes(include=["object", "string"]).columns:
        clean_df[col] = clean_df[col].str.strip()

    clean_df = clean_df.replace(
        {
            "?": np.nan,
            "": np.nan,
            "-9": np.nan,
            "-9.0": np.nan,
            "-9.00": np.nan,
        }
    )

    for col in numeric_columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df["domain"] = clean_df["domain"].astype(str)
    return clean_df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_bin"] = np.nan
    valid_target_mask = out["target"].notna()
    out.loc[valid_target_mask, "target_bin"] = (
        out.loc[valid_target_mask, "target"] > 0
    ).astype(int)
    return out


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss_count = df.isna().sum()
    miss_pct = (df.isna().mean() * 100).round(2)
    summary = pd.DataFrame({"missing_count": miss_count, "missing_pct": miss_pct})
    return summary.sort_values(by="missing_pct", ascending=False)


def print_and_save_domain_overview(
    domain_dfs: Dict[str, pd.DataFrame], combined_df: pd.DataFrame, tables_dir: Path
) -> None:
    print_section("4) Data cleaning, controllo tipi e missingness")

    for domain, df in domain_dfs.items():
        print(f"\n[DOMINIO] {domain}")
        print(f"- righe: {df.shape[0]}")
        print(f"- colonne: {df.shape[1]}")
        print("- tipi colonne:")
        print(df.dtypes.to_string())

        miss_tbl = missing_summary(df)
        print("- missing per colonna:")
        print(miss_tbl.to_string())
        miss_tbl.to_csv(tables_dir / f"missing_summary_{domain}.csv", index=True)

    print("\n[DATASET GLOBALE CONCATENATO]")
    print(f"- righe: {combined_df.shape[0]}")
    print(f"- colonne: {combined_df.shape[1]}")
    print("- tipi colonne:")
    print(combined_df.dtypes.to_string())

    miss_tbl_global = missing_summary(combined_df)
    print("- missing per colonna:")
    print(miss_tbl_global.to_string())
    miss_tbl_global.to_csv(tables_dir / "missing_summary_global.csv", index=True)


def save_cleaned_data(
    domain_dfs: Dict[str, pd.DataFrame], combined_df: pd.DataFrame, data_dir: Path
) -> None:
    for domain, df in domain_dfs.items():
        df.to_csv(data_dir / f"cleaned_{domain}.csv", index=False)
    combined_df.to_csv(data_dir / "cleaned_all_domains.csv", index=False)
    print(f"[INFO] DataFrame puliti salvati in: {data_dir}")


def target_distribution_table(
    domain_dfs: Dict[str, pd.DataFrame], combined_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    for domain, df in domain_dfs.items():
        valid = df["target_bin"].dropna()
        n = int(valid.shape[0])
        n0 = int((valid == 0).sum())
        n1 = int((valid == 1).sum())
        prev1 = (n1 / n) if n > 0 else np.nan
        rows.append(
            {
                "domain": domain,
                "n_valid_target": n,
                "n_target_0": n0,
                "n_target_1": n1,
                "target_1_prevalence": prev1,
            }
        )

    valid_all = combined_df["target_bin"].dropna()
    n_all = int(valid_all.shape[0])
    n0_all = int((valid_all == 0).sum())
    n1_all = int((valid_all == 1).sum())
    prev1_all = (n1_all / n_all) if n_all > 0 else np.nan
    rows.append(
        {
            "domain": "global",
            "n_valid_target": n_all,
            "n_target_0": n0_all,
            "n_target_1": n1_all,
            "target_1_prevalence": prev1_all,
        }
    )

    out = pd.DataFrame(rows)
    return out


def plot_domain_sizes(
    combined_df: pd.DataFrame, domain_order: List[str], fig_path: Path
) -> None:
    counts = combined_df["domain"].value_counts().reindex(domain_order)

    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        ax = sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    else:
        ax = plt.gca()
        colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        ax.bar(counts.index.tolist(), counts.values, color=colors)
    ax.set_title("Numerosita` campioni per dominio")
    ax.set_xlabel("Dominio")
    ax.set_ylabel("Numero campioni")
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(counts.values):
        ax.text(i, value + max(counts.values) * 0.01, f"{int(value)}", ha="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_target_distribution(
    combined_df: pd.DataFrame, domain_order: List[str], fig_path: Path
) -> None:
    plot_df = combined_df.dropna(subset=["target_bin"]).copy()
    plot_df["target_bin"] = plot_df["target_bin"].astype(int)

    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        ax = sns.countplot(
            data=plot_df,
            x="domain",
            hue="target_bin",
            order=domain_order,
            palette="Set1",
        )
        ax.legend(title="target_bin", labels=["0", "1"])
    else:
        ax = plt.gca()
        grouped = (
            plot_df.groupby(["domain", "target_bin"])
            .size()
            .unstack(fill_value=0)
            .reindex(domain_order)
        )
        if 0 not in grouped.columns:
            grouped[0] = 0
        if 1 not in grouped.columns:
            grouped[1] = 0
        grouped = grouped[[0, 1]]

        x = np.arange(len(domain_order))
        width = 0.38
        ax.bar(
            x - width / 2, grouped[0].values, width=width, label="0", color="#4C72B0"
        )
        ax.bar(
            x + width / 2, grouped[1].values, width=width, label="1", color="#DD8452"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(domain_order, rotation=20)
        ax.legend(title="target_bin")

    ax.set_title("Distribuzione target_bin per dominio")
    ax.set_xlabel("Dominio")
    ax.set_ylabel("Conteggio")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def descriptive_stats_by_domain(
    combined_df: pd.DataFrame, numeric_features: List[str]
) -> pd.DataFrame:
    stats_df = (
        combined_df.groupby("domain")[numeric_features]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .round(3)
    )
    return stats_df


def plot_feature_distributions(
    combined_df: pd.DataFrame,
    features: List[str],
    domain_order: List[str],
    output_dir: Path,
    bins: int,
) -> None:
    for feature in features:
        valid = combined_df.dropna(subset=[feature]).copy()
        if valid.empty:
            print(f"[WARN] Feature '{feature}' completamente mancante: plot saltato.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        if HAS_SEABORN:
            sns.boxplot(
                data=valid,
                x="domain",
                y=feature,
                order=domain_order,
                ax=axes[0],
                palette="Set3",
            )
        else:
            box_data = [
                valid.loc[valid["domain"] == domain, feature].dropna().values
                for domain in domain_order
            ]
            axes[0].boxplot(box_data, labels=domain_order, patch_artist=True)
        axes[0].set_title(f"Boxplot - {feature}")
        axes[0].tick_params(axis="x", rotation=20)

        if HAS_SEABORN:
            sns.violinplot(
                data=valid,
                x="domain",
                y=feature,
                order=domain_order,
                ax=axes[1],
                palette="Pastel1",
                cut=0,
                inner="quartile",
            )
        else:
            violin_data = [
                valid.loc[valid["domain"] == domain, feature].dropna().values
                for domain in domain_order
            ]
            axes[1].violinplot(violin_data, showmeans=False, showmedians=True)
            axes[1].set_xticks(np.arange(1, len(domain_order) + 1))
            axes[1].set_xticklabels(domain_order, rotation=20)
        axes[1].set_title(f"Violin - {feature}")
        axes[1].tick_params(axis="x", rotation=20)

        if HAS_SEABORN:
            sns.histplot(
                data=valid,
                x=feature,
                hue="domain",
                bins=bins,
                stat="density",
                common_norm=False,
                element="step",
                ax=axes[2],
                palette="tab10",
            )
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(domain_order)))
            for idx, domain in enumerate(domain_order):
                values = valid.loc[valid["domain"] == domain, feature].dropna().values
                if len(values) == 0:
                    continue
                axes[2].hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    label=domain,
                    color=colors[idx],
                )
            axes[2].legend()
        axes[2].set_title(f"Istogramma per dominio - {feature}")

        fig.suptitle(f"Confronto distribuzioni - {feature}", y=1.03, fontsize=13)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"distribution_{feature}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def run_eda(
    combined_df: pd.DataFrame,
    config: WorkflowConfig,
    fig_eda_dir: Path,
    tables_dir: Path,
) -> None:
    print_section("6) EDA comparativa tra domini")

    plot_domain_sizes(
        combined_df, config.domain_order, fig_eda_dir / "domain_sizes.png"
    )
    plot_target_distribution(
        combined_df,
        config.domain_order,
        fig_eda_dir / "target_distribution_by_domain.png",
    )

    stats = descriptive_stats_by_domain(combined_df, config.key_numeric_for_eda)
    stats.to_csv(tables_dir / "descriptive_stats_key_numeric_by_domain.csv")
    print("[INFO] Statistiche descrittive (feature numeriche chiave) salvate.")

    plot_feature_distributions(
        combined_df=combined_df,
        features=config.key_numeric_for_eda,
        domain_order=config.domain_order,
        output_dir=fig_eda_dir,
        bins=config.histogram_bins,
    )
    print(f"[INFO] Figure EDA salvate in: {fig_eda_dir}")


def missing_matrix_by_domain(
    combined_df: pd.DataFrame, columns: List[str], domain_order: List[str]
) -> pd.DataFrame:
    rows = []
    for domain in domain_order:
        subset = combined_df.loc[combined_df["domain"] == domain, columns]
        pct = subset.isna().mean() * 100
        pct.name = domain
        rows.append(pct)
    matrix = pd.DataFrame(rows)
    matrix.index.name = "domain"
    return matrix


def plot_heatmap_generic(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str,
    value_fmt: str,
    cbar_label: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(
        figsize=(max(8, 0.9 * matrix_df.shape[1]), max(5, 0.6 * matrix_df.shape[0] + 2))
    )
    if HAS_SEABORN:
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt=value_fmt,
            cmap=cmap,
            cbar_kws={"label": cbar_label},
        )
    else:
        ax = plt.gca()
        mat = matrix_df.to_numpy(dtype=float)
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(matrix_df.shape[1]))
        ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(matrix_df.shape[0]))
        ax.set_yticklabels(matrix_df.index)

        fmt = "{:" + value_fmt + "}"
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text = "nan" if np.isnan(value) else fmt.format(value)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_data_quality(
    combined_df: pd.DataFrame,
    config: WorkflowConfig,
    fig_quality_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    print_section("7) Analisi qualita` dati e missingness")

    quality_cols = config.predictor_columns + ["target"]
    missing_matrix = missing_matrix_by_domain(
        combined_df=combined_df,
        columns=quality_cols,
        domain_order=config.domain_order,
    )
    missing_matrix.to_csv(tables_dir / "missing_pct_domain_x_feature.csv")

    plot_heatmap_generic(
        matrix_df=missing_matrix,
        title="Missingness (%) per dominio e variabile",
        output_path=fig_quality_dir / "missingness_heatmap_domain_feature.png",
        cmap="Reds",
        value_fmt=".1f",
        cbar_label="Missing %",
        xlabel="Variabile",
        ylabel="Dominio",
    )

    missing_range = (
        missing_matrix.max(axis=0) - missing_matrix.min(axis=0)
    ).sort_values(ascending=False)
    missing_range_df = missing_range.rename("missing_range_pct").to_frame()
    missing_range_df.to_csv(tables_dir / "missingness_range_by_feature.csv")

    domain_completeness = missing_matrix.mean(axis=1).sort_values()
    completeness_df = domain_completeness.rename("avg_missing_pct").to_frame()
    completeness_df.to_csv(tables_dir / "domain_completeness_summary.csv")

    print("[SINTESI QUALITA` DATI]")
    print("- Domini piu` completi (minor missing medio):")
    print(completeness_df.to_string())
    print("- Variabili con maggiore differenza di missingness tra domini:")
    print(missing_range_df.head(8).to_string())

    best_domain = domain_completeness.index[0]
    worst_domain = domain_completeness.index[-1]
    top_problem_features = ", ".join(missing_range.head(5).index.tolist())

    print(
        f"- Dominio mediamente piu` completo: {best_domain}. "
        f"Dominio mediamente meno completo: {worst_domain}."
    )
    print(
        f"- Feature potenzialmente problematiche per transportability: {top_problem_features}. "
        "Differenze di missingness possono alterare robustezza cross-domain."
    )

    return missing_matrix


def select_features_for_dashi(
    combined_df: pd.DataFrame, config: WorkflowConfig, tables_dir: Path
) -> List[str]:
    print_section(
        "8) Preparazione per DASHI (selezione feature e versioni raw/preprocessate)"
    )

    candidate_features = config.predictor_columns
    global_missing = combined_df[candidate_features].isna().mean()

    per_domain_missing = (
        missing_matrix_by_domain(
            combined_df=combined_df,
            columns=candidate_features,
            domain_order=config.domain_order,
        )
        / 100.0
    )

    rows = []
    selected = []

    for feat in candidate_features:
        reasons = []
        gm = float(global_missing[feat])
        if gm > config.missing_threshold_for_feature_drop:
            reasons.append(
                f"missing globale {gm:.1%} > soglia {config.missing_threshold_for_feature_drop:.0%}"
            )

        fully_missing_domains = per_domain_missing.index[
            per_domain_missing[feat] >= 1.0
        ].tolist()
        if fully_missing_domains:
            reasons.append(
                "completamente mancante in dominio/i: "
                + ", ".join(fully_missing_domains)
            )

        status = "selected" if len(reasons) == 0 else "excluded"
        if status == "selected":
            selected.append(feat)

        rows.append(
            {
                "feature": feat,
                "status": status,
                "global_missing_pct": round(gm * 100, 2),
                "reason_if_excluded": " | ".join(reasons),
            }
        )

    report = pd.DataFrame(rows).sort_values(["status", "global_missing_pct"])
    report.to_csv(tables_dir / "feature_selection_for_dashi.csv", index=False)

    if not selected:
        raise RuntimeError(
            "[ERRORE] Nessuna feature selezionata per DASHI. "
            "Controllare soglia missing o qualita` dati."
        )

    print(f"[INFO] Feature candidate: {len(candidate_features)}")
    print(f"[INFO] Feature selezionate per DASHI: {len(selected)} -> {selected}")
    excluded = report.loc[
        report["status"] == "excluded", ["feature", "reason_if_excluded"]
    ]
    if not excluded.empty:
        print("[INFO] Feature escluse e motivazione:")
        print(excluded.to_string(index=False))
    else:
        print("[INFO] Nessuna feature esclusa.")

    return selected


def median_impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def zscore_standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True, ddof=0).replace(0, 1.0)
    standardized = (df - means) / stds
    return standardized


def prepare_domain_matrices(
    domain_dfs: Dict[str, pd.DataFrame],
    selected_features: List[str],
    data_dir: Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    domain_matrices: Dict[str, Dict[str, pd.DataFrame]] = {}

    for domain, df in domain_dfs.items():
        valid_df = df.dropna(subset=["target_bin"]).copy()
        valid_df["target_bin"] = valid_df["target_bin"].astype(int)

        x_raw = valid_df[selected_features].copy()
        y = valid_df["target_bin"].copy()

        if HAS_SKLEARN:
            imputer = SimpleImputer(strategy="median")
            x_imputed = pd.DataFrame(
                imputer.fit_transform(x_raw),
                columns=selected_features,
                index=x_raw.index,
            )
            scaler = StandardScaler()
            x_standardized = pd.DataFrame(
                scaler.fit_transform(x_imputed),
                columns=selected_features,
                index=x_imputed.index,
            )
        else:
            # Fallback senza sklearn: imputazione mediana + z-score con pandas/numpy.
            x_imputed = median_impute_dataframe(x_raw)
            x_standardized = zscore_standardize_dataframe(x_imputed)

        domain_matrices[domain] = {
            "X_raw": x_raw,
            "X_imputed": x_imputed,
            "X_standardized": x_standardized,
            "y": y.to_frame("target_bin"),
        }

        x_raw.to_csv(data_dir / f"{domain}_X_raw.csv", index=False)
        x_imputed.to_csv(data_dir / f"{domain}_X_imputed.csv", index=False)
        x_standardized.to_csv(data_dir / f"{domain}_X_standardized.csv", index=False)
        y.to_frame("target_bin").to_csv(data_dir / f"{domain}_y.csv", index=False)

        print(
            f"[INFO] Matrici salvate per {domain}: "
            f"X_raw {x_raw.shape}, X_imputed {x_imputed.shape}, y {y.shape}"
        )

    return domain_matrices


def build_pooled_descriptive_view(
    combined_df: pd.DataFrame, selected_features: List[str], data_dir: Path
) -> pd.DataFrame:
    pooled_view = combined_df[
        ["domain", "target", "target_bin"] + selected_features
    ].copy()

    # Vista preprocessata pooled solo per confronti descrittivi/visivi.
    # Non deve essere usata come input diretto per valutazione LODO.
    if HAS_SKLEARN:
        imputer = SimpleImputer(strategy="median")
        pooled_view[selected_features] = imputer.fit_transform(
            pooled_view[selected_features]
        )
    else:
        pooled_view[selected_features] = median_impute_dataframe(
            pooled_view[selected_features]
        )

    pooled_view.to_csv(data_dir / "pooled_descriptive_imputed.csv", index=False)
    print(
        "[INFO] Salvata vista pooled imputata (uso descrittivo, non per eval predittiva)."
    )
    return pooled_view


def standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt((var_x + var_y) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(x) - np.mean(y)) / pooled_std)


def jensen_shannon_divergence(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan

    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    if np.isclose(min_val, max_val):
        return 0.0

    hist_range = (min_val, max_val)
    p, _ = np.histogram(x, bins=bins, range=hist_range, density=False)
    q, _ = np.histogram(y, bins=bins, range=hist_range, density=False)

    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(js)


def compute_domain_shift(
    combined_df: pd.DataFrame,
    features: List[str],
    domain_order: List[str],
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows = []

    for dom_a, dom_b in combinations(domain_order, 2):
        subset_a = combined_df.loc[combined_df["domain"] == dom_a]
        subset_b = combined_df.loc[combined_df["domain"] == dom_b]

        for feat in features:
            x = subset_a[feat].dropna().to_numpy(dtype=float)
            y = subset_b[feat].dropna().to_numpy(dtype=float)

            if len(x) == 0 or len(y) == 0:
                smd = np.nan
                js = np.nan
                abs_mean_diff = np.nan
                abs_std_diff = np.nan
            else:
                smd = standardized_mean_difference(x, y)
                js = jensen_shannon_divergence(x, y, bins=bins)
                abs_mean_diff = float(abs(np.mean(x) - np.mean(y)))
                abs_std_diff = float(abs(np.std(x, ddof=1) - np.std(y, ddof=1)))

            feature_rows.append(
                {
                    "domain_a": dom_a,
                    "domain_b": dom_b,
                    "feature": feat,
                    "n_a": len(x),
                    "n_b": len(y),
                    "smd": smd,
                    "js_divergence": js,
                    "abs_mean_diff": abs_mean_diff,
                    "abs_std_diff": abs_std_diff,
                }
            )

    feature_shift_df = pd.DataFrame(feature_rows)

    pair_summary = (
        feature_shift_df.groupby(["domain_a", "domain_b"], as_index=False)
        .agg(
            mean_smd=("smd", "mean"),
            mean_js=("js_divergence", "mean"),
            mean_abs_mean_diff=("abs_mean_diff", "mean"),
            mean_abs_std_diff=("abs_std_diff", "mean"),
            features_used=("feature", "count"),
        )
        .copy()
    )

    # JS e` in [0, ln(2)] -> normalizzazione per combinarla con SMD.
    pair_summary["combined_shift_score"] = pair_summary["mean_smd"].fillna(
        0.0
    ) + pair_summary["mean_js"].fillna(0.0) / np.log(2.0)

    matrix = pd.DataFrame(np.nan, index=domain_order, columns=domain_order, dtype=float)
    for d in domain_order:
        matrix.loc[d, d] = 0.0

    for _, row in pair_summary.iterrows():
        d_a = row["domain_a"]
        d_b = row["domain_b"]
        score = row["combined_shift_score"]
        matrix.loc[d_a, d_b] = score
        matrix.loc[d_b, d_a] = score

    return feature_shift_df, pair_summary, matrix


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame, title: str, output_path: Path, cmap: str = "viridis"
) -> None:
    plot_heatmap_generic(
        matrix_df=matrix_df,
        title=title,
        output_path=output_path,
        cmap=cmap,
        value_fmt=".3f",
        cbar_label="Score",
        xlabel="Dominio",
        ylabel="Dominio",
    )


def run_domain_shift_analysis(
    combined_df: pd.DataFrame,
    selected_features: List[str],
    config: WorkflowConfig,
    fig_shift_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    print_section("9) Analisi quantitativa proxy del domain shift")

    feature_shift_df, pair_summary, shift_matrix = compute_domain_shift(
        combined_df=combined_df,
        features=selected_features,
        domain_order=config.domain_order,
        bins=config.histogram_bins,
    )

    feature_shift_df.to_csv(tables_dir / "domain_shift_feature_level.csv", index=False)
    pair_summary.to_csv(tables_dir / "domain_shift_pair_summary.csv", index=False)
    shift_matrix.to_csv(
        tables_dir / "domain_shift_matrix_domain_x_domain.csv", index=True
    )

    plot_matrix_heatmap(
        shift_matrix,
        "Domain Shift Matrix (combined score)",
        fig_shift_dir / "domain_shift_matrix_heatmap.png",
        cmap="magma_r",
    )

    print("[INFO] Prime righe summary domain shift:")
    print(
        pair_summary.sort_values("combined_shift_score", ascending=False)
        .head()
        .to_string(index=False)
    )
    print(f"[INFO] Matrice domain-vs-domain salvata in: {tables_dir}")
    return shift_matrix


def build_model_pipelines(seed: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def compute_classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def run_leave_one_domain_out(
    combined_df: pd.DataFrame,
    features: List[str],
    domain_order: List[str],
    pipelines: Dict[str, Pipeline],
) -> pd.DataFrame:
    rows = []

    work_df = combined_df.dropna(subset=["target_bin"]).copy()
    work_df["target_bin"] = work_df["target_bin"].astype(int)

    for test_domain in domain_order:
        train_df = work_df.loc[work_df["domain"] != test_domain].copy()
        test_df = work_df.loc[work_df["domain"] == test_domain].copy()

        if train_df.empty or test_df.empty:
            print(f"[WARN] Fold LODO saltato per {test_domain}: train o test vuoto.")
            continue
        if train_df["target_bin"].nunique() < 2:
            print(
                f"[WARN] Fold LODO saltato per {test_domain}: train con una sola classe."
            )
            continue

        x_train = train_df[features]
        y_train = train_df["target_bin"]
        x_test = test_df[features]
        y_test = test_df["target_bin"]

        print(
            f"[INFO] LODO fold test={test_domain}: n_train={len(train_df)}, "
            f"n_test={len(test_df)}, train_domains={sorted(train_df['domain'].unique())}"
        )

        for model_name, pipeline in pipelines.items():
            clf = clone(pipeline)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            y_score = None
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(x_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(x_test)

            metrics = compute_classification_metrics(y_test, y_pred, y_score)
            rows.append(
                {
                    "evaluation": "leave_one_domain_out",
                    "model": model_name,
                    "test_domain": test_domain,
                    "train_domains": ",".join(sorted(train_df["domain"].unique())),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "test_positive_prevalence": float(y_test.mean()),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def run_pooled_cv(
    combined_df: pd.DataFrame,
    features: List[str],
    pipelines: Dict[str, Pipeline],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    work_df = combined_df.dropna(subset=["target_bin"]).copy()
    work_df["target_bin"] = work_df["target_bin"].astype(int)

    x = work_df[features]
    y = work_df["target_bin"]

    min_class_count = int(y.value_counts().min())
    n_splits_eff = min(n_splits, min_class_count)
    if n_splits_eff < 2:
        raise ValueError(
            "[ERRORE] Impossibile fare pooled CV: classe minoritaria con meno di 2 campioni."
        )

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for model_name, pipeline in pipelines.items():
            clf = clone(pipeline)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            y_score = None
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(x_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(x_test)

            metrics = compute_classification_metrics(y_test, y_pred, y_score)
            rows.append(
                {
                    "evaluation": "pooled_stratified_cv",
                    "model": model_name,
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def plot_lodo_metric_bars(lodo_df: pd.DataFrame, fig_modeling_dir: Path) -> None:
    if lodo_df.empty:
        return

    plot_df = lodo_df.melt(
        id_vars=["model", "test_domain"],
        value_vars=["accuracy", "balanced_accuracy", "roc_auc", "f1"],
        var_name="metric",
        value_name="value",
    )

    if HAS_SEABORN:
        g = sns.catplot(
            data=plot_df,
            x="test_domain",
            y="value",
            hue="model",
            col="metric",
            kind="bar",
            col_wrap=2,
            height=4,
            aspect=1.3,
            sharey=False,
            palette="Set2",
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Test domain", "Score")
        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(fig_modeling_dir / "lodo_metrics_by_domain.png", dpi=300)
        plt.close()
        return

    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]
    models = sorted(plot_df["model"].unique())
    domains = sorted(plot_df["test_domain"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for idx_metric, metric in enumerate(metrics):
        ax = axes[idx_metric]
        metric_df = plot_df.loc[plot_df["metric"] == metric].copy()
        pivot = metric_df.pivot(
            index="test_domain", columns="model", values="value"
        ).reindex(domains)

        x = np.arange(len(domains))
        width = 0.75 / max(1, len(models))

        for idx_model, model in enumerate(models):
            values = (
                pivot[model].values
                if model in pivot.columns
                else np.zeros(len(domains))
            )
            pos = x - 0.375 + width / 2 + idx_model * width
            ax.bar(pos, values, width=width, label=model, color=colors[idx_model])

        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=20)
        ax.set_title(metric)
        ax.set_xlabel("Test domain")
        ax.set_ylabel("Score")
        if idx_metric == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(fig_modeling_dir / "lodo_metrics_by_domain.png", dpi=300)
    plt.close(fig)


def run_modeling(
    combined_df: pd.DataFrame,
    selected_features: List[str],
    config: WorkflowConfig,
    fig_modeling_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    print_section("10) Modellazione baseline per transportability")
    if not HAS_SKLEARN:
        print(
            "[WARN] `scikit-learn` non disponibile: step di modeling saltato. "
            "Per abilitarlo usa l'interprete corretto o installa scikit-learn in "
            "`/usr/local/bin/python3`."
        )
        skipped = pd.DataFrame(
            [
                {
                    "evaluation": "modeling_skipped",
                    "reason": "missing_scikit_learn",
                    "install_hint": "python -m pip install scikit-learn",
                }
            ]
        )
        skipped.to_csv(tables_dir / "modeling_skipped_missing_sklearn.csv", index=False)
        return skipped

    pipelines = build_model_pipelines(seed=config.random_seed)

    lodo_results = run_leave_one_domain_out(
        combined_df=combined_df,
        features=selected_features,
        domain_order=config.domain_order,
        pipelines=pipelines,
    )

    pooled_cv_results = run_pooled_cv(
        combined_df=combined_df,
        features=selected_features,
        pipelines=pipelines,
        n_splits=config.pooled_cv_folds,
        seed=config.random_seed,
    )

    all_results = pd.concat(
        [lodo_results, pooled_cv_results], ignore_index=True, sort=False
    )
    all_results.to_csv(tables_dir / "modeling_all_results.csv", index=False)

    lodo_summary = summarize_metrics(lodo_results, group_cols=["model", "test_domain"])
    pooled_summary = summarize_metrics(pooled_cv_results, group_cols=["model"])

    lodo_summary.to_csv(tables_dir / "modeling_lodo_summary.csv", index=False)
    pooled_summary.to_csv(tables_dir / "modeling_pooled_cv_summary.csv", index=False)

    plot_lodo_metric_bars(lodo_results, fig_modeling_dir)

    print("[INFO] Risultati LODO (prime righe):")
    print(lodo_results.head().to_string(index=False))
    print("[INFO] Summary LODO:")
    print(lodo_summary.to_string(index=False))
    print("[INFO] Summary pooled CV:")
    print(pooled_summary.to_string(index=False))

    return all_results


def build_domain_summary_table(
    domain_dfs: Dict[str, pd.DataFrame], combined_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for domain, df in domain_dfs.items():
        pred_missing_rate = float(df.isna().mean().mean() * 100)
        valid_y = df["target_bin"].dropna()
        prevalence = float(valid_y.mean()) if not valid_y.empty else np.nan
        rows.append(
            {
                "domain": domain,
                "n_samples": int(len(df)),
                "overall_missing_pct": round(pred_missing_rate, 3),
                "target1_prevalence": round(prevalence, 4)
                if not np.isnan(prevalence)
                else np.nan,
            }
        )

    valid_y_global = combined_df["target_bin"].dropna()
    rows.append(
        {
            "domain": "global",
            "n_samples": int(len(combined_df)),
            "overall_missing_pct": round(
                float(combined_df.isna().mean().mean() * 100), 3
            ),
            "target1_prevalence": round(float(valid_y_global.mean()), 4)
            if not valid_y_global.empty
            else np.nan,
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    set_plot_theme()

    config = WorkflowConfig()

    print_section("CONFIG")
    print(f"[INFO] Output root: {config.output_root}")
    print("[INFO] Domini e path utilizzati:")
    for d in config.domain_order:
        print(f"  - {d}: {config.domain_paths[d]}")
    print(
        f"[INFO] Soglia esclusione feature per missing globale: "
        f"{config.missing_threshold_for_feature_drop:.0%}"
    )
    print(
        "[INFO] scikit-learn disponibile: "
        f"{HAS_SKLEARN} (necessario per baseline LODO e pooled CV)."
    )

    set_reproducibility(config.random_seed)
    out_dirs = create_output_dirs(config.output_root)

    # 1-3) Load schema + domini
    raw_domain_dfs, _ = load_all_domains(config)

    # 4-5) Cleaning + target binario
    cleaned_domain_dfs: Dict[str, pd.DataFrame] = {}
    for domain in config.domain_order:
        cleaned = clean_dataframe(raw_domain_dfs[domain], config.numeric_columns)
        cleaned = add_binary_target(cleaned)
        cleaned_domain_dfs[domain] = cleaned

    cleaned_combined_df = pd.concat(
        [cleaned_domain_dfs[d] for d in config.domain_order],
        ignore_index=True,
        sort=False,
    )

    print_and_save_domain_overview(
        domain_dfs=cleaned_domain_dfs,
        combined_df=cleaned_combined_df,
        tables_dir=out_dirs["tables"],
    )

    save_cleaned_data(
        domain_dfs=cleaned_domain_dfs,
        combined_df=cleaned_combined_df,
        data_dir=out_dirs["data"],
    )

    print_section("5) Target binario e distribuzione classi")
    target_dist = target_distribution_table(cleaned_domain_dfs, cleaned_combined_df)
    print(target_dist.to_string(index=False))
    target_dist.to_csv(
        out_dirs["tables"] / "target_distribution_by_domain.csv", index=False
    )

    # 6) EDA comparativa
    run_eda(
        combined_df=cleaned_combined_df,
        config=config,
        fig_eda_dir=out_dirs["fig_eda"],
        tables_dir=out_dirs["tables"],
    )

    # 7) Qualita` dati
    analyze_data_quality(
        combined_df=cleaned_combined_df,
        config=config,
        fig_quality_dir=out_dirs["fig_quality"],
        tables_dir=out_dirs["tables"],
    )

    # 8) Preparazione DASHI
    selected_features = select_features_for_dashi(
        combined_df=cleaned_combined_df,
        config=config,
        tables_dir=out_dirs["tables"],
    )
    prepare_domain_matrices(
        domain_dfs=cleaned_domain_dfs,
        selected_features=selected_features,
        data_dir=out_dirs["data"],
    )
    build_pooled_descriptive_view(
        combined_df=cleaned_combined_df,
        selected_features=selected_features,
        data_dir=out_dirs["data"],
    )

    # 9) Domain shift
    run_domain_shift_analysis(
        combined_df=cleaned_combined_df,
        selected_features=selected_features,
        config=config,
        fig_shift_dir=out_dirs["fig_shift"],
        tables_dir=out_dirs["tables"],
    )

    # 10) Baseline modeling transportability
    run_modeling(
        combined_df=cleaned_combined_df,
        selected_features=selected_features,
        config=config,
        fig_modeling_dir=out_dirs["fig_modeling"],
        tables_dir=out_dirs["tables"],
    )

    # 11) Tabella finale sintetica dominio
    print_section("11) Output finale e tabella sintetica per dominio")
    domain_summary = build_domain_summary_table(cleaned_domain_dfs, cleaned_combined_df)
    print(domain_summary.to_string(index=False))
    domain_summary.to_csv(out_dirs["tables"] / "domain_summary_table.csv", index=False)

    print("\n[COMPLETATO]")
    print(f"- Output root: {out_dirs['root']}")
    print(f"- Dati puliti: {out_dirs['data']}")
    print(f"- Tabelle CSV: {out_dirs['tables']}")
    print(f"- Figure: {out_dirs['figures']}")
    print(
        "- Nota: i 4 dataset restano domini distinti in tutte le analisi; "
        "la concatenazione e` solo tecnica e preserva `domain`."
    )


if __name__ == "__main__":
    main()
