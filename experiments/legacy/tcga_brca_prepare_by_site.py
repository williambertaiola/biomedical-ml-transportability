#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preparazione TCGA-BRCA per site (senza classificazione)
========================================================

Obiettivo dello script:
- leggere il dataset completo TCGA-BRCA (.tsv)
- riconoscere l'orientamento (geni x campioni oppure campioni x feature)
- ottenere una tabella finale con una riga per campione
- estrarre dal barcode:
  - site (TSS code)
  - sample_type_code (2 cifre: 01, 06, 11, ...)
- identificare i site "grandi" (>= MIN_SAMPLES_PER_SITE)
- salvare un CSV separato per ogni site grande
- salvare un file di summary per tutti i site

Nota importante:
- Qui NON facciamo classificazione/transductive learning.
- Non modifichiamo i counts e non facciamo feature selection.
- Lo scopo e solo preparare database separati per dominio/site.
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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Config globale
# ============================================================================

# Soglia minima per definire un site "grande"
MIN_SAMPLES_PER_SITE = 30

# Percorsi di default richiesti
DEFAULT_INPUT = Path(_configured_path("tcga_expression"))
DEFAULT_OUTPUT_DIR = Path(_configured_path("legacy_tcga_brca_by_site"))

# Nomi plausibili della colonna barcode quando il dataset e gia orientato per campioni
BARCODE_CANDIDATE_NAMES = [
    "sample_barcode",
    "sample_id",
    "barcode",
    "tcga_barcode",
    "aliquot_barcode",
    "id",
]


# ============================================================================
# Utility barcode
# ============================================================================


def is_tcga_like_barcode(value: object) -> bool:
    """
    Controlla se una stringa assomiglia a un barcode TCGA.

    Formato atteso minimo:
    TCGA-XX-YYYY-...
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    text = str(value).strip().upper()
    return bool(re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-", text))


def parse_tcga_barcode(
    barcode: object,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Estrae dal barcode TCGA:
    - site (TSS) -> secondo blocco
    - sample_type_code -> prime 2 cifre del quarto blocco

    Esempio:
    TCGA-ZZ-0000-01A -> site=BH, sample_type_code=01

    Restituisce anche un eventuale messaggio di errore in caso di parsing fallito.
    """
    if barcode is None or (isinstance(barcode, float) and np.isnan(barcode)):
        return None, None, "barcode_missing"

    text = str(barcode).strip().upper()
    if text == "":
        return None, None, "barcode_empty"

    # Normalizziamo separatori eventuali
    normalized = text.replace("_", "-").replace(".", "-")
    parts = normalized.split("-")

    if len(parts) < 4:
        return None, None, "barcode_too_short"

    if parts[0] != "TCGA":
        return None, None, "barcode_not_tcga"

    site = parts[1]
    sample_token = parts[3]

    match_sample_type = re.match(r"(\d{2})", sample_token)
    if match_sample_type is None:
        return site, None, "sample_type_not_found"

    sample_type_code = match_sample_type.group(1)
    return site, sample_type_code, None


# ============================================================================
# Riconoscimento orientamento e barcode
# ============================================================================


def detect_orientation(df: pd.DataFrame) -> str:
    """
    Riconosce in modo euristico l'orientamento del file.

    Regola pratica:
    - se molte colonne hanno nome barcode TCGA e la prima colonna NON e barcode,
      assumiamo orientamento geni x campioni.
    - altrimenti assumiamo campioni x feature.
    """
    col_names = [str(c) for c in df.columns]
    first_col = col_names[0] if len(col_names) > 0 else ""

    n_sample_like_cols = sum(is_tcga_like_barcode(c) for c in col_names)

    # Soglia prudente: almeno 10 colonne simili a barcode
    if n_sample_like_cols >= 10 and not is_tcga_like_barcode(first_col):
        return "genes_x_samples"

    return "samples_x_features"


def find_barcode_column_in_samples_df(df: pd.DataFrame) -> Optional[str]:
    """
    Cerca una colonna barcode quando il dataframe e gia in formato campioni x feature.

    Strategia:
    - priorita ai nomi noti
    - fallback: colonna con alta quota di valori simili a barcode TCGA
    """
    # 1) Match diretto per nome
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    for candidate in BARCODE_CANDIDATE_NAMES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    # 2) Scoring per quota barcode-like
    best_col = None
    best_ratio = 0.0

    for col in df.columns:
        series = df[col]

        # Consideriamo soprattutto colonne testuali/categorical
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue

        values = series.astype("string")
        ratio = float(values.map(is_tcga_like_barcode).mean())

        if ratio > best_ratio:
            best_ratio = ratio
            best_col = str(col)

    if best_col is not None and best_ratio >= 0.5:
        return best_col

    return None


def convert_to_samples_by_rows(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Restituisce un dataframe con una riga per campione e una colonna esplicita sample_barcode.

    Output:
    - dataframe in formato campioni x feature
    - orientamento rilevato
    - nome colonna usata/creata come barcode
    """
    orientation = detect_orientation(df_raw)

    if orientation == "genes_x_samples":
        # Caso tipico TCGA star_counts.tsv:
        # prima colonna = ID gene, altre colonne = barcode campioni
        gene_id_col = str(df_raw.columns[0])

        # Impostiamo i geni come indice, poi trasponiamo
        expr = df_raw.set_index(gene_id_col)
        samples_df = expr.transpose().copy()

        # Il barcode e nell'indice dopo la trasposizione: lo rendiamo colonna esplicita
        samples_df.index.name = "sample_barcode"
        samples_df = samples_df.reset_index()

        barcode_col = "sample_barcode"
        return samples_df, orientation, barcode_col

    # Caso gia campioni x feature
    samples_df = df_raw.copy()

    # Se l'indice sembra contenere barcode, lo trasformiamo in colonna
    index_values = pd.Series(samples_df.index.astype(str))
    index_ratio = (
        float(index_values.map(is_tcga_like_barcode).mean())
        if len(index_values) > 0
        else 0.0
    )

    if index_ratio >= 0.5:
        samples_df = samples_df.reset_index().rename(
            columns={"index": "sample_barcode"}
        )
        return samples_df, orientation, "sample_barcode"

    # Altrimenti cerchiamo una colonna barcode
    barcode_col = find_barcode_column_in_samples_df(samples_df)
    if barcode_col is None:
        raise RuntimeError(
            "Impossibile identificare una colonna barcode TCGA valida nel dataset orientato per campioni."
        )

    # Uniformiamo il nome della colonna barcode per output coerente
    if barcode_col != "sample_barcode":
        samples_df = samples_df.rename(columns={barcode_col: "sample_barcode"})
        barcode_col = "sample_barcode"

    return samples_df, orientation, barcode_col


# ============================================================================
# Costruzione summary e salvataggi
# ============================================================================


def build_sites_summary(
    df_samples: pd.DataFrame, min_samples_per_site: int
) -> pd.DataFrame:
    """
    Costruisce tabella riepilogativa per site con conteggi sample_type.
    """
    rows: List[Dict[str, object]] = []

    for site_value, group in df_samples.groupby("site", dropna=False):
        n_samples = int(len(group))

        # Conteggi sample type specifici richiesti
        st_counts = group["sample_type_code"].value_counts(dropna=False)
        count_01 = int(st_counts.get("01", 0))
        count_06 = int(st_counts.get("06", 0))
        count_11 = int(st_counts.get("11", 0))

        other_count = n_samples - count_01 - count_06 - count_11

        is_large = bool(pd.notna(site_value) and n_samples >= min_samples_per_site)

        rows.append(
            {
                "site": site_value,
                "n_samples": n_samples,
                "is_large_site": int(is_large),
                "sample_type_01_count": count_01,
                "sample_type_06_count": count_06,
                "sample_type_11_count": count_11,
                "other_sample_types_count": int(other_count),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("n_samples", ascending=False).reset_index(drop=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split TCGA-BRCA per site grande con sample_type da barcode"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Percorso file TSV completo"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Cartella output"
    )
    parser.add_argument(
        "--min-samples-per-site",
        type=int,
        default=MIN_SAMPLES_PER_SITE,
        help="Soglia minima per definire un site grande",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    min_samples_per_site: int = int(args.min_samples_per_site)

    # Creiamo cartella output se non esiste
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("TCGA-BRCA PREPARAZIONE PER SITE")
    print("=" * 90)
    print(f"Input file: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"MIN_SAMPLES_PER_SITE: {min_samples_per_site}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input non trovato: {input_path}")

    # ---------------------------------------------------------------------
    # 1) Caricamento file TSV
    # ---------------------------------------------------------------------
    print("\n[1/8] Lettura file TSV...")
    df_raw = pd.read_csv(input_path, sep="\t", low_memory=False)

    print(f"Shape raw: {df_raw.shape}")
    print(f"Prime 15 colonne: {df_raw.columns[:15].tolist()}")
    print("Prime 3 righe (prime 8 colonne):")
    print(df_raw.iloc[:3, :8].to_string(index=False))

    # ---------------------------------------------------------------------
    # 2) Conversione in formato campioni x feature + barcode esplicito
    # ---------------------------------------------------------------------
    print("\n[2/8] Rilevamento orientamento e conversione...")
    df_samples, orientation, barcode_col = convert_to_samples_by_rows(df_raw)

    print(f"Orientamento rilevato: {orientation}")
    print(f"Colonna barcode usata: {barcode_col}")
    print(f"Shape campioni x feature: {df_samples.shape}")

    # Normalizziamo il nome della colonna barcode per coerenza
    if barcode_col != "sample_barcode":
        df_samples = df_samples.rename(columns={barcode_col: "sample_barcode"})

    # ---------------------------------------------------------------------
    # 3) Parsing barcode -> site e sample_type_code
    # ---------------------------------------------------------------------
    print("\n[3/8] Estrazione site e sample_type_code dal barcode...")

    # Applichiamo parsing riga per riga, senza interrompere in caso di barcode non validi
    parsed = df_samples["sample_barcode"].apply(parse_tcga_barcode)

    df_samples["site"] = parsed.map(lambda x: x[0])
    df_samples["sample_type_code"] = parsed.map(lambda x: x[1])
    parse_error = parsed.map(lambda x: x[2])

    # Log barcode non parseabili
    invalid_mask = parse_error.notna()
    n_invalid = int(invalid_mask.sum())

    if n_invalid > 0:
        invalid_df = df_samples.loc[invalid_mask, ["sample_barcode"]].copy()
        invalid_df["parse_error"] = parse_error[invalid_mask].values
        invalid_log_path = output_dir / "tcga_brca_invalid_barcodes.csv"
        invalid_df.to_csv(invalid_log_path, index=False)

        print(f"WARNING: barcode non parseabili: {n_invalid}")
        print(f"Dettagli salvati in: {invalid_log_path}")
        print("Esempi barcode non parseabili:")
        print(invalid_df.head(10).to_string(index=False))
    else:
        print("Tutti i barcode risultano parseabili secondo il formato TCGA atteso.")

    # ---------------------------------------------------------------------
    # 4) Distribuzioni site e sample_type
    # ---------------------------------------------------------------------
    print("\n[4/8] Distribuzione campioni per site...")
    site_counts = (
        df_samples["site"].value_counts(dropna=False).sort_values(ascending=False)
    )
    print(site_counts.to_string())

    print("\nDistribuzione sample_type_code:")
    sample_type_counts = (
        df_samples["sample_type_code"]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
    )
    print(sample_type_counts.to_string())

    # ---------------------------------------------------------------------
    # 5) Definizione site grandi
    # ---------------------------------------------------------------------
    print("\n[5/8] Identificazione site grandi...")
    valid_site_counts = (
        df_samples.loc[df_samples["site"].notna(), "site"]
        .value_counts()
        .sort_values(ascending=False)
    )
    large_sites = valid_site_counts[valid_site_counts >= min_samples_per_site]

    n_total_sites = int(valid_site_counts.shape[0])
    n_large_sites = int(large_sites.shape[0])

    print(f"Numero totale site (non-null): {n_total_sites}")
    print(f"Numero site grandi (>= {min_samples_per_site}): {n_large_sites}")

    if n_large_sites > 0:
        print("Elenco site grandi con numerosita:")
        print(large_sites.to_string())
    else:
        print(
            "Nessun site raggiunge la soglia minima: nessun file site-specific verra creato."
        )

    # ---------------------------------------------------------------------
    # 6) Summary globale per site
    # ---------------------------------------------------------------------
    print("\n[6/8] Creazione file summary site...")
    summary_df = build_sites_summary(df_samples, min_samples_per_site)
    summary_path = output_dir / "tcga_brca_sites_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary salvato in: {summary_path}")

    # ---------------------------------------------------------------------
    # 7) Creazione database separati per ciascun site grande
    # ---------------------------------------------------------------------
    print("\n[7/8] Salvataggio file separati per site grande...")

    # Manteniamo ordine colonne richiesto: metadata davanti, poi tutte le feature
    metadata_cols = ["sample_barcode", "site", "sample_type_code"]
    feature_cols = [c for c in df_samples.columns if c not in metadata_cols]
    ordered_cols = metadata_cols + feature_cols

    created_files: List[Path] = []

    for site_code, n_samples in large_sites.items():
        site_df = df_samples.loc[df_samples["site"] == site_code, ordered_cols].copy()

        # Controlli finali richiesti prima di salvare
        n_unique_site = int(site_df["site"].nunique(dropna=False))
        has_sample_type = "sample_type_code" in site_df.columns

        if n_unique_site != 1:
            print(
                f"WARNING: site {site_code} ha n_unique_site={n_unique_site}, file non salvato."
            )
            continue

        if not has_sample_type:
            print(
                f"WARNING: site {site_code} senza colonna sample_type_code, file non salvato."
            )
            continue

        file_name = f"tcga_brca_site_{site_code}.csv"
        out_path = output_dir / file_name
        site_df.to_csv(out_path, index=False)

        created_files.append(out_path)

        print(
            f"Salvato: {out_path.name} | shape={site_df.shape} | "
            f"site_unico={site_df['site'].iloc[0]} | sample_type_col_presente={has_sample_type}"
        )

    # ---------------------------------------------------------------------
    # 8) Report finale
    # ---------------------------------------------------------------------
    print("\n[8/8] Completato")
    print(f"Database separati creati: {len(created_files)}")
    print(f"Cartella output: {output_dir}")

    if len(created_files) > 0:
        print("File creati:")
        for p in created_files:
            print(f"- {p.name}")


if __name__ == "__main__":
    main()
