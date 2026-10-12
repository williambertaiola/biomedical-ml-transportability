#!/usr/bin/env python3
"""
Analisi dataset shift multi-source con DASHI su dataset UCI unificato.

Esecuzione:
    python3 run_dashi_uci_multisource.py
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


import os
from pathlib import Path
from typing import Optional

# Evita problemi in ambienti headless/sandbox.
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import dashi


# =============================================================================
# CONFIGURAZIONE (modifica facilmente qui)
# =============================================================================
DATASET_PATH = Path(_configured_path("legacy_input_uci.csv"))
OUTPUT_DIR = Path("dashi_results")

# Parametri preprocessing
DROP_ROWS_WITH_MISSING_SOURCE = True
REMOVE_SOURCES_WITH_LT2_SAMPLES = True
STANDARDIZE_FEATURES = True

# Parametri DASHI
KDE_RESOLUTION = 20
MDS_DIMENSIONS = 2
DIM_REDUCTION = "PCA"


def _find_column_case_insensitive(
    df: pd.DataFrame, candidates: list[str]
) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _coerce_numeric_features(df: pd.DataFrame, excluded_cols: set[str]) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
            features.append(col)
    return features


def _js_distance_matrix(probability_map: np.ndarray) -> np.ndarray:
    """Calcola matrice distanze JS (stessa distanza usata in MSV)."""
    p = np.asarray(probability_map, dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    n = p.shape[0]
    out = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            pi = p[i]
            pj = p[j]
            m = 0.5 * (pi + pj)
            kl_pm = np.sum(pi * np.log(pi / m))
            kl_qm = np.sum(pj * np.log(pj / m))
            js = 0.5 * (kl_pm + kl_qm)
            d = float(np.sqrt(max(js, 0.0)))
            out[i, j] = d
            out[j, i] = d
    return out


def _save_plotly_with_fallback(
    fig,
    png_path: Path,
    html_path: Path,
    fallback_png_fn,
    label: str,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
    except Exception as exc:
        print(
            f"[WARN] PNG plotly non disponibile per {label} ({type(exc).__name__}). Uso fallback matplotlib."
        )
        fallback_png_fn()


def _plot_msv_fallback(
    vertices: np.ndarray,
    sources: list[str],
    spo: np.ndarray,
    n_by_source: np.ndarray,
    gpd: Optional[float],
    out_png: Path,
) -> None:
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return

    v = vertices.copy()
    if v.shape[1] == 1:
        v = np.hstack([v, np.zeros((v.shape[0], 1))])
    elif v.shape[1] > 2:
        v = v[:, :2]

    sizes = np.asarray(n_by_source, dtype=float)
    if sizes.size != v.shape[0]:
        sizes = np.ones(v.shape[0], dtype=float)
    sizes = 100 * (sizes / max(np.nanmax(sizes), 1.0)) + 40

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        v[:, 0], v[:, 1], c=spo, s=sizes, cmap="viridis", alpha=0.9, edgecolor="k"
    )
    for i, src in enumerate(sources):
        ax.annotate(
            src,
            (v[i, 0], v[i, 1]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )
    title = "MSV (fallback matplotlib)"
    if gpd is not None:
        title += f" | GPD={gpd:.4f}"
    ax.set_title(title)
    ax.set_xlabel("D1")
    ax.set_ylabel("D2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("SPO")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_dsm_fallback(
    multivariate_probability_map: np.ndarray, sources: list[str], out_png: Path
) -> None:
    maps = np.asarray(multivariate_probability_map, dtype=float)
    n_sources = maps.shape[0]
    n_cols = min(3, n_sources)
    n_rows = int(np.ceil(n_sources / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_sources):
        ax = axes[i]
        im = ax.imshow(maps[i], cmap="magma", aspect="auto")
        ax.set_title(str(sources[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_sources, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Dataset Shift Map (fallback matplotlib)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_distance_heatmap(distance_df: pd.DataFrame, out_png: Path) -> None:
    mat = distance_df.values.astype(float)
    labels = distance_df.index.tolist()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Heatmap distanze JS tra sorgenti")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distanza Jensen-Shannon")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> int:
    print("\n=== DASHI Multi-Source Dataset Shift Analysis ===")
    print(f"[INFO] Dataset path: {DATASET_PATH}")
    print(f"[INFO] Output dir:   {OUTPUT_DIR.resolve()}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Caricamento dataset
    df = pd.read_csv(DATASET_PATH, na_values=["?", "NA", "N/A", "null", "None", ""])
    print(f"\n[INFO] Shape dataset originale: {df.shape}")

    # 2) Identificazione colonne source/target/features
    source_col = _find_column_case_insensitive(df, ["source"])
    if source_col is None:
        raise ValueError("Colonna 'source' non trovata nel dataset.")

    target_col = _find_column_case_insensitive(df, ["target", "num"])
    if target_col is not None:
        print(f"[INFO] Colonna target individuata: {target_col}")
    else:
        print(
            "[INFO] Nessuna colonna target trovata ('target'/'num'). Analisi shift non supervisionata."
        )

    excluded = {source_col}
    if target_col is not None:
        excluded.add(target_col)

    feature_cols = _coerce_numeric_features(df, excluded_cols=excluded)
    if not feature_cols:
        raise ValueError(
            "Nessuna feature numerica disponibile dopo selezione/conversione."
        )

    # 3) Preprocessing: missing + standardizzazione
    work_cols = [source_col, *feature_cols]
    if target_col is not None:
        work_cols.append(target_col)
    work_df = df[work_cols].copy()

    if DROP_ROWS_WITH_MISSING_SOURCE:
        work_df = work_df.dropna(subset=[source_col])

    work_df[source_col] = work_df[source_col].astype(str)
    if REMOVE_SOURCES_WITH_LT2_SAMPLES:
        counts = work_df[source_col].value_counts()
        valid_sources = counts[counts >= 2].index
        work_df = work_df[work_df[source_col].isin(valid_sources)].copy()

    # Elimina feature interamente mancanti.
    feature_cols = [c for c in feature_cols if work_df[c].notna().any()]
    if not feature_cols:
        raise ValueError("Tutte le feature candidate sono interamente mancanti.")

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(work_df[feature_cols])

    if STANDARDIZE_FEATURES:
        scaler = StandardScaler()
        x_processed = scaler.fit_transform(x_imputed)
    else:
        x_processed = x_imputed

    processed_df = pd.DataFrame(x_processed, columns=feature_cols, index=work_df.index)
    dashi_df = pd.concat([work_df[[source_col]], processed_df], axis=1)
    # DASHI richiede dtype object/category per colonna sorgente.
    dashi_df[source_col] = dashi_df[source_col].astype(object)

    # 4) DASHI multi-source/domain shift
    source_map = dashi.estimate_multivariate_data_source_map(
        data=dashi_df,
        source_column_name=source_col,
        kde_resolution=KDE_RESOLUTION,
        dimensions=MDS_DIMENSIONS,
        dim_reduction=DIM_REDUCTION,
        scale=False,  # gia' standardizzato sopra (se STANDARDIZE_FEATURES=True)
        scatter_plot=False,
        verbose=False,
    )
    msv = dashi.estimate_MSV_metrics(source_map)

    sources = [str(s) for s in np.asarray(msv.sources)]
    spo = np.asarray(msv.SPO, dtype=float)
    vertices = np.asarray(msv.vertices, dtype=float)
    n_by_source = np.asarray(msv.nBySource, dtype=float)
    gpd = None if msv.GPD is None else float(msv.GPD)

    # 5) Metriche principali + heatmap distanze tra sorgenti
    prob_map = np.asarray(source_map.probability_map, dtype=float)
    distance_matrix = _js_distance_matrix(prob_map)
    distance_df = pd.DataFrame(distance_matrix, index=sources, columns=sources)

    msv_df = pd.DataFrame(
        {
            "source": sources,
            "SPO": spo,
            "n_samples_dashi": n_by_source
            if len(n_by_source) == len(sources)
            else np.nan,
            "n_samples_empirical": work_df[source_col]
            .value_counts()
            .reindex(sources)
            .fillna(0)
            .astype(int)
            .values,
        }
    )

    # 6) Salvataggio visualizzazioni in dashi_results
    dsm_html = OUTPUT_DIR / "dsm_multivariate.html"
    dsm_png = OUTPUT_DIR / "dsm_multivariate.png"
    msv_html = OUTPUT_DIR / "msv_plot.html"
    msv_png = OUTPUT_DIR / "msv_plot.png"
    heatmap_png = OUTPUT_DIR / "source_distance_heatmap.png"

    dsm_fig = dashi.plot_multivariate_data_source_map(source_map, absolute=False)
    _save_plotly_with_fallback(
        fig=dsm_fig,
        png_path=dsm_png,
        html_path=dsm_html,
        fallback_png_fn=lambda: _plot_dsm_fallback(
            source_map.multivariate_probability_map, sources, dsm_png
        ),
        label="DSM",
    )

    msv_fig = dashi.plot_MSV(msv_metrics=msv, dimensions=2)
    _save_plotly_with_fallback(
        fig=msv_fig,
        png_path=msv_png,
        html_path=msv_html,
        fallback_png_fn=lambda: _plot_msv_fallback(
            vertices, sources, spo, n_by_source, gpd, msv_png
        ),
        label="MSV",
    )

    _plot_distance_heatmap(distance_df, heatmap_png)

    # Salvataggio tabelle metriche
    msv_df.to_csv(OUTPUT_DIR / "msv_metrics_by_source.csv", index=False)
    distance_df.to_csv(OUTPUT_DIR / "pairwise_js_distance_matrix.csv")

    # 7) Stampa output richiesto nel terminale
    print("\n=== RIEPILOGO ===")
    print(
        f"Dimensione dataset (originale): {df.shape[0]} righe x {df.shape[1]} colonne"
    )
    print(
        f"Dimensione dataset (usato DASHI): {dashi_df.shape[0]} righe x {dashi_df.shape[1]} colonne"
    )

    print("\nCampioni per source:")
    print(work_df[source_col].value_counts().sort_index().to_string())

    print(f"\nFeature numeriche usate ({len(feature_cols)}):")
    print(", ".join(feature_cols))

    print("\nMetriche dataset shift:")
    if gpd is not None:
        print(f"- GPD: {gpd:.6f}")
    else:
        print("- GPD: non disponibile")
    print(f"- Mean SPO: {np.nanmean(spo):.6f}")
    print(f"- Std SPO:  {np.nanstd(spo):.6f}")

    print("\nSPO per source:")
    print(msv_df[["source", "SPO"]].to_string(index=False))

    print("\nMatrice distanze JS tra source:")
    print(distance_df.round(4).to_string())

    print(f"\n[OK] Risultati salvati in: {OUTPUT_DIR.resolve()}")
    print(f" - {dsm_html.name}")
    print(f" - {dsm_png.name}")
    print(f" - {msv_html.name}")
    print(f" - {msv_png.name}")
    print(f" - {heatmap_png.name}")
    print(f" - msv_metrics_by_source.csv")
    print(f" - pairwise_js_distance_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
