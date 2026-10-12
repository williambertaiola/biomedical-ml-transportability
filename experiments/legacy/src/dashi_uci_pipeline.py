"""Transportability experiment: dashi uci pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import ssl
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if "XDG_CACHE_HOME" not in os.environ:
    cache_root = (Path.cwd() / ".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = (Path.cwd() / ".mplconfig").resolve()
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib.pyplot as plt


LOGGER = logging.getLogger("dashi_uci_pipeline")

UCI_COLUMNS = [
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
DEFAULT_FEATURE_COLUMNS = UCI_COLUMNS[:-1]


@dataclass(frozen=True)
class SourceSpec:
    key: str
    source_name: str
    filename: str
    url: str


def get_source_specs() -> list[SourceSpec]:
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
    return [
        SourceSpec(
            key="cleveland",
            source_name="Cleveland",
            filename="processed.cleveland.data",
            url=f"{base}/processed.cleveland.data",
        ),
        SourceSpec(
            key="hungarian",
            source_name="Hungary",
            filename="processed.hungarian.data",
            url=f"{base}/processed.hungarian.data",
        ),
        SourceSpec(
            key="switzerland",
            source_name="Switzerland",
            filename="processed.switzerland.data",
            url=f"{base}/processed.switzerland.data",
        ),
        SourceSpec(
            key="va",
            source_name="VA Long Beach",
            filename="processed.va.data",
            url=f"{base}/processed.va.data",
        ),
    ]


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_project_structure(project_root: Path) -> dict[str, Path]:
    paths = {
        "project_root": project_root,
        "data": project_root / "data",
        "outputs": project_root / "outputs",
        "tables": project_root / "outputs" / "tables",
        "figures": project_root / "outputs" / "figures",
        "reports": project_root / "outputs" / "reports",
        "src": project_root / "src",
        "mplconfig": project_root / ".mplconfig",
    }
    for path in paths.values():
        if path == project_root:
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def inspect_dashi_api() -> dict[str, Any]:
    api: dict[str, Any] = {"package": "dashi"}
    try:
        from importlib.metadata import version

        api["version"] = version("dashi")
    except Exception:
        api["version"] = "unknown"

    try:
        import dashi

        candidate_functions = [
            "format_data",
            "estimate_multivariate_data_source_map",
            "estimate_conditional_data_source_map",
            "estimate_MSV_metrics",
            "plot_MSV",
            "plot_multivariate_data_source_map",
            "estimate_multibatch_models",
            "arrange_performance_metrics",
            "plot_multibatch_performance",
        ]
        api["functions"] = {
            name: callable(getattr(dashi, name, None)) for name in candidate_functions
        }
    except Exception as exc:
        api["import_error"] = str(exc)

    return api


def _urlopen_with_optional_ssl(url: str, timeout: int, verify_ssl: bool) -> Any:
    if verify_ssl:
        return urllib.request.urlopen(url, timeout=timeout)
    insecure_ctx = ssl._create_unverified_context()
    return urllib.request.urlopen(url, timeout=timeout, context=insecure_ctx)


def verify_remote_files(
    source_specs: list[SourceSpec], timeout: int = 30
) -> pd.DataFrame:
    verification_rows: list[dict[str, Any]] = []
    for spec in source_specs:
        row = {
            "source": spec.source_name,
            "filename": spec.filename,
            "url": spec.url,
            "available": False,
            "ssl_mode": None,
            "http_status": None,
            "bytes_preview": None,
            "error": None,
        }
        for verify_ssl, ssl_mode in ((True, "strict"), (False, "insecure_fallback")):
            try:
                with _urlopen_with_optional_ssl(
                    spec.url, timeout=timeout, verify_ssl=verify_ssl
                ) as response:
                    preview = response.read(256)
                    row["available"] = True
                    row["ssl_mode"] = ssl_mode
                    row["http_status"] = getattr(response, "status", None)
                    row["bytes_preview"] = len(preview)
                    row["error"] = None
                    break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        verification_rows.append(row)
    return pd.DataFrame(verification_rows)


def download_uci_file(spec: SourceSpec, destination: Path, timeout: int = 60) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("File already present, skip download: %s", destination.name)
        return "already_present"

    last_error: Exception | None = None
    for verify_ssl, ssl_mode in ((True, "strict"), (False, "insecure_fallback")):
        try:
            LOGGER.info(
                "Downloading %s from UCI with SSL mode: %s", spec.filename, ssl_mode
            )
            with _urlopen_with_optional_ssl(
                spec.url, timeout=timeout, verify_ssl=verify_ssl
            ) as response:
                payload = response.read()
            destination.write_bytes(payload)
            LOGGER.info("Saved %s (%d bytes)", destination.name, len(payload))
            return ssl_mode
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Download failed for %s with SSL mode %s: %s",
                spec.filename,
                ssl_mode,
                exc,
            )

    raise RuntimeError(
        f"Unable to download {spec.filename} after strict + fallback attempts. "
        f"Last error: {last_error}"
    )


def slugify_source(source_name: str) -> str:
    slug = source_name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def parse_uci_source_file(
    path: Path, source_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[list[str]] = []
    malformed_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = [chunk.strip() for chunk in stripped.split(",")]
            if len(parts) != len(UCI_COLUMNS):
                alt_parts = [x for x in re.split(r"[,\s]+", stripped) if x]
                if len(alt_parts) == len(UCI_COLUMNS):
                    parts = alt_parts
                else:
                    malformed_rows.append(
                        {
                            "source": source_name,
                            "filename": path.name,
                            "line_number": line_number,
                            "observed_fields": len(parts),
                            "raw_line": stripped[:300],
                        }
                    )
                    continue

            records.append(parts)

    if not records:
        raise RuntimeError(f"No valid rows parsed from file: {path}")

    source_slug = slugify_source(source_name)
    df = pd.DataFrame(records, columns=UCI_COLUMNS)
    df.insert(
        0,
        "sample_id",
        [f"{source_slug}_{idx:04d}" for idx in range(1, len(df) + 1)],
    )
    df.insert(1, "source", source_name)
    df.insert(2, "source_row_id", np.arange(1, len(df) + 1, dtype=int))

    df = df.replace("?", np.nan)
    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    malformed_df = pd.DataFrame(malformed_rows)
    return df, malformed_df


def build_raw_dataset(
    data_dir: Path, source_specs: list[SourceSpec]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_frames: list[pd.DataFrame] = []
    malformed_frames: list[pd.DataFrame] = []

    for spec in source_specs:
        local_path = data_dir / spec.filename
        source_df, malformed_df = parse_uci_source_file(local_path, spec.source_name)
        source_frames.append(source_df)
        if not malformed_df.empty:
            malformed_frames.append(malformed_df)

    raw_df = pd.concat(source_frames, ignore_index=True)
    malformed_all = (
        pd.concat(malformed_frames, ignore_index=True)
        if malformed_frames
        else pd.DataFrame(
            columns=["source", "filename", "line_number", "observed_fields", "raw_line"]
        )
    )
    return raw_df, malformed_all


def prepare_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw_df.copy()
    prep_meta: dict[str, Any] = {}

    if df["sample_id"].duplicated().any():
        prep_meta["duplicate_sample_id_detected"] = True
        df["sample_id"] = [f"sample_{idx:05d}" for idx in range(1, len(df) + 1)]
    else:
        prep_meta["duplicate_sample_id_detected"] = False

    df["target"] = np.where(df["num"].isna(), np.nan, (df["num"] > 0).astype(int))
    missing_target_rows = int(df["target"].isna().sum())
    prep_meta["rows_with_missing_target_removed"] = missing_target_rows
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    candidate_features = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    non_empty_features = [c for c in candidate_features if df[c].notna().any()]
    dropped_all_missing = sorted(set(candidate_features) - set(non_empty_features))
    prep_meta["dropped_all_missing_features"] = dropped_all_missing

    for column in non_empty_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_pct_before = (
        df[non_empty_features]
        .isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .reset_index()
    )
    missing_pct_before = missing_pct_before.rename(columns={"index": "column"})

    imputation_rows: list[dict[str, Any]] = []
    for column in non_empty_features:
        missing_before = int(df[column].isna().sum())
        if missing_before > 0:
            median_value = df[column].median()
            if pd.isna(median_value):
                median_value = 0.0
            df[column] = df[column].fillna(median_value)
            strategy = "median"
            fill_value = float(median_value)
        else:
            strategy = "none_needed"
            fill_value = np.nan
        imputation_rows.append(
            {
                "column": column,
                "missing_before": missing_before,
                "strategy": strategy,
                "fill_value": fill_value,
            }
        )

    df["source"] = df["source"].astype("object")
    final_columns = ["sample_id", "source", *non_empty_features, "target"]
    final_df = df[final_columns].copy()

    prep_meta["n_rows_raw"] = int(len(raw_df))
    prep_meta["n_rows_final"] = int(len(final_df))
    prep_meta["n_features_final"] = int(len(non_empty_features))
    prep_meta["feature_columns"] = non_empty_features
    prep_meta["imputation_table"] = pd.DataFrame(imputation_rows)
    prep_meta["missing_pct_before_imputation"] = missing_pct_before

    return final_df, prep_meta


def save_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def compute_and_save_eda(
    raw_df: pd.DataFrame,
    final_df: pd.DataFrame,
    prep_meta: dict[str, Any],
    tables_dir: Path,
    reports_dir: Path,
) -> dict[str, pd.DataFrame]:
    shapes_df = pd.DataFrame(
        [
            {
                "dataset": "raw_combined",
                "n_rows": raw_df.shape[0],
                "n_columns": raw_df.shape[1],
            },
            {
                "dataset": "final_for_dashi",
                "n_rows": final_df.shape[0],
                "n_columns": final_df.shape[1],
            },
        ]
    )

    samples_per_source = (
        final_df["source"]
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="n_samples")
        .sort_values("source")
    )

    target_global = (
        final_df["target"]
        .value_counts(dropna=False)
        .rename_axis("target")
        .reset_index(name="n_samples")
        .sort_values("target")
    )
    target_global["pct"] = (
        target_global["n_samples"] / target_global["n_samples"].sum() * 100
    )

    target_by_source = (
        final_df.groupby("source", dropna=False)["target"]
        .value_counts(dropna=False)
        .rename("n_samples")
        .reset_index()
    )
    target_by_source["pct_within_source"] = (
        target_by_source["n_samples"]
        / target_by_source.groupby("source")["n_samples"].transform("sum")
        * 100
    )

    missing_pct_before = prep_meta["missing_pct_before_imputation"].copy()
    missing_pct_before = missing_pct_before.sort_values("missing_pct", ascending=False)

    imputation_df = prep_meta["imputation_table"].copy()

    save_dataframe(shapes_df, tables_dir / "eda_dataset_shapes.csv")
    save_dataframe(samples_per_source, tables_dir / "eda_samples_per_source.csv")
    save_dataframe(target_global, tables_dir / "eda_target_distribution_global.csv")
    save_dataframe(
        target_by_source, tables_dir / "eda_target_distribution_by_source.csv"
    )
    save_dataframe(
        missing_pct_before,
        tables_dir / "eda_missing_pct_by_column_before_imputation.csv",
    )
    save_dataframe(imputation_df, tables_dir / "imputation_summary.csv")

    summary_lines = [
        "UCI Heart Disease - EDA Summary",
        "",
        f"Raw combined shape: {raw_df.shape}",
        f"Final shape (for DASHI): {final_df.shape}",
        "",
        "Samples per source:",
        samples_per_source.to_string(index=False),
        "",
        "Global target distribution:",
        target_global.to_string(index=False),
        "",
        "Target distribution by source:",
        target_by_source.to_string(index=False),
        "",
        "Missing percentage by column (before imputation):",
        missing_pct_before.to_string(index=False),
    ]
    eda_summary_path = reports_dir / "eda_summary.txt"
    eda_summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Dataset shape (raw): {raw_df.shape}")
    print(f"Dataset shape (final): {final_df.shape}")
    print("\nNumero di campioni per source:")
    print(samples_per_source.to_string(index=False))
    print("\nDistribuzione target globale:")
    print(target_global.to_string(index=False))
    print("\nDistribuzione target per source:")
    print(target_by_source.to_string(index=False))
    print("\nPercentuale di missing per colonna (prima imputazione):")
    print(missing_pct_before.to_string(index=False))

    return {
        "shapes": shapes_df,
        "samples_per_source": samples_per_source,
        "target_global": target_global,
        "target_by_source": target_by_source,
        "missing_pct_before": missing_pct_before,
        "imputation_summary": imputation_df,
    }


def save_matplotlib_data_source_map(
    probability_maps: np.ndarray, sources: list[str], output_png: Path
) -> None:
    n_sources = len(sources)
    n_cols = 2
    n_rows = math.ceil(n_sources / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    vmin = float(np.nanmin(probability_maps))
    vmax = float(np.nanmax(probability_maps))
    for idx, source in enumerate(sources):
        ax = axes[idx]
        img = ax.imshow(
            probability_maps[idx],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Source: {source}")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_sources, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("DASHI Data Source Map (matplotlib fallback)", fontsize=13)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_matplotlib_msv_plot(
    vertices: np.ndarray,
    sources: list[str],
    spo: np.ndarray,
    n_by_source: np.ndarray,
    gpd: float | None,
    output_png: Path,
) -> None:
    if vertices.ndim != 2 or vertices.shape[0] != len(sources):
        raise ValueError(
            "Vertices array has unexpected shape for MSV fallback plotting."
        )

    x = vertices[:, 0]
    y = vertices[:, 1] if vertices.shape[1] > 1 else np.zeros_like(x)
    sizes = np.clip(n_by_source.astype(float), 1, None)
    sizes = 100 + 12 * (sizes - sizes.min())

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=spo, s=sizes, cmap="plasma", alpha=0.9, edgecolor="k")
    for idx, src in enumerate(sources):
        ax.annotate(
            src, (x[idx], y[idx]), xytext=(6, 6), textcoords="offset points", fontsize=9
        )
    title = "MSV Plot (matplotlib fallback)"
    if gpd is not None and np.isfinite(gpd):
        title += f" | GPD={gpd:.4f}"
    ax.set_title(title)
    ax.set_xlabel("Vertex dim 1")
    ax.set_ylabel("Vertex dim 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("SPO")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_matplotlib_heatmap(
    matrix_df: pd.DataFrame, metric_name: str, output_png: Path, title_prefix: str
) -> None:
    matrix = matrix_df.astype(float).values
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(matrix, cmap="viridis", aspect="auto")

    ax.set_xticks(range(matrix_df.shape[1]))
    ax.set_yticks(range(matrix_df.shape[0]))
    ax.set_xticklabels([str(c) for c in matrix_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in matrix_df.index])
    ax.set_xlabel("Test source")
    ax.set_ylabel("Train source")
    ax.set_title(f"{title_prefix}: {metric_name}")

    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(metric_name)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plotly_with_fallback(
    figure: Any,
    output_png: Path,
    output_html: Path,
    fallback_png_fn: Callable[[], None],
    label: str,
) -> dict[str, Any]:
    out = {
        "png_saved": False,
        "html_saved": False,
        "fallback_used": False,
        "error": None,
    }
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    enable_plotly_png = os.environ.get("DASHI_ENABLE_PLOTLY_PNG", "0").strip() == "1"
    if enable_plotly_png:
        try:
            figure.write_image(str(output_png))
            out["png_saved"] = True
        except Exception as exc:
            out["error"] = (
                f"write_image failed for {label}: {type(exc).__name__}: {exc}"
            )
            LOGGER.warning(out["error"])
    else:
        LOGGER.info(
            "Skipping plotly write_image for %s (DASHI_ENABLE_PLOTLY_PNG!=1). "
            "Using matplotlib PNG fallback for stability.",
            label,
        )

    try:
        figure.write_html(str(output_html))
        out["html_saved"] = True
    except Exception as exc:
        html_error = f"write_html failed for {label}: {type(exc).__name__}: {exc}"
        LOGGER.warning(html_error)
        if out["error"] is None:
            out["error"] = html_error

    if not out["png_saved"]:
        fallback_png_fn()
        out["png_saved"] = True
        out["fallback_used"] = True
    return out


def _get_callable(module: Any, name: str) -> Callable[..., Any] | None:
    candidate = getattr(module, name, None)
    return candidate if callable(candidate) else None


def run_unsupervised_dashi(
    final_df: pd.DataFrame,
    feature_columns: list[str],
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    import dashi

    LOGGER.info("Running DASHI unsupervised characterization...")
    data = final_df[["source", *feature_columns]].copy()
    data["source"] = data["source"].astype("object")

    estimate_map_fn = _get_callable(dashi, "estimate_multivariate_data_source_map")
    estimate_msv_fn = _get_callable(dashi, "estimate_MSV_metrics")
    if estimate_map_fn is None or estimate_msv_fn is None:
        raise RuntimeError(
            "DASHI required unsupervised functions are missing "
            "(estimate_multivariate_data_source_map / estimate_MSV_metrics)."
        )

    data_source_map = estimate_map_fn(
        data=data,
        source_column_name="source",
        kde_resolution=30,
        dimensions=2,
        dim_reduction="PCA",
        scale=True,
        scatter_plot=False,
        verbose=False,
    )
    msv_metrics = estimate_msv_fn(data_source_map)

    sources = [str(s) for s in np.asarray(msv_metrics.sources)]
    spo = np.asarray(msv_metrics.SPO, dtype=float)
    vertices = np.asarray(msv_metrics.vertices, dtype=float)
    n_by_source = np.asarray(msv_metrics.nBySource, dtype=float)
    source_counts = (
        data["source"].value_counts().reindex(sources).fillna(0).to_numpy(dtype=float)
    )

    if (
        n_by_source.size == 0
        or np.isnan(n_by_source).any()
        or np.allclose(n_by_source, 0)
    ):
        LOGGER.warning(
            "DASHI returned empty/zero nBySource; replacing with empirical counts per source."
        )
        n_by_source = source_counts
        msv_metrics.nBySource = n_by_source

    gpd_value = None if msv_metrics.GPD is None else float(msv_metrics.GPD)
    msv_table = pd.DataFrame(
        {
            "source": sources,
            "SPO": spo,
            "n_samples": n_by_source,
            "source_count_empirical": source_counts,
        }
    )
    save_dataframe(msv_table, tables_dir / "unsupervised_msv_by_source.csv")

    vertex_columns = [f"vertex_dim_{i + 1}" for i in range(vertices.shape[1])]
    vertices_df = pd.DataFrame(vertices, columns=vertex_columns)
    vertices_df.insert(0, "source", sources)
    save_dataframe(vertices_df, tables_dir / "unsupervised_msv_vertices.csv")

    global_metrics = {
        "GPD": gpd_value,
        "n_sources": len(sources),
        "sources": sources,
        "mean_SPO": float(np.nanmean(spo)),
        "std_SPO": float(np.nanstd(spo)),
    }
    save_json(global_metrics, tables_dir / "unsupervised_global_metrics.json")

    probability_maps = np.asarray(
        data_source_map.multivariate_probability_map, dtype=float
    )
    save_matplotlib_data_source_map(
        probability_maps=probability_maps,
        sources=sources,
        output_png=figures_dir / "unsupervised_data_source_map_fallback.png",
    )

    save_matplotlib_msv_plot(
        vertices=vertices,
        sources=sources,
        spo=spo,
        n_by_source=n_by_source,
        gpd=gpd_value,
        output_png=figures_dir / "unsupervised_msv_plot_fallback.png",
    )

    plot_data_source_map_fn = _get_callable(dashi, "plot_multivariate_data_source_map")
    plot_msv_fn = _get_callable(dashi, "plot_MSV")
    plot_status: dict[str, Any] = {}

    if plot_data_source_map_fn is not None:
        try:
            dsm_figure = plot_data_source_map_fn(data_source_map, absolute=False)
            plot_status["data_source_map"] = save_plotly_with_fallback(
                figure=dsm_figure,
                output_png=figures_dir / "unsupervised_data_source_map.png",
                output_html=figures_dir / "unsupervised_data_source_map.html",
                fallback_png_fn=lambda: save_matplotlib_data_source_map(
                    probability_maps=probability_maps,
                    sources=sources,
                    output_png=figures_dir / "unsupervised_data_source_map.png",
                ),
                label="data_source_map",
            )
        except Exception as exc:
            LOGGER.warning("plot_multivariate_data_source_map failed: %s", exc)
            save_matplotlib_data_source_map(
                probability_maps=probability_maps,
                sources=sources,
                output_png=figures_dir / "unsupervised_data_source_map.png",
            )
            plot_status["data_source_map"] = {
                "png_saved": True,
                "html_saved": False,
                "fallback_used": True,
                "error": str(exc),
            }
    else:
        save_matplotlib_data_source_map(
            probability_maps=probability_maps,
            sources=sources,
            output_png=figures_dir / "unsupervised_data_source_map.png",
        )
        plot_status["data_source_map"] = {
            "png_saved": True,
            "html_saved": False,
            "fallback_used": True,
            "error": "plot_multivariate_data_source_map not available in installed dashi",
        }

    if plot_msv_fn is not None:
        try:
            msv_figure = plot_msv_fn(msv_metrics=msv_metrics, dimensions=2)
            plot_status["msv_plot"] = save_plotly_with_fallback(
                figure=msv_figure,
                output_png=figures_dir / "unsupervised_msv_plot.png",
                output_html=figures_dir / "unsupervised_msv_plot.html",
                fallback_png_fn=lambda: save_matplotlib_msv_plot(
                    vertices=vertices,
                    sources=sources,
                    spo=spo,
                    n_by_source=n_by_source,
                    gpd=gpd_value,
                    output_png=figures_dir / "unsupervised_msv_plot.png",
                ),
                label="msv_plot",
            )
        except Exception as exc:
            LOGGER.warning("plot_MSV failed: %s", exc)
            save_matplotlib_msv_plot(
                vertices=vertices,
                sources=sources,
                spo=spo,
                n_by_source=n_by_source,
                gpd=gpd_value,
                output_png=figures_dir / "unsupervised_msv_plot.png",
            )
            plot_status["msv_plot"] = {
                "png_saved": True,
                "html_saved": False,
                "fallback_used": True,
                "error": str(exc),
            }
    else:
        save_matplotlib_msv_plot(
            vertices=vertices,
            sources=sources,
            spo=spo,
            n_by_source=n_by_source,
            gpd=gpd_value,
            output_png=figures_dir / "unsupervised_msv_plot.png",
        )
        plot_status["msv_plot"] = {
            "png_saved": True,
            "html_saved": False,
            "fallback_used": True,
            "error": "plot_MSV not available in installed dashi",
        }

    save_json(plot_status, tables_dir / "unsupervised_plot_status.json")
    return {
        "gpd": gpd_value,
        "msv_table": msv_table,
        "plot_status": plot_status,
    }


def _flatten_supervised_metrics(
    metrics: dict[tuple[Any, Any, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, metric_dict in metrics.items():
        train_source, test_source, subset = key
        if isinstance(train_source, (list, tuple, set, np.ndarray)):
            train_source_repr = "|".join(str(x) for x in train_source)
        else:
            train_source_repr = str(train_source)

        for metric_name, value in metric_dict.items():
            rows.append(
                {
                    "train_source": train_source_repr,
                    "test_source": str(test_source),
                    "subset": subset,
                    "metric": metric_name,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def supervised_fallback_sklearn(
    final_df: pd.DataFrame, feature_columns: list[str]
) -> dict[tuple[Any, Any, str], dict[str, float]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    LOGGER.warning("Using sklearn fallback for supervised characterization.")
    metrics: dict[tuple[Any, Any, str], dict[str, float]] = {}
    sources = sorted(final_df["source"].astype(str).unique())

    for train_source in sources:
        train_mask = final_df["source"].astype(str) == train_source
        X_train = final_df.loc[train_mask, feature_columns].to_numpy()
        y_train = final_df.loc[train_mask, "target"].to_numpy()

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        for test_source in sources:
            test_mask = final_df["source"].astype(str) == test_source
            X_test = final_df.loc[test_mask, feature_columns].to_numpy()
            y_test = final_df.loc[test_mask, "target"].to_numpy()

            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            metric_dict = {
                "ACCURACY": float(accuracy_score(y_test, y_pred)),
                "F1-SCORE_1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            try:
                metric_dict["AUC_1"] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                metric_dict["AUC_1"] = np.nan

            metrics[((train_source,), test_source, "test")] = metric_dict
    return metrics


def run_supervised_dashi(
    final_df: pd.DataFrame,
    feature_columns: list[str],
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    import dashi

    LOGGER.info("Running DASHI supervised characterization...")
    data = final_df[["source", *feature_columns, "target"]].copy()
    data["source"] = data["source"].astype("object")
    data["target"] = data["target"].astype(int)

    estimate_models_fn = _get_callable(dashi, "estimate_multibatch_models")
    arrange_metrics_fn = _get_callable(dashi, "arrange_performance_metrics")
    plot_multibatch_fn = _get_callable(dashi, "plot_multibatch_performance")

    if estimate_models_fn is None:
        metrics = supervised_fallback_sklearn(final_df, feature_columns)
        used_fallback_modeling = True
    else:
        try:
            metrics = estimate_models_fn(
                data=data,
                inputs_numerical_column_names=feature_columns,
                inputs_categorical_column_names=None,
                output_classification_column_name="target",
                output_regression_column_name=None,
                source_column_name="source",
                date_column_name=None,
                period=None,
                model_type="histogram_gradient_boosting",
                learning_strategy="from_scratch",
            )
            used_fallback_modeling = False
        except Exception as exc:
            LOGGER.warning(
                "estimate_multibatch_models failed. Fallback sklearn: %s", exc
            )
            metrics = supervised_fallback_sklearn(final_df, feature_columns)
            used_fallback_modeling = True

    long_metrics_df = _flatten_supervised_metrics(metrics)
    if long_metrics_df.empty:
        raise RuntimeError("No supervised metrics were produced.")
    save_dataframe(long_metrics_df, tables_dir / "supervised_metrics_long.csv")

    available_metrics = sorted(long_metrics_df["metric"].dropna().unique().tolist())
    save_json(
        {
            "available_metrics": available_metrics,
            "used_fallback_modeling": used_fallback_modeling,
        },
        tables_dir / "supervised_available_metrics.json",
    )

    preferred_metrics = [
        "ACCURACY",
        "AUC_MACRO",
        "AUC_1",
        "F1-SCORE_MACRO",
        "F1-SCORE_1",
        "LOGLOSS",
    ]
    chosen_metrics = [m for m in preferred_metrics if m in available_metrics]
    if not chosen_metrics:
        chosen_metrics = available_metrics[:3]

    metric_matrices: dict[str, pd.DataFrame] = {}
    if arrange_metrics_fn is not None:
        for metric_name in chosen_metrics:
            try:
                matrix_df = arrange_metrics_fn(metrics=metrics, metric_name=metric_name)
                metric_matrices[metric_name] = matrix_df
                save_dataframe(
                    matrix_df.reset_index().rename(columns={"index": "train_source"}),
                    tables_dir / f"supervised_matrix_{metric_name}.csv",
                )
            except Exception as exc:
                LOGGER.warning(
                    "arrange_performance_metrics failed for %s: %s", metric_name, exc
                )
    else:
        LOGGER.warning(
            "arrange_performance_metrics missing; creating manual ACCURACY matrix fallback."
        )
        subset = long_metrics_df[
            (long_metrics_df["subset"] == "test")
            & (long_metrics_df["metric"] == "ACCURACY")
        ]
        if not subset.empty:
            matrix_df = (
                subset.pivot(
                    index="train_source", columns="test_source", values="value"
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            metric_matrices["ACCURACY"] = matrix_df
            save_dataframe(
                matrix_df.reset_index().rename(columns={"index": "train_source"}),
                tables_dir / "supervised_matrix_ACCURACY.csv",
            )

    if not metric_matrices:
        raise RuntimeError("Unable to build supervised train-vs-test matrices.")

    primary_metric = (
        chosen_metrics[0]
        if chosen_metrics[0] in metric_matrices
        else list(metric_matrices)[0]
    )
    primary_matrix = metric_matrices[primary_metric]

    save_matplotlib_heatmap(
        matrix_df=primary_matrix,
        metric_name=primary_metric,
        output_png=figures_dir / "supervised_heatmap_train_vs_test.png",
        title_prefix="DASHI supervised train-vs-test",
    )

    plot_status: dict[str, Any] = {
        "heatmap_png_saved": True,
        "heatmap_html_saved": False,
        "fallback_used": True,
        "error": None,
        "metric": primary_metric,
    }

    if plot_multibatch_fn is not None:
        try:
            plotly_heatmap = plot_multibatch_fn(
                metrics=metrics, metric_name=primary_metric
            )
            status = save_plotly_with_fallback(
                figure=plotly_heatmap,
                output_png=figures_dir / "supervised_heatmap_train_vs_test.png",
                output_html=figures_dir / "supervised_heatmap_train_vs_test.html",
                fallback_png_fn=lambda: save_matplotlib_heatmap(
                    matrix_df=primary_matrix,
                    metric_name=primary_metric,
                    output_png=figures_dir / "supervised_heatmap_train_vs_test.png",
                    title_prefix="DASHI supervised train-vs-test",
                ),
                label="supervised_heatmap",
            )
            plot_status = {
                "heatmap_png_saved": status["png_saved"],
                "heatmap_html_saved": status["html_saved"],
                "fallback_used": status["fallback_used"],
                "error": status["error"],
                "metric": primary_metric,
            }
        except Exception as exc:
            LOGGER.warning("plot_multibatch_performance failed: %s", exc)
            plot_status["error"] = str(exc)

    save_json(plot_status, tables_dir / "supervised_plot_status.json")
    return {
        "metrics_long": long_metrics_df,
        "available_metrics": available_metrics,
        "primary_metric": primary_metric,
        "primary_matrix": primary_matrix,
        "plot_status": plot_status,
        "used_fallback_modeling": used_fallback_modeling,
    }


def generate_interpretive_report(
    final_df: pd.DataFrame,
    prep_meta: dict[str, Any],
    unsupervised_result: dict[str, Any],
    supervised_result: dict[str, Any],
    report_path: Path,
) -> str:
    sources = sorted(final_df["source"].astype(str).unique().tolist())
    n_sources = len(sources)

    msv_table = unsupervised_result["msv_table"].copy()
    if not msv_table.empty and "SPO" in msv_table.columns:
        msv_sorted = msv_table.sort_values("SPO", ascending=False)
        top_outlier_source = str(msv_sorted.iloc[0]["source"])
        top_outlier_spo = float(msv_sorted.iloc[0]["SPO"])
    else:
        top_outlier_source = "n/a"
        top_outlier_spo = np.nan

    primary_metric = supervised_result["primary_metric"]
    matrix_df = supervised_result["primary_matrix"].astype(float)
    matrix_values = matrix_df.values
    diag = np.diag(matrix_values)
    mask_offdiag = ~np.eye(matrix_values.shape[0], dtype=bool)
    offdiag_values = matrix_values[mask_offdiag]
    offdiag_values = offdiag_values[np.isfinite(offdiag_values)]

    if offdiag_values.size > 0:
        generalization_range = float(
            np.nanmax(offdiag_values) - np.nanmin(offdiag_values)
        )
        generalization_std = float(np.nanstd(offdiag_values))
    else:
        generalization_range = float("nan")
        generalization_std = float("nan")

    if np.isfinite(generalization_range):
        if generalization_range < 0.08:
            generalization_statement = (
                "La generalizzazione train->test appare relativamente omogenea "
                f"(range off-diagonal {generalization_range:.3f} su {primary_metric})."
            )
        else:
            generalization_statement = (
                "La generalizzazione train->test appare variabile tra source "
                f"(range off-diagonal {generalization_range:.3f} su {primary_metric})."
            )
    else:
        generalization_statement = (
            "Non e' stato possibile stimare in modo stabile la variabilita' off-diagonal "
            "della matrice supervised."
        )

    report_lines = [
        "# Report DASHI - UCI Heart Disease (Multi-Source)",
        "",
        "## Sintesi",
        f"- Numero di source analizzati: **{n_sources}** ({', '.join(sources)}).",
        f"- GPD (unsupervised): **{unsupervised_result['gpd']:.4f}**."
        if unsupervised_result["gpd"] is not None
        else "- GPD (unsupervised): non disponibile.",
        (
            f"- Source con SPO piu' alto (potenziale outlier): **{top_outlier_source}** "
            f"(SPO={top_outlier_spo:.4f})."
            if np.isfinite(top_outlier_spo)
            else "- SPO per-source non disponibile."
        ),
        f"- {generalization_statement}",
        "",
        "## Limiti e note metodologiche",
        "- Dataset piccolo e sbilanciato tra source, con possibili effetti sulla stabilita' delle metriche.",
        "- Missing values gestiti con imputazione mediana per mantenere la comparabilita' multi-source.",
        "- Obiettivo del workflow: characterization/transportability, non tuning prestazionale del classificatore.",
        "- In questa macchina, l'export PNG dei plot Plotly puo' fallire; la pipeline salva fallback matplotlib in PNG e, quando possibile, anche HTML interattivi.",
        "",
        "## Dettagli tecnici",
        f"- Features finali usate: {', '.join(prep_meta['feature_columns'])}",
        f"- Righe rimosse per target mancante: {prep_meta['rows_with_missing_target_removed']}",
        f"- Metodologia supervised fallback attivata: {supervised_result['used_fallback_modeling']}",
        "",
    ]

    report_text = "\n".join(report_lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_pipeline(project_root: Path, force_download: bool = False) -> dict[str, Any]:
    paths = ensure_project_structure(project_root)
    os.environ.setdefault("MPLCONFIGDIR", str(paths["mplconfig"]))

    LOGGER.info("Inspecting DASHI API...")
    dashi_api = inspect_dashi_api()
    save_json(dashi_api, paths["tables"] / "dashi_api_inspection.json")

    source_specs = get_source_specs()
    LOGGER.info("Verifying remote UCI files before execution...")
    verification_df = verify_remote_files(source_specs)
    save_dataframe(
        verification_df, paths["tables"] / "uci_remote_file_verification.csv"
    )

    if not verification_df["available"].all():
        unavailable = verification_df.loc[
            ~verification_df["available"], "filename"
        ].tolist()
        local_ready = all(
            (paths["data"] / spec.filename).exists()
            and (paths["data"] / spec.filename).stat().st_size > 0
            for spec in source_specs
        )
        if local_ready:
            LOGGER.warning(
                "Remote verification failed for %s, but local raw files are already present. "
                "Proceeding in offline mode.",
                unavailable,
            )
        else:
            raise RuntimeError(
                f"Some required UCI files are unavailable: {unavailable}. "
                "See verification CSV and ensure network access."
            )

    download_rows: list[dict[str, Any]] = []
    for spec in source_specs:
        target_path = paths["data"] / spec.filename
        status = (
            download_uci_file(spec, target_path)
            if force_download or not target_path.exists()
            else "already_present"
        )
        download_rows.append(
            {
                "source": spec.source_name,
                "filename": spec.filename,
                "local_path": str(target_path),
                "download_status": status,
                "size_bytes": target_path.stat().st_size if target_path.exists() else 0,
            }
        )
    save_dataframe(
        pd.DataFrame(download_rows), paths["tables"] / "uci_download_log.csv"
    )

    LOGGER.info("Building combined raw dataset...")
    raw_df, malformed_df = build_raw_dataset(paths["data"], source_specs)
    save_dataframe(raw_df, paths["data"] / "uci_heart_combined_raw.csv")
    save_dataframe(malformed_df, paths["tables"] / "uci_parsing_malformed_rows.csv")

    LOGGER.info("Preparing final dataset for DASHI...")
    final_df, prep_meta = prepare_dataset(raw_df)
    save_dataframe(final_df, paths["data"] / "uci_heart_for_dashi.csv")
    save_json(
        {
            key: value
            for key, value in prep_meta.items()
            if key not in {"imputation_table", "missing_pct_before_imputation"}
        },
        paths["tables"] / "preparation_metadata.json",
    )

    LOGGER.info("Computing EDA outputs...")
    eda_outputs = compute_and_save_eda(
        raw_df=raw_df,
        final_df=final_df,
        prep_meta=prep_meta,
        tables_dir=paths["tables"],
        reports_dir=paths["reports"],
    )

    LOGGER.info("Running unsupervised DASHI...")
    unsupervised_result = run_unsupervised_dashi(
        final_df=final_df,
        feature_columns=prep_meta["feature_columns"],
        figures_dir=paths["figures"],
        tables_dir=paths["tables"],
    )

    LOGGER.info("Running supervised DASHI...")
    supervised_result = run_supervised_dashi(
        final_df=final_df,
        feature_columns=prep_meta["feature_columns"],
        figures_dir=paths["figures"],
        tables_dir=paths["tables"],
    )

    LOGGER.info("Generating final interpretive report...")
    report_text = generate_interpretive_report(
        final_df=final_df,
        prep_meta=prep_meta,
        unsupervised_result=unsupervised_result,
        supervised_result=supervised_result,
        report_path=paths["reports"] / "final_transportability_report.md",
    )

    print("\n--- REPORT FINALE (sintesi) ---")
    print(report_text)

    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "eda": {k: v.to_dict(orient="records") for k, v in eda_outputs.items()},
        "unsupervised": {
            "gpd": unsupervised_result["gpd"],
            "plot_status": unsupervised_result["plot_status"],
        },
        "supervised": {
            "primary_metric": supervised_result["primary_metric"],
            "available_metrics": supervised_result["available_metrics"],
            "plot_status": supervised_result["plot_status"],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete DASHI pipeline on UCI Heart Disease (multi-source)."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root folder where data/outputs/src should exist (default: cwd).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of UCI raw files even if present locally.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging(verbose=True)
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    LOGGER.info("Project root: %s", project_root)

    try:
        run_pipeline(project_root=project_root, force_download=args.force_download)
        LOGGER.info("Pipeline completed successfully.")
        return 0
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
