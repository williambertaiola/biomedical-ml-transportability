"""Post-experiment analysis: run uci dashi."""

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
import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Optional

# Matplotlib/Plotly cache configuration before importing plotting libraries.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(_configured_path("results_root"))
DEFAULT_INPUT_PATH = WORKSPACE_ROOT / "Data" / "UCI_Heart_preprocessing"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT
DEFAULT_OUTPUT_NAME = "UCI_Heart_DASHI_faithful"

DROP_ROWS_WITH_MISSING_SOURCE = True
REMOVE_SOURCES_WITH_LT2_SAMPLES = True
STANDARDIZE_FEATURES = True
KDE_RESOLUTION = 20
MDS_DIMENSIONS = 3
DIM_REDUCTION = "PCA"
SUPERVISED_MODEL_TYPE = "histogram_gradient_boosting"
SUPERVISED_LEARNING_STRATEGY = "from_scratch"
SUPERVISED_METRICS_TO_PLOT = [
    "ACCURACY",
    "F1-SCORE_MACRO",
    "RECALL_MACRO",
    "PRECISION_MACRO",
    "AUC_MACRO",
    "PR-AUC_MACRO",
    "LOGLOSS",
]
STATIC_FIG_DPI = 420
STATIC_FONT_SIZE = 14
STATIC_TITLE_SIZE = 18
STATIC_LABEL_SIZE = 15
STATIC_TICK_SIZE = 11
STATIC_LEGEND_SIZE = 11

plt.rcParams.update(
    {
        "figure.dpi": STATIC_FIG_DPI,
        "savefig.dpi": STATIC_FIG_DPI,
        "font.size": STATIC_FONT_SIZE,
        "axes.titlesize": STATIC_TITLE_SIZE,
        "axes.labelsize": STATIC_LABEL_SIZE,
        "xtick.labelsize": STATIC_TICK_SIZE,
        "ytick.labelsize": STATIC_TICK_SIZE,
        "legend.fontsize": STATIC_LEGEND_SIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_dataset_path(input_arg: str | Path) -> Path:
    input_path = Path(input_arg).expanduser().resolve()
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        candidates = [
            input_path / "data" / "processed" / "uci_heart_analysis_ready.csv",
            input_path / "data" / "interim" / "uci_heart_unified.csv",
            input_path / "uci_heart_analysis_ready.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not resolve the UCI analysis CSV from the provided input path. "
        f"Looked under: {input_path}"
    )


def resolve_output_directory(output_arg: str | Path) -> Path:
    output_path = Path(output_arg).expanduser().resolve()
    if output_path.exists() and output_path.is_dir():
        if output_path.name == DEFAULT_OUTPUT_NAME:
            return output_path
        if (output_path / "unsupervised").exists() or (
            output_path / "supervised"
        ).exists():
            return output_path
        return output_path / DEFAULT_OUTPUT_NAME
    return output_path


def setup_logging(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("dashi_uci_preprocessed")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _find_column_case_insensitive(
    df: pd.DataFrame, candidates: list[str]
) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _save_plotly_html(fig: Any, html_path: Path) -> None:
    ensure_dir(html_path.parent)
    fig.write_html(str(html_path), include_plotlyjs="cdn")


def _save_matplotlib_bundle(fig: plt.Figure, base_path: Path) -> list[Path]:
    ensure_dir(base_path.parent)
    created: list[Path] = []
    for suffix in [".png", ".pdf", ".svg"]:
        path = base_path.with_suffix(suffix)
        fig.savefig(
            path,
            dpi=STATIC_FIG_DPI if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        created.append(path)
    plt.close(fig)
    return created


def _style_plotly_figure(fig: Any, title: str, is_3d: bool = False) -> Any:
    common_layout = dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
        template="plotly_white",
        font=dict(size=16, family="Arial"),
        legend=dict(font=dict(size=13), title_font=dict(size=14)),
        margin=dict(l=60, r=40, t=80, b=70),
    )
    fig.update_layout(**common_layout)
    if is_3d:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="D1", title_font=dict(size=16), tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="D2", title_font=dict(size=16), tickfont=dict(size=12)
                ),
                zaxis=dict(
                    title="D3", title_font=dict(size=16), tickfont=dict(size=12)
                ),
                camera=dict(eye=dict(x=1.55, y=1.35, z=1.15)),
                aspectmode="cube",
            )
        )
    else:
        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=12))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=12))
    return fig


def _extract_support_array(data_source_map) -> np.ndarray:
    support = data_source_map.support
    if isinstance(support, pd.DataFrame):
        values = support.iloc[:, 0].to_numpy()
    elif isinstance(support, pd.Series):
        values = support.to_numpy()
    else:
        values = np.asarray(support)
    return values.astype(float, copy=False)


def _compute_sparse_tick_positions(
    values: np.ndarray, max_ticks: int = 8
) -> tuple[np.ndarray, list[str]]:
    if len(values) <= max_ticks:
        tick_positions = values
    else:
        idx = np.linspace(0, len(values) - 1, num=max_ticks, dtype=int)
        tick_positions = values[idx]
    labels: list[str] = []
    for value in tick_positions:
        if np.isclose(value, round(value), atol=1e-6):
            labels.append(f"{int(round(value))}")
        else:
            labels.append(f"{value:.1f}")
    return tick_positions, labels


def _plot_univariate_dsm_static(
    data_source_map, base_path: Path, absolute: bool
) -> list[Path]:
    matrix = np.asarray(
        data_source_map.counts_map if absolute else data_source_map.probability_map,
        dtype=float,
    )
    sources = [str(s) for s in np.asarray(data_source_map.sources)]
    support = _extract_support_array(data_source_map)
    plotly_palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]
    title = f"Evolution of {data_source_map.variable_name}"

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    for idx, source in enumerate(sources):
        ax.plot(
            support,
            matrix[idx],
            linewidth=2.0,
            color=plotly_palette[idx % len(plotly_palette)],
            label=source,
        )

    tick_positions, tick_labels = _compute_sparse_tick_positions(support, max_ticks=10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, color="#E5ECF6", linewidth=1.0)

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color("#444444")
        ax.spines[spine_name].set_linewidth(0.8)

    ax.set_xlabel(str(data_source_map.variable_name))
    ax.set_ylabel("Absolute frequency" if absolute else "Relative frequency")
    ax.set_title(title, pad=14)
    ax.tick_params(
        axis="both", which="major", direction="out", length=4, width=0.8, colors="black"
    )
    ax.margins(x=0.01)
    ax.legend(
        title="Source",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    return _save_matplotlib_bundle(fig, base_path)


def _plot_multivariate_dsm_static(
    multivariate_probability_map: np.ndarray, sources: list[str], base_path: Path
) -> list[Path]:
    maps = np.asarray(multivariate_probability_map, dtype=float)
    if maps.ndim == 4:
        central_idx = maps.shape[-1] // 2
        sliced_maps = maps[:, :, :, central_idx]
        subtitle = f"central slice at D3 index {central_idx}"
    elif maps.ndim == 3:
        sliced_maps = maps
        subtitle = "density slices"
    else:
        raise ValueError(f"Unexpected multivariate map shape: {maps.shape}")

    n_sources = sliced_maps.shape[0]
    n_cols = min(2, n_sources)
    n_rows = int(np.ceil(n_sources / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    vmax = np.nanmax(sliced_maps)
    for idx, source in enumerate(sources):
        ax = axes[idx]
        im = ax.imshow(
            sliced_maps[idx],
            cmap="magma",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(str(source))
        ax.set_xlabel("D2 support")
        ax.set_ylabel("D1 support")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(n_sources, len(axes)):
        axes[idx].axis("off")
    fig.suptitle(f"Multivariate data source map ({subtitle})", y=0.99)
    fig.tight_layout()
    return _save_matplotlib_bundle(fig, base_path)


def _plot_msv_2d_static(
    vertices: np.ndarray,
    sources: list[str],
    spo: np.ndarray,
    n_by_source: np.ndarray,
    gpd: Optional[float],
    base_path: Path,
) -> list[Path]:
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return []
    coords = vertices.copy()
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    coords = coords[:, :2]

    sizes = np.asarray(n_by_source, dtype=float)
    if sizes.size != coords.shape[0]:
        sizes = np.ones(coords.shape[0], dtype=float)
    sizes = 350 * (sizes / max(np.nanmax(sizes), 1.0)) + 70

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=spo,
        s=sizes,
        cmap="viridis",
        alpha=0.92,
        edgecolor="black",
        linewidth=0.8,
    )
    for i, src in enumerate(sources):
        ax.annotate(
            src,
            (coords[i, 0], coords[i, 1]),
            fontsize=11,
            xytext=(6, 6),
            textcoords="offset points",
        )
    title = "Multi-Source Variability (2D)"
    if gpd is not None:
        title += f" | GPD={gpd:.4f}"
    ax.set_title(title)
    ax.set_xlabel("D1")
    ax.set_ylabel("D2")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("SPO")
    fig.tight_layout()
    return _save_matplotlib_bundle(fig, base_path)


def _plot_msv_3d_static(
    vertices: np.ndarray,
    sources: list[str],
    spo: np.ndarray,
    n_by_source: np.ndarray,
    gpd: Optional[float],
    base_path: Path,
) -> list[Path]:
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return []
    coords = vertices.copy()
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2))])
    elif coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    else:
        coords = coords[:, :3]

    sizes = np.asarray(n_by_source, dtype=float)
    if sizes.size != coords.shape[0]:
        sizes = np.ones(coords.shape[0], dtype=float)
    sizes = 420 * (sizes / max(np.nanmax(sizes), 1.0)) + 60

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=spo,
        s=sizes,
        cmap="viridis",
        alpha=0.92,
        edgecolors="black",
        linewidths=0.8,
        depthshade=True,
    )
    for i, src in enumerate(sources):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], src, fontsize=10)
    ax.set_title(
        "Multi-Source Variability (3D)"
        + (f" | GPD={gpd:.4f}" if gpd is not None else ""),
        pad=20,
    )
    ax.set_xlabel("D1", labelpad=10)
    ax.set_ylabel("D2", labelpad=10)
    ax.set_zlabel("D3", labelpad=10)
    ax.view_init(elev=22, azim=38)
    span = np.ptp(coords, axis=0)
    span[span == 0] = 1.0
    try:
        ax.set_box_aspect(span)
    except Exception:
        pass
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label("SPO")
    fig.tight_layout()
    return _save_matplotlib_bundle(fig, base_path)


def _normalize_heatmap_train_label(label: Any) -> str:
    text = str(label)
    if text.startswith("From ") and " to " in text:
        return text[len("From ") :].split(" to ", 1)[0].strip()
    return text


def _plot_supervised_heatmap_static(
    metric_frame: pd.DataFrame, metric_name: str, base_path: Path
) -> list[Path]:
    matrix = metric_frame.to_numpy(dtype=float)
    reverse = metric_name in {
        "MEAN_ABSOLUTE_ERROR",
        "MEAN_SQUARED_ERROR",
        "ROOT_MEAN_SQUARED_ERROR",
        "LOGLOSS",
    }
    cmap = "RdYlGn_r" if reverse else "RdYlGn"
    fig, ax = plt.subplots(
        figsize=(max(8, 1.8 * matrix.shape[1]), max(6, 1.2 * matrix.shape[0]))
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} heatmap", pad=16)
    ax.set_xlabel("Test batch")
    ax.set_ylabel("Training batch")
    ax.set_xticks(np.arange(metric_frame.shape[1]))
    ax.set_xticklabels(metric_frame.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(metric_frame.shape[0]))
    ax.set_yticklabels(
        [_normalize_heatmap_train_label(label) for label in metric_frame.index.tolist()]
    )
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
                color = "black"
            else:
                label = f"{value:.3f}"
                color = "black" if 0.25 < value < 0.75 else "white"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)
    fig.tight_layout()
    return _save_matplotlib_bundle(fig, base_path)


def _configure_dashi_module(
    root_module_name: str,
    local_dashi_root: Path | None,
    logger: logging.Logger,
) -> Path:
    import dashi

    installed_path = Path(dashi.__file__).resolve().parent / root_module_name
    if local_dashi_root is None:
        logger.info(
            "Using installed DASHI module for %s from %s",
            root_module_name,
            installed_path,
        )
        return installed_path

    local_path = local_dashi_root / root_module_name
    if not local_path.exists():
        logger.warning(
            "Requested local DASHI override for %s but path does not exist: %s. Falling back to installed module.",
            root_module_name,
            local_path,
        )
        return installed_path

    for module_name in list(sys.modules):
        if module_name == f"dashi.{root_module_name}" or module_name.startswith(
            f"dashi.{root_module_name}."
        ):
            del sys.modules[module_name]

    local_module = types.ModuleType(f"dashi.{root_module_name}")
    local_module.__path__ = [str(local_path), str(installed_path)]
    local_module.__package__ = f"dashi.{root_module_name}"
    sys.modules[f"dashi.{root_module_name}"] = local_module
    setattr(dashi, root_module_name, local_module)
    logger.info(
        "Configured local DASHI package override for %s -> %s",
        root_module_name,
        local_path,
    )
    return local_path


def _log_module_origin(label: str, fn: Any, logger: logging.Logger) -> None:
    module_path = Path(inspect.getfile(fn)).resolve()
    logger.info("%s resolved to %s", label, module_path)


def load_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    df = pd.read_csv(input_path, na_values=["?", "NA", "N/A", "null", "None", ""])
    logger.info("Loaded UCI preprocessed dataset with shape %s", df.shape)
    return df


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    source_col = _find_column_case_insensitive(
        df, ["source", "domain_id", "domain", "site", "hospital", "center"]
    )
    target_col = _find_column_case_insensitive(
        df, ["target", "num", "target_binary", "label"]
    )
    target_original_col = _find_column_case_insensitive(
        df, ["target_original", "num_original", "original_target"]
    )
    subject_col = _find_column_case_insensitive(
        df, ["subject_id", "sample_id", "patient_id", "id"]
    )

    if source_col is None:
        raise ValueError(
            "Could not identify the source/domain column in the input CSV."
        )
    if target_col is None:
        raise ValueError(
            "Could not identify the binary target column in the input CSV."
        )

    return {
        "source_col": source_col,
        "target_col": target_col,
        "target_original_col": target_original_col or "",
        "subject_col": subject_col or "",
    }


def detect_numeric_features(df: pd.DataFrame, excluded_cols: set[str]) -> list[str]:
    numeric_features: list[str] = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
            numeric_features.append(col)
    return numeric_features


def build_unsupervised_dataframe(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_original_col: str,
    subject_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    working_df = df.copy()
    excluded = {source_col, target_col}
    if target_original_col:
        excluded.add(target_original_col)
    if subject_col:
        excluded.add(subject_col)

    feature_cols = detect_numeric_features(working_df, excluded_cols=excluded)
    if not feature_cols:
        raise ValueError(
            "No usable numeric features available for DASHI unsupervised analysis."
        )

    work_cols = [source_col, *feature_cols]
    work_df = working_df[work_cols].copy()

    if DROP_ROWS_WITH_MISSING_SOURCE:
        work_df = work_df.dropna(subset=[source_col])
    work_df[source_col] = work_df[source_col].astype(str)

    if REMOVE_SOURCES_WITH_LT2_SAMPLES:
        counts = work_df[source_col].value_counts()
        valid_sources = counts[counts >= 2].index
        work_df = work_df.loc[work_df[source_col].isin(valid_sources)].copy()

    feature_cols = [c for c in feature_cols if work_df[c].notna().any()]
    if not feature_cols:
        raise ValueError(
            "All candidate unsupervised features became empty after filtering."
        )

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(work_df[feature_cols])

    if STANDARDIZE_FEATURES:
        scaler = StandardScaler()
        x_processed = scaler.fit_transform(x_imputed)
    else:
        scaler = None
        x_processed = x_imputed

    processed_df = pd.DataFrame(x_processed, columns=feature_cols, index=work_df.index)
    dashi_df = pd.concat([work_df[[source_col]], processed_df], axis=1)
    dashi_df[source_col] = dashi_df[source_col].astype(object)

    summary = {
        "n_rows_original": int(len(df)),
        "n_rows_used": int(len(dashi_df)),
        "n_sources": int(dashi_df[source_col].nunique()),
        "sources": sorted(dashi_df[source_col].astype(str).unique().tolist()),
        "n_features": int(len(feature_cols)),
    }
    logger.info(
        "Prepared unsupervised DASHI frame with %s rows, %s features, %s sources",
        summary["n_rows_used"],
        summary["n_features"],
        summary["n_sources"],
    )
    return dashi_df, feature_cols, summary


def build_supervised_dataframe(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_original_col: str,
    subject_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    working_df = df.copy()
    excluded = {source_col, target_col}
    if target_original_col:
        excluded.add(target_original_col)
    if subject_col:
        excluded.add(subject_col)

    numeric_features = detect_numeric_features(working_df, excluded_cols=excluded)
    if not numeric_features:
        raise ValueError(
            "No usable numeric features available for DASHI supervised analysis."
        )

    model_df = working_df[[source_col, target_col] + numeric_features].copy()
    model_df = model_df.dropna(subset=[source_col, target_col]).copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype("int64")
    model_df[source_col] = model_df[source_col].astype("object")
    missing_cells_before = int(model_df[numeric_features].isna().sum().sum())
    if missing_cells_before > 0:
        imputer = SimpleImputer(strategy="median")
        model_df.loc[:, numeric_features] = imputer.fit_transform(
            model_df[numeric_features]
        )
        logger.info(
            "Supervised DASHI input required median imputation for %s missing numeric cells.",
            missing_cells_before,
        )

    summary = {
        "n_rows_used": int(len(model_df)),
        "n_sources": int(model_df[source_col].nunique()),
        "sources": sorted(model_df[source_col].astype(str).unique().tolist()),
        "n_features": int(len(numeric_features)),
        "class_counts": model_df[target_col].value_counts().sort_index().to_dict(),
        "missing_numeric_cells_imputed": missing_cells_before,
    }
    logger.info(
        "Prepared supervised DASHI frame with %s rows, %s features, %s sources",
        summary["n_rows_used"],
        summary["n_features"],
        summary["n_sources"],
    )
    return model_df, numeric_features, summary


def run_unsupervised_analysis(
    dashi_df: pd.DataFrame,
    feature_cols: list[str],
    source_col: str,
    output_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    local_dashi_root: Path | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    import dashi

    module_root = _configure_dashi_module(
        "unsupervised_characterization", local_dashi_root, logger
    )
    from dashi.unsupervised_characterization.data_source_map.data_source_map import (
        estimate_multivariate_data_source_map,
        estimate_univariate_data_source_map,
    )
    from dashi.unsupervised_characterization.data_source_map.data_source_map_plotter import (
        plot_multivariate_data_source_map,
        plot_univariate_data_source_map,
    )
    from dashi.unsupervised_characterization.variability_metrics.estimate_msv_metrics import (
        estimate_MSV_metrics,
    )
    from dashi.unsupervised_characterization.variability_metrics.plot_msv import (
        plot_MSV,
    )

    _log_module_origin(
        "estimate_univariate_data_source_map",
        estimate_univariate_data_source_map,
        logger,
    )
    _log_module_origin(
        "estimate_multivariate_data_source_map",
        estimate_multivariate_data_source_map,
        logger,
    )
    _log_module_origin("estimate_MSV_metrics", estimate_MSV_metrics, logger)
    _log_module_origin(
        "plot_univariate_data_source_map", plot_univariate_data_source_map, logger
    )
    _log_module_origin(
        "plot_multivariate_data_source_map", plot_multivariate_data_source_map, logger
    )
    _log_module_origin("plot_MSV", plot_MSV, logger)

    anchor_feature = feature_cols[0]
    output_files: list[Path] = []

    dsm_univariate_all = estimate_univariate_data_source_map(
        data=dashi_df,
        source_column=source_col,
        numeric_smoothing=True,
        verbose=False,
    )
    dsm_univariate = (
        dsm_univariate_all[anchor_feature]
        if isinstance(dsm_univariate_all, dict)
        else dsm_univariate_all
    )

    dsm_multivariate = estimate_multivariate_data_source_map(
        data=dashi_df,
        source_column_name=source_col,
        kde_resolution=KDE_RESOLUTION,
        dimensions=MDS_DIMENSIONS,
        dim_reduction=DIM_REDUCTION,
        scale=False,
        scatter_plot=False,
        verbose=False,
    )
    msv = estimate_MSV_metrics(dsm_multivariate)

    sources = [str(s) for s in np.asarray(msv.sources)]
    spo = np.asarray(msv.SPO, dtype=float)
    vertices = np.asarray(msv.vertices, dtype=float)
    n_by_source = np.asarray(msv.nBySource, dtype=float)
    gpd = None if msv.GPD is None else float(msv.GPD)
    support_values = _extract_support_array(dsm_univariate)
    sparse_tick_positions, sparse_tick_labels = _compute_sparse_tick_positions(
        support_values, max_ticks=8
    )

    fig1_html = output_dir / f"01_dsm_univariate_relative_{anchor_feature}.html"
    fig1_base = output_dir / f"01_dsm_univariate_relative_{anchor_feature}"
    fig1 = _style_plotly_figure(
        plot_univariate_data_source_map(dsm_univariate, absolute=False),
        title=f"Univariate data source map (relative) - {anchor_feature}",
    )
    fig1.update_xaxes(
        tickmode="array",
        tickvals=sparse_tick_positions.tolist(),
        ticktext=sparse_tick_labels,
    )
    _save_plotly_html(fig1, fig1_html)
    output_files.append(fig1_html)
    output_files.extend(
        _plot_univariate_dsm_static(dsm_univariate, fig1_base, absolute=False)
    )

    fig2_html = output_dir / f"02_dsm_univariate_absolute_{anchor_feature}.html"
    fig2_base = output_dir / f"02_dsm_univariate_absolute_{anchor_feature}"
    fig2 = _style_plotly_figure(
        plot_univariate_data_source_map(dsm_univariate, absolute=True),
        title=f"Univariate data source map (absolute) - {anchor_feature}",
    )
    fig2.update_xaxes(
        tickmode="array",
        tickvals=sparse_tick_positions.tolist(),
        ticktext=sparse_tick_labels,
    )
    _save_plotly_html(fig2, fig2_html)
    output_files.append(fig2_html)
    output_files.extend(
        _plot_univariate_dsm_static(dsm_univariate, fig2_base, absolute=True)
    )

    fig3_html = output_dir / "03_dsm_multivariate_relative.html"
    fig3_base = output_dir / "03_dsm_multivariate_relative"
    fig3 = _style_plotly_figure(
        plot_multivariate_data_source_map(dsm_multivariate, absolute=False),
        title="Multivariate data source map",
    )
    _save_plotly_html(fig3, fig3_html)
    output_files.append(fig3_html)
    output_files.extend(
        _plot_multivariate_dsm_static(
            dsm_multivariate.multivariate_probability_map, sources, fig3_base
        )
    )

    fig4_html = output_dir / "04_msv_2d.html"
    fig4_base = output_dir / "04_msv_2d"
    fig4 = _style_plotly_figure(
        plot_MSV(msv_metrics=msv, dimensions=2), title="Multi-Source Variability (2D)"
    )
    _save_plotly_html(fig4, fig4_html)
    output_files.append(fig4_html)
    output_files.extend(
        _plot_msv_2d_static(vertices, sources, spo, n_by_source, gpd, fig4_base)
    )

    msv_dims_for_plot = 3 if vertices.ndim == 2 and vertices.shape[1] >= 3 else 2
    fig5_html = output_dir / f"05_msv_{msv_dims_for_plot}d.html"
    fig5_base = output_dir / f"05_msv_{msv_dims_for_plot}d"
    fig5 = _style_plotly_figure(
        plot_MSV(msv_metrics=msv, dimensions=msv_dims_for_plot),
        title=f"Multi-Source Variability ({msv_dims_for_plot}D)",
        is_3d=msv_dims_for_plot == 3,
    )
    _save_plotly_html(fig5, fig5_html)
    output_files.append(fig5_html)
    if msv_dims_for_plot == 3:
        output_files.extend(
            _plot_msv_3d_static(vertices, sources, spo, n_by_source, gpd, fig5_base)
        )
    else:
        output_files.extend(
            _plot_msv_2d_static(vertices, sources, spo, n_by_source, gpd, fig5_base)
        )

    # Aliases to stay close to the older output naming.
    alias_map = {
        "data_source_map_univariate.html": fig1_html,
        "data_source_map_multivariate.html": fig3_html,
        "msv_plot_3d.html": fig5_html,
        "interactive_msv_open_me.html": fig5_html,
    }
    for alias_name, source_path in alias_map.items():
        shutil.copy2(source_path, output_dir / alias_name)
        output_files.append(output_dir / alias_name)

    alias_static_map = {
        "data_source_map_univariate": fig1_base,
        "data_source_map_multivariate": fig3_base,
        "msv_plot_3d": fig5_base,
    }
    for alias_name, source_base in alias_static_map.items():
        for suffix in [".png", ".pdf", ".svg"]:
            source_path = source_base.with_suffix(suffix)
            if source_path.exists():
                alias_path = output_dir / f"{alias_name}{suffix}"
                shutil.copy2(source_path, alias_path)
                output_files.append(alias_path)

    msv_table = pd.DataFrame(
        {
            "source": sources,
            "SPO": spo,
            "nBySource": n_by_source if len(n_by_source) == len(sources) else np.nan,
            "GPD": gpd,
        }
    )
    msv_csv = output_dir / "msv_metrics_by_source.csv"
    msv_table.to_csv(msv_csv, index=False)
    output_files.append(msv_csv)

    msv_source_tsv = output_dir / "msv_source_metrics.tsv"
    msv_table.to_csv(msv_source_tsv, sep="\t", index=False)
    output_files.append(msv_source_tsv)

    gpd_tsv = output_dir / "msv_gpd.tsv"
    pd.DataFrame(
        [{"GPD": gpd, "n_sources": len(sources), "anchor_feature": anchor_feature}]
    ).to_csv(gpd_tsv, sep="\t", index=False)
    output_files.append(gpd_tsv)

    univar_prob = pd.DataFrame(
        np.asarray(dsm_univariate.probability_map, dtype=float),
        index=[str(s) for s in np.asarray(dsm_univariate.sources)],
    )
    univar_csv = output_dir / f"univariate_probability_map_{anchor_feature}.csv"
    univar_prob.to_csv(univar_csv)
    output_files.append(univar_csv)

    multivar_prob_array = np.asarray(
        getattr(
            dsm_multivariate,
            "multivariate_probability_map",
            dsm_multivariate.probability_map,
        ),
        dtype=float,
    )
    multivar_prob = pd.DataFrame(
        multivar_prob_array.reshape(multivar_prob_array.shape[0], -1),
        index=[str(s) for s in np.asarray(dsm_multivariate.sources)],
    )
    multivar_csv = output_dir / "multivariate_probability_map.csv"
    multivar_prob.to_csv(multivar_csv)
    output_files.append(multivar_csv)

    source_counts = (
        dashi_df[source_col]
        .value_counts()
        .sort_index()
        .rename_axis("source")
        .reset_index(name="n_samples")
    )
    source_counts_path = output_dir / "source_counts.csv"
    source_counts.to_csv(source_counts_path, index=False)
    output_files.append(source_counts_path)

    prepared_input_path = output_dir / "unsupervised_input_standardized.csv"
    dashi_df.to_csv(prepared_input_path, index=False)
    output_files.append(prepared_input_path)

    # Copy key outputs into common figures/tables folders.
    for path in output_files:
        if path.suffix.lower() in {".png", ".pdf", ".svg", ".html"}:
            shutil.copy2(path, figures_dir / path.name)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            shutil.copy2(path, tables_dir / path.name)

    return {
        "anchor_feature": anchor_feature,
        "generated_files": [str(p) for p in sorted(output_files, key=lambda x: x.name)],
        "source_counts": source_counts.to_dict(orient="records"),
        "gpd": gpd,
        "n_sources": len(sources),
        "module_root": str(module_root),
        "interactive_3d_html": str(fig5_html),
    }


def run_supervised_analysis(
    model_df: pd.DataFrame,
    numeric_features: list[str],
    source_col: str,
    target_col: str,
    output_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    local_dashi_root: Path | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    import dashi

    module_root = _configure_dashi_module(
        "supervised_characterization", local_dashi_root, logger
    )
    from dashi.supervised_characterization.arrange_metrics import (
        arrange_performance_metrics,
    )
    from dashi.supervised_characterization.estimate_models import (
        estimate_multibatch_models,
    )
    from dashi.supervised_characterization.plot_performance import (
        plot_multibatch_performance,
    )

    _log_module_origin(
        "arrange_performance_metrics", arrange_performance_metrics, logger
    )
    _log_module_origin("estimate_multibatch_models", estimate_multibatch_models, logger)
    _log_module_origin(
        "plot_multibatch_performance", plot_multibatch_performance, logger
    )

    metrics = estimate_multibatch_models(
        data=model_df,
        inputs_numerical_column_names=numeric_features,
        inputs_categorical_column_names=None,
        output_regression_column_name=None,
        output_classification_column_name=target_col,
        date_column_name=None,
        period=None,
        source_column_name=source_col,
        learning_strategy=SUPERVISED_LEARNING_STRATEGY,
        model_type=SUPERVISED_MODEL_TYPE,
    )

    metrics_rows: list[dict[str, Any]] = []
    for (train_batches, test_batch, split_name), metric_values in metrics.items():
        train_repr = (
            "|".join(str(x) for x in train_batches)
            if isinstance(train_batches, tuple)
            else str(train_batches)
        )
        for metric_name, metric_value in metric_values.items():
            metrics_rows.append(
                {
                    "train_batches": train_repr,
                    "test_batch": str(test_batch),
                    "split": split_name,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    metrics_long = pd.DataFrame(metrics_rows)
    metrics_long_path = output_dir / "supervised_multibatch_metrics_long.tsv"
    metrics_long.to_csv(metrics_long_path, sep="\t", index=False)

    available_metrics = sorted(
        {metric for values in metrics.values() for metric in values.keys()}
    )
    available_metrics_path = output_dir / "supervised_available_metrics.tsv"
    pd.DataFrame({"metric": available_metrics}).to_csv(
        available_metrics_path, sep="\t", index=False
    )

    matrix_paths: list[Path] = []
    figure_paths: list[Path] = []
    for metric_name in SUPERVISED_METRICS_TO_PLOT:
        if metric_name not in available_metrics:
            logger.warning(
                "Skipping supervised metric %s because it is not available.",
                metric_name,
            )
            continue

        performance_frame = arrange_performance_metrics(
            metrics=metrics, metric_name=metric_name
        )
        matrix_path = (
            output_dir
            / f"supervised_{metric_name.lower().replace('-', '_')}_matrix.tsv"
        )
        performance_frame.to_csv(matrix_path, sep="\t", index=True)
        matrix_paths.append(matrix_path)

        figure = _style_plotly_figure(
            plot_multibatch_performance(metrics=metrics, metric_name=metric_name),
            title=f"{metric_name.replace('_', ' ').title()} heatmap",
        )
        figure_path = (
            output_dir
            / f"supervised_{metric_name.lower().replace('-', '_')}_heatmap.html"
        )
        figure.write_html(str(figure_path), include_plotlyjs="cdn")
        figure_paths.append(figure_path)
        static_paths = _plot_supervised_heatmap_static(
            performance_frame,
            metric_name,
            output_dir / f"supervised_{metric_name.lower().replace('-', '_')}_heatmap",
        )
        figure_paths.extend(static_paths)

    prepared_input_path = output_dir / "supervised_input.csv"
    model_df.to_csv(prepared_input_path, index=False)

    source_target_counts = (
        model_df.groupby(source_col)[target_col]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
        .reset_index()
    )
    source_target_counts_path = output_dir / "source_target_counts.csv"
    source_target_counts.to_csv(source_target_counts_path, index=False)

    key_paths = [
        metrics_long_path,
        available_metrics_path,
        prepared_input_path,
        source_target_counts_path,
        *matrix_paths,
        *figure_paths,
    ]
    for path in key_paths:
        if path.suffix.lower() in {".html", ".png", ".pdf", ".svg"}:
            shutil.copy2(path, figures_dir / path.name)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            shutil.copy2(path, tables_dir / path.name)

    return {
        "available_metrics": available_metrics,
        "generated_files": [str(p) for p in sorted(key_paths, key=lambda x: x.name)],
        "n_metric_rows": int(len(metrics_long)),
        "missing_numeric_cells_imputed": int(
            model_df[numeric_features].isna().sum().sum()
        ),
        "module_root": str(module_root),
    }


def write_summary_report(
    output_dir: Path,
    input_arg: Path,
    resolved_input_path: Path,
    column_map: dict[str, str],
    unsup_summary: dict[str, Any],
    sup_summary: dict[str, Any],
    dataset_df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    report_lines = [
        "# DASHI UCI Preprocessed Pipeline",
        "",
        "## Reference workflow",
        "- Unsupervised section replicated the logic of `unsupervised_uci.py`: source detection, numeric-feature selection, median imputation, optional standardization, univariate DSM, multivariate DSM, and MSV estimation/plots.",
        "- Supervised section replicated the logic of `supervised_uci.py`: multi-batch estimation with source-based batching, long metrics table, arranged metric matrices, and Plotly heatmaps.",
        "- The main adaptation was column mapping from the new preprocessed CSV: `domain_id -> source`, `target -> classification target`, `subject_id -> ID to exclude`, and `target_original` kept as metadata but excluded from modeling.",
        "- A second necessary adaptation was supervised numeric imputation before calling `estimate_multibatch_models`, because the local DASHI supervised module rejects missing values while the preprocessed UCI CSV still contains them.",
        "- In this environment the pipeline prefers the installed `dashi` package and only uses a local override if explicitly provided and available.",
        "",
        "## Input dataset",
        f"- Input argument: `{input_arg}`",
        f"- Resolved CSV path: `{resolved_input_path}`",
        f"- Shape: `{dataset_df.shape[0]} rows x {dataset_df.shape[1]} columns`",
        f"- Source/domain column used: `{column_map['source_col']}`",
        f"- Target column used: `{column_map['target_col']}`",
        f"- Subject ID column excluded: `{column_map['subject_col'] or 'not found'}`",
        f"- Original target column excluded: `{column_map['target_original_col'] or 'not found'}`",
        "",
        "## Unsupervised outputs",
        f"- Anchor feature for univariate DSM: `{unsup_summary['anchor_feature']}`",
        f"- Number of sources analyzed: `{unsup_summary['n_sources']}`",
        f"- GPD: `{unsup_summary['gpd']}`",
        f"- Generated files: `{len(unsup_summary['generated_files'])}`",
        f"- Unsupervised module root used: `{unsup_summary.get('module_root', 'n/a')}`",
        f"- Interactive 3D figure: `{unsup_summary.get('interactive_3d_html', 'n/a')}`",
        "",
        "## Supervised outputs",
        f"- Available supervised metrics: `{', '.join(sup_summary['available_metrics'])}`",
        f"- Number of metric rows saved: `{sup_summary['n_metric_rows']}`",
        f"- Generated files: `{len(sup_summary['generated_files'])}`",
        f"- Missing numeric cells imputed before supervised DASHI: `{sup_summary.get('missing_numeric_cells_imputed', 'n/a')}`",
        f"- Supervised module root used: `{sup_summary.get('module_root', 'n/a')}`",
        "",
        "## Output structure",
        "- `unsupervised/`: unsupervised DASHI outputs and intermediate tables.",
        "- `supervised/`: supervised DASHI outputs and arranged metric matrices.",
        "- `figures/`: consolidated figures copied from both sections.",
        "- `tables/`: consolidated CSV/TSV tables copied from both sections.",
        "- `logs/`: pipeline log.",
        "",
        "## Quick open",
        "- `unsupervised/05_msv_3d.html`: main interactive 3D MSV figure.",
        "- `unsupervised/interactive_msv_open_me.html`: alias to the same interactive 3D figure.",
        "- `figures/05_msv_3d.html`: consolidated copy of the interactive 3D figure.",
    ]
    report_path = output_dir / "README.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary report to %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DASHI unsupervised and supervised pipelines on UCI preprocessed data."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the UCI preprocessing folder or directly to the analysis-ready CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Existing output root directory or a dedicated output directory path.",
    )
    parser.add_argument(
        "--local-dashi-root",
        default="",
        help="Optional root directory containing local DASHI module folders to override the installed package.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_arg = Path(args.input).expanduser().resolve()
    input_path = resolve_input_dataset_path(args.input)
    output_dir = resolve_output_directory(args.output)
    local_dashi_root = (
        Path(args.local_dashi_root).expanduser().resolve()
        if args.local_dashi_root
        else None
    )

    if output_dir.exists():
        for child_name in ["unsupervised", "supervised", "figures", "tables", "logs"]:
            child = output_dir / child_name
            if child.exists():
                shutil.rmtree(child)
        for file_name in ["README.md", "run_summary.json", "run_uci_dashi_faithful.py"]:
            file_path = output_dir / file_name
            if file_path.exists():
                file_path.unlink()

    unsup_output = ensure_dir(output_dir / "unsupervised")
    sup_output = ensure_dir(output_dir / "supervised")
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")
    logs_dir = ensure_dir(output_dir / "logs")

    logger = setup_logging(logs_dir / "uci_dashi_preprocessed.log")
    logger.info("Starting DASHI UCI preprocessed pipeline")
    logger.info("Input argument: %s", input_arg)
    logger.info("Resolved input CSV: %s", input_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Local DASHI override root: %s", local_dashi_root or "not used")

    dataset_df = load_dataset(input_path, logger)
    column_map = detect_columns(dataset_df)
    logger.info("Detected columns: %s", column_map)

    unsup_df, unsup_features, unsup_data_summary = build_unsupervised_dataframe(
        dataset_df,
        source_col=column_map["source_col"],
        target_col=column_map["target_col"],
        target_original_col=column_map["target_original_col"],
        subject_col=column_map["subject_col"],
        logger=logger,
    )
    sup_df, sup_features, sup_data_summary = build_supervised_dataframe(
        dataset_df,
        source_col=column_map["source_col"],
        target_col=column_map["target_col"],
        target_original_col=column_map["target_original_col"],
        subject_col=column_map["subject_col"],
        logger=logger,
    )

    unsup_summary = run_unsupervised_analysis(
        unsup_df,
        unsup_features,
        source_col=column_map["source_col"],
        output_dir=unsup_output,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        local_dashi_root=local_dashi_root,
        logger=logger,
    )
    sup_summary = run_supervised_analysis(
        sup_df,
        sup_features,
        source_col=column_map["source_col"],
        target_col=column_map["target_col"],
        output_dir=sup_output,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        local_dashi_root=local_dashi_root,
        logger=logger,
    )
    sup_summary["missing_numeric_cells_imputed"] = sup_data_summary.get(
        "missing_numeric_cells_imputed", 0
    )

    summary_payload = {
        "input_argument": str(input_arg),
        "resolved_input_path": str(input_path),
        "output_dir": str(output_dir),
        "column_map": column_map,
        "unsupervised_data_summary": unsup_data_summary,
        "supervised_data_summary": sup_data_summary,
        "unsupervised_outputs": unsup_summary,
        "supervised_outputs": sup_summary,
        "local_dashi_root": str(local_dashi_root)
        if local_dashi_root is not None
        else "",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    shutil.copy2(Path(__file__), output_dir / "run_uci_dashi_faithful.py")
    write_summary_report(
        output_dir,
        input_arg,
        input_path,
        column_map,
        unsup_summary,
        sup_summary,
        dataset_df,
        logger,
    )
    logger.info("DASHI UCI preprocessed pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
