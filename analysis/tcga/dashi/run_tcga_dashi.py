"""Post-experiment analysis: run tcga dashi."""

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
DEFAULT_INPUT_PATH = WORKSPACE_ROOT / "Data" / "TCGA_BRCA_preprocessing"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR
DEFAULT_OUTPUT_NAME = "TCGA_BRCA_DASHI_faithful"

DROP_ROWS_WITH_MISSING_SOURCE = True
REMOVE_SOURCES_WITH_LT2_SAMPLES = True
STANDARDIZE_FEATURES = True
KDE_RESOLUTION = 20
MDS_DIMENSIONS = 3
DIM_REDUCTION = "PCA"
MAX_NUMERIC_FEATURES = 500
GENE_PREFIX = "ENSG"
TCGA_BINARY_TARGET_VALUES = {0, 1}
APPLY_TCGA_CLASS_SITE_FILTER = True
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
            input_path
            / "TCGA_BRCA_preprocessing"
            / "data"
            / "processed"
            / "tcga_brca_stage_transportability_dataset.csv",
            input_path
            / "data"
            / "processed"
            / "tcga_brca_stage_transportability_dataset.csv",
            input_path / "tcga_brca_stage_transportability_dataset.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not resolve the TCGA transportability CSV from the provided input path. "
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
    logger = logging.getLogger("dashi_tcga_preprocessed")
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


def _stringify_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    payload = {str(k): int(v) for k, v in counts.items()}
    return json.dumps(payload, sort_keys=True)


def _normalize_train_batches(train_batches: Any) -> tuple[str, ...]:
    if isinstance(train_batches, tuple):
        return tuple(str(x) for x in train_batches)
    if isinstance(train_batches, list):
        return tuple(str(x) for x in train_batches)
    return (str(train_batches),)


def _train_batch_repr(train_batches: Any) -> str:
    return "|".join(_normalize_train_batches(train_batches))


def _normalize_heatmap_train_label(label: Any) -> str:
    text = str(label)
    if text.startswith("From ") and " to " in text:
        return text[len("From ") :].split(" to ", 1)[0].strip()
    return text


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


def _select_top_variance_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
    context: str,
    max_features: Optional[int] = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not feature_cols:
        return [], []
    if max_features is None:
        max_features = MAX_NUMERIC_FEATURES

    variances = frame[feature_cols].var(axis=0, skipna=True).replace({np.nan: -np.inf})
    ranked = variances.sort_values(ascending=False)
    selected = ranked.head(min(max_features, len(ranked))).index.tolist()
    manifest: list[dict[str, Any]] = []
    for rank, feature_name in enumerate(selected, start=1):
        value = ranked.loc[feature_name]
        manifest.append(
            {
                "feature": feature_name,
                "variance": None if not np.isfinite(value) else float(value),
                "rank": rank,
                "context": context,
            }
        )

    if len(feature_cols) > len(selected):
        logger.info(
            "Selected top %s variable numeric features for %s from %s candidates.",
            len(selected),
            context,
            len(feature_cols),
        )
    else:
        logger.info("Using all %s numeric features for %s.", len(selected), context)
    return selected, manifest


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
    sizes = 520 * (sizes / max(np.nanmax(sizes), 1.0)) + 140

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
        linewidths=1.0,
        depthshade=True,
    )
    for i, src in enumerate(sources):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], src, fontsize=10, zorder=5)
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


def _normalized_msv_coordinates(vertices: np.ndarray, dimensions: int) -> np.ndarray:
    coords = np.asarray(vertices, dtype=float).copy()
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros((0, dimensions), dtype=float)
    if coords.shape[1] < dimensions:
        coords = np.hstack(
            [coords, np.zeros((coords.shape[0], dimensions - coords.shape[1]))]
        )
    coords = coords[:, :dimensions]
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def _normalized_msv_marker_sizes(
    n_by_source: np.ndarray, n_points: int, *, min_size: float, spread: float
) -> np.ndarray:
    sizes = np.asarray(n_by_source, dtype=float).reshape(-1)
    if sizes.size != n_points:
        sizes = np.ones(n_points, dtype=float)
    sizes = np.nan_to_num(sizes, nan=0.0, posinf=0.0, neginf=0.0)
    sizes[sizes < 0] = 0.0
    max_size = float(np.max(sizes)) if sizes.size else 0.0
    if max_size <= 0.0:
        return np.full(n_points, min_size + spread * 0.5, dtype=float)
    return min_size + spread * (sizes / max_size)


def _build_plotly_msv_figure(
    *,
    vertices: np.ndarray,
    sources: list[str],
    spo: np.ndarray,
    n_by_source: np.ndarray,
    gpd: Optional[float],
    dimensions: int,
):
    from plotly import graph_objects as go

    coords = _normalized_msv_coordinates(vertices, dimensions)
    spo_values = np.asarray(spo, dtype=float).reshape(-1)
    if spo_values.size != coords.shape[0]:
        spo_values = np.zeros(coords.shape[0], dtype=float)
    spo_values = np.nan_to_num(spo_values, nan=0.0, posinf=0.0, neginf=0.0)
    marker_sizes = _normalized_msv_marker_sizes(
        n_by_source,
        coords.shape[0],
        min_size=16.0 if dimensions == 3 else 14.0,
        spread=42.0 if dimensions == 3 else 34.0,
    )
    title = f"Multi-Source Variability ({dimensions}D)" + (
        f" | GPD={gpd:.4f}" if gpd is not None else ""
    )
    hover_text = [
        f"<b>{src}</b><br>D1: {coords[idx, 0]:.3f}<br>D2: {coords[idx, 1]:.3f}"
        + (f"<br>D3: {coords[idx, 2]:.3f}" if dimensions == 3 else "")
        + f"<br>SPO: {spo_values[idx]:.4f}"
        for idx, src in enumerate(sources)
    ]

    common_marker = dict(
        size=marker_sizes.tolist(),
        color=spo_values.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="SPO"),
        opacity=0.88,
        line=dict(color="black", width=1.0),
    )

    if dimensions == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers+text",
                    text=sources,
                    textposition="top center",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    marker=common_marker,
                )
            ]
        )
        return _style_plotly_figure(fig, title=title, is_3d=True)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+text",
                text=sources,
                textposition="top center",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_text,
                marker=common_marker,
            )
        ]
    )
    fig.update_xaxes(title="D1")
    fig.update_yaxes(title="D2")
    return _style_plotly_figure(fig, title=title, is_3d=False)


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


def _configure_local_package_path(
    root_module_name: str, local_path: Path, logger: logging.Logger
) -> None:
    import dashi

    installed_path = Path(dashi.__file__).resolve().parent / root_module_name
    if not local_path.exists():
        raise FileNotFoundError(f"Local DASHI module folder not found: {local_path}")

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


def _assert_local_module(fn: Any, expected_root: Path) -> None:
    module_path = Path(inspect.getfile(fn)).resolve()
    if not str(module_path).startswith(str(expected_root.resolve())):
        raise RuntimeError(
            f"Expected local DASHI module under {expected_root}, got {module_path}"
        )


def load_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    df = pd.read_csv(input_path, na_values=["?", "NA", "N/A", "null", "None", ""])
    logger.info("Loaded TCGA-BRCA transportability dataset with shape %s", df.shape)
    return df


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    source_col = _find_column_case_insensitive(
        df, ["source", "domain_id", "domain", "site", "hospital", "center"]
    )
    target_col = _find_column_case_insensitive(
        df,
        ["target", "num", "target_binary", "label", "tumor_level", "sample_type_code"],
    )
    target_original_col = _find_column_case_insensitive(
        df,
        [
            "target_label",
            "ajcc_pathologic_stage_raw",
            "target_original",
            "num_original",
            "original_target",
            "sample_type_label",
        ],
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


def detect_gene_features(df: pd.DataFrame) -> list[str]:
    gene_features = [col for col in df.columns if str(col).startswith(GENE_PREFIX)]
    if not gene_features:
        raise ValueError(
            f"No gene-expression columns found with prefix '{GENE_PREFIX}'."
        )
    for col in gene_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return gene_features


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

    feature_cols = [
        col for col in detect_gene_features(working_df) if col not in excluded
    ]
    if not feature_cols:
        raise ValueError(
            "No usable gene-expression features available for DASHI unsupervised analysis."
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

    candidate_feature_cols = [c for c in feature_cols if work_df[c].notna().any()]
    if not candidate_feature_cols:
        raise ValueError(
            "All candidate unsupervised features became empty after filtering."
        )
    feature_cols, selected_manifest = _select_top_variance_features(
        work_df,
        candidate_feature_cols,
        logger=logger,
        context="unsupervised TCGA DASHI",
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
        "n_candidate_features": int(len(candidate_feature_cols)),
        "feature_selection_method": "top_variance",
        "feature_selection_top_k": int(MAX_NUMERIC_FEATURES),
        "selected_features_manifest": selected_manifest,
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

    numeric_features = [
        col for col in detect_gene_features(working_df) if col not in excluded
    ]
    if not numeric_features:
        raise ValueError(
            "No usable gene-expression features available for DASHI supervised analysis."
        )
    candidate_numeric_features = list(numeric_features)

    model_df = working_df[[source_col, target_col] + numeric_features].copy()
    model_df = model_df.dropna(subset=[source_col, target_col]).copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype("int64")
    model_df[source_col] = model_df[source_col].astype("object")
    numeric_features = [c for c in numeric_features if model_df[c].notna().any()]
    numeric_features, selected_manifest = _select_top_variance_features(
        model_df,
        numeric_features,
        logger=logger,
        context="supervised TCGA DASHI",
    )
    model_df = model_df[[source_col, target_col] + numeric_features].copy()
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
        "n_candidate_features": int(len(candidate_numeric_features)),
        "feature_selection_method": "top_variance",
        "feature_selection_top_k": int(MAX_NUMERIC_FEATURES),
        "selected_features_manifest": selected_manifest,
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


def summarize_input_missingness(
    df: pd.DataFrame, column_map: dict[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    protected_cols = {v for v in column_map.values() if v}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0 and col not in protected_cols:
            continue
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_rate": float(missing_count / len(df)) if len(df) else 0.0,
                "is_key_column": col in protected_cols,
            }
        )
    if not rows:
        rows.append(
            {
                "column": "__no_missing_columns__",
                "dtype": "",
                "missing_count": 0,
                "missing_rate": 0.0,
                "is_key_column": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_key_column", "missing_count", "column"], ascending=[False, False, True]
    )


def prepare_tcga_cohort(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_original_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_df = df.copy()
    before_site_target = pd.crosstab(
        working_df[source_col], working_df[target_col], dropna=False
    ).sort_index()

    working_df[target_col] = pd.to_numeric(working_df[target_col], errors="coerce")
    filtered_df = working_df.loc[
        working_df[target_col].isin(TCGA_BINARY_TARGET_VALUES)
    ].copy()

    after_target_filter = pd.crosstab(
        filtered_df[source_col], filtered_df[target_col], dropna=False
    ).sort_index()
    valid_sites = after_target_filter.index[
        (after_target_filter.get(0, 0) > 0) & (after_target_filter.get(1, 0) > 0)
    ].tolist()
    filtered_df = filtered_df.loc[filtered_df[source_col].isin(valid_sites)].copy()
    filtered_df[target_col] = filtered_df[target_col].astype("int64")

    if target_original_col and target_original_col in filtered_df.columns:
        filtered_df[target_original_col] = filtered_df[target_original_col].astype(str)

    after_site_filter = pd.crosstab(
        filtered_df[source_col], filtered_df[target_col], dropna=False
    ).sort_index()

    def _records_with_string_keys(frame: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in frame.reset_index().to_dict(orient="records"):
            records.append({str(k): v for k, v in row.items()})
        return records

    summary = {
        "n_rows_before": int(len(df)),
        "n_rows_after_target_filter": int(sum(after_target_filter.sum(axis=1))),
        "n_rows_after_site_filter": int(len(filtered_df)),
        "n_sites_before": int(df[source_col].nunique()),
        "n_sites_after": int(filtered_df[source_col].nunique()),
        "target_values_kept": sorted(int(x) for x in TCGA_BINARY_TARGET_VALUES),
        "retained_sites": [str(x) for x in valid_sites],
        "site_target_counts_before": _records_with_string_keys(before_site_target),
        "site_target_counts_after_target_filter": _records_with_string_keys(
            after_target_filter
        ),
        "site_target_counts_after_site_filter": _records_with_string_keys(
            after_site_filter
        ),
    }
    logger.info(
        "Prepared TCGA cohort for DASHI: kept target values %s and retained %s sites with both classes (%s rows).",
        sorted(TCGA_BINARY_TARGET_VALUES),
        len(valid_sites),
        len(filtered_df),
    )
    return filtered_df, summary


def diagnose_metric_nans(
    metrics: dict[Any, dict[str, Any]],
    model_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    global_classes = sorted(model_df[target_col].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for (train_batches, test_batch, split_name), metric_values in metrics.items():
        train_batches_tuple = _normalize_train_batches(train_batches)
        train_repr = _train_batch_repr(train_batches)
        test_batch_str = str(test_batch)
        train_mask = model_df[source_col].astype(str).isin(train_batches_tuple)
        test_mask = model_df[source_col].astype(str) == test_batch_str
        train_targets = model_df.loc[train_mask, target_col].astype(str)
        test_targets = model_df.loc[test_mask, target_col].astype(str)
        missing_train_classes = [
            cls for cls in global_classes if cls not in set(train_targets.tolist())
        ]
        missing_test_classes = [
            cls for cls in global_classes if cls not in set(test_targets.tolist())
        ]

        for metric_name, metric_value in metric_values.items():
            if not pd.isna(metric_value):
                continue

            reasons: list[str] = []
            if train_targets.empty:
                reasons.append("empty_training_pool")
            if test_targets.empty:
                reasons.append("empty_test_site")
            if train_targets.nunique() < 2:
                reasons.append("single_class_training_pool")
            if test_targets.nunique() < 2:
                reasons.append("single_class_test_site")
            if metric_name.startswith("AUC"):
                if test_targets.nunique() < 2:
                    reasons.append("auc_not_defined_single_test_class")
                if missing_test_classes:
                    reasons.append("auc_missing_test_classes")
            if metric_name.startswith("PR-AUC"):
                if test_targets.nunique() < 2:
                    reasons.append("pr_auc_not_defined_single_test_class")
                if missing_test_classes:
                    reasons.append("pr_auc_missing_test_classes")
            if missing_train_classes:
                reasons.append(
                    "missing_train_classes:" + "|".join(missing_train_classes)
                )
            if missing_test_classes:
                reasons.append("missing_test_classes:" + "|".join(missing_test_classes))
            if not reasons:
                reasons.append("nan_returned_by_dashi_metric")

            rows.append(
                {
                    "train_batches": train_repr,
                    "test_batch": test_batch_str,
                    "split": str(split_name),
                    "metric": metric_name,
                    "reason": ";".join(reasons),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "train_class_counts": _stringify_counts(train_targets),
                    "test_class_counts": _stringify_counts(test_targets),
                    "n_train_classes": int(train_targets.nunique()),
                    "n_test_classes": int(test_targets.nunique()),
                }
            )

    nan_df = pd.DataFrame(rows)
    if not nan_df.empty:
        logger.warning(
            "Detected %s NaN metric values in supervised TCGA DASHI outputs.",
            len(nan_df),
        )
    else:
        logger.info("No NaN metric values detected in supervised TCGA DASHI outputs.")
    return nan_df


def diagnose_heatmap_nans(
    performance_frame: pd.DataFrame,
    metric_name: str,
    metrics_long: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = metrics_long.loc[metrics_long["metric"] == metric_name].copy()
    for train_label in performance_frame.index.tolist():
        normalized_train_label = _normalize_heatmap_train_label(train_label)
        for test_label in performance_frame.columns.tolist():
            value = performance_frame.loc[train_label, test_label]
            if not pd.isna(value):
                continue
            matches = subset.loc[
                (subset["train_batches"].astype(str) == str(normalized_train_label))
                & (subset["test_batch"].astype(str) == str(test_label))
            ]
            if matches.empty:
                reason = "pivot_missing_combination"
                split_values = ""
            elif matches["value"].isna().all():
                reason = "upstream_metric_nan"
                split_values = "|".join(
                    sorted(matches["split"].astype(str).unique().tolist())
                )
            else:
                reason = "nan_introduced_after_arrangement"
                split_values = "|".join(
                    sorted(matches["split"].astype(str).unique().tolist())
                )
            rows.append(
                {
                    "metric": metric_name,
                    "train_batches": str(train_label),
                    "normalized_train_batches": str(normalized_train_label),
                    "test_batch": str(test_label),
                    "reason": reason,
                    "available_splits": split_values,
                }
            )
    return pd.DataFrame(rows)


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
    fig4 = _build_plotly_msv_figure(
        vertices=vertices,
        sources=sources,
        spo=spo,
        n_by_source=n_by_source,
        gpd=gpd,
        dimensions=2,
    )
    _save_plotly_html(fig4, fig4_html)
    output_files.append(fig4_html)
    output_files.extend(
        _plot_msv_2d_static(vertices, sources, spo, n_by_source, gpd, fig4_base)
    )

    msv_dims_for_plot = 3 if vertices.ndim == 2 and vertices.shape[1] >= 3 else 2
    fig5_html = output_dir / f"05_msv_{msv_dims_for_plot}d.html"
    fig5_base = output_dir / f"05_msv_{msv_dims_for_plot}d"
    fig5 = _build_plotly_msv_figure(
        vertices=vertices,
        sources=sources,
        spo=spo,
        n_by_source=n_by_source,
        gpd=gpd,
        dimensions=msv_dims_for_plot,
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
        train_repr = _train_batch_repr(train_batches)
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

    nan_metrics_debug = diagnose_metric_nans(
        metrics=metrics,
        model_df=model_df,
        source_col=source_col,
        target_col=target_col,
        logger=logger,
    )
    nan_metrics_debug_path = output_dir / "supervised_nan_metric_debug.tsv"
    if nan_metrics_debug.empty:
        nan_metrics_debug = pd.DataFrame(
            [
                {
                    "train_batches": "",
                    "test_batch": "",
                    "split": "",
                    "metric": "",
                    "reason": "no_nan_metrics_detected",
                    "n_train": 0,
                    "n_test": 0,
                    "train_class_counts": "{}",
                    "test_class_counts": "{}",
                    "n_train_classes": 0,
                    "n_test_classes": 0,
                }
            ]
        )
    nan_metrics_debug.to_csv(nan_metrics_debug_path, sep="\t", index=False)

    available_metrics = sorted(
        {metric for values in metrics.values() for metric in values.keys()}
    )
    available_metrics_path = output_dir / "supervised_available_metrics.tsv"
    pd.DataFrame({"metric": available_metrics}).to_csv(
        available_metrics_path, sep="\t", index=False
    )

    matrix_paths: list[Path] = []
    figure_paths: list[Path] = []
    heatmap_nan_paths: list[Path] = []
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

        heatmap_nan_debug = diagnose_heatmap_nans(
            performance_frame=performance_frame,
            metric_name=metric_name,
            metrics_long=metrics_long,
        )
        heatmap_nan_path = (
            output_dir
            / f"supervised_{metric_name.lower().replace('-', '_')}_nan_debug.tsv"
        )
        if heatmap_nan_debug.empty:
            heatmap_nan_debug = pd.DataFrame(
                [
                    {
                        "metric": metric_name,
                        "train_batches": "",
                        "test_batch": "",
                        "reason": "no_heatmap_nan_cells",
                        "available_splits": "",
                    }
                ]
            )
        else:
            logger.warning(
                "Metric %s produced %s NaN cells in the arranged heatmap matrix.",
                metric_name,
                len(heatmap_nan_debug),
            )
        heatmap_nan_debug.to_csv(heatmap_nan_path, sep="\t", index=False)
        heatmap_nan_paths.append(heatmap_nan_path)

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
        nan_metrics_debug_path,
        *matrix_paths,
        *heatmap_nan_paths,
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
        "nan_metric_rows": 0
        if nan_metrics_debug_path is None
        else int(
            0
            if "no_nan_metrics_detected" in nan_metrics_debug.iloc[0].get("reason", "")
            and len(nan_metrics_debug) == 1
            else len(nan_metrics_debug)
        ),
        "heatmap_nan_debug_files": [
            str(p) for p in sorted(heatmap_nan_paths, key=lambda x: x.name)
        ],
        "module_root": str(module_root),
    }


def write_summary_report(
    output_dir: Path,
    input_path: Path,
    column_map: dict[str, str],
    unsup_summary: dict[str, Any],
    sup_summary: dict[str, Any],
    dataset_df: pd.DataFrame,
    tcga_filter_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    report_lines = [
        "# DASHI TCGA-BRCA Preprocessed Pipeline",
        "",
        "## Reference workflow",
        "- Unsupervised section follows the same Python DASHI workflow used for UCI: source detection, top-variance gene selection, median imputation, optional standardization, univariate DSM, multivariate DSM, and MSV estimation/plots.",
        "- Supervised section follows the same Python DASHI workflow used for UCI: multi-batch estimation with source-based batching, long metrics table, arranged metric matrices, and Plotly heatmaps.",
        "- For TCGA we keep the high-dimensionality safeguard already used in the older TCGA DASHI script: only the top variable gene-expression features are retained before running DASHI to keep the workflow tractable and stable.",
        "- Column mapping on the current TCGA transportability CSV is: `site -> source`, `target -> classification target`, `target_label/ajcc_pathologic_stage_raw -> target metadata`, and `subject_id -> ID to exclude`.",
        "- Supervised numeric imputation is still performed before calling `estimate_multibatch_models`, because the local DASHI supervised module rejects missing values.",
        "- Extra NaN diagnostics were added for TCGA to explain whether missing cells in metrics/heatmaps come from the input, from class-missing site splits, or from metric-specific undefined cases.",
        "- Interactive 3D outputs are exported as Plotly HTML and the static companion figure is rendered with a real Matplotlib 3D axis.",
        "",
        "## Input dataset",
        f"- Input path: `{input_path}`",
        f"- Shape: `{dataset_df.shape[0]} rows x {dataset_df.shape[1]} columns`",
        f"- Source/domain column used: `{column_map['source_col']}`",
        f"- Target column used: `{column_map['target_col']}`",
        f"- Subject ID column excluded: `{column_map['subject_col'] or 'not found'}`",
        f"- Original target column excluded: `{column_map['target_original_col'] or 'not found'}`",
        (
            f"- TCGA cohort restriction applied: kept target values `{tcga_filter_summary['target_values_kept']}` and sites `{', '.join(tcga_filter_summary['retained_sites'])}`"
            if tcga_filter_summary.get("applied_filter")
            else f"- TCGA cohort restriction applied: `none` (full multisite cohort retained; {tcga_filter_summary['n_rows_after_site_filter']} rows, {tcga_filter_summary['n_sites_after']} sites)"
        ),
        "",
        "## Unsupervised outputs",
        f"- Anchor feature for univariate DSM: `{unsup_summary['anchor_feature']}`",
        f"- Number of sources analyzed: `{unsup_summary['n_sources']}`",
        f"- GPD: `{unsup_summary['gpd']}`",
        f"- Numeric features kept after variance filtering: `{unsup_summary.get('n_candidate_features', 'n/a')}` candidates -> `{unsup_summary.get('n_features', 'n/a')}` selected.",
        f"- Generated files: `{len(unsup_summary['generated_files'])}`",
        f"- Unsupervised module root used: `{unsup_summary.get('module_root', 'n/a')}`",
        f"- Interactive 3D figure: `{unsup_summary.get('interactive_3d_html', 'n/a')}`",
        "",
        "## Supervised outputs",
        f"- Available supervised metrics: `{', '.join(sup_summary['available_metrics'])}`",
        f"- Number of metric rows saved: `{sup_summary['n_metric_rows']}`",
        f"- Generated files: `{len(sup_summary['generated_files'])}`",
        f"- Missing numeric cells imputed before supervised DASHI: `{sup_summary.get('missing_numeric_cells_imputed', 'n/a')}`",
        f"- NaN metric rows detected and logged: `{sup_summary.get('nan_metric_rows', 'n/a')}`",
        f"- Supervised module root used: `{sup_summary.get('module_root', 'n/a')}`",
        "",
        "## Output structure",
        "- `unsupervised/`: unsupervised DASHI outputs and intermediate tables.",
        "- `supervised/`: supervised DASHI outputs and arranged metric matrices.",
        "- `figures/`: consolidated figures copied from both sections.",
        "- `tables/`: consolidated CSV/TSV tables copied from both sections.",
        "- `logs/`: pipeline log.",
        "- `tables/input_missingness_summary.csv`: missing values already present in the raw preprocessed input.",
        "- `supervised/supervised_nan_metric_debug.tsv`: per-metric NaN diagnostics with train/test class counts and reasons.",
        "- `supervised/*_nan_debug.tsv`: heatmap-cell NaN diagnostics after arranging each performance matrix.",
        "- `unsupervised/05_msv_3d.html`: main interactive 3D MSV figure.",
        "- `unsupervised/interactive_msv_open_me.html`: alias to the same interactive 3D figure.",
        "- `figures/05_msv_3d.html`: consolidated copy of the interactive 3D figure.",
    ]
    report_path = output_dir / "README.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary report to %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DASHI unsupervised and supervised pipelines on TCGA-BRCA preprocessed data."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the TCGA preprocessing folder, the Data folder, or directly to the stage-transportability CSV.",
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
        for file_name in [
            "README.md",
            "run_summary.json",
            "run_tcga_dashi_faithful.py",
        ]:
            file_path = output_dir / file_name
            if file_path.exists():
                file_path.unlink()

    unsup_output = ensure_dir(output_dir / "unsupervised")
    sup_output = ensure_dir(output_dir / "supervised")
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")
    logs_dir = ensure_dir(output_dir / "logs")

    logger = setup_logging(logs_dir / "tcga_dashi_preprocessed.log")
    logger.info("Starting DASHI TCGA-BRCA preprocessed pipeline")
    logger.info("Input argument: %s", input_arg)
    logger.info("Resolved input CSV: %s", input_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Local DASHI override root: %s", local_dashi_root or "not used")

    dataset_df = load_dataset(input_path, logger)
    column_map = detect_columns(dataset_df)
    logger.info("Detected columns: %s", column_map)

    input_missingness = summarize_input_missingness(dataset_df, column_map)
    input_missingness_path = tables_dir / "input_missingness_summary.csv"
    input_missingness.to_csv(input_missingness_path, index=False)

    if APPLY_TCGA_CLASS_SITE_FILTER:
        tcga_filtered_df, tcga_filter_summary = prepare_tcga_cohort(
            dataset_df,
            source_col=column_map["source_col"],
            target_col=column_map["target_col"],
            target_original_col=column_map["target_original_col"],
            logger=logger,
        )
        site_target_before = tcga_filter_summary["site_target_counts_before"]
        site_target_after_target = tcga_filter_summary[
            "site_target_counts_after_target_filter"
        ]
        site_target_after_site = tcga_filter_summary[
            "site_target_counts_after_site_filter"
        ]
        tcga_filter_summary["applied_filter"] = True
    else:
        tcga_filtered_df = dataset_df.copy()
        site_target_full = [
            {str(k): v for k, v in row.items()}
            for row in pd.crosstab(
                tcga_filtered_df[column_map["source_col"]],
                pd.to_numeric(
                    tcga_filtered_df[column_map["target_col"]], errors="coerce"
                ),
                dropna=False,
            )
            .sort_index()
            .reset_index()
            .to_dict(orient="records")
        ]
        tcga_filter_summary = {
            "applied_filter": False,
            "n_rows_before": int(len(dataset_df)),
            "n_rows_after_target_filter": int(len(dataset_df)),
            "n_rows_after_site_filter": int(len(dataset_df)),
            "n_sites_before": int(dataset_df[column_map["source_col"]].nunique()),
            "n_sites_after": int(dataset_df[column_map["source_col"]].nunique()),
            "target_values_kept": sorted(
                pd.to_numeric(dataset_df[column_map["target_col"]], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
            "retained_sites": sorted(
                dataset_df[column_map["source_col"]].astype(str).unique().tolist()
            ),
            "site_target_counts_before": site_target_full,
            "site_target_counts_after_target_filter": site_target_full,
            "site_target_counts_after_site_filter": site_target_full,
        }
        site_target_before = site_target_full
        site_target_after_target = site_target_full
        site_target_after_site = site_target_full

    pd.DataFrame(site_target_before).to_csv(
        tables_dir / "tcga_site_target_counts_before_filter.csv",
        index=False,
    )
    pd.DataFrame(site_target_after_target).to_csv(
        tables_dir / "tcga_site_target_counts_after_target_filter.csv",
        index=False,
    )
    pd.DataFrame(site_target_after_site).to_csv(
        tables_dir / "tcga_site_target_counts_after_site_filter.csv",
        index=False,
    )
    pd.DataFrame([tcga_filter_summary]).drop(
        columns=[
            "site_target_counts_before",
            "site_target_counts_after_target_filter",
            "site_target_counts_after_site_filter",
        ]
    ).to_csv(
        tables_dir / "tcga_filtering_summary.csv",
        index=False,
    )

    unsup_df, unsup_features, unsup_data_summary = build_unsupervised_dataframe(
        tcga_filtered_df,
        source_col=column_map["source_col"],
        target_col=column_map["target_col"],
        target_original_col=column_map["target_original_col"],
        subject_col=column_map["subject_col"],
        logger=logger,
    )
    sup_df, sup_features, sup_data_summary = build_supervised_dataframe(
        tcga_filtered_df,
        source_col=column_map["source_col"],
        target_col=column_map["target_col"],
        target_original_col=column_map["target_original_col"],
        subject_col=column_map["subject_col"],
        logger=logger,
    )

    pd.DataFrame(unsup_data_summary.get("selected_features_manifest", [])).to_csv(
        tables_dir / "unsupervised_selected_feature_manifest.csv",
        index=False,
    )
    pd.DataFrame(sup_data_summary.get("selected_features_manifest", [])).to_csv(
        tables_dir / "supervised_selected_feature_manifest.csv",
        index=False,
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
    unsup_summary["n_features"] = unsup_data_summary.get("n_features")
    unsup_summary["n_candidate_features"] = unsup_data_summary.get(
        "n_candidate_features"
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
        "tcga_filter_summary": tcga_filter_summary,
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

    shutil.copy2(Path(__file__), output_dir / "run_tcga_dashi_faithful.py")
    write_summary_report(
        output_dir,
        input_path,
        column_map,
        unsup_summary,
        sup_summary,
        dataset_df,
        tcga_filter_summary,
        logger,
    )
    logger.info("DASHI TCGA-BRCA preprocessed pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
