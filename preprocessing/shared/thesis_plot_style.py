from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

matplotlib.use("Agg")


def apply_base_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "savefig.bbox": "tight",
        }
    )


def thesis_cmap(reverse: bool = False):
    return plt.get_cmap("RdYlGn_r" if reverse else "RdYlGn")


def value_colors(
    values: Iterable[float], *, reverse: bool = False
) -> list[tuple[float, float, float, float]]:
    finite_values = pd.Series(list(values), dtype="float64").replace(
        [np.inf, -np.inf], np.nan
    )
    if finite_values.notna().sum() == 0:
        return [thesis_cmap(reverse=reverse)(0.5)] * len(finite_values)

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if np.isclose(vmin, vmax):
        norm = colors.Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = thesis_cmap(reverse=reverse)
    return [
        cmap(norm(value)) if pd.notna(value) else (0.8, 0.8, 0.8, 1.0)
        for value in finite_values
    ]


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sorted_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    reverse_colors: bool = False,
    rotation: int = 45,
) -> None:
    ordered = df.sort_values(y, ascending=False).reset_index(drop=True)
    apply_base_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        ordered[x].astype(str),
        ordered[y].astype(float),
        color=value_colors(ordered[y].astype(float).tolist(), reverse=reverse_colors),
        edgecolor="#666666",
        linewidth=0.6,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    save_figure(fig, output_path)


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str | None = None,
) -> None:
    apply_base_style()
    fig, ax = plt.subplots(
        figsize=(max(8, 0.9 * matrix_df.shape[1]), max(5, 0.45 * matrix_df.shape[0]))
    )
    image = ax.imshow(
        matrix_df.to_numpy(dtype=float),
        aspect="auto",
        cmap=thesis_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index.astype(str))

    if fmt is not None and matrix_df.shape[0] <= 30 and matrix_df.shape[1] <= 30:
        for row_idx in range(matrix_df.shape[0]):
            for col_idx in range(matrix_df.shape[1]):
                value = matrix_df.iat[row_idx, col_idx]
                if pd.notna(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        format(float(value), fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#222222",
                    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel("Value", rotation=90)
    save_figure(fig, output_path)
