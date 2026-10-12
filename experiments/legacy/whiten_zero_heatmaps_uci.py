"""Transportability experiment: whiten zero heatmaps uci."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def masked_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap


def annotation_color(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "#333333"
    if np.isclose(value, 0.0):
        return "#111111"
    return "white" if value >= threshold else "#111111"


def draw_single_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    cmap_name: str,
    fmt: str,
) -> None:
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_where(np.isclose(values, 0.0), values)
    cmap = masked_cmap(cmap_name)

    image = ax.imshow(masked_values, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())

    finite_non_zero = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    threshold = float(np.nanmean(finite_non_zero)) if finite_non_zero.size else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                color=annotation_color(value, threshold),
                fontsize=8,
            )

    return image


def regenerate_weighting_distance_heatmap(root: Path) -> None:
    matrix = pd.read_csv(root / "weighting" / "domain_distance_matrix.csv", index_col=0)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * matrix.shape[1]), max(5, 0.9 * matrix.shape[0]))
    )
    image = draw_single_heatmap(
        ax,
        matrix,
        title="Pairwise Domain Distance Heatmap",
        cmap_name="RdYlGn_r",
        fmt="{:.3f}",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=90)
    fig.tight_layout()
    fig.savefig(
        root / "weighting" / "domain_distance_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def regenerate_combat_distance_heatmap(root: Path) -> None:
    raw_matrix = pd.read_csv(
        root / "combat" / "site_distance_matrix_raw.csv", index_col=0
    )
    combat_matrix = pd.read_csv(
        root / "combat" / "site_distance_matrix_combat.csv", index_col=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, matrix in zip(
        axes,
        ("Raw Site Distances", "ComBat Site Distances"),
        (raw_matrix, combat_matrix),
        strict=False,
    ):
        image = draw_single_heatmap(
            ax,
            matrix,
            title=title,
            cmap_name="RdYlGn_r",
            fmt="{:.2f}",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(
        root / "combat" / "plots" / "site_distance_heatmap_raw_vs_combat.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    root = Path(_configured_path("legacy_uci_final"))
    regenerate_weighting_distance_heatmap(root)
    regenerate_combat_distance_heatmap(root)


if __name__ == "__main__":
    main()
