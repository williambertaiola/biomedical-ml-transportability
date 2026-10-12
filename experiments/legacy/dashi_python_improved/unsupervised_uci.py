"""Transportability experiment: unsupervised uci."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dashi

from common_pipeline import dataset_output_dir, write_text
from dataset_loaders import UCI_SOURCE_COLUMN, prepare_uci_unsupervised

KDE_RESOLUTION = 20
MDS_DIMENSIONS = 3
DIM_REDUCTION = "PCA"


def _save_plotly_with_fallback(
    fig, png_path: Path, html_path: Path, fallback_png_fn, label: str
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
    except Exception as exc:
        print(
            f"[WARN] PNG plotly non disponibile per {label} ({type(exc).__name__}). Uso fallback minimo."
        )
        fallback_png_fn()


def _plot_univariate_dsm_fallback(
    data_source_map, out_png: Path, absolute: bool
) -> None:
    matrix = np.asarray(
        data_source_map.counts_map if absolute else data_source_map.probability_map,
        dtype=float,
    )
    sources = [str(source) for source in np.asarray(data_source_map.sources)]

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources)
    ax.set_xlabel(str(data_source_map.variable_name))
    ax.set_title(
        f"Univariate DSM ({'absolute' if absolute else 'relative'}) - fallback"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("count" if absolute else "probability")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_multivariate_dsm_fallback(
    multivariate_probability_map: np.ndarray, sources: list[str], out_png: Path
) -> None:
    maps = np.asarray(multivariate_probability_map, dtype=float)
    n_sources = maps.shape[0]
    n_cols = min(3, n_sources)
    n_rows = int(np.ceil(n_sources / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for index in range(n_sources):
        ax = axes[index]
        projected_map = np.asarray(maps[index], dtype=float)
        while projected_map.ndim > 2:
            projected_map = projected_map.sum(axis=0)
        im = ax.imshow(projected_map, cmap="magma", aspect="auto")
        ax.set_title(str(sources[index]))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for index in range(n_sources, len(axes)):
        axes[index].axis("off")

    fig.suptitle("Multivariate DSM (fallback)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


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

    coords = vertices.copy()
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    elif coords.shape[1] > 2:
        coords = coords[:, :2]

    sizes = np.asarray(n_by_source, dtype=float)
    if sizes.size != coords.shape[0]:
        sizes = np.ones(coords.shape[0], dtype=float)
    sizes = 100 * (sizes / max(np.nanmax(sizes), 1.0)) + 40

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=spo,
        s=sizes,
        cmap="viridis",
        alpha=0.9,
        edgecolor="k",
    )
    for index, source in enumerate(sources):
        ax.annotate(
            source,
            (coords[index, 0], coords[index, 1]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )
    title = "MSV fallback (2D)"
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


def main() -> int:
    output_dir = dataset_output_dir("uci", "unsupervised")
    analysis_df, feature_cols, metadata = prepare_uci_unsupervised()
    dashi_df = analysis_df[feature_cols + [UCI_SOURCE_COLUMN]].copy()

    anchor_feature = feature_cols[0]
    dsm_univariate_all = dashi.estimate_univariate_data_source_map(
        data=dashi_df,
        source_column=UCI_SOURCE_COLUMN,
        numeric_smoothing=True,
        verbose=False,
    )
    dsm_univariate = (
        dsm_univariate_all[anchor_feature]
        if isinstance(dsm_univariate_all, dict)
        else dsm_univariate_all
    )

    dsm_multivariate = dashi.estimate_multivariate_data_source_map(
        data=dashi_df,
        source_column_name=UCI_SOURCE_COLUMN,
        kde_resolution=KDE_RESOLUTION,
        dimensions=MDS_DIMENSIONS,
        dim_reduction=DIM_REDUCTION,
        scale=False,
        scatter_plot=False,
        verbose=False,
    )
    msv = dashi.estimate_MSV_metrics(dsm_multivariate)

    sources = [str(source) for source in np.asarray(msv.sources)]
    spo = np.asarray(msv.SPO, dtype=float)
    vertices = np.asarray(msv.vertices, dtype=float)
    n_by_source = np.asarray(msv.nBySource, dtype=float)
    gpd = None if msv.GPD is None else float(msv.GPD)

    fig1_html = output_dir / f"01_dsm_univariate_relative_{anchor_feature}.html"
    fig1_png = output_dir / f"01_dsm_univariate_relative_{anchor_feature}.png"
    fig1 = dashi.plot_univariate_data_source_map(dsm_univariate, absolute=False)
    _save_plotly_with_fallback(
        fig=fig1,
        png_path=fig1_png,
        html_path=fig1_html,
        fallback_png_fn=lambda: _plot_univariate_dsm_fallback(
            dsm_univariate, fig1_png, absolute=False
        ),
        label="DSM univariate relative",
    )

    fig2_html = output_dir / f"02_dsm_univariate_absolute_{anchor_feature}.html"
    fig2_png = output_dir / f"02_dsm_univariate_absolute_{anchor_feature}.png"
    fig2 = dashi.plot_univariate_data_source_map(dsm_univariate, absolute=True)
    _save_plotly_with_fallback(
        fig=fig2,
        png_path=fig2_png,
        html_path=fig2_html,
        fallback_png_fn=lambda: _plot_univariate_dsm_fallback(
            dsm_univariate, fig2_png, absolute=True
        ),
        label="DSM univariate absolute",
    )

    fig3_html = output_dir / "03_dsm_multivariate_relative.html"
    fig3_png = output_dir / "03_dsm_multivariate_relative.png"
    fig3 = dashi.plot_multivariate_data_source_map(dsm_multivariate, absolute=False)
    _save_plotly_with_fallback(
        fig=fig3,
        png_path=fig3_png,
        html_path=fig3_html,
        fallback_png_fn=lambda: _plot_multivariate_dsm_fallback(
            dsm_multivariate.multivariate_probability_map, sources, fig3_png
        ),
        label="DSM multivariate relative",
    )

    fig4_html = output_dir / "04_msv_2d.html"
    fig4_png = output_dir / "04_msv_2d.png"
    fig4 = dashi.plot_MSV(msv_metrics=msv, dimensions=2)
    _save_plotly_with_fallback(
        fig=fig4,
        png_path=fig4_png,
        html_path=fig4_html,
        fallback_png_fn=lambda: _plot_msv_fallback(
            vertices, sources, spo, n_by_source, gpd, fig4_png
        ),
        label="MSV 2D",
    )

    msv_dims_for_plot = 3 if vertices.ndim == 2 and vertices.shape[1] >= 3 else 2
    fig5_html = output_dir / f"05_msv_{msv_dims_for_plot}d.html"
    fig5_png = output_dir / f"05_msv_{msv_dims_for_plot}d.png"
    fig5 = dashi.plot_MSV(msv_metrics=msv, dimensions=msv_dims_for_plot)
    _save_plotly_with_fallback(
        fig=fig5,
        png_path=fig5_png,
        html_path=fig5_html,
        fallback_png_fn=lambda: _plot_msv_fallback(
            vertices, sources, spo, n_by_source, gpd, fig5_png
        ),
        label=f"MSV {msv_dims_for_plot}D",
    )

    pd.DataFrame(
        {
            "source": sources,
            "SPO": spo,
            "nBySource": n_by_source if len(n_by_source) == len(sources) else np.nan,
            "GPD": gpd,
        }
    ).to_csv(output_dir / "msv_metrics_by_source.csv", index=False)

    pd.DataFrame(
        np.asarray(dsm_univariate.probability_map, dtype=float),
        index=[str(source) for source in np.asarray(dsm_univariate.sources)],
    ).to_csv(output_dir / f"univariate_probability_map_{anchor_feature}.csv")

    pd.DataFrame(
        np.asarray(dsm_multivariate.probability_map, dtype=float),
        index=[str(source) for source in np.asarray(dsm_multivariate.sources)],
    ).to_csv(output_dir / "multivariate_probability_map.csv")

    summary = "\n".join(
        [
            "# UCI Unsupervised",
            "",
            f"- Input: `{metadata['input_path']}`",
            f"- Samples analizzati: {metadata['n_samples']}",
            f"- Feature numeriche: {metadata['n_features']}",
            f"- GPD: {gpd:.6f}" if gpd is not None else "- GPD: NA",
            "",
            "## Conteggi per source",
            *(
                f"- {source}: {count}"
                for source, count in metadata["source_counts"].items()
            ),
        ]
    )
    write_text(summary, output_dir / "analysis_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
