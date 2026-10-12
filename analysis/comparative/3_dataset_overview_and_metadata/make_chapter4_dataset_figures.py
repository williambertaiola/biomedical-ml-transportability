"""Post-experiment analysis: make chapter4 dataset figures."""

#!/usr/bin/env python3
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
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


UCI_PATH = Path(
    _configured_path(
        "results_data_uci_heart_preprocessing_data_processed_uci_heart_analysis_ready.csv"
    )
)
ABIDE_PATH = Path(
    _configured_path(
        "results_data_abide2_preprocessing_data_processed_abide2_core_multisite_transportability_dataset_strict.csv"
    )
)
TCGA_PATH = Path(
    _configured_path(
        "results_data_tcga_brca_preprocessing_data_processed_tcga_brca_stage_transportability_dataset.csv"
    )
)

UCI_ORDER = ["cleveland", "hungarian", "long_beach_va", "switzerland"]
TCGA_MIN_SITE_SAMPLES = 30
TCGA_MIN_SITE_CLASS_COUNT = 2

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 500,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pretty_uci_site(site: str) -> str:
    mapping = {
        "cleveland": "Cleveland",
        "hungarian": "Hungarian",
        "long_beach_va": "Long Beach VA",
        "switzerland": "Switzerland",
    }
    return mapping.get(str(site), str(site))


def pretty_abide_site(site: str) -> str:
    label = str(site).replace("ABIDEII-", "").replace("_", " ")
    return " ".join(label.split())


def pretty_tcga_site(site: str) -> str:
    return str(site)


def map_abide_target(df: pd.DataFrame) -> pd.Series:
    if "target" in df.columns:
        target = pd.to_numeric(df["target"], errors="coerce")
        if target.isna().any():
            raise ValueError("ABIDE target column contains non-numeric values.")
        return target.astype(int)

    if "disease_level_code" in df.columns:
        codes = pd.to_numeric(df["disease_level_code"], errors="coerce")
        if codes.isna().any():
            raise ValueError("ABIDE disease_level_code contains non-numeric values.")
        return codes.astype(int).eq(1).astype(int)

    if "disease_level" in df.columns:
        labels = df["disease_level"].astype("string").str.strip().str.lower()
        return labels.eq("asd").astype(int)

    raise ValueError("Unable to derive the ABIDE binary target.")


def summarize_dataset(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    site_column: str,
    target_column: str,
    pretty_site,
    class_0_label: str,
    class_1_label: str,
    site_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working_df = df[[site_column, target_column]].copy()
    working_df[site_column] = working_df[site_column].astype(str)
    working_df[target_column] = pd.to_numeric(
        working_df[target_column], errors="raise"
    ).astype(int)

    sample_size = (
        working_df.groupby(site_column, sort=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .rename(columns={site_column: "site_id"})
    )

    class_counts = (
        working_df.groupby(site_column, sort=False)[target_column]
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis("site_id")
        .reset_index()
    )
    for required_class in (0, 1):
        if required_class not in class_counts.columns:
            class_counts[required_class] = 0

    class_counts = class_counts.rename(columns={0: "class_0_count", 1: "class_1_count"})
    class_counts["total_samples"] = (
        class_counts["class_0_count"] + class_counts["class_1_count"]
    )
    class_counts["class_1_prevalence"] = (
        class_counts["class_1_count"] / class_counts["total_samples"]
    ).round(4)

    if site_order:
        order_map = {site: index for index, site in enumerate(site_order)}
        sample_size["sort_order"] = (
            sample_size["site_id"].map(order_map).fillna(len(order_map)).astype(int)
        )
        class_counts["sort_order"] = (
            class_counts["site_id"].map(order_map).fillna(len(order_map)).astype(int)
        )
        sample_size = sample_size.sort_values(["sort_order", "site_id"]).drop(
            columns="sort_order"
        )
        class_counts = class_counts.sort_values(["sort_order", "site_id"]).drop(
            columns="sort_order"
        )
    else:
        sample_size = sample_size.sort_values("site_id").reset_index(drop=True)
        class_counts = class_counts.sort_values("site_id").reset_index(drop=True)

    sample_size["dataset"] = dataset_name
    sample_size["site_label"] = sample_size["site_id"].map(pretty_site)

    class_counts["dataset"] = dataset_name
    class_counts["site_label"] = class_counts["site_id"].map(pretty_site)
    class_counts["class_0_label"] = class_0_label
    class_counts["class_1_label"] = class_1_label

    sample_size = sample_size[["dataset", "site_id", "site_label", "n_samples"]]
    class_counts = class_counts[
        [
            "dataset",
            "site_id",
            "site_label",
            "class_0_count",
            "class_1_count",
            "class_0_label",
            "class_1_label",
            "total_samples",
            "class_1_prevalence",
        ]
    ]
    return sample_size, class_counts


def load_uci_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(UCI_PATH)
    return summarize_dataset(
        dataset_name="UCI Heart Disease",
        df=df,
        site_column="domain_id",
        target_column="target",
        pretty_site=pretty_uci_site,
        class_0_label="Absence of heart disease",
        class_1_label="Presence of heart disease",
        site_order=UCI_ORDER,
    )


def load_abide_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ABIDE_PATH)
    df = df.copy()
    df["target"] = map_abide_target(df)

    # Match the transportability and DASHI analyses by retaining only sites with both classes observed.
    site_class_counts = (
        df.groupby("site")["target"].value_counts().unstack(fill_value=0)
    )
    eligible_sites = site_class_counts.index[
        (site_class_counts.get(0, 0) > 0) & (site_class_counts.get(1, 0) > 0)
    ].tolist()
    filtered_df = df.loc[df["site"].isin(eligible_sites)].copy()

    return summarize_dataset(
        dataset_name="ABIDE II",
        df=filtered_df,
        site_column="site",
        target_column="target",
        pretty_site=pretty_abide_site,
        class_0_label="Control",
        class_1_label="ASD",
        site_order=sorted(eligible_sites),
    )


def load_tcga_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TCGA_PATH, usecols=["site", "target"])
    df = df.copy()
    df["site"] = df["site"].astype("string")
    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)

    site_counts = df.groupby("site")["target"].value_counts().unstack(fill_value=0)
    valid_sites = site_counts.index[
        (site_counts.sum(axis=1) >= TCGA_MIN_SITE_SAMPLES)
        & (site_counts.min(axis=1) >= TCGA_MIN_SITE_CLASS_COUNT)
    ].tolist()
    filtered_df = df.loc[df["site"].isin(valid_sites)].copy()

    return summarize_dataset(
        dataset_name="TCGA-BRCA",
        df=filtered_df,
        site_column="site",
        target_column="target",
        pretty_site=pretty_tcga_site,
        class_0_label="Stage I--II",
        class_1_label="Stage III--IV",
        site_order=sorted(valid_sites),
    )


def style_axis(ax: plt.Axes, *, ylabel: str | None = None) -> None:
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def draw_sample_size_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    title: str,
    color: str,
) -> None:
    ax.bar(
        summary_df["site_label"], summary_df["n_samples"], color=color, edgecolor="none"
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Site / domain")
    ax.tick_params(axis="x", rotation=45)
    style_axis(ax, ylabel="Number of subjects")


def draw_class_distribution_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    title: str,
    class_0_color: str,
    class_1_color: str,
) -> None:
    ax.bar(
        summary_df["site_label"],
        summary_df["class_0_count"],
        color=class_0_color,
        edgecolor="none",
        label="Class 0",
    )
    ax.bar(
        summary_df["site_label"],
        summary_df["class_1_count"],
        bottom=summary_df["class_0_count"],
        color=class_1_color,
        edgecolor="none",
        label="Class 1",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Site / domain")
    ax.tick_params(axis="x", rotation=45)
    style_axis(ax, ylabel="Number of subjects")


def save_multi_panel_sample_size(
    sample_tables: list[pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1, 3, figsize=(18.5, 6.0), constrained_layout=True, sharey=True
    )
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    titles = ["A) UCI Heart Disease", "B) ABIDE II", "C) TCGA-BRCA"]

    for ax, summary_df, color, title in zip(
        axes, sample_tables, colors, titles, strict=True
    ):
        draw_sample_size_panel(ax, summary_df, title=title, color=color)

    fig.suptitle(
        "Sample size by site/domain across the study datasets",
        y=1.04,
        fontsize=20,
        fontweight="bold",
    )

    output_paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        output_path = output_dir / f"fig_dataset_sample_size_by_site{suffix}"
        fig.savefig(
            output_path,
            dpi=500 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def save_multi_panel_class_distribution(
    class_tables: list[pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1, 3, figsize=(18.5, 6.0), constrained_layout=True, sharey=True
    )
    titles = ["A) UCI Heart Disease", "B) ABIDE II", "C) TCGA-BRCA"]
    class_0_color = "#A6CEE3"
    class_1_color = "#FB9A99"

    for ax, summary_df, title in zip(axes, class_tables, titles, strict=True):
        draw_class_distribution_panel(
            ax,
            summary_df,
            title=title,
            class_0_color=class_0_color,
            class_1_color=class_1_color,
        )

    legend_handles = [
        Patch(facecolor=class_0_color, edgecolor="none", label="Class 0"),
        Patch(facecolor=class_1_color, edgecolor="none", label="Class 1"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Class distribution by site/domain across the study datasets",
        y=1.14,
        fontsize=20,
        fontweight="bold",
    )

    output_paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        output_path = output_dir / f"fig_dataset_class_distribution_by_site{suffix}"
        fig.savefig(
            output_path,
            dpi=500 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Chapter 4 descriptive dataset figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "chapter4_dataset_figures",
        help="Directory where figures and CSV summaries will be written.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir.resolve())

    uci_sample, uci_class = load_uci_summaries()
    abide_sample, abide_class = load_abide_summaries()
    tcga_sample, tcga_class = load_tcga_summaries()

    sample_summary = pd.concat(
        [uci_sample, abide_sample, tcga_sample], ignore_index=True
    )
    class_summary = pd.concat([uci_class, abide_class, tcga_class], ignore_index=True)

    sample_summary.to_csv(output_dir / "dataset_sample_size_by_site.csv", index=False)
    class_summary.to_csv(
        output_dir / "dataset_class_distribution_by_site.csv", index=False
    )

    save_multi_panel_sample_size([uci_sample, abide_sample, tcga_sample], output_dir)
    save_multi_panel_class_distribution(
        [uci_class, abide_class, tcga_class], output_dir
    )

    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
