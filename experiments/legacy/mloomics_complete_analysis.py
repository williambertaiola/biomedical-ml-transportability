#!/usr/bin/env python3
"""
Comprehensive structural/data-quality analysis for MLOmics Main_Dataset.

Focus:
- completeness and file-level integrity
- multi-omics modality coverage
- per-cohort alignment consistency (sample-count consistency across omics)
- descriptive shift of dataset scale across tasks/cohorts/variants
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOmics complete analysis")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to MLOmics_database root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./mloomics_analysis_output_full"),
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=300,
        help="Rows to sample for preview missingness/ID checks per file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_plot_style() -> None:
    if HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")


def infer_modality(name: str) -> str:
    n = name.lower()
    if "survival" in n:
        return "survival"
    if "subtype" in n or "label" in n:
        return "label"
    if "mirna" in n:
        return "miRNA"
    if "mrna" in n:
        return "mRNA"
    if "methy" in n or "methyl" in n:
        return "Methy"
    if "cnv" in n:
        return "CNV"
    if "protein" in n:
        return "Protein"
    return "Other"


def count_rows_wc(path: Path) -> Optional[int]:
    try:
        out = subprocess.run(
            ["wc", "-l", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        token = out.stdout.strip().split()[0]
        total_lines = int(token)
        return max(total_lines - 1, 0)
    except Exception:
        return None


def count_cols_from_header(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        return len(header)
    except Exception:
        return None


def sample_profile(path: Path, sample_rows: int) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "sample_missing_pct": np.nan,
        "sample_first_col_unique_ratio": np.nan,
        "sample_first_col_missing_pct": np.nan,
        "sample_read_ok": False,
    }
    try:
        df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
        if df.shape[0] == 0:
            profile["sample_read_ok"] = True
            return profile
        miss_pct = float(df.isna().mean().mean() * 100.0)
        first = df.columns[0]
        fc = df[first]
        profile["sample_missing_pct"] = miss_pct
        profile["sample_first_col_unique_ratio"] = float(
            fc.nunique(dropna=True) / max(len(fc), 1)
        )
        profile["sample_first_col_missing_pct"] = float(fc.isna().mean() * 100.0)
        profile["sample_read_ok"] = True
        return profile
    except Exception:
        return profile


def catalog_files(main_dataset_dir: Path, sample_rows: int) -> pd.DataFrame:
    csv_files = sorted(main_dataset_dir.rglob("*.csv"))
    rows: List[Dict[str, object]] = []
    for p in csv_files:
        rel = p.relative_to(main_dataset_dir)
        parts = rel.parts
        task = parts[0] if len(parts) > 0 else "UNKNOWN"
        cohort = parts[1] if len(parts) > 1 else "UNKNOWN"
        variant = parts[2] if len(parts) > 2 else "UNKNOWN"
        modality = infer_modality(p.name)

        n_rows = count_rows_wc(p)
        n_cols = count_cols_from_header(p)
        size_mb = p.stat().st_size / 1024 / 1024
        prof = sample_profile(p, sample_rows=sample_rows)

        rows.append(
            {
                "path": str(p),
                "relative_path": str(rel),
                "task": task,
                "cohort": cohort,
                "variant": variant,
                "file_name": p.name,
                "modality": modality,
                "size_mb": size_mb,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_samples_est": (n_cols - 1) if (n_cols and n_cols > 0) else np.nan,
                **prof,
            }
        )
    return pd.DataFrame(rows)


def build_unit_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = catalog.groupby(["task", "cohort", "variant"], dropna=False)
    for (task, cohort, variant), g in grouped:
        omics = g[~g["modality"].isin(["survival", "label", "Other"])]
        omics_modalities = sorted(omics["modality"].unique().tolist())
        omics_rows = omics["n_rows"].dropna().astype(int).tolist()
        omics_samples = omics["n_samples_est"].dropna().astype(int).tolist()
        n_modalities = len(omics_modalities)
        if len(omics_rows) > 0:
            row_min = int(np.min(omics_rows))
            row_max = int(np.max(omics_rows))
            row_med = float(np.median(omics_rows))
            row_range = int(row_max - row_min)
        else:
            row_min, row_max, row_med, row_range = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        if len(omics_samples) > 0:
            sample_min = int(np.min(omics_samples))
            sample_max = int(np.max(omics_samples))
            sample_med = float(np.median(omics_samples))
            sample_consistency = int(sample_min == sample_max)
            sample_range = int(sample_max - sample_min)
        else:
            sample_min, sample_max, sample_med, sample_consistency, sample_range = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        rows.append(
            {
                "task": task,
                "cohort": cohort,
                "variant": variant,
                "n_files": int(len(g)),
                "n_modalities_omics": n_modalities,
                "omics_modalities": ",".join(omics_modalities),
                "has_survival": int((g["modality"] == "survival").any()),
                "has_label": int((g["modality"] == "label").any()),
                "omics_feature_rows_min": row_min,
                "omics_feature_rows_max": row_max,
                "omics_feature_rows_median": row_med,
                "omics_rows_range": row_range,
                "omics_samples_min": sample_min,
                "omics_samples_max": sample_max,
                "omics_samples_median": sample_med,
                "omics_samples_range": sample_range,
                "omics_sample_consistency_flag": sample_consistency,
                "total_size_mb": float(g["size_mb"].sum()),
                "mean_sample_missing_pct": float(
                    g["sample_missing_pct"].mean(skipna=True)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "cohort", "variant"])


def build_task_summary(
    catalog: pd.DataFrame, unit_summary: pd.DataFrame
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()
    c = (
        catalog.groupby("task", dropna=False)
        .agg(
            n_files=("path", "count"),
            n_cohorts=("cohort", "nunique"),
            total_size_mb=("size_mb", "sum"),
            median_rows=("n_rows", "median"),
            median_cols=("n_cols", "median"),
        )
        .reset_index()
    )
    if not unit_summary.empty:
        u = (
            unit_summary.groupby("task", dropna=False)
            .agg(
                n_units=("cohort", "count"),
                mean_modalities_omics=("n_modalities_omics", "mean"),
                consistency_rate=("omics_sample_consistency_flag", "mean"),
            )
            .reset_index()
        )
        c = c.merge(u, on="task", how="left")
    return c.sort_values("n_files", ascending=False)


def create_figures(
    catalog: pd.DataFrame, unit_summary: pd.DataFrame, output_fig_dir: Path
) -> List[str]:
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    # 1) Files by task
    task_counts = (
        catalog["task"].value_counts().rename_axis("task").reset_index(name="n_files")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(task_counts["task"], task_counts["n_files"], color="#457b9d")
    ax.set_title("Number of CSV files by task")
    ax.set_xlabel("Task")
    ax.set_ylabel("N files")
    fig.tight_layout()
    p = output_fig_dir / "01_files_by_task.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 2) Modality counts by task
    modality_task = pd.crosstab(catalog["task"], catalog["modality"])
    fig, ax = plt.subplots(figsize=(12, 7))
    modality_task.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Modality distribution by task")
    ax.set_xlabel("Task")
    ax.set_ylabel("N files")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    p = output_fig_dir / "02_modality_by_task.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 3) Row counts by modality (log scale)
    tmp = catalog.dropna(subset=["n_rows"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAVE_SEABORN:
        sns.boxplot(data=tmp, x="modality", y="n_rows", ax=ax)
    else:
        order = tmp["modality"].value_counts().index.tolist()
        data = [tmp.loc[tmp["modality"] == m, "n_rows"] for m in order]
        ax.boxplot(data, tick_labels=order)
    ax.set_yscale("log")
    ax.set_title("Row-count distribution by modality (log scale)")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = output_fig_dir / "03_rows_by_modality.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 4) Feature counts by modality (log scale)
    tmp = catalog.dropna(subset=["n_samples_est"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAVE_SEABORN:
        sns.boxplot(data=tmp, x="modality", y="n_samples_est", ax=ax)
    else:
        order = tmp["modality"].value_counts().index.tolist()
        data = [tmp.loc[tmp["modality"] == m, "n_samples_est"] for m in order]
        ax.boxplot(data, tick_labels=order)
    ax.set_yscale("log")
    ax.set_title("Estimated sample-count distribution by modality (log scale)")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Estimated #samples")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = output_fig_dir / "04_samples_by_modality.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 5) Alignment consistency by variant
    if not unit_summary.empty:
        cons = (
            unit_summary.groupby("variant", dropna=False)[
                "omics_sample_consistency_flag"
            ]
            .mean()
            .reset_index(name="consistency_rate")
            .sort_values("consistency_rate", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(cons["variant"], cons["consistency_rate"], color="#2a9d8f")
        ax.set_ylim(0, 1.05)
        ax.set_title("Sample-count consistency rate across omics (by variant)")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Consistency rate")
        fig.tight_layout()
        p = output_fig_dir / "05_alignment_consistency_by_variant.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        created.append(p.name)

    # 6) Top cohorts by total size
    cohort_size = (
        catalog.groupby(["task", "cohort"], dropna=False)["size_mb"]
        .sum()
        .reset_index(name="total_size_mb")
        .sort_values("total_size_mb", ascending=False)
        .head(20)
    )
    cohort_size["task_cohort"] = cohort_size["task"] + ":" + cohort_size["cohort"]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(cohort_size["task_cohort"], cohort_size["total_size_mb"], color="#8d99ae")
    ax.set_title("Top 20 task/cohort groups by total file size")
    ax.set_xlabel("Total size (MB)")
    ax.invert_yaxis()
    fig.tight_layout()
    p = output_fig_dir / "06_top_cohorts_by_size.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 7) Sample missingness preview by modality
    tmp = catalog.dropna(subset=["sample_missing_pct"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAVE_SEABORN:
        sns.boxplot(data=tmp, x="modality", y="sample_missing_pct", ax=ax)
    else:
        order = tmp["modality"].value_counts().index.tolist()
        data = [tmp.loc[tmp["modality"] == m, "sample_missing_pct"] for m in order]
        ax.boxplot(data, tick_labels=order)
    ax.set_title("Preview missingness (% from sampled rows) by modality")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Missing % (sample preview)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = output_fig_dir / "07_preview_missingness_by_modality.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 8) Largest files
    top_large = catalog.sort_values("size_mb", ascending=False).head(20).copy()
    top_large["short_name"] = top_large["cohort"] + "/" + top_large["file_name"]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(top_large["short_name"], top_large["size_mb"], color="#e76f51")
    ax.set_title("Top 20 largest files")
    ax.set_xlabel("Size (MB)")
    ax.invert_yaxis()
    fig.tight_layout()
    p = output_fig_dir / "08_largest_files.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    created.append(p.name)

    # 9) Units per task and variant
    if not unit_summary.empty:
        tv = pd.crosstab(unit_summary["task"], unit_summary["variant"])
        fig, ax = plt.subplots(figsize=(12, 7))
        tv.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title("Dataset units by task and variant")
        ax.set_xlabel("Task")
        ax.set_ylabel("N units")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        p = output_fig_dir / "09_units_by_task_variant.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        created.append(p.name)

    # 10) Omics modality presence heatmap (Top-30 units by size)
    if not unit_summary.empty:
        u = unit_summary.copy()
        u["unit"] = u["task"] + "|" + u["cohort"] + "|" + u["variant"]
        top_units = u.sort_values("total_size_mb", ascending=False).head(30)
        records = []
        for _, row in top_units.iterrows():
            mods = (
                set(str(row["omics_modalities"]).split(","))
                if pd.notna(row["omics_modalities"])
                else set()
            )
            for m in ["mRNA", "miRNA", "Methy", "CNV", "Protein"]:
                records.append(
                    {"unit": row["unit"], "modality": m, "present": int(m in mods)}
                )
        mat = pd.DataFrame(records).pivot(
            index="unit", columns="modality", values="present"
        )
        fig, ax = plt.subplots(figsize=(10, 11))
        if HAVE_SEABORN:
            sns.heatmap(
                mat, cmap="YlGnBu", cbar_kws={"label": "Modality present"}, ax=ax
            )
        else:
            im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="Modality present")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=45)
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_title("Omics modality presence (top 30 units by size)")
        ax.set_xlabel("Modality")
        ax.set_ylabel("Task|Cohort|Variant")
        fig.tight_layout()
        p = output_fig_dir / "10_modality_presence_heatmap.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        created.append(p.name)

    return created


def write_report(
    data_dir: Path,
    output_dir: Path,
    catalog: pd.DataFrame,
    unit_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    figure_names: List[str],
) -> None:
    report = output_dir / "analysis_report.md"
    n_files = len(catalog)
    n_tasks = catalog["task"].nunique() if not catalog.empty else 0
    n_cohorts = catalog["cohort"].nunique() if not catalog.empty else 0
    size_total_gb = catalog["size_mb"].sum() / 1024 if not catalog.empty else 0.0
    variants = (
        sorted(catalog["variant"].dropna().unique().tolist())
        if not catalog.empty
        else []
    )

    cons_rate = (
        float(unit_summary["omics_sample_consistency_flag"].mean())
        if not unit_summary.empty
        else np.nan
    )
    top_missing = (
        catalog.sort_values("sample_missing_pct", ascending=False)[
            ["task", "cohort", "variant", "file_name", "sample_missing_pct"]
        ]
        .head(10)
        .round({"sample_missing_pct": 2})
    )
    top_large = (
        catalog.sort_values("size_mb", ascending=False)[
            ["task", "cohort", "variant", "file_name", "size_mb", "n_rows", "n_cols"]
        ]
        .head(10)
        .round({"size_mb": 2})
    )

    lines = [
        "# MLOmics Main_Dataset - Analysis Report",
        "",
        "## 1) Scope",
        f"- Input analyzed: `{data_dir}`",
        f"- Total CSV files: **{n_files}**",
        f"- Tasks: **{n_tasks}**",
        f"- Cohorts (folder-level): **{n_cohorts}**",
        f"- Total data size (CSV): **{size_total_gb:.2f} GB**",
        f"- Variants detected: **{', '.join(variants)}**",
        "",
        "## 2) Key quality findings",
        f"- Omics sample-count consistency rate across dataset units: **{cons_rate:.3f}**",
        "- Detailed tables:",
        "- `file_catalog.csv`",
        "- `dataset_unit_summary.csv`",
        "- `task_summary.csv`",
        "- `modality_summary.csv`",
        "- `top_largest_files.csv`",
        "",
        "## 3) Preview of potential issues",
        "- Top files by sampled missingness and largest files are in CSV outputs.",
        "",
        "## 4) Figures",
    ]
    for fname in figure_names:
        lines.append(f"- ![fig]({('figures/' + fname)})")

    report.write_text("\n".join(lines), encoding="utf-8")

    top_missing.to_csv(output_dir / "top_missingness_preview.csv", index=False)
    top_large.to_csv(output_dir / "top_largest_files.csv", index=False)


def run_analysis(args: argparse.Namespace) -> None:
    setup_plot_style()

    root = args.input_dir
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")

    main_dataset = root / "MLOmics" / "Main_Dataset"
    if not main_dataset.exists():
        raise FileNotFoundError(
            f"Main_Dataset not found at expected path: {main_dataset}. "
            "Check input path."
        )

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Main_Dataset: {main_dataset}")
    print(f"[INFO] Output dir: {output_dir}")

    catalog = catalog_files(main_dataset, sample_rows=args.sample_rows)
    if catalog.empty:
        raise RuntimeError("No CSV files found under Main_Dataset.")

    unit_summary = build_unit_summary(catalog)
    task_summary = build_task_summary(catalog, unit_summary)
    modality_summary = (
        catalog.groupby("modality", dropna=False)
        .agg(
            n_files=("path", "count"),
            total_size_mb=("size_mb", "sum"),
            median_rows=("n_rows", "median"),
            median_cols=("n_cols", "median"),
            mean_sample_missing_pct=("sample_missing_pct", "mean"),
        )
        .reset_index()
        .sort_values("n_files", ascending=False)
    )

    catalog.to_csv(output_dir / "file_catalog.csv", index=False)
    unit_summary.to_csv(output_dir / "dataset_unit_summary.csv", index=False)
    task_summary.to_csv(output_dir / "task_summary.csv", index=False)
    modality_summary.to_csv(output_dir / "modality_summary.csv", index=False)

    fig_names = create_figures(catalog, unit_summary, figures_dir)
    write_report(
        main_dataset, output_dir, catalog, unit_summary, task_summary, fig_names
    )

    print("[INFO] Analysis completed.")
    print(f"[INFO] Report: {output_dir / 'analysis_report.md'}")
    print(f"[INFO] Figures: {figures_dir}")


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
