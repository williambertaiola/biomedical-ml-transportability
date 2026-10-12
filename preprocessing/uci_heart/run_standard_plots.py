"""Preprocessing: run standard plots."""

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


import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_ROOT = PROJECT_ROOT.parent / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from thesis_plot_style import plot_matrix_heatmap, plot_sorted_bar  # noqa: E402


def main() -> None:
    output_root = (
        Path(_configured_path("results_root")) / "Data" / "UCI_Heart_preprocessing"
    )
    dataset_path = output_root / "data" / "processed" / "uci_heart_analysis_ready.csv"
    report_dir = output_root / "reports"
    figure_dir = report_dir / "figures_standard"
    table_dir = report_dir / "tables_standard"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    feature_cols = [
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
    ]

    domain_counts = (
        df.groupby("domain_id", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
        .reset_index(drop=True)
    )
    domain_counts.to_csv(table_dir / "domain_sample_counts.csv", index=False)
    plot_sorted_bar(
        domain_counts,
        x="domain_id",
        y="sample_count",
        title="UCI Heart Disease samples per domain",
        xlabel="Domain",
        ylabel="Samples",
        output_path=figure_dir / "01_domain_sample_counts.png",
        rotation=0,
    )

    class_balance = (
        pd.crosstab(df["domain_id"], df["target"])
        .rename(columns={0: "Negative", 1: "Positive"})
        .reindex(columns=["Negative", "Positive"], fill_value=0)
        .sort_index()
    )
    class_balance.to_csv(table_dir / "domain_class_balance.csv")
    plot_matrix_heatmap(
        class_balance,
        title="UCI Heart Disease class balance by domain",
        xlabel="Class",
        ylabel="Domain",
        output_path=figure_dir / "02_domain_class_balance_heatmap.png",
        vmin=float(class_balance.to_numpy().min()),
        vmax=float(class_balance.to_numpy().max()),
        fmt=".0f",
    )

    completeness = (
        df.groupby("domain_id", dropna=False)[feature_cols]
        .agg(lambda series: float(series.notna().mean()))
        .sort_index()
    )
    completeness.to_csv(table_dir / "domain_feature_completeness.csv")
    plot_matrix_heatmap(
        completeness,
        title="UCI Heart Disease feature completeness by domain",
        xlabel="Feature",
        ylabel="Domain",
        output_path=figure_dir / "03_domain_feature_completeness_heatmap.png",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
    )

    note_lines = [
        "UCI Heart Disease preprocessing note",
        "",
        "What was done:",
        "- Unified the four original centers into one analysis-ready multi-domain table.",
        "- Preserved domain_id explicitly for later leave-one-domain-out validation.",
        "- Standardized missing-value tokens and kept preprocessing leakage-safe.",
        "- Built a binary target from the original disease severity labels.",
        "",
        "Why this matters theoretically:",
        "- Transportability requires domain labels to remain explicit rather than being dropped too early.",
        "- Dataset-level cleaning is acceptable only when it does not estimate model parameters from the full cohort.",
        "- Imputation, scaling, encoding, and variance filtering must remain train-only steps in later modeling.",
        "- A simple benchmark dataset is useful for method debugging, but it should not dominate thesis conclusions over harder biomedical settings.",
        "",
        f"Analysis-ready dataset path: {dataset_path}",
        f"Number of retained samples: {len(df)}",
        f"Number of domains: {df['domain_id'].nunique()}",
        f"Predictive feature columns: {len(feature_cols)}",
    ]
    (report_dir / "UCI_Heart_preprocessing_note.txt").write_text(
        "\n".join(note_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
