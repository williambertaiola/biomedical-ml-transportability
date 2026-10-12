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
        Path(_configured_path("results_root")) / "Data" / "ABIDE2_preprocessing"
    )
    dataset_path = (
        output_root
        / "data"
        / "processed"
        / "abide2_core_multisite_transportability_dataset.csv"
    )
    report_dir = output_root / "reports"
    figure_dir = report_dir / "figures_standard"
    table_dir = report_dir / "tables_standard"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    feature_cols = [
        column
        for column in df.columns
        if column not in {"subject_id", "site", "disease_level", "disease_level_code"}
    ]

    site_counts = (
        df.groupby("site", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
        .reset_index(drop=True)
    )
    site_counts.to_csv(table_dir / "site_sample_counts.csv", index=False)
    plot_sorted_bar(
        site_counts,
        x="site",
        y="sample_count",
        title="ABIDE II samples per site",
        xlabel="Site",
        ylabel="Samples",
        output_path=figure_dir / "01_site_sample_counts.png",
        rotation=60,
    )

    class_balance = (
        pd.crosstab(df["site"], df["disease_level"])
        .reindex(columns=["Control", "ASD"], fill_value=0)
        .sort_index()
    )
    class_balance.to_csv(table_dir / "site_class_balance.csv")
    plot_matrix_heatmap(
        class_balance,
        title="ABIDE II class balance by site",
        xlabel="Class",
        ylabel="Site",
        output_path=figure_dir / "02_site_class_balance_heatmap.png",
        vmin=float(class_balance.to_numpy().min()),
        vmax=float(class_balance.to_numpy().max()),
        fmt=".0f",
    )

    completeness = (
        df.groupby("site", dropna=False)[feature_cols]
        .agg(lambda series: float(series.notna().mean()))
        .sort_index()
    )
    completeness.to_csv(table_dir / "site_feature_completeness.csv")
    plot_matrix_heatmap(
        completeness,
        title="ABIDE II feature completeness by site",
        xlabel="Feature",
        ylabel="Site",
        output_path=figure_dir / "03_site_metadata_completeness_heatmap.png",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
    )

    note_lines = [
        "ABIDE II preprocessing note",
        "",
        "What was done:",
        "- Built the composite subject-level table from ABIDE II phenotypic sources and QAP files.",
        "- Created the rich multisite transportability dataset used as the main preprocessing output.",
        "- Preserved site and diagnosis labels explicitly for later leave-one-site-out analyses.",
        "- Avoided global imputation, encoding, and scaling at preprocessing stage.",
        "",
        "Why this matters theoretically:",
        "- In a transportability thesis, preprocessing must separate dataset assembly from train-only transformations.",
        "- Site is preserved as metadata because it defines the external validation split.",
        "- Variables too close to the target should be discussed explicitly to avoid circular interpretation.",
        "- Any later imputer, scaler, or encoder must be fit only inside the training folds.",
        "",
        f"Final rich dataset path: {dataset_path}",
        f"Number of retained subjects: {len(df)}",
        f"Number of retained sites: {df['site'].nunique()}",
        f"Feature columns kept in the rich dataset: {len(feature_cols)}",
    ]
    (report_dir / "ABIDE2_preprocessing_note.txt").write_text(
        "\n".join(note_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
