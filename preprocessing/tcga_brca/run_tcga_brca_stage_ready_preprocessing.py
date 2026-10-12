"""Preprocessing: run tcga brca stage ready preprocessing."""

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
from thesis_tools.paths import resolve_config_paths


import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SHARED_ROOT = PROJECT_ROOT.parent / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from thesis_plot_style import plot_matrix_heatmap, plot_sorted_bar  # noqa: E402
from src.preprocessing.stage_ready_dataset import build_stage_ready_artifacts  # noqa: E402
from src.utils.io import init_logger, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stage-ready TCGA-BRCA preprocessing dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tcga_brca_stage_ready_config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return resolve_config_paths(json.load(handle))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_root = Path(config["output_root"])
    logger = init_logger(
        output_root / "logs" / "tcga_brca_stage_ready_preprocessing.log"
    )
    logger.info("Starting TCGA-BRCA stage-ready preprocessing")

    artifacts = build_stage_ready_artifacts(
        expression_matrix=Path(config["expression_matrix"]),
        clinical_dir=Path(config["clinical_dir"]),
        output_root=output_root,
        min_per_class_per_site=int(config["min_per_class_per_site"]),
        logger=logger,
    )

    figure_dir = output_root / "reports" / "figures_standard"
    table_dir = output_root / "reports" / "tables_standard"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    plot_sorted_bar(
        artifacts.site_summary,
        x="site",
        y="n_samples",
        title="TCGA-BRCA stage-ready samples per site",
        xlabel="Site",
        ylabel="Samples",
        output_path=figure_dir / "01_site_sample_counts.png",
        rotation=45,
    )

    class_balance = (
        pd.crosstab(
            artifacts.transportability_df["site"],
            artifacts.transportability_df["target_label"],
        )
        .reindex(columns=["Stage I-II", "Stage III-IV"], fill_value=0)
        .sort_index()
    )
    class_balance.to_csv(table_dir / "tcga_brca_stage_class_balance.csv")
    plot_matrix_heatmap(
        class_balance,
        title="TCGA-BRCA stage balance by site",
        xlabel="Stage group",
        ylabel="Site",
        output_path=figure_dir / "02_site_stage_balance_heatmap.png",
        vmin=float(class_balance.to_numpy().min()),
        vmax=float(class_balance.to_numpy().max()),
        fmt=".0f",
    )

    plot_matrix_heatmap(
        artifacts.metadata_completeness,
        title="TCGA-BRCA metadata completeness by site",
        xlabel="Metadata field",
        ylabel="Site",
        output_path=figure_dir / "03_site_metadata_completeness_heatmap.png",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
    )

    write_json(
        artifacts.summary_payload,
        output_root / "data" / "processed" / "tcga_brca_stage_ready_summary.json",
    )

    note_lines = [
        "TCGA-BRCA stage-ready preprocessing note",
        "",
        "What was done:",
        "- Standardized the expression matrix to one row per sample and one column per gene.",
        "- Parsed TCGA barcodes to recover subject-level and site-level metadata.",
        "- Restricted the stage-ready cohort to primary tumors only.",
        "- Deduplicated subjects with multiple primary-tumor samples using a deterministic QC-aware rule.",
        "- Merged clinical AJCC pathologic stage information and collapsed it to Stage I-II vs Stage III-IV.",
        "- Built a transportability-ready dataset by keeping only sites with both classes represented at the chosen minimum count threshold.",
        "",
        "Why this matters theoretically:",
        "- A transportability benchmark is only meaningful if the prediction task is clinically non-trivial and defined consistently across sites.",
        "- Deduplication must occur before modeling to avoid multiple correlated rows from the same subject.",
        "- Stage labels are metadata-derived and must be merged before any train-only transformation, but later feature selection and scaling must still remain fold-specific.",
        "- Preserving site explicitly is essential because out-of-domain validity is defined at the site level in this thesis.",
        "",
        f"Stage-annotated dataset path: {output_root / 'data' / 'processed' / 'tcga_brca_stage_annotated_primary_tumors.csv'}",
        f"Transportability-ready dataset path: {output_root / 'data' / 'processed' / 'tcga_brca_stage_transportability_dataset.csv'}",
        f"Subjects in the stage-annotated primary-tumor cohort: {artifacts.summary_payload['n_primary_subject_level_rows']}",
        f"Samples with usable stage labels: {artifacts.summary_payload['n_samples_with_stage_labels']}",
        f"Valid sites after class-balance filtering: {artifacts.summary_payload['n_valid_sites_after_min_class_filter']}",
    ]
    (
        output_root / "reports" / "TCGA_BRCA_stage_ready_preprocessing_note.txt"
    ).write_text(
        "\n".join(note_lines) + "\n",
        encoding="utf-8",
    )
    logger.info("Finished TCGA-BRCA stage-ready preprocessing")


if __name__ == "__main__":
    main()
