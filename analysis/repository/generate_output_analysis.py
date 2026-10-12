"""Post-experiment analysis: generate output analysis."""

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


import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(_configured_path("results_root"))
OUTPUT_DIR = ROOT / "Output"
EXCLUDE_TOP_LEVEL = {"Output"}
DATASET_NAMES = ["UCI", "TCGA", "ABIDE"]
CANONICAL_SECTIONS = [
    "Methods",
    "Quality_report",
    "Final_table",
    "Entropy",
    "Sample_size",
    "Site_distance",
    "Heatmap_cluster",
    "Pattern_discovery",
    "Metadata_discussion",
    "Dashi",
]


def relpath(path: Path) -> str:
    return str(path.relative_to(ROOT))


def build_tree_lines(root: Path, max_depth: int = 3) -> list[str]:
    lines = [root.name]

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(
            [p for p in path.iterdir() if p.name != ".DS_Store"],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name)
            if entry.is_dir():
                extension = "    " if idx == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)

    _walk(root, "", 1)
    return lines


def collect_file_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if OUTPUT_DIR in path.parents:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        relative = relpath(path)
        top_level = relative.split("/", 1)[0]
        rows.append(
            {
                "relative_path": relative,
                "top_level_area": top_level,
                "parent_directory": relpath(path.parent),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["top_level_area", "parent_directory", "filename"])
        .reset_index(drop=True)
    )


def collect_directory_summary(root: Path, manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    directories = [
        p
        for p in root.rglob("*")
        if p.is_dir() and OUTPUT_DIR not in p.parents and p != OUTPUT_DIR
    ]
    for directory in sorted(directories):
        relative = relpath(directory)
        file_subset = manifest_df[manifest_df["parent_directory"] == relative]
        ext_counts = Counter(file_subset["extension"].tolist())
        rows.append(
            {
                "directory": relative,
                "n_files_direct": int(len(file_subset)),
                "n_csv_direct": int(ext_counts.get(".csv", 0)),
                "n_xlsx_direct": int(ext_counts.get(".xlsx", 0)),
                "n_png_direct": int(ext_counts.get(".png", 0)),
                "n_pdf_direct": int(ext_counts.get(".pdf", 0)),
                "n_md_direct": int(ext_counts.get(".md", 0)),
                "n_txt_direct": int(ext_counts.get(".txt", 0)),
                "n_py_direct": int(ext_counts.get(".py", 0)),
            }
        )
    return pd.DataFrame(rows)


def read_pattern_master(dataset_root: Path) -> pd.DataFrame | None:
    path = dataset_root / "Pattern_discovery" / "pattern_discovery_master_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def read_metadata_shift(dataset_root: Path) -> pd.DataFrame | None:
    path = dataset_root / "Metadata_discussion" / "metadata_shift_by_target_site.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def dataset_component_status(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        dataset_root = root / dataset
        for section in CANONICAL_SECTIONS:
            section_path = dataset_root / section
            exists = section_path.exists()
            n_files = 0
            if exists:
                n_files = sum(1 for p in section_path.rglob("*") if p.is_file())
            rows.append(
                {
                    "dataset": dataset,
                    "section": section,
                    "exists": bool(exists),
                    "n_files_recursive": int(n_files),
                    "path": str(section_path),
                }
            )
    return pd.DataFrame(rows)


def dataset_master_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        dataset_root = root / dataset
        master_df = read_pattern_master(dataset_root)
        shift_df = read_metadata_shift(dataset_root)
        if master_df is None or master_df.empty:
            continue
        sites = sorted(master_df["target_site"].dropna().unique().tolist())
        models = sorted(master_df["model"].dropna().unique().tolist())
        methods = sorted(master_df["method"].dropna().unique().tolist())
        metrics = sorted(master_df["metric"].dropna().unique().tolist())
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": str(master_df["dataset"].iloc[0]),
                "n_rows": int(len(master_df)),
                "n_target_sites": int(len(sites)),
                "target_sites": ", ".join(sites),
                "n_models": int(len(models)),
                "models": ", ".join(models),
                "n_methods": int(len(methods)),
                "methods": ", ".join(methods),
                "n_metrics": int(len(metrics)),
                "metrics": ", ".join(metrics),
                "mean_ood_overall": float(
                    pd.to_numeric(
                        master_df["mean_ood_performance"], errors="coerce"
                    ).mean()
                ),
                "mean_gap_overall": float(
                    pd.to_numeric(master_df["id_ood_gap"], errors="coerce").mean()
                ),
                "mean_ece_overall": float(
                    pd.to_numeric(
                        master_df.get("ece", master_df.get("mean_ece")), errors="coerce"
                    ).mean()
                ),
                "mean_normalized_entropy": float(
                    pd.to_numeric(master_df["normalized_entropy"], errors="coerce")
                    .drop_duplicates()
                    .mean()
                ),
                "mean_site_distance_euclidean": float(
                    pd.to_numeric(
                        master_df["euclidean_centroid_distance"], errors="coerce"
                    )
                    .drop_duplicates()
                    .mean()
                ),
                "mean_metadata_shift": float(
                    pd.to_numeric(
                        shift_df["overall_metadata_shift_score"], errors="coerce"
                    ).mean()
                )
                if shift_df is not None and not shift_df.empty
                else float("nan"),
            }
        )
        method_group = (
            master_df.groupby("method", as_index=False)
            .agg(
                mean_ood_performance=("mean_ood_performance", "mean"),
                mean_std_ood_performance=("std_ood_performance", "mean"),
                mean_id_ood_gap=("id_ood_gap", "mean"),
                mean_ece=("ece", "mean")
                if "ece" in master_df.columns
                else ("mean_ece", "mean"),
            )
            .sort_values("method")
        )
        for _, row in method_group.iterrows():
            method_rows.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "mean_ood_performance": float(row["mean_ood_performance"]),
                    "mean_std_ood_performance": float(row["mean_std_ood_performance"]),
                    "mean_id_ood_gap": float(row["mean_id_ood_gap"]),
                    "mean_ece": float(row["mean_ece"]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(method_rows)


def key_reports_index(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mappings = [
        ("MASTER_README", root / "MASTER_README.md", "master_index"),
        (
            "Cross-dataset summary",
            root / "Final_summary" / "cross_dataset_summary.md",
            "cross_dataset_summary",
        ),
        (
            "Method robustness final summary",
            root
            / "Final_discussion_best_method"
            / "method_robustness_final_summary.txt",
            "final_discussion",
        ),
    ]
    for dataset in DATASET_NAMES:
        dataset_root = root / dataset
        mappings.extend(
            [
                (
                    f"{dataset} quality report",
                    dataset_root / "Quality_report",
                    "quality_report_folder",
                ),
                (
                    f"{dataset} final table",
                    dataset_root / "Final_table",
                    "final_table_folder",
                ),
                (f"{dataset} entropy", dataset_root / "Entropy", "entropy_folder"),
                (
                    f"{dataset} site distance",
                    dataset_root / "Site_distance",
                    "site_distance_folder",
                ),
                (
                    f"{dataset} metadata discussion",
                    dataset_root / "Metadata_discussion",
                    "metadata_folder",
                ),
                (
                    f"{dataset} pattern discovery",
                    dataset_root / "Pattern_discovery",
                    "pattern_folder",
                ),
            ]
        )
    for label, path, category in mappings:
        rows.append(
            {
                "label": label,
                "category": category,
                "exists": path.exists(),
                "path": str(path),
                "path_relative_to_root": relpath(path)
                if path.exists()
                else str(path).replace(str(ROOT) + "/", ""),
            }
        )
    return pd.DataFrame(rows)


def cleanup_candidates(manifest_df: pd.DataFrame) -> pd.DataFrame:
    patterns = [".DS_Store", ".Rhistory", "__pycache__"]
    rows: list[dict[str, Any]] = []
    for _, row in manifest_df.iterrows():
        path = row["relative_path"]
        if any(token in path for token in patterns):
            rows.append(
                {
                    "relative_path": path,
                    "extension": row["extension"],
                    "size_kb": row["size_kb"],
                    "note": "Potential cleanup candidate or hidden/provenance artifact",
                }
            )
    return pd.DataFrame(rows)


def build_markdown_summary(
    *,
    dataset_summary_df: pd.DataFrame,
    component_df: pd.DataFrame,
    method_df: pd.DataFrame,
    key_index_df: pd.DataFrame,
    cleanup_df: pd.DataFrame,
) -> str:
    lines = [
        "# Output Analysis Summary",
        "",
        "This folder contains a generated audit of the thesis workspace in `final work`.",
        "",
        "## High-level structure",
        "",
        f"- Top-level datasets present: {', '.join(DATASET_NAMES)}.",
        "- Shared top-level synthesis present: `Final_summary` and `Final_discussion_best_method`.",
        "- Preprocessing provenance present under `Data`.",
        "",
        "## Dataset overview",
        "",
    ]
    for _, row in dataset_summary_df.iterrows():
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                f"- Label: {row['dataset_label']}",
                f"- Target sites: {row['n_target_sites']}",
                f"- Models: {row['n_models']}",
                f"- Methods: {row['n_methods']}",
                f"- Metrics: {row['n_metrics']}",
                f"- Mean OOD performance across the master table: {row['mean_ood_overall']:.3f}",
                f"- Mean ID-OOD gap across the master table: {row['mean_gap_overall']:.3f}",
                f"- Mean ECE across the master table: {row['mean_ece_overall']:.3f}",
            ]
        )
        if pd.notna(row["mean_metadata_shift"]):
            lines.append(
                f"- Mean metadata shift score: {row['mean_metadata_shift']:.3f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Analysis blocks detected",
            "",
        ]
    )
    for dataset in DATASET_NAMES:
        subset = component_df[component_df["dataset"] == dataset]
        present_sections = subset.loc[subset["exists"], "section"].tolist()
        missing_sections = subset.loc[~subset["exists"], "section"].tolist()
        lines.append(f"- {dataset}: present -> {', '.join(present_sections)}.")
        if missing_sections:
            lines.append(f"  Missing -> {', '.join(missing_sections)}.")
    lines.extend(
        [
            "",
            "## Method-level snapshot",
            "",
        ]
    )
    for dataset in DATASET_NAMES:
        subset = method_df[method_df["dataset"] == dataset]
        if subset.empty:
            continue
        best = subset.sort_values("mean_ood_performance", ascending=False).iloc[0]
        lines.append(
            f"- {dataset}: highest average OOD among methods in the pattern master table is {best['method']} "
            f"({best['mean_ood_performance']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Key reports",
            "",
        ]
    )
    for _, row in key_index_df.iterrows():
        if row["exists"]:
            lines.append(f"- {row['label']}: `{row['path_relative_to_root']}`")
    lines.extend(
        [
            "",
            "## Cleanup / artifact notes",
            "",
            f"- Cleanup candidates detected: {len(cleanup_df)} files or artifacts.",
            "- Hidden files and provenance artifacts are listed separately and were not modified.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_contents_report(
    *,
    manifest_df: pd.DataFrame,
    dir_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    method_df: pd.DataFrame,
) -> str:
    total_files = len(manifest_df)
    ext_counts = manifest_df["extension"].value_counts().head(15)
    top_areas = manifest_df["top_level_area"].value_counts()
    lines = [
        "FINAL WORK CONTENT ANALYSIS",
        "",
        f"Total files indexed (excluding Output): {total_files}",
        "",
        "Top-level area file counts:",
    ]
    for area, count in top_areas.items():
        lines.append(f"- {area}: {count}")
    lines.extend(["", "Most common file extensions:"])
    for ext, count in ext_counts.items():
        label = ext if ext else "[no extension]"
        lines.append(f"- {label}: {count}")
    lines.extend(["", "Dataset summaries:"])
    for _, row in dataset_summary_df.iterrows():
        lines.append(
            f"- {row['dataset']}: {row['n_target_sites']} target sites, {row['n_models']} models, "
            f"{row['n_methods']} methods, {row['n_metrics']} metrics, mean OOD {row['mean_ood_overall']:.3f}"
        )
    lines.extend(["", "Method summary by dataset:"])
    for dataset in DATASET_NAMES:
        subset = method_df[method_df["dataset"] == dataset].sort_values("method")
        if subset.empty:
            continue
        lines.append(f"- {dataset}:")
        for _, row in subset.iterrows():
            lines.append(
                f"  {row['method']}: OOD {row['mean_ood_performance']:.3f}, gap {row['mean_id_ood_gap']:.3f}, "
                f"ECE {row['mean_ece']:.3f}, variability {row['mean_std_ood_performance']:.3f}"
            )
    busiest = dir_summary_df.sort_values("n_files_direct", ascending=False).head(15)
    lines.extend(["", "Directories with the most direct files:"])
    for _, row in busiest.iterrows():
        lines.append(f"- {row['directory']}: {row['n_files_direct']} direct files")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = collect_file_manifest(ROOT)
    dir_summary_df = collect_directory_summary(ROOT, manifest_df)
    component_df = dataset_component_status(ROOT)
    dataset_summary_df, method_df = dataset_master_summary(ROOT)
    key_index_df = key_reports_index(ROOT)
    cleanup_df = cleanup_candidates(manifest_df)

    manifest_df.to_csv(
        OUTPUT_DIR / "file_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    dir_summary_df.to_csv(OUTPUT_DIR / "directory_summary.csv", index=False)
    component_df.to_csv(OUTPUT_DIR / "dataset_component_status.csv", index=False)
    dataset_summary_df.to_csv(OUTPUT_DIR / "dataset_master_summary.csv", index=False)
    method_df.to_csv(OUTPUT_DIR / "dataset_method_snapshot.csv", index=False)
    key_index_df.to_csv(OUTPUT_DIR / "key_reports_index.csv", index=False)
    cleanup_df.to_csv(OUTPUT_DIR / "cleanup_candidates.csv", index=False)

    write_text(
        OUTPUT_DIR / "workspace_tree.txt",
        "\n".join(build_tree_lines(ROOT, max_depth=3)) + "\n",
    )
    write_text(
        OUTPUT_DIR / "workspace_content_analysis.txt",
        build_contents_report(
            manifest_df=manifest_df,
            dir_summary_df=dir_summary_df,
            dataset_summary_df=dataset_summary_df,
            method_df=method_df,
        ),
    )
    write_text(
        OUTPUT_DIR / "workspace_summary.md",
        build_markdown_summary(
            dataset_summary_df=dataset_summary_df,
            component_df=component_df,
            method_df=method_df,
            key_index_df=key_index_df,
            cleanup_df=cleanup_df,
        ),
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "n_manifest_files": int(len(manifest_df)),
        "n_directories_summarized": int(len(dir_summary_df)),
        "datasets": DATASET_NAMES,
    }
    write_text(
        OUTPUT_DIR / "output_analysis_metadata.json", json.dumps(metadata, indent=2)
    )

    print(f"Wrote output analysis package to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
