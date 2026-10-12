"""Transportability experiment: generate best method table."""

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


import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uci_transportability.config import (
    METHOD_ORDER,
    MODEL_ORDER,
    TARGET_SITE_ORDER,
    pretty_model_name,
    pretty_target_site,
)


PRIMARY_SELECTION_RULE = (
    "Highest OOD ROC-AUC, then highest OOD Balanced Accuracy, then highest OOD F1, "
    "then lowest OOD ECE, then smallest ROC-AUC gap."
)

METRIC_DIRECTIONS = {
    "roc_auc_ood": False,
    "f1_ood": False,
    "balanced_accuracy_ood": False,
    "ece_ood": True,
    "gap_roc_auc": True,
    "gap_f1": True,
    "gap_balanced_accuracy": True,
}

METRIC_LABELS = {
    "roc_auc_ood": "Best ROC-AUC",
    "f1_ood": "Best F1",
    "balanced_accuracy_ood": "Best Balanced Accuracy",
    "ece_ood": "Lowest ECE",
    "gap_roc_auc": "Smallest ROC-AUC gap",
    "gap_f1": "Smallest F1 gap",
    "gap_balanced_accuracy": "Smallest Balanced Accuracy gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate best-method summary tables for UCI transportability results."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to target_site_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination folder for summary tables",
    )
    return parser.parse_args()


def _prepare_results(results_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "method",
        "model",
        "target_site",
        "roc_auc_ood",
        "f1_ood",
        "balanced_accuracy_ood",
        "ece_ood",
        "gap_roc_auc",
        "gap_f1",
        "gap_balanced_accuracy",
    }
    missing = sorted(required_columns.difference(results_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    prepared = results_df.copy()
    prepared["method"] = prepared["method"].astype(str)
    prepared["model"] = prepared["model"].astype(str)
    prepared["target_site"] = prepared["target_site"].astype(str)
    prepared["model_display"] = prepared["model"].map(pretty_model_name)
    prepared["target_site_display"] = prepared["target_site"].map(pretty_target_site)
    method_rank = {method: rank for rank, method in enumerate(METHOD_ORDER)}
    model_rank = {model: rank for rank, model in enumerate(MODEL_ORDER)}
    site_rank = {site: rank for rank, site in enumerate(TARGET_SITE_ORDER)}
    prepared["_method_rank"] = (
        prepared["method"].map(method_rank).fillna(len(method_rank)).astype(int)
    )
    prepared["_model_rank"] = (
        prepared["model"].map(model_rank).fillna(len(model_rank)).astype(int)
    )
    prepared["_site_rank"] = (
        prepared["target_site"].map(site_rank).fillna(len(site_rank)).astype(int)
    )
    return prepared


def _select_primary_best(subset: pd.DataFrame) -> pd.Series:
    ranked = subset.sort_values(
        by=[
            "roc_auc_ood",
            "balanced_accuracy_ood",
            "f1_ood",
            "ece_ood",
            "gap_roc_auc",
            "_method_rank",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def build_primary_best_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = results_df.groupby(["model", "target_site"], sort=False, dropna=False)
    for (model_name, target_site), subset in grouped:
        best_row = _select_primary_best(subset)
        rows.append(
            {
                "model": model_name,
                "model_display": pretty_model_name(model_name),
                "target_site": target_site,
                "target_site_display": pretty_target_site(target_site),
                "best_method": best_row["method"],
                "roc_auc_ood": float(best_row["roc_auc_ood"]),
                "f1_ood": float(best_row["f1_ood"]),
                "balanced_accuracy_ood": float(best_row["balanced_accuracy_ood"]),
                "ece_ood": float(best_row["ece_ood"]),
                "gap_roc_auc": float(best_row["gap_roc_auc"]),
                "gap_f1": float(best_row["gap_f1"]),
                "gap_balanced_accuracy": float(best_row["gap_balanced_accuracy"]),
                "selection_rule": PRIMARY_SELECTION_RULE,
            }
        )

    primary_df = pd.DataFrame(rows)
    if primary_df.empty:
        return primary_df
    return primary_df.sort_values(["model_display", "target_site_display"]).reset_index(
        drop=True
    )


def build_best_by_metric_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = results_df.groupby(["model", "target_site"], sort=False, dropna=False)
    for (model_name, target_site), subset in grouped:
        row: dict[str, object] = {
            "model": model_name,
            "model_display": pretty_model_name(model_name),
            "target_site": target_site,
            "target_site_display": pretty_target_site(target_site),
        }
        for metric_name, ascending in METRIC_DIRECTIONS.items():
            best_metric_row = (
                subset.sort_values(
                    by=[metric_name, "_method_rank"],
                    ascending=[ascending, True],
                )
                .reset_index(drop=True)
                .iloc[0]
            )
            metric_stub = metric_name.replace("_ood", "").replace("gap_", "gap_")
            row[f"{metric_stub}_best_method"] = best_metric_row["method"]
            row[f"{metric_stub}_best_value"] = float(best_metric_row[metric_name])
        rows.append(row)

    metric_df = pd.DataFrame(rows)
    if metric_df.empty:
        return metric_df
    return metric_df.sort_values(["model_display", "target_site_display"]).reset_index(
        drop=True
    )


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows available."]
    preview = df.fillna("NA").copy()
    headers = [str(column) for column in preview.columns]
    rows = [[str(value) for value in row] for row in preview.to_numpy()]
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return lines


def write_markdown_summary(
    *, primary_df: pd.DataFrame, metric_df: pd.DataFrame, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# UCI best-method summary",
        "",
        "## Overall best method per model/site",
        "",
        f"Selection rule: {PRIMARY_SELECTION_RULE}",
        "",
    ]
    lines.extend(
        _markdown_table(
            primary_df[
                [
                    "model_display",
                    "target_site_display",
                    "best_method",
                    "roc_auc_ood",
                    "f1_ood",
                    "balanced_accuracy_ood",
                    "ece_ood",
                    "gap_roc_auc",
                ]
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Best method by metric",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            metric_df[
                [
                    "model_display",
                    "target_site_display",
                    "roc_auc_best_method",
                    "f1_best_method",
                    "balanced_accuracy_best_method",
                    "ece_best_method",
                    "gap_roc_auc_best_method",
                ]
            ]
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_csv)
    prepared_df = _prepare_results(results_df)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_df = build_primary_best_table(prepared_df)
    metric_df = build_best_by_metric_table(prepared_df)

    primary_path = output_dir / "best_method_per_model_site.csv"
    metric_path = output_dir / "best_method_per_model_site_by_metric.csv"
    markdown_path = output_dir / "best_method_summary.md"

    primary_df.to_csv(primary_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    write_markdown_summary(
        primary_df=primary_df, metric_df=metric_df, output_path=markdown_path
    )

    print(f"Saved: {primary_path}")
    print(f"Saved: {metric_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
