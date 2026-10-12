"""Post-experiment analysis: generate dataset goal interpretation."""

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


import math
from pathlib import Path

import pandas as pd


ROOT = Path(_configured_path("results_final_discussion_best_method"))
BEST_METHOD_PATH = ROOT / "best_method_by_site.csv"
BUNDLED_EXCEL_PYTHON = Path(_configured_path("python_executable"))

OBJECTIVES = [
    ("ood", "best_method_ood", "maximize raw OOD performance"),
    ("gap", "best_method_gap", "minimize ID-OOD gap"),
    ("ece", "best_method_ece", "minimize ECE / improve calibration"),
    ("variability", "best_method_variability", "minimize OOD variability"),
]

METHOD_LABELS = {
    "M0": "M0: Baseline",
    "M1": "M1: ComBat",
    "M2": "M2: Weighting",
    "M3": "M3: Transductive",
    "M4": "M4: ComBat + Transductive",
}

DATASET_LABELS = {
    "UCI": "UCI Heart Disease",
    "TCGA": "TCGA-BRCA",
    "ABIDE": "ABIDE II",
}


def write_excel_with_fallback(
    df: pd.DataFrame, path: Path, *, sheet_name: str = "Summary"
) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return
    except ModuleNotFoundError:
        if not BUNDLED_EXCEL_PYTHON.exists():
            raise

    temp_csv = path.with_suffix(".tmp.csv")
    df.to_csv(temp_csv, index=False)
    helper = f"""
import pandas as pd
df = pd.read_csv(r'{temp_csv}')
with pd.ExcelWriter(r'{path}') as writer:
    df.to_excel(writer, sheet_name='{sheet_name[:31]}', index=False)
"""
    import subprocess

    subprocess.run([str(BUNDLED_EXCEL_PYTHON), "-c", helper], check=True)
    temp_csv.unlink(missing_ok=True)


def label_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def format_support(count: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{count}/{total} ({count / total:.1%})"


def summarize_objective(
    dataset_short: str, subset: pd.DataFrame, column: str, goal_text: str
) -> dict[str, object]:
    counts = subset[column].value_counts()
    total = int(counts.sum())
    top_count = int(counts.iloc[0])
    top_methods = sorted(counts[counts == top_count].index.tolist())
    recommended_method = " / ".join(label_method(method) for method in top_methods)

    if len(counts) > len(top_methods):
        runner_up_count = int(counts.iloc[len(top_methods)])
        runner_up_methods = sorted(counts[counts == runner_up_count].index.tolist())
        runner_up = " / ".join(label_method(method) for method in runner_up_methods)
    else:
        runner_up_count = 0
        runner_up = "none"

    margin = top_count - runner_up_count

    dataset_sentence = {
        "UCI": {
            "ood": "On UCI, adaptation-oriented methods tend to help more on raw generalization than on the harder datasets.",
            "gap": "On UCI, gap control is not monopolized by one strategy, but domain correction helps more than target adaptation alone.",
            "ece": "On UCI, calibration tends to improve when the target-adaptation step is active.",
            "variability": "On UCI, the most stable method is not necessarily the one with the best OOD score.",
        },
        "TCGA": {
            "ood": "On TCGA, the baseline remains surprisingly competitive for raw OOD performance, suggesting that stronger adaptation does not automatically help in the highest-shift genomic setting.",
            "gap": "On TCGA, ComBat-like correction tends to reduce the train-test mismatch more consistently than the other strategies.",
            "ece": "On TCGA, the best calibration signal comes from the transductive family rather than from the baseline.",
            "variability": "On TCGA, ComBat emerges mainly as a stabilizer rather than as the strongest OOD booster.",
        },
        "ABIDE": {
            "ood": "On ABIDE, weighting and target-aware adaptation are more competitive than the pure baseline for raw OOD performance.",
            "gap": "On ABIDE, the combined ComBat + transductive strategy helps most often when the objective is to compress the ID-OOD gap.",
            "ece": "On ABIDE, calibration tends to favor transductive adaptation more than the other strategies.",
            "variability": "On ABIDE, ComBat is most useful as a variance-reduction strategy rather than as the best raw OOD performer.",
        },
    }

    if len(top_methods) == 1:
        interpretation = (
            f"For {DATASET_LABELS.get(dataset_short, dataset_short)}, if the goal is to {goal_text}, "
            f"the most recurrent recommendation is {label_method(top_methods[0])}, with support in "
            f"{format_support(top_count, total)} of the site-model-metric comparisons. "
            f"{dataset_sentence.get(dataset_short, {}).get(column.replace('best_method_', ''), '')}"
        ).strip()
    else:
        interpretation = (
            f"For {DATASET_LABELS.get(dataset_short, dataset_short)}, no single method dominates the objective to {goal_text}. "
            f"The top pattern is a tie between {recommended_method}, each supported in {format_support(top_count, total)} "
            f"of the comparisons. This suggests the ranking is condition-dependent rather than absolute."
        )

    return {
        "dataset_short": dataset_short,
        "dataset": str(subset["dataset"].iloc[0]),
        "objective_key": column.replace("best_method_", ""),
        "analysis_goal": goal_text,
        "recommended_method": recommended_method,
        "top_methods_raw": ", ".join(top_methods),
        "top_count": top_count,
        "total_comparisons": total,
        "top_share": top_count / total if total else math.nan,
        "runner_up_method": runner_up,
        "runner_up_count": runner_up_count,
        "margin_vs_runner_up": margin,
        "interpretation": interpretation,
    }


def build_goal_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_short, subset in df.groupby("dataset_short"):
        for objective_key, column, goal_text in OBJECTIVES:
            rows.append(summarize_objective(dataset_short, subset, column, goal_text))
    return pd.DataFrame(rows)


def build_model_level_notes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_short, dataset_subset in df.groupby("dataset_short"):
        for model_display, model_subset in dataset_subset.groupby("model_display"):
            for objective_key, column, _ in OBJECTIVES:
                counts = model_subset[column].value_counts()
                top_count = int(counts.iloc[0])
                top_methods = sorted(counts[counts == top_count].index.tolist())
                rows.append(
                    {
                        "dataset_short": dataset_short,
                        "model_display": model_display,
                        "objective_key": objective_key,
                        "dominant_method": " / ".join(
                            label_method(method) for method in top_methods
                        ),
                        "support": format_support(top_count, int(counts.sum())),
                    }
                )
    return pd.DataFrame(rows)


def build_discussion_text(
    summary_df: pd.DataFrame, model_notes_df: pd.DataFrame
) -> str:
    lines = [
        "DISCUSSION-READY INTERPRETATION OF best_method_by_site",
        "",
        "This note summarizes the pattern emerging from best_method_by_site across dataset x target site x model x metric.",
        "The key point is that there is no universally dominant method: the winner changes depending on the objective.",
        "",
    ]

    for dataset_short in ["UCI", "TCGA", "ABIDE"]:
        dataset_rows = summary_df[summary_df["dataset_short"] == dataset_short]
        lines.append(f"{DATASET_LABELS.get(dataset_short, dataset_short)}")
        lines.append("")
        for objective in ["ood", "gap", "ece", "variability"]:
            row = dataset_rows[dataset_rows["objective_key"] == objective].iloc[0]
            lines.append(
                f"- If the goal is to {row['analysis_goal']}, the leading recommendation is {row['recommended_method']}."
            )
            lines.append(
                f"  Evidence: {int(row['top_count'])}/{int(row['total_comparisons'])} winning comparisons."
            )
            if row["runner_up_method"] != "none":
                lines.append(
                    f"  Runner-up: {row['runner_up_method']} with {int(row['runner_up_count'])}/{int(row['total_comparisons'])} wins."
                )
            lines.append(f"  Interpretation: {row['interpretation']}")
        lines.append("")
        model_rows = model_notes_df[model_notes_df["dataset_short"] == dataset_short]
        lines.append("Model-level signal:")
        for model_display in sorted(model_rows["model_display"].unique().tolist()):
            model_subset = model_rows[
                (model_rows["model_display"] == model_display)
                & (model_rows["objective_key"] == "ood")
            ]
            if model_subset.empty:
                continue
            note = model_subset.iloc[0]
            lines.append(
                f"- For {model_display}, the dominant raw OOD winner is {note['dominant_method']} ({note['support']})."
            )
        lines.append("")

    lines.extend(
        [
            "Cross-dataset takeaway",
            "",
            "- Raw OOD performance, ID-OOD gap, calibration and variability do not reward the same strategy.",
            "- In practice, method choice should be conditioned on the target objective, not only on average performance.",
            "- A method can look weak in raw OOD score but still be attractive when calibration or stability is the priority.",
            "- This supports a conditional-robustness interpretation rather than a single global ranking.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(BEST_METHOD_PATH)
    summary_df = build_goal_summary(df)
    model_notes_df = build_model_level_notes(df)

    summary_csv = ROOT / "dataset_goal_best_method_summary.csv"
    summary_xlsx = ROOT / "dataset_goal_best_method_summary.xlsx"
    notes_csv = ROOT / "dataset_goal_best_method_model_notes.csv"
    notes_xlsx = ROOT / "dataset_goal_best_method_model_notes.xlsx"
    discussion_txt = ROOT / "dataset_goal_best_method_discussion.txt"

    summary_df.to_csv(summary_csv, index=False)
    model_notes_df.to_csv(notes_csv, index=False)
    write_excel_with_fallback(summary_df, summary_xlsx, sheet_name="Goal summary")
    write_excel_with_fallback(model_notes_df, notes_xlsx, sheet_name="Model notes")
    discussion_txt.write_text(
        build_discussion_text(summary_df, model_notes_df), encoding="utf-8"
    )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_xlsx}")
    print(f"Wrote {notes_csv}")
    print(f"Wrote {notes_xlsx}")
    print(f"Wrote {discussion_txt}")


if __name__ == "__main__":
    main()
