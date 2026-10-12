"""Post-experiment analysis: generate abide quality report."""

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


import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


STRICT_CSV = Path(
    _configured_path(
        "results_data_abide2_preprocessing_data_processed_abide2_core_multisite_transportability_dataset_strict.csv"
    )
)
RICH_CSV = Path(
    _configured_path(
        "results_data_abide2_preprocessing_data_processed_abide2_core_multisite_transportability_dataset.csv"
    )
)
BLUEPRINT_JSON = Path(
    _configured_path(
        "results_data_abide2_preprocessing_data_processed_abide2_core_multisite_preprocessing_blueprint.json"
    )
)
OUTPUT_DIR = Path(_configured_path("results_abide_quality_report"))


COLUMN_DESCRIPTIONS = {
    "subject_id": {
        "role": "identifier",
        "modeling_group": "identifier",
        "conceptual_type": "string identifier",
        "description": "Subject-level identifier carried into the final ABIDE strict transportability dataset.",
        "notes": "Should remain unique per row and is excluded from predictive modeling.",
    },
    "site": {
        "role": "metadata",
        "modeling_group": "domain_metadata",
        "conceptual_type": "site/domain label",
        "description": "ABIDE site indicator used as the domain variable for cross-site transportability.",
        "notes": "This is the grouping variable used for target-site splits and source-domain validation.",
    },
    "disease_level": {
        "role": "label",
        "modeling_group": "target",
        "conceptual_type": "binary categorical string",
        "description": "Human-readable ASD versus Control label.",
        "notes": "Observed values in this strict dataset are ASD and Control.",
    },
    "disease_level_code": {
        "role": "label",
        "modeling_group": "target",
        "conceptual_type": "binary coded label",
        "description": "Numeric encoding paired with the human-readable diagnosis label.",
        "notes": "Observed mapping in this dataset is 1 = ASD and 2 = Control.",
    },
    "age_at_scan": {
        "role": "predictive_feature",
        "modeling_group": "strict_numeric_feature",
        "conceptual_type": "numeric continuous",
        "description": "Age at scan in years.",
        "notes": "Fully observed in the strict dataset, but strongly heterogeneous across sites.",
    },
    "sex_code": {
        "role": "predictive_feature",
        "modeling_group": "strict_categorical_feature",
        "conceptual_type": "coded categorical",
        "description": "Sex code from the source phenotype table.",
        "notes": "Stored numerically but should be modeled as categorical rather than continuous.",
    },
    "handedness_category_code": {
        "role": "predictive_feature",
        "modeling_group": "strict_categorical_feature",
        "conceptual_type": "coded categorical",
        "description": "Handedness category code from the source phenotype table.",
        "notes": "Low-cardinality feature with some missingness; should be treated as categorical.",
    },
    "eye_status_at_scan_code": {
        "role": "predictive_feature",
        "modeling_group": "strict_categorical_feature",
        "conceptual_type": "coded categorical",
        "description": "Eye-status-at-scan code from the source phenotype table.",
        "notes": "Low-cardinality feature with almost complete coverage.",
    },
    "fiq": {
        "role": "predictive_feature",
        "modeling_group": "strict_numeric_feature",
        "conceptual_type": "numeric continuous",
        "description": "Full-scale IQ score.",
        "notes": "Clinically meaningful but incompletely observed and highly site-dependent in coverage.",
    },
}


def first_values(series: pd.Series, n: int = 8) -> str:
    values = series.dropna().unique().tolist()[:n]
    rendered: list[str] = []
    for value in values:
        if isinstance(value, (np.integer, int)):
            rendered.append(str(int(value)))
        elif isinstance(value, (np.floating, float)) and float(value).is_integer():
            rendered.append(str(int(value)))
        else:
            rendered.append(str(value))
    return ", ".join(rendered)


def render_scalar(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    *,
    round_cols: dict[str, int] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    if df.empty:
        return ["No rows available."]

    preview = df.head(max_rows).copy() if max_rows is not None else df.copy()
    if round_cols:
        for column, digits in round_cols.items():
            if column in preview.columns:
                preview[column] = preview[column].apply(
                    lambda x: round(float(x), digits) if pd.notna(x) else x
                )

    preview = preview.fillna("NA")
    headers = [str(col) for col in preview.columns]
    rows = [
        [render_scalar(value) for value in row]
        for row in preview.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        if rows
        else len(headers[idx])
        for idx in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    return lines


def render_html(report_text: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #faf9f7;
      color: #1f2328;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 28px 40px;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>{escape(report_text)}</main>
</body>
</html>
"""


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    strict_df = pd.read_csv(STRICT_CSV)
    rich_df = pd.read_csv(RICH_CSV)
    blueprint = json.loads(BLUEPRINT_JSON.read_text(encoding="utf-8"))
    return strict_df, rich_df, blueprint


def build_variant_summary(
    strict_df: pd.DataFrame, rich_df: pd.DataFrame, blueprint: dict
) -> pd.DataFrame:
    rows = []
    for variant_name, df, key in [
        ("strict", strict_df, "strict_variant"),
        ("rich", rich_df, "rich_variant"),
    ]:
        feature_columns = blueprint[key]["feature_columns"]
        rows.append(
            {
                "variant": variant_name,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "n_predictive_features": int(len(feature_columns)),
                "path": blueprint[key]["path"],
                "recommended_for": blueprint[key].get("recommended_for", ""),
            }
        )
    return pd.DataFrame(rows)


def build_variable_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        desc = COLUMN_DESCRIPTIONS.get(
            column,
            {
                "role": "unknown",
                "modeling_group": "unknown",
                "conceptual_type": "unknown",
                "description": "",
                "notes": "",
            },
        )
        row = {
            "column_name": column,
            "role": desc["role"],
            "modeling_group": desc["modeling_group"],
            "conceptual_type": desc["conceptual_type"],
            "pandas_dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100.0),
            "unique_non_null": int(series.nunique(dropna=True)),
            "example_values": first_values(series),
            "description": desc["description"],
            "notes": desc["notes"],
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "min": float(series.min()) if series.notna().any() else np.nan,
                    "q25": float(series.quantile(0.25))
                    if series.notna().any()
                    else np.nan,
                    "median": float(series.median())
                    if series.notna().any()
                    else np.nan,
                    "q75": float(series.quantile(0.75))
                    if series.notna().any()
                    else np.nan,
                    "max": float(series.max()) if series.notna().any() else np.nan,
                    "mean": float(series.mean()) if series.notna().any() else np.nan,
                    "std": float(series.std()) if series.notna().any() else np.nan,
                }
            )
        else:
            row.update(
                {
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["role", "modeling_group", "column_name"])
        .reset_index(drop=True)
    )


def build_missingness_summary(variable_inventory: pd.DataFrame) -> pd.DataFrame:
    return (
        variable_inventory[
            [
                "column_name",
                "role",
                "modeling_group",
                "missing_count",
                "missing_pct",
                "unique_non_null",
            ]
        ]
        .sort_values(["missing_count", "column_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "column_name": column,
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100.0),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "std": float(series.std()) if series.notna().any() else np.nan,
                "min": float(series.min()) if series.notna().any() else np.nan,
                "q25": float(series.quantile(0.25)) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "q75": float(series.quantile(0.75)) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_selected_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "site",
        "disease_level",
        "disease_level_code",
        "sex_code",
        "handedness_category_code",
        "eye_status_at_scan_code",
    ]
    rows: list[dict[str, object]] = []
    for column in selected_columns:
        counts = df[column].value_counts(dropna=False)
        rendered = []
        for key, value in counts.items():
            label = (
                "NA"
                if pd.isna(key)
                else str(int(key))
                if isinstance(key, float) and float(key).is_integer()
                else str(key)
            )
            rendered.append(f"{label}: {int(value)}")
        rows.append(
            {
                "column_name": column,
                "missing_count": int(df[column].isna().sum()),
                "unique_non_null": int(df[column].nunique(dropna=True)),
                "levels_and_counts": "; ".join(rendered),
            }
        )
    return pd.DataFrame(rows)


def build_row_missingness_summary(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    row_missing = df[feature_cols].isna().sum(axis=1).value_counts().sort_index()
    return pd.DataFrame(
        {
            "n_missing_feature_values": row_missing.index.astype(int),
            "n_subjects": row_missing.values.astype(int),
            "share_subjects": row_missing.values / len(df),
        }
    )


def build_target_eligibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site, group in df.groupby("site", dropna=False):
        asd_count = int((group["disease_level_code"] == 1).sum())
        control_count = int((group["disease_level_code"] == 2).sum())
        class_counts = [count for count in [asd_count, control_count] if count > 0]
        min_class_count = int(min(class_counts)) if class_counts else 0
        eligible = len(class_counts) == 2 and min_class_count >= 2
        reasons: list[str] = []
        if len(class_counts) < 2:
            reasons.append("single_class_site")
        if min_class_count < 2:
            reasons.append("min_class_count_below_2")
        rows.append(
            {
                "site": str(site),
                "n_subjects": int(len(group)),
                "n_asd": asd_count,
                "n_control": control_count,
                "asd_rate": float((group["disease_level_code"] == 1).mean()),
                "min_class_count": min_class_count,
                "eligible_as_target": bool(eligible),
                "exclusion_reason": "; ".join(reasons),
            }
        )
    summary = (
        pd.DataFrame(rows)
        .sort_values(
            ["eligible_as_target", "n_subjects", "site"], ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )
    for column in ["n_subjects", "n_asd", "n_control", "min_class_count"]:
        summary[column] = summary[column].astype(int)
    summary["eligible_as_target"] = summary["eligible_as_target"].astype(bool)
    return summary


def build_site_quality_summary(
    df: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    grouped = df.groupby("site", dropna=False)
    summary = grouped.apply(
        lambda x: pd.Series(
            {
                "n_subjects": len(x),
                "n_unique_subject_id": x["subject_id"].nunique(dropna=True),
                "n_asd": int((x["disease_level_code"] == 1).sum()),
                "n_control": int((x["disease_level_code"] == 2).sum()),
                "asd_rate": float((x["disease_level_code"] == 1).mean()),
                "age_mean": float(x["age_at_scan"].mean()),
                "age_sd": float(x["age_at_scan"].std()),
                "age_min": float(x["age_at_scan"].min()),
                "age_max": float(x["age_at_scan"].max()),
                "fiq_non_missing_n": int(x["fiq"].notna().sum()),
                "fiq_missing_pct": float(x["fiq"].isna().mean() * 100.0),
                "handedness_missing_pct": float(
                    x["handedness_category_code"].isna().mean() * 100.0
                ),
                "eye_status_missing_pct": float(
                    x["eye_status_at_scan_code"].isna().mean() * 100.0
                ),
            }
        ),
        include_groups=False,
    ).reset_index()
    summary = summary.merge(
        eligibility[["site", "eligible_as_target", "exclusion_reason"]],
        on="site",
        how="left",
    )
    summary = summary.sort_values("n_subjects", ascending=False).reset_index(drop=True)
    for column in [
        "n_subjects",
        "n_unique_subject_id",
        "n_asd",
        "n_control",
        "fiq_non_missing_n",
    ]:
        summary[column] = summary[column].astype(int)
    summary["eligible_as_target"] = summary["eligible_as_target"].astype(bool)
    return summary


def build_quality_flags(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_summary: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_rows = int(df.duplicated().sum())
    duplicate_subjects = int(df["subject_id"].duplicated().sum())
    mono_class_sites = eligibility.loc[
        ~eligibility["eligible_as_target"], "site"
    ].tolist()
    fiq_fully_missing_sites = site_summary.loc[
        site_summary["fiq_non_missing_n"] == 0, "site"
    ].tolist()
    fiq_high_missing_sites = site_summary.loc[
        site_summary["fiq_missing_pct"] >= 25.0, "site"
    ].tolist()
    subjects_with_any_missing = int(df[feature_cols].isna().any(axis=1).sum())
    age_mean_range = float(
        site_summary["age_mean"].max() - site_summary["age_mean"].min()
    )
    eligible_site_summary = site_summary.loc[
        site_summary["eligible_as_target"].fillna(False)
    ].copy()
    eligible_asd_range = float(
        eligible_site_summary["asd_rate"].max()
        - eligible_site_summary["asd_rate"].min()
    )
    all_site_asd_range = float(
        site_summary["asd_rate"].max() - site_summary["asd_rate"].min()
    )

    rows = [
        {
            "flag": "final_csv_exact_duplicate_rows",
            "value": duplicate_rows,
            "severity": "info" if duplicate_rows == 0 else "warning",
            "notes": "Direct duplicate-row check on the strict ABIDE CSV used by the transportability pipeline.",
        },
        {
            "flag": "duplicated_subject_id_rows",
            "value": duplicate_subjects,
            "severity": "info" if duplicate_subjects == 0 else "warning",
            "notes": "Duplicated subject identifiers in the strict dataset.",
        },
        {
            "flag": "target_ineligible_sites",
            "value": int((~eligibility["eligible_as_target"]).sum()),
            "severity": "review" if mono_class_sites else "info",
            "notes": "Sites excluded as target domains under the 50/50 transportability split: "
            + (", ".join(mono_class_sites) if mono_class_sites else "none"),
        },
        {
            "flag": "fiq_missing_subjects",
            "value": int(df["fiq"].isna().sum()),
            "severity": "review" if int(df["fiq"].isna().sum()) > 0 else "info",
            "notes": "Subjects missing FIQ in the strict dataset.",
        },
        {
            "flag": "fiq_fully_missing_sites",
            "value": int(len(fiq_fully_missing_sites)),
            "severity": "review" if fiq_fully_missing_sites else "info",
            "notes": "Sites with no observed FIQ values: "
            + (
                ", ".join(fiq_fully_missing_sites)
                if fiq_fully_missing_sites
                else "none"
            ),
        },
        {
            "flag": "fiq_high_missing_sites_ge_25pct",
            "value": int(len(fiq_high_missing_sites)),
            "severity": "review" if fiq_high_missing_sites else "info",
            "notes": "Sites with at least 25% FIQ missingness: "
            + (", ".join(fiq_high_missing_sites) if fiq_high_missing_sites else "none"),
        },
        {
            "flag": "handedness_missing_subjects",
            "value": int(df["handedness_category_code"].isna().sum()),
            "severity": "review"
            if int(df["handedness_category_code"].isna().sum()) > 0
            else "info",
            "notes": "Subjects with missing handedness code.",
        },
        {
            "flag": "eye_status_missing_subjects",
            "value": int(df["eye_status_at_scan_code"].isna().sum()),
            "severity": "review"
            if int(df["eye_status_at_scan_code"].isna().sum()) > 0
            else "info",
            "notes": "Subjects with missing eye-status-at-scan code.",
        },
        {
            "flag": "subjects_with_any_missing_strict_feature",
            "value": subjects_with_any_missing,
            "severity": "review" if subjects_with_any_missing > 0 else "info",
            "notes": "Subjects missing at least one of the strict predictive features.",
        },
        {
            "flag": "site_age_mean_range_years",
            "value": round(age_mean_range, 6),
            "severity": "review" if age_mean_range >= 10.0 else "info",
            "notes": "Difference between the maximum and minimum site-level mean age.",
        },
        {
            "flag": "eligible_target_site_asd_rate_range",
            "value": round(eligible_asd_range, 6),
            "severity": "review" if eligible_asd_range >= 0.15 else "info",
            "notes": "Range of ASD prevalence across the sites that are actually eligible as targets.",
        },
        {
            "flag": "all_site_asd_rate_range",
            "value": round(all_site_asd_range, 6),
            "severity": "review" if all_site_asd_range >= 0.15 else "info",
            "notes": "Range of ASD prevalence across all observed sites, including mono-class sites.",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    *,
    strict_df: pd.DataFrame,
    rich_df: pd.DataFrame,
    blueprint: dict,
    variant_summary: pd.DataFrame,
    variable_inventory: pd.DataFrame,
    missingness_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    row_missingness_summary: pd.DataFrame,
    target_eligibility_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    quality_flags: pd.DataFrame,
    feature_cols: list[str],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    eligible_targets = int(target_eligibility_summary["eligible_as_target"].sum())
    target_exclusions = target_eligibility_summary.loc[
        ~target_eligibility_summary["eligible_as_target"], "site"
    ].tolist()
    dictionary_view = variable_inventory[
        [
            "column_name",
            "role",
            "modeling_group",
            "conceptual_type",
            "pandas_dtype",
            "missing_count",
            "missing_pct",
            "unique_non_null",
            "description",
            "notes",
        ]
    ]

    lines = [
        "# ABIDE II strict phenotype quality report",
        "",
        "## Scope",
        f"- Generated at: `{generated_at}`",
        f"- Primary input file: `{STRICT_CSV}`",
        f"- Companion rich file: `{RICH_CSV}`",
        f"- Blueprint file: `{BLUEPRINT_JSON}`",
        "- This report describes the exact strict ABIDE dataset currently used by the baseline transportability pipeline.",
        "",
        "## Dataset snapshot",
        f"- Strict dataset shape: `{strict_df.shape[0]} rows x {strict_df.shape[1]} columns`.",
        f"- Rich companion dataset shape: `{rich_df.shape[0]} rows x {rich_df.shape[1]} columns`.",
        f"- Observed sites: `{strict_df['site'].nunique()}`.",
        f"- Eligible target sites under the 50/50 split protocol: `{eligible_targets}`.",
        f"- Sites excluded only as targets: `{', '.join(target_exclusions) if target_exclusions else 'None'}`.",
        f"- Overall ASD prevalence: `{(strict_df['disease_level_code'] == 1).mean():.4f}`.",
        f"- Exact duplicate rows: `{int(strict_df.duplicated().sum())}`.",
        f"- Duplicated subject_id rows: `{int(strict_df['subject_id'].duplicated().sum())}`.",
        "- Transportability target mapping used downstream: derive binary `target` as `ASD = 1` and `Control = 0` from `disease_level_code`.",
        "",
        "## Variant summary",
        "",
    ]
    lines.extend(markdown_table(variant_summary))
    lines.extend(
        [
            "",
            "## Strict feature set",
            f"- Predictive features in the strict variant: `{', '.join(feature_cols)}`.",
            f"- Blueprint recommendation for the strict variant: `{blueprint['strict_variant'].get('recommended_for', '')}`.",
            f"- Blueprint recommendation for the rich variant: `{blueprint['rich_variant'].get('recommended_for', '')}`.",
            "- The strict bundle is appropriate for the main non-circular baseline because it excludes behavior scales too close to the ASD/control label.",
            "",
            "## Target-site eligibility",
            "",
        ]
    )
    lines.extend(markdown_table(target_eligibility_summary, round_cols={"asd_rate": 4}))
    lines.extend(
        [
            "",
            "## Site quality summary",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            site_summary[
                [
                    "site",
                    "n_subjects",
                    "n_asd",
                    "n_control",
                    "asd_rate",
                    "eligible_as_target",
                    "age_mean",
                    "age_sd",
                    "fiq_missing_pct",
                    "handedness_missing_pct",
                    "eye_status_missing_pct",
                ]
            ],
            round_cols={
                "asd_rate": 4,
                "age_mean": 2,
                "age_sd": 2,
                "fiq_missing_pct": 2,
                "handedness_missing_pct": 2,
                "eye_status_missing_pct": 2,
            },
        )
    )
    lines.extend(
        [
            "",
            "## Variable dictionary",
            "",
        ]
    )
    lines.extend(markdown_table(dictionary_view, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Numeric summary",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            numeric_summary,
            round_cols={
                "missing_pct": 2,
                "mean": 3,
                "std": 3,
                "min": 3,
                "q25": 3,
                "median": 3,
                "q75": 3,
                "max": 3,
            },
        )
    )
    lines.extend(
        [
            "",
            "## Selected categorical summaries",
            "",
        ]
    )
    lines.extend(markdown_table(categorical_summary))
    lines.extend(
        [
            "",
            "## Feature-level missingness summary",
            "",
        ]
    )
    lines.extend(markdown_table(missingness_summary, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Row missingness across strict predictive features",
            "",
        ]
    )
    lines.extend(
        markdown_table(row_missingness_summary, round_cols={"share_subjects": 4})
    )
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
        ]
    )
    lines.extend(markdown_table(quality_flags))
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "- The strict ABIDE dataset is compact and leakage-conscious, but it is not fully clean: several encoded categorical variables are stored as floats because missingness was intentionally preserved for train-fold-only preprocessing.",
            "- Site heterogeneity is substantial. Mean age varies sharply across centers, and ASD prevalence is not uniform even before considering the two mono-class sites.",
            "- `fiq` is the most fragile strict feature: it is entirely missing at `ABIDEII-EMC_1` and heavily incomplete at `ABIDEII-IP_1`, so train-only imputation is essential.",
            "- `handedness_category_code` and `eye_status_at_scan_code` are low-cardinality clinical codes and should be modeled as categorical variables, not as continuous measures.",
            "- Because `ABIDEII-KUL_3` and `ABIDEII-NYU_2` are mono-class, they can remain among source domains but are not valid target sites for the requested 50/50 transportability protocol.",
            "",
            "## Companion files",
            "- `ABIDE2_quality_report.rdm`: requested human-readable report file.",
            "- `ABIDE2_quality_report.md`: markdown twin of the same report.",
            "- `ABIDE2_quality_report.html`: simple HTML preview of the same report content.",
            "- `abide_variant_summary.csv`: strict-versus-rich dataset summary.",
            "- `abide_variable_inventory.csv`: per-column dictionary with roles, types, missingness, and notes.",
            "- `abide_missingness_summary.csv`: compact missingness table for every column.",
            "- `abide_numeric_summary.csv`: descriptive statistics for numeric strict variables.",
            "- `abide_selected_categorical_summary.csv`: readable counts for the main categorical variables.",
            "- `abide_row_missingness_summary.csv`: number of missing strict features per subject.",
            "- `abide_target_eligibility_summary.csv`: target-site eligibility under the split protocol used in the pipeline.",
            "- `abide_site_quality_summary.csv`: site-level counts, prevalence, age summary, and missingness markers.",
            "- `abide_quality_flags.csv`: compact QC attention-point table.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strict_df, rich_df, blueprint = load_inputs()

    feature_cols = list(blueprint["strict_variant"]["feature_columns"])
    numeric_cols = [
        column for column in feature_cols if column in {"age_at_scan", "fiq"}
    ]

    variant_summary = build_variant_summary(strict_df, rich_df, blueprint)
    variable_inventory = build_variable_inventory(strict_df)
    missingness_summary = build_missingness_summary(variable_inventory)
    numeric_summary = build_numeric_summary(strict_df, numeric_cols)
    categorical_summary = build_selected_categorical_summary(strict_df)
    row_missingness_summary = build_row_missingness_summary(strict_df, feature_cols)
    target_eligibility_summary = build_target_eligibility_summary(strict_df)
    site_summary = build_site_quality_summary(strict_df, target_eligibility_summary)
    quality_flags = build_quality_flags(
        strict_df, feature_cols, site_summary, target_eligibility_summary
    )

    variant_summary.to_csv(OUTPUT_DIR / "abide_variant_summary.csv", index=False)
    variable_inventory.to_csv(OUTPUT_DIR / "abide_variable_inventory.csv", index=False)
    missingness_summary.to_csv(
        OUTPUT_DIR / "abide_missingness_summary.csv", index=False
    )
    numeric_summary.to_csv(OUTPUT_DIR / "abide_numeric_summary.csv", index=False)
    categorical_summary.to_csv(
        OUTPUT_DIR / "abide_selected_categorical_summary.csv", index=False
    )
    row_missingness_summary.to_csv(
        OUTPUT_DIR / "abide_row_missingness_summary.csv", index=False
    )
    target_eligibility_summary.to_csv(
        OUTPUT_DIR / "abide_target_eligibility_summary.csv", index=False
    )
    site_summary.to_csv(OUTPUT_DIR / "abide_site_quality_summary.csv", index=False)
    quality_flags.to_csv(OUTPUT_DIR / "abide_quality_flags.csv", index=False)

    report_text = build_report(
        strict_df=strict_df,
        rich_df=rich_df,
        blueprint=blueprint,
        variant_summary=variant_summary,
        variable_inventory=variable_inventory,
        missingness_summary=missingness_summary,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        row_missingness_summary=row_missingness_summary,
        target_eligibility_summary=target_eligibility_summary,
        site_summary=site_summary,
        quality_flags=quality_flags,
        feature_cols=feature_cols,
    )
    (OUTPUT_DIR / "ABIDE2_quality_report.rdm").write_text(report_text, encoding="utf-8")
    (OUTPUT_DIR / "ABIDE2_quality_report.md").write_text(report_text, encoding="utf-8")
    (OUTPUT_DIR / "ABIDE2_quality_report.html").write_text(
        render_html(report_text, "ABIDE II strict phenotype quality report"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
