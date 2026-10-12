"""Post-experiment analysis: generate uci quality report."""

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
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(_configured_path("results_root"))
INPUT_CSV = (
    WORKSPACE_ROOT
    / "Data"
    / "UCI_Heart_preprocessing"
    / "data"
    / "processed"
    / "uci_heart_analysis_ready.csv"
)
SCHEMA_JSON = (
    WORKSPACE_ROOT
    / "Data"
    / "UCI_Heart_preprocessing"
    / "data"
    / "processed"
    / "uci_heart_feature_schema.json"
)
QUALITY_JSON = (
    WORKSPACE_ROOT
    / "Data"
    / "UCI_Heart_preprocessing"
    / "data"
    / "processed"
    / "uci_heart_quality_checks.json"
)
OUTPUT_DIR = WORKSPACE_ROOT / "UCI" / "Quality_report"


COLUMN_DESCRIPTIONS = {
    "subject_id": {
        "role": "identifier",
        "conceptual_type": "string identifier",
        "description": "Harmonized subject identifier carried into the analysis-ready dataset.",
        "notes": "Should be unique per row; useful for auditing and joins, excluded from predictive modeling.",
    },
    "domain_id": {
        "role": "metadata",
        "conceptual_type": "site/domain label",
        "description": "Site/domain indicator for cross-site transportability analyses.",
        "notes": "Observed domains: cleveland, hungarian, long_beach_va, switzerland.",
    },
    "target_original": {
        "role": "label",
        "conceptual_type": "ordinal multiclass label",
        "description": "Original UCI heart disease severity label before binarization.",
        "notes": "Observed values are 0-4; downstream binary target collapses 1-4 into 1.",
    },
    "target": {
        "role": "label",
        "conceptual_type": "binary target",
        "description": "Binary endpoint used for modeling in the current pipeline.",
        "notes": "0 = no disease, 1 = disease present after collapsing original severities 1-4.",
    },
    "age": {
        "role": "predictive_feature",
        "conceptual_type": "numeric continuous",
        "description": "Patient age in years.",
        "notes": "No missing values in the final analysis-ready CSV.",
    },
    "sex": {
        "role": "predictive_feature",
        "conceptual_type": "binary categorical encoded as numeric",
        "description": "Sex code from the original UCI schema.",
        "notes": "Stored numerically but semantically categorical.",
    },
    "cp": {
        "role": "predictive_feature",
        "conceptual_type": "categorical encoded as numeric",
        "description": "Chest pain type code.",
        "notes": "Observed categories 1-4; should be treated as categorical in most models.",
    },
    "trestbps": {
        "role": "predictive_feature",
        "conceptual_type": "numeric continuous",
        "description": "Resting blood pressure.",
        "notes": "Contains missing values and one zero value that is likely non-physiological.",
    },
    "chol": {
        "role": "predictive_feature",
        "conceptual_type": "numeric continuous",
        "description": "Serum cholesterol measurement.",
        "notes": "Contains missing values and many zero values, which likely encode unavailable measurements in some domains.",
    },
    "fbs": {
        "role": "predictive_feature",
        "conceptual_type": "binary categorical encoded as numeric",
        "description": "Fasting blood sugar indicator (>120 mg/dl in the original UCI schema).",
        "notes": "Observed as 0/1 with additional missingness.",
    },
    "restecg": {
        "role": "predictive_feature",
        "conceptual_type": "categorical encoded as numeric",
        "description": "Resting electrocardiographic result code.",
        "notes": "Low-cardinality categorical variable stored as numeric.",
    },
    "thalach": {
        "role": "predictive_feature",
        "conceptual_type": "numeric continuous",
        "description": "Maximum heart rate achieved.",
        "notes": "Contains moderate missingness.",
    },
    "exang": {
        "role": "predictive_feature",
        "conceptual_type": "binary categorical encoded as numeric",
        "description": "Exercise-induced angina indicator.",
        "notes": "Observed as 0/1 with missing values.",
    },
    "oldpeak": {
        "role": "predictive_feature",
        "conceptual_type": "numeric continuous",
        "description": "ST depression induced by exercise relative to rest.",
        "notes": "Contains missing values and some negative values that deserve review.",
    },
    "slope": {
        "role": "predictive_feature",
        "conceptual_type": "categorical encoded as numeric",
        "description": "Slope of the peak exercise ST segment.",
        "notes": "High missingness; one observed zero code may be non-standard depending on UCI coding convention.",
    },
    "ca": {
        "role": "predictive_feature",
        "conceptual_type": "categorical/ordinal encoded as numeric",
        "description": "Number of major vessels colored by fluoroscopy.",
        "notes": "Very high missingness; one observed value 9 is suspicious and should be audited.",
    },
    "thal": {
        "role": "predictive_feature",
        "conceptual_type": "categorical encoded as numeric",
        "description": "Thalassemia / perfusion defect code from the original UCI schema.",
        "notes": "High missingness and heterogeneous coding values across sites; best treated as categorical.",
    },
}


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def first_values(series: pd.Series, n: int = 8) -> str:
    values = series.dropna().unique().tolist()[:n]
    rendered = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            rendered.append(str(int(value)))
        else:
            rendered.append(str(value))
    return ", ".join(rendered)


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(INPUT_CSV)
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    qc = json.loads(QUALITY_JSON.read_text(encoding="utf-8"))
    return df, schema, qc


def build_variable_inventory(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    excluded_columns = set(schema["excluded_columns"])

    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        desc = COLUMN_DESCRIPTIONS.get(
            column,
            {
                "role": "unknown",
                "conceptual_type": "unknown",
                "description": "",
                "notes": "",
            },
        )

        modeling_group = "excluded"
        if column in numeric_features:
            modeling_group = "numeric_feature"
        elif column in categorical_features:
            modeling_group = "categorical_feature"
        elif column not in excluded_columns:
            modeling_group = "other_feature"

        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        missing_pct = float(series.isna().mean() * 100.0)
        unique_non_null = int(series.nunique(dropna=True))

        row = {
            "column_name": column,
            "role": desc["role"],
            "modeling_group": modeling_group,
            "conceptual_type": desc["conceptual_type"],
            "pandas_dtype": str(series.dtype),
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": round(missing_pct, 4),
            "unique_non_null": unique_non_null,
            "example_values": first_values(series),
            "description": desc["description"],
            "notes": desc["notes"],
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "min": float(series.min()) if non_null else np.nan,
                    "q25": float(series.quantile(0.25)) if non_null else np.nan,
                    "median": float(series.median()) if non_null else np.nan,
                    "q75": float(series.quantile(0.75)) if non_null else np.nan,
                    "max": float(series.max()) if non_null else np.nan,
                    "mean": float(series.mean()) if non_null else np.nan,
                    "std": float(series.std()) if non_null else np.nan,
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

    return pd.DataFrame(rows)


def build_domain_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("domain_id", dropna=False)
    summary = grouped.apply(
        lambda x: pd.Series(
            {
                "n_rows": len(x),
                "n_unique_subject_id": x["subject_id"].nunique(dropna=True),
                "positive_cases": int((x["target"] == 1).sum()),
                "negative_cases": int((x["target"] == 0).sum()),
                "positive_prevalence": float(x["target"].mean()),
                "missing_pct_ca": float(x["ca"].isna().mean() * 100.0),
                "missing_pct_thal": float(x["thal"].isna().mean() * 100.0),
                "missing_pct_slope": float(x["slope"].isna().mean() * 100.0),
                "chol_zero_count": int((x["chol"] == 0).sum()),
                "trestbps_zero_count": int((x["trestbps"] == 0).sum()),
            }
        ),
        include_groups=False,
    )
    summary = summary.reset_index().sort_values("domain_id").reset_index(drop=True)
    int_columns = [
        "n_rows",
        "n_unique_subject_id",
        "positive_cases",
        "negative_cases",
        "chol_zero_count",
        "trestbps_zero_count",
    ]
    for column in int_columns:
        summary[column] = summary[column].astype(int)
    return summary


def build_numeric_summary(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in schema["numeric_features"]:
        series = df[column]
        rows.append(
            {
                "column_name": column,
                "count_non_null": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def build_categorical_summary(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in schema["categorical_features"]:
        series = df[column]
        counts = series.value_counts(dropna=False).sort_index()
        rows.append(
            {
                "column_name": column,
                "missing_count": int(series.isna().sum()),
                "unique_non_null": int(series.nunique(dropna=True)),
                "levels_and_counts": "; ".join(
                    f"{idx}: {int(val)}" for idx, val in counts.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_quality_flags(df: pd.DataFrame, qc: dict) -> pd.DataFrame:
    duplicate_subjects = df.loc[
        df["subject_id"].duplicated(keep=False), ["subject_id", "domain_id"]
    ].drop_duplicates()
    rows = [
        {
            "flag": "final_csv_exact_duplicate_rows",
            "value": int(df.duplicated().sum()),
            "severity": "info" if int(df.duplicated().sum()) == 0 else "warning",
            "notes": "Direct duplicate-row check on the current analysis-ready CSV.",
        },
        {
            "flag": "quality_json_exact_duplicate_rows_detected",
            "value": int(qc.get("exact_duplicate_rows_detected", 0)),
            "severity": "warning"
            if int(qc.get("exact_duplicate_rows_detected", 0)) > 0
            else "info",
            "notes": "Value reported by the preprocessing quality-check JSON.",
        },
        {
            "flag": "duplicated_subject_id_rows",
            "value": int(len(duplicate_subjects)),
            "severity": "warning" if len(duplicate_subjects) > 0 else "info",
            "notes": "Current CSV contains duplicated subject_id values.",
        },
        {
            "flag": "chol_equal_zero",
            "value": int((df["chol"] == 0).sum()),
            "severity": "warning" if int((df["chol"] == 0).sum()) > 0 else "info",
            "notes": "Zero cholesterol values are likely placeholders or missing-code artifacts in some sites.",
        },
        {
            "flag": "trestbps_equal_zero",
            "value": int((df["trestbps"] == 0).sum()),
            "severity": "warning" if int((df["trestbps"] == 0).sum()) > 0 else "info",
            "notes": "Zero resting blood pressure is likely non-physiological.",
        },
        {
            "flag": "oldpeak_negative",
            "value": int((df["oldpeak"] < 0).sum()),
            "severity": "review" if int((df["oldpeak"] < 0).sum()) > 0 else "info",
            "notes": "Negative oldpeak values exist; these may be valid depending on preprocessing choice but deserve review.",
        },
        {
            "flag": "ca_equal_9",
            "value": int((df["ca"] == 9).sum()),
            "severity": "warning" if int((df["ca"] == 9).sum()) > 0 else "info",
            "notes": "Value 9 is outside the usual low-cardinality range for ca and should be audited.",
        },
        {
            "flag": "slope_equal_0",
            "value": int((df["slope"] == 0).sum()),
            "severity": "review" if int((df["slope"] == 0).sum()) > 0 else "info",
            "notes": "Slope usually appears as low-cardinality coded values; zero may be a remapped or placeholder code.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(
    df: pd.DataFrame,
    round_cols: dict[str, int] | None = None,
    max_rows: int | None = None,
) -> list[str]:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    if round_cols:
        for column, digits in round_cols.items():
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda x: round(float(x), digits) if pd.notna(x) else x
                )

    df = df.fillna("NA")
    headers = [str(col) for col in df.columns]
    rows = [
        [str(value) for value in row] for row in df.itertuples(index=False, name=None)
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


def build_report(
    df: pd.DataFrame,
    schema: dict,
    qc: dict,
    variable_inventory: pd.DataFrame,
    domain_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    quality_flags: pd.DataFrame,
) -> str:
    duplicate_subjects = df.loc[df["subject_id"].duplicated(keep=False)].sort_values(
        "subject_id"
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    high_missing = variable_inventory.loc[
        variable_inventory["missing_pct"] >= 20.0, ["column_name", "missing_pct"]
    ]
    total_missing_cells = int(df.isna().sum().sum())

    lines: list[str] = [
        "# UCI Heart Disease quality report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Dataset audited",
        f"- Input CSV: `{INPUT_CSV}`",
        f"- Schema JSON: `{SCHEMA_JSON}`",
        f"- Quality-check JSON: `{QUALITY_JSON}`",
        f"- Rows: `{len(df)}`",
        f"- Columns: `{df.shape[1]}`",
        f"- Unique subject_id values: `{df['subject_id'].nunique(dropna=True)}`",
        f"- Domains/sites: `{df['domain_id'].nunique(dropna=True)}`",
        f"- Binary target prevalence: `{df['target'].mean():.4f}`",
        f"- Total missing cells: `{total_missing_cells}`",
        "",
        "## What this file contains",
        "- A plain-language description of the current analysis-ready UCI dataset.",
        "- Variable roles, semantic types, and practical modeling notes.",
        "- Domain-level class balance and domain-specific quality asymmetries.",
        "- Attention points that may matter for transportability and future harmonization.",
        "",
        "## Variable groups",
        f"- Identifier columns: `{', '.join(schema['id_columns'])}`",
        f"- Metadata columns: `{', '.join(schema['metadata_columns'])}`",
        f"- Target columns: `{schema['target_column']}`, `{schema['target_original_column']}`",
        f"- Predictive numeric features: `{', '.join(schema['numeric_features'])}`",
        f"- Predictive categorical-coded features: `{', '.join(schema['categorical_features'])}`",
        "",
        "## Dataset content summary",
        "- `subject_id` is a harmonized row identifier.",
        "- `domain_id` identifies the acquisition/collection site and is central for cross-site experiments.",
        "- `target_original` preserves the original 0-4 disease severity coding from UCI.",
        "- `target` is the binary label currently used downstream, with class 1 formed by collapsing original severities 1-4.",
        "- The predictive table contains 13 features: 5 continuous numeric variables and 8 low-cardinality variables stored as numeric codes.",
        "",
        "## Domain summary",
        "",
    ]
    lines.extend(markdown_table(domain_summary, round_cols={"positive_prevalence": 4}))
    lines.extend(
        [
            "",
            "## High-missingness columns",
            "",
        ]
    )
    lines.extend(markdown_table(high_missing, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Numeric feature summary",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            numeric_summary,
            round_cols={
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
            "## Categorical-coded feature summary",
            "",
        ]
    )
    lines.extend(markdown_table(categorical_summary))
    lines.extend(
        [
            "",
            "## Main quality findings",
            f"- The final analysis-ready CSV has `{int(df.duplicated().sum())}` exact duplicated rows, while the preprocessing QC JSON reported `{int(qc.get('exact_duplicate_rows_detected', 0))}` duplicates detected during QC.",
            f"- There is `{int(df['subject_id'].duplicated().sum())}` duplicated `subject_id` value in the final CSV, corresponding to `{duplicate_subjects['subject_id'].nunique()}` repeated identifier(s).",
            f"- `ca`, `thal`, and `slope` are the most incomplete columns, with missingness concentrated especially outside Cleveland.",
            f"- `chol` contains `{int((df['chol'] == 0).sum())}` zero values; these are heavily site-dependent and likely reflect missing or placeholder measurements rather than true physiology.",
            f"- `trestbps` contains `{int((df['trestbps'] == 0).sum())}` zero value, which is likely non-physiological.",
            f"- `oldpeak` contains `{int((df['oldpeak'] < 0).sum())}` negative values; these may or may not be acceptable depending on the preprocessing/clinical interpretation chosen downstream.",
            f"- `ca` contains `{int((df['ca'] == 9).sum())}` record with value 9 and `slope` contains `{int((df['slope'] == 0).sum())}` record with value 0, both worth auditing before strict transportability experiments.",
            f"- Class prevalence differs substantially by site: Cleveland `{df.loc[df['domain_id'] == 'cleveland', 'target'].mean():.3f}`, Hungarian `{df.loc[df['domain_id'] == 'hungarian', 'target'].mean():.3f}`, Long Beach VA `{df.loc[df['domain_id'] == 'long_beach_va', 'target'].mean():.3f}`, Switzerland `{df.loc[df['domain_id'] == 'switzerland', 'target'].mean():.3f}`.",
            "",
            "## Variable dictionary",
            "",
        ]
    )
    dict_view = variable_inventory[
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
    lines.extend(markdown_table(dict_view, round_cols={"missing_pct": 2}))
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
        ]
    )
    lines.extend(markdown_table(quality_flags))

    if not duplicate_subjects.empty:
        lines.extend(
            [
                "",
                "## Duplicated subject_id rows",
                "",
            ]
        )
        lines.extend(markdown_table(duplicate_subjects))

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "- This is already an analysis-ready dataset, not the raw UCI dump. Some quality issues were intentionally preserved rather than globally imputed or encoded, which is appropriate for leakage-safe downstream preprocessing.",
            "- Several low-cardinality variables are stored as numeric dtypes but are semantically categorical and should generally be modeled as such.",
            "- Domain-specific missingness and domain-specific placeholder patterns are strong candidates to affect transportability unless handled inside the training loop only.",
            "",
            "## Companion files",
            "- `UCI_Heart_quality_report.rdm`: requested human-readable report file.",
            "- `UCI_Heart_quality_report.md`: markdown twin of the same report for easier preview.",
            "- `uci_variable_inventory.csv`: per-column dictionary with roles, types, missingness, and notes.",
            "- `uci_domain_summary.csv`: site-level sample counts, prevalence, and selected quality markers.",
            "- `uci_numeric_summary.csv`: descriptive statistics for numeric features.",
            "- `uci_categorical_summary.csv`: low-cardinality feature levels and counts.",
            "- `uci_quality_flags.csv`: compact table of quality flags and attention points.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, schema, qc = load_inputs()

    variable_inventory = build_variable_inventory(df, schema)
    domain_summary = build_domain_summary(df)
    numeric_summary = build_numeric_summary(df, schema)
    categorical_summary = build_categorical_summary(df, schema)
    quality_flags = build_quality_flags(df, qc)

    variable_inventory.to_csv(OUTPUT_DIR / "uci_variable_inventory.csv", index=False)
    domain_summary.to_csv(OUTPUT_DIR / "uci_domain_summary.csv", index=False)
    numeric_summary.to_csv(OUTPUT_DIR / "uci_numeric_summary.csv", index=False)
    categorical_summary.to_csv(OUTPUT_DIR / "uci_categorical_summary.csv", index=False)
    quality_flags.to_csv(OUTPUT_DIR / "uci_quality_flags.csv", index=False)

    report_text = build_report(
        df=df,
        schema=schema,
        qc=qc,
        variable_inventory=variable_inventory,
        domain_summary=domain_summary,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        quality_flags=quality_flags,
    )
    (OUTPUT_DIR / "UCI_Heart_quality_report.rdm").write_text(
        report_text, encoding="utf-8"
    )
    (OUTPUT_DIR / "UCI_Heart_quality_report.md").write_text(
        report_text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
