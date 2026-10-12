"""Transportability experiment: dataset loaders."""

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


from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from common_pipeline import resolve_existing_path

UCI_SOURCE_COLUMN = "source"
UCI_TARGET_COLUMN = "V58"
UCI_MISSING_TARGET_CODE = -9

ABIDE_SOURCE_COLUMN = "Site_ID"
ABIDE_LABEL_COLUMN = "Series"
ABIDE_TARGET_COLUMN = "mFD"

TCGA_SOURCE_COLUMN = "source"
TCGA_TARGET_COLUMN = "target"
TCGA_LABEL_COLUMN = "label"
TCGA_SAMPLE_TYPE_COLUMN = "sample_type"


def get_uci_input_path() -> Path:
    return resolve_existing_path(
        [
            _configured_path("legacy_input_uci.csv"),
            _configured_path("legacy_input_uci.csv"),
        ],
        label="dataset UCI",
    )


def get_abide2_input_path() -> Path:
    return resolve_existing_path(
        [
            _configured_path(
                "abide_raw_abideii_mri_quality_metrics_functional_qap.csv"
            ),
        ],
        label="dataset ABIDE II",
    )


def get_tcga_input_path() -> Path:
    return resolve_existing_path(
        [
            _configured_path("tcga_expression"),
            _configured_path("tcga_expression"),
            _configured_path("tcga_expression"),
            _configured_path("tcga_expression"),
        ],
        label="dataset TCGA-BRCA",
    )


@lru_cache(maxsize=1)
def load_uci_raw() -> pd.DataFrame:
    return pd.read_csv(get_uci_input_path())


@lru_cache(maxsize=1)
def load_abide2_raw() -> pd.DataFrame:
    return pd.read_csv(get_abide2_input_path())


@lru_cache(maxsize=1)
def load_tcga_brca_log_cpm() -> pd.DataFrame:
    raw_df = pd.read_csv(get_tcga_input_path(), sep="\t")
    if "Ensembl_ID" not in raw_df.columns:
        raise ValueError("La colonna 'Ensembl_ID' non è presente nel file TCGA.")

    grouped_df = raw_df.groupby("Ensembl_ID", sort=False, as_index=False).sum(
        numeric_only=True
    )
    matrix_df = grouped_df.set_index("Ensembl_ID").T
    matrix_df.index.name = "sample_id"
    matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")

    library_sizes = matrix_df.sum(axis=1).replace(0, np.nan)
    log_cpm = np.log1p(matrix_df.div(library_sizes, axis=0) * 1_000_000.0)
    log_cpm = log_cpm.astype("float32")
    return log_cpm


def prepare_uci_unsupervised() -> tuple[pd.DataFrame, list[str], dict]:
    df = load_uci_raw().copy()

    if UCI_SOURCE_COLUMN not in df.columns:
        raise ValueError(
            f"La colonna '{UCI_SOURCE_COLUMN}' non è presente nel dataset UCI."
        )

    numeric_features = [
        column
        for column in df.columns
        if column not in (UCI_SOURCE_COLUMN, UCI_TARGET_COLUMN)
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_features:
        raise ValueError("UCI non contiene feature numeriche utilizzabili.")

    work_df = df[[UCI_SOURCE_COLUMN] + numeric_features].copy()
    work_df = work_df.dropna(subset=[UCI_SOURCE_COLUMN]).copy()
    work_df[UCI_SOURCE_COLUMN] = work_df[UCI_SOURCE_COLUMN].astype(str)

    source_counts = work_df[UCI_SOURCE_COLUMN].value_counts()
    valid_sources = source_counts[source_counts >= 2].index
    work_df = work_df[work_df[UCI_SOURCE_COLUMN].isin(valid_sources)].copy()

    numeric_features = [
        column for column in numeric_features if work_df[column].notna().any()
    ]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    features_imputed = imputer.fit_transform(work_df[numeric_features])
    features_scaled = scaler.fit_transform(features_imputed)

    analysis_df = pd.DataFrame(
        features_scaled, columns=numeric_features, index=work_df.index
    )
    analysis_df[UCI_SOURCE_COLUMN] = work_df[UCI_SOURCE_COLUMN].astype("object")

    if UCI_TARGET_COLUMN in df.columns:
        labels = pd.to_numeric(
            df.loc[work_df.index, UCI_TARGET_COLUMN], errors="coerce"
        )
        valid_labels = labels.notna() & (labels != UCI_MISSING_TARGET_CODE)
        analysis_df["label"] = (
            labels.where(valid_labels).astype("Int64").astype("string")
        )

    metadata = {
        "input_path": str(get_uci_input_path()),
        "source_counts": work_df[UCI_SOURCE_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "n_samples": int(len(analysis_df)),
        "n_features": int(len(numeric_features)),
    }
    return analysis_df, numeric_features, metadata


def prepare_uci_supervised() -> tuple[pd.DataFrame, list[str], dict]:
    raw_df = load_uci_raw().copy()
    if UCI_SOURCE_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{UCI_SOURCE_COLUMN}' non è presente nel dataset UCI."
        )
    if UCI_TARGET_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{UCI_TARGET_COLUMN}' non è presente nel dataset UCI."
        )

    numeric_features = [
        column
        for column in raw_df.columns
        if column not in (UCI_SOURCE_COLUMN, UCI_TARGET_COLUMN)
        and pd.api.types.is_numeric_dtype(raw_df[column])
    ]
    if not numeric_features:
        raise ValueError("UCI non contiene feature numeriche utilizzabili.")

    model_df = raw_df[[UCI_SOURCE_COLUMN, UCI_TARGET_COLUMN] + numeric_features].copy()
    model_df = model_df.dropna(
        subset=[UCI_SOURCE_COLUMN, UCI_TARGET_COLUMN] + numeric_features
    ).copy()
    model_df = model_df[model_df[UCI_TARGET_COLUMN] != UCI_MISSING_TARGET_CODE].copy()
    model_df[UCI_SOURCE_COLUMN] = model_df[UCI_SOURCE_COLUMN].astype("object")
    model_df[UCI_TARGET_COLUMN] = pd.to_numeric(
        model_df[UCI_TARGET_COLUMN], errors="coerce"
    ).astype("int64")

    metadata = {
        "input_path": str(get_uci_input_path()),
        "source_counts": model_df[UCI_SOURCE_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "target_counts": model_df[UCI_TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "n_samples": int(len(model_df)),
        "n_features": int(len(numeric_features)),
    }
    return model_df, numeric_features, metadata


def prepare_abide2_unsupervised() -> tuple[pd.DataFrame, list[str], dict]:
    raw_df = load_abide2_raw().copy()
    if ABIDE_SOURCE_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{ABIDE_SOURCE_COLUMN}' non è presente nel dataset ABIDE II."
        )
    if ABIDE_LABEL_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{ABIDE_LABEL_COLUMN}' non è presente nel dataset ABIDE II."
        )

    feature_columns = raw_df.select_dtypes(include=["number"]).columns.tolist()
    if "Sub_ID" in feature_columns:
        feature_columns.remove("Sub_ID")
    if not feature_columns:
        raise ValueError("ABIDE II non contiene feature numeriche utilizzabili.")

    analysis_df = (
        raw_df[[ABIDE_SOURCE_COLUMN, ABIDE_LABEL_COLUMN] + feature_columns]
        .dropna(subset=[ABIDE_SOURCE_COLUMN, ABIDE_LABEL_COLUMN] + feature_columns)
        .copy()
    )
    analysis_df = analysis_df.rename(
        columns={
            ABIDE_SOURCE_COLUMN: "source",
            ABIDE_LABEL_COLUMN: "label",
        }
    )
    analysis_df["source"] = analysis_df["source"].astype("object")
    analysis_df["label"] = analysis_df["label"].astype("object")

    metadata = {
        "input_path": str(get_abide2_input_path()),
        "source_counts": analysis_df["source"].value_counts().sort_index().to_dict(),
        "label_counts": analysis_df["label"].value_counts().sort_index().to_dict(),
        "n_samples": int(len(analysis_df)),
        "n_features": int(len(feature_columns)),
    }
    return analysis_df, feature_columns, metadata


def prepare_abide2_supervised() -> tuple[pd.DataFrame, list[str], dict]:
    raw_df = load_abide2_raw().copy()
    if ABIDE_SOURCE_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{ABIDE_SOURCE_COLUMN}' non è presente nel dataset ABIDE II."
        )
    if ABIDE_TARGET_COLUMN not in raw_df.columns:
        raise ValueError(
            f"La colonna '{ABIDE_TARGET_COLUMN}' non è presente nel dataset ABIDE II."
        )

    numeric_features = raw_df.select_dtypes(include=["number"]).columns.tolist()
    if "Sub_ID" in numeric_features:
        numeric_features.remove("Sub_ID")
    if ABIDE_TARGET_COLUMN in numeric_features:
        numeric_features.remove(ABIDE_TARGET_COLUMN)
    if not numeric_features:
        raise ValueError(
            "ABIDE II non contiene feature numeriche utilizzabili per la regressione."
        )

    model_df = (
        raw_df[[ABIDE_SOURCE_COLUMN, ABIDE_TARGET_COLUMN] + numeric_features]
        .dropna(subset=[ABIDE_SOURCE_COLUMN, ABIDE_TARGET_COLUMN] + numeric_features)
        .copy()
    )
    model_df = model_df.rename(
        columns={ABIDE_SOURCE_COLUMN: "source", ABIDE_TARGET_COLUMN: "target"}
    )
    model_df["source"] = model_df["source"].astype("object")
    model_df["target"] = pd.to_numeric(model_df["target"], errors="coerce")
    model_df = model_df.dropna(subset=["target"]).copy()

    metadata = {
        "input_path": str(get_abide2_input_path()),
        "source_counts": model_df["source"].value_counts().sort_index().to_dict(),
        "n_samples": int(len(model_df)),
        "n_features": int(len(numeric_features)),
        "target_summary": model_df["target"].describe().to_dict(),
    }
    return model_df, numeric_features, metadata


def _extract_tcga_tss(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) < 2:
        raise ValueError(f"Barcode TCGA non valido: {sample_id}")
    return parts[1]


def _extract_tcga_sample_type(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) < 4:
        raise ValueError(f"Barcode TCGA non valido: {sample_id}")
    return parts[3][:2]


@lru_cache(maxsize=8)
def prepare_tcga_dataset(
    *,
    min_source_samples: int = 20,
    require_both_binary_classes: bool = True,
    min_binary_class_count: int = 4,
    global_top_variable_genes: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    expression_df = load_tcga_brca_log_cpm().copy()
    sample_ids = expression_df.index.to_series()

    source_values = sample_ids.map(_extract_tcga_tss)
    sample_type_values = sample_ids.map(_extract_tcga_sample_type)
    binary_target_values = sample_type_values.map(
        lambda value: "normal" if value == "11" else "tumor"
    )

    annotations_df = pd.DataFrame(
        {
            TCGA_SOURCE_COLUMN: source_values.to_numpy(dtype=object),
            TCGA_SAMPLE_TYPE_COLUMN: sample_type_values.to_numpy(dtype=object),
            TCGA_TARGET_COLUMN: binary_target_values.to_numpy(dtype=object),
            TCGA_LABEL_COLUMN: binary_target_values.to_numpy(dtype=object),
        },
        index=expression_df.index,
    )
    model_df = pd.concat([expression_df, annotations_df], axis=1)

    source_counts = model_df[TCGA_SOURCE_COLUMN].value_counts()
    valid_sources = source_counts[source_counts >= min_source_samples].index
    model_df = model_df[model_df[TCGA_SOURCE_COLUMN].isin(valid_sources)].copy()

    if require_both_binary_classes:
        crosstab = pd.crosstab(
            model_df[TCGA_SOURCE_COLUMN], model_df[TCGA_TARGET_COLUMN]
        )
        if "tumor" not in crosstab.columns:
            crosstab["tumor"] = 0
        if "normal" not in crosstab.columns:
            crosstab["normal"] = 0
        balanced_sources = crosstab[
            (crosstab["tumor"] >= min_binary_class_count)
            & (crosstab["normal"] >= min_binary_class_count)
        ].index
        model_df = model_df[model_df[TCGA_SOURCE_COLUMN].isin(balanced_sources)].copy()

    feature_columns = [
        column
        for column in model_df.columns
        if column
        not in (
            TCGA_SOURCE_COLUMN,
            TCGA_TARGET_COLUMN,
            TCGA_LABEL_COLUMN,
            TCGA_SAMPLE_TYPE_COLUMN,
        )
    ]
    if (
        global_top_variable_genes is not None
        and len(feature_columns) > global_top_variable_genes
    ):
        variances = model_df[feature_columns].var(axis=0)
        feature_columns = variances.nlargest(global_top_variable_genes).index.tolist()
        model_df = model_df[
            feature_columns
            + [
                TCGA_SOURCE_COLUMN,
                TCGA_TARGET_COLUMN,
                TCGA_LABEL_COLUMN,
                TCGA_SAMPLE_TYPE_COLUMN,
            ]
        ].copy()

    for column in (
        TCGA_SOURCE_COLUMN,
        TCGA_TARGET_COLUMN,
        TCGA_LABEL_COLUMN,
        TCGA_SAMPLE_TYPE_COLUMN,
    ):
        model_df[column] = model_df[column].astype("object")

    metadata = {
        "input_path": str(get_tcga_input_path()),
        "source_counts": model_df[TCGA_SOURCE_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "sample_type_counts": model_df[TCGA_SAMPLE_TYPE_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "binary_target_counts": model_df[TCGA_TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "n_samples": int(len(model_df)),
        "n_features": int(len(feature_columns)),
    }
    return model_df, feature_columns, metadata
