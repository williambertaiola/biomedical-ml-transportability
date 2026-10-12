"""Transportability experiment: datasets."""

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


import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_environment

PROJECT_ROOT = bootstrap_environment()

import numpy as np
import pandas as pd


ABIDE_DATA_DIR = Path(_configured_path("abide_raw_root"))
ABIDE_PHENO_FILE = Path(_configured_path("abide_raw_abideii_composite_phenotypic.csv"))
ABIDE_QAP_DIR = ABIDE_DATA_DIR / "ABIDEII_MRI_Quality_Metrics"

UCI_INPUT_FILE = Path(_configured_path("legacy_input_uci_preprocessed.csv"))
TCGA_INPUT_FILE = Path(_configured_path("tcga_expression"))


@dataclass
class DatasetBundle:
    dataset_id: str
    display_name: str
    data: pd.DataFrame
    sample_id_col: str
    domain_col: str
    label_col: str
    feature_columns: list[str]
    harmonizable_features: list[str]
    passthrough_features: list[str]
    positive_label_name: str
    domain_kind: str
    domain_real: bool
    harmonization_method: str
    harmonization_allowed: bool
    harmonization_reason: str
    interpretability_note: str
    audit: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _clean_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    cleaned: list[str] = []
    for col in columns:
        key = re.sub(r"[^0-9A-Za-z]+", "_", str(col).strip()).strip("_").lower() or "x"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = f"{key}_{seen[key]}"
        cleaned.append(key)
    return cleaned


def _make_unique(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for value in values:
        seen[value] = seen.get(value, 0) + 1
        out.append(value if seen[value] == 1 else f"{value}__dup{seen[value]}")
    return out


def _read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    last_exc: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Cannot read {path}: {last_exc}")


def _safe_numeric_summary(df_mod: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df_mod.copy()
    df.columns = _clean_columns(df.columns.tolist())
    if "site_id" not in df.columns or "sub_id" not in df.columns:
        raise RuntimeError(f"{prefix} summary requires site_id and sub_id columns.")
    keep = ["site_id", "sub_id"]
    out = df[keep].copy()
    for column in df.columns:
        if column in keep:
            continue
        if column.startswith("unnamed") or column in {
            "session",
            "series",
            "unique_id",
            "unique_id_1",
        }:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            out[f"{prefix}_{column}"] = numeric
    return out.groupby(["site_id", "sub_id"], as_index=False).mean(numeric_only=True)


def load_abide2_dataset() -> DatasetBundle:
    if not ABIDE_PHENO_FILE.exists():
        raise FileNotFoundError(f"ABIDE II phenotypic file missing: {ABIDE_PHENO_FILE}")

    phen = _read_table(ABIDE_PHENO_FILE, sep=",")
    phen.columns = _clean_columns(phen.columns.tolist())
    required = {"site_id", "sub_id", "dx_group"}
    missing = required.difference(phen.columns)
    if missing:
        raise RuntimeError(
            f"ABIDE II phenotypic file missing columns: {sorted(missing)}"
        )

    phen = phen.copy()
    phen["site_id"] = phen["site_id"].astype("string")
    phen["sub_id"] = pd.to_numeric(phen["sub_id"], errors="coerce")
    phen["dx_group"] = pd.to_numeric(phen["dx_group"], errors="coerce")
    phen = phen.loc[
        phen["dx_group"].isin([1, 2]) & phen["site_id"].notna() & phen["sub_id"].notna()
    ].copy()
    phen["label"] = (phen["dx_group"] == 1).astype("int64")
    phen["diagnosis_label"] = phen["label"].map({1: "ASD", 0: "Control"})

    anat = _safe_numeric_summary(
        _read_table(ABIDE_QAP_DIR / "anat_qap.csv", sep=","), "anat"
    )
    dti = _safe_numeric_summary(
        _read_table(ABIDE_QAP_DIR / "dti_qap.csv", sep=","), "dti"
    )
    func = _safe_numeric_summary(
        _read_table(ABIDE_QAP_DIR / "functional_qap.csv", sep=","), "func"
    )

    merged = phen.merge(anat, on=["site_id", "sub_id"], how="left")
    merged = merged.merge(dti, on=["site_id", "sub_id"], how="left")
    merged = merged.merge(func, on=["site_id", "sub_id"], how="left")

    summary = (
        merged.groupby("site_id", dropna=False)["label"]
        .agg(
            n_samples="size",
            n_asd=lambda s: int((s == 1).sum()),
            n_ctrl=lambda s: int((s == 0).sum()),
        )
        .reset_index()
        .sort_values(["n_samples", "site_id"], ascending=[False, True])
    )
    eligible = summary.loc[(summary["n_asd"] >= 15) & (summary["n_ctrl"] >= 15)].copy()
    if len(eligible) < 2:
        raise RuntimeError(
            "ABIDE II does not have enough eligible multi-site domains for LODO."
        )
    selected_sites = eligible.head(14)["site_id"].astype(str).tolist()
    merged = merged.loc[merged["site_id"].isin(selected_sites)].copy()

    merged["sample_id"] = merged.apply(
        lambda row: f"abide2_{row['site_id']}_{int(row['sub_id'])}", axis=1
    )
    merged["domain"] = merged["site_id"].astype(str)

    sex_raw = (
        merged["sex"]
        if "sex" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    sex_num = pd.to_numeric(sex_raw, errors="coerce")
    merged["sex_binary"] = np.where(
        sex_num == 1, 1.0, np.where(sex_num == 2, 0.0, np.nan)
    )

    base_features = [
        c
        for c in ("age_at_scan", "fiq", "viq", "piq", "handedness_scores")
        if c in merged.columns
    ]
    qap_features = [c for c in merged.columns if re.match(r"^(anat|dti|func)_", c)]
    feature_columns = ["sex_binary", *base_features, *qap_features]
    feature_columns = [c for c in feature_columns if c in merged.columns]
    harmonizable_features = qap_features
    passthrough = [c for c in feature_columns if c not in harmonizable_features]

    data = merged[["sample_id", "domain", "label", *feature_columns]].copy()
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["label"] = data["label"].astype("int64")

    audit = {
        "files": {
            "phenotypic_file": str(ABIDE_PHENO_FILE),
            "anat_qap": str(ABIDE_QAP_DIR / "anat_qap.csv"),
            "dti_qap": str(ABIDE_QAP_DIR / "dti_qap.csv"),
            "functional_qap": str(ABIDE_QAP_DIR / "functional_qap.csv"),
        },
        "selected_sites": selected_sites,
        "site_counts": data["domain"].value_counts().sort_index().to_dict(),
        "class_counts": data["label"].value_counts().sort_index().to_dict(),
        "n_samples": int(len(data)),
        "n_features": int(len(feature_columns)),
    }

    return DatasetBundle(
        dataset_id="abide2",
        display_name="ABIDE II",
        data=data,
        sample_id_col="sample_id",
        domain_col="domain",
        label_col="label",
        feature_columns=feature_columns,
        harmonizable_features=harmonizable_features,
        passthrough_features=passthrough,
        positive_label_name="ASD",
        domain_kind="site",
        domain_real=True,
        harmonization_method="neuroCombat",
        harmonization_allowed=True,
        harmonization_reason="Site is a real multi-site acquisition domain and the selected ABIDE II features are continuous MRI-derived metrics.",
        interpretability_note="Cross-site transportability is interpretable as a real multi-site MRI benchmark. Harmonization is applied only on source domains; strict LODO prevents exact ComBat correction on the unseen holdout site without leakage.",
        audit=audit,
        extra={"abide_site_min_per_class": 15, "abide_max_sites": 14},
    )


def load_uci_dataset() -> DatasetBundle:
    df = _read_table(UCI_INPUT_FILE, sep=",")
    if "source" not in df.columns or "V58" not in df.columns:
        raise RuntimeError("UCI input must contain 'source' and 'V58'.")

    work = df.copy()
    work["label"] = pd.to_numeric(work["V58"], errors="coerce")
    work = work.loc[work["label"].isin([0, 1])].copy()
    work["domain"] = work["source"].astype("string")
    work["sample_id"] = [f"uci_{idx:04d}" for idx in range(1, len(work) + 1)]

    feature_columns = [
        c
        for c in work.columns
        if c not in {"source", "V58", "label", "domain", "sample_id"}
    ]
    numeric_features = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(work[c])
    ]
    work[numeric_features] = work[numeric_features].replace({-9: np.nan, -99: np.nan})

    dropped: dict[str, str] = {}
    kept_features: list[str] = []
    for column in numeric_features:
        series = work[column]
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = float(n_unique / max(len(series), 1))
        missing_rate = float(series.isna().mean())
        if n_unique < 2:
            dropped[column] = "constant_after_sentinel_cleanup"
        elif unique_ratio > 0.70 and n_unique > 100:
            dropped[column] = "quasi_identifier"
        elif missing_rate > 0.80:
            dropped[column] = "too_many_missing_values"
        else:
            kept_features.append(column)

    if len(kept_features) < 5:
        raise RuntimeError("Too few usable UCI features after cleaning.")

    harmonizable_features: list[str] = []
    passthrough_features: list[str] = []
    for column in kept_features:
        n_unique = int(work[column].nunique(dropna=True))
        if n_unique >= 10:
            harmonizable_features.append(column)
        else:
            passthrough_features.append(column)

    data = work[["sample_id", "domain", "label", *kept_features]].copy()
    for column in kept_features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["label"] = data["label"].astype("int64")

    audit = {
        "files": {"input_file": str(UCI_INPUT_FILE)},
        "site_counts": data["domain"].value_counts().sort_index().to_dict(),
        "class_counts": data["label"].value_counts().sort_index().to_dict(),
        "n_samples": int(len(data)),
        "n_features_kept": int(len(kept_features)),
        "n_features_harmonized": int(len(harmonizable_features)),
        "dropped_columns": dropped,
    }

    return DatasetBundle(
        dataset_id="uci_heart",
        display_name="UCI Heart Disease",
        data=data,
        sample_id_col="sample_id",
        domain_col="domain",
        label_col="label",
        feature_columns=kept_features,
        harmonizable_features=harmonizable_features,
        passthrough_features=passthrough_features,
        positive_label_name="heart_disease",
        domain_kind="center",
        domain_real=True,
        harmonization_method="ComBat_standard",
        harmonization_allowed=True,
        harmonization_reason="The merged dataset exposes a real center/site variable through 'source'; ComBat is restricted to continuous-like numeric features and leaves low-cardinality coded variables untouched.",
        interpretability_note="Cross-center transportability is real because 'source' encodes actual collection centers. Harmonization should be interpreted cautiously because UCI is mixed clinical tabular data rather than a homogeneous assay matrix.",
        audit=audit,
        extra={"sentinel_values": [-9, -99]},
    )


def load_tcga_dataset() -> DatasetBundle:
    raw = _read_table(TCGA_INPUT_FILE, sep="\t")
    gene_col = raw.columns[0]
    gene_names = _make_unique(
        raw[gene_col].astype("string").fillna("missing_gene").tolist()
    )
    sample_ids = raw.columns[1:].tolist()
    values = raw.iloc[:, 1:].to_numpy(dtype=np.float32).T
    del raw
    gc.collect()

    expression = pd.DataFrame(values, columns=gene_names, index=sample_ids)
    del values
    gc.collect()

    frac_non_integer = float(
        np.mean(
            np.abs(
                expression.to_numpy(dtype=np.float32)
                - np.round(expression.to_numpy(dtype=np.float32))
            )
            > 1e-8
        )
    )
    input_is_raw_counts = frac_non_integer < 0.01

    metadata = pd.DataFrame({"sample_id": expression.index.astype(str)})
    split_parts = metadata["sample_id"].str.split("-", expand=True)
    metadata["domain"] = split_parts[1].astype("string")
    metadata["sample_type_code"] = split_parts[3].str.slice(0, 2)
    metadata["label"] = np.where(
        metadata["sample_type_code"].isin(["01", "06"]),
        1,
        np.where(metadata["sample_type_code"] == "11", 0, np.nan),
    )
    metadata = metadata.loc[
        metadata["label"].notna() & metadata["domain"].notna()
    ].copy()
    metadata["label"] = metadata["label"].astype("int64")

    domain_table = (
        metadata.groupby("domain", dropna=False)["label"]
        .agg(
            n_samples="size",
            n_tumor=lambda s: int((s == 1).sum()),
            n_normal=lambda s: int((s == 0).sum()),
        )
        .reset_index()
        .sort_values(["n_samples", "domain"], ascending=[False, True])
    )
    eligible = domain_table.loc[
        (domain_table["n_samples"] >= 20)
        & (domain_table["n_tumor"] >= 4)
        & (domain_table["n_normal"] >= 4)
    ].copy()
    if len(eligible) < 2:
        raise RuntimeError(
            "TCGA-BRCA does not have enough eligible TSS domains with both classes."
        )
    selected_domains = eligible["domain"].astype(str).tolist()

    metadata = metadata.loc[metadata["domain"].isin(selected_domains)].copy()
    expression = expression.loc[metadata["sample_id"]].copy()
    expression.reset_index(drop=True, inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    data = pd.concat(
        [metadata[["sample_id", "domain", "label"]], expression.reset_index(drop=True)],
        axis=1,
    )
    feature_columns = expression.columns.tolist()

    harmonization_allowed = bool(input_is_raw_counts)
    harmonization_reason = (
        "Raw integer counts detected; ComBat-seq is appropriate on source domains using TSS as the real center proxy."
        if harmonization_allowed
        else "Current local matrix is not a raw integer count matrix, so exact ComBat-seq would be methodologically invalid and is skipped."
    )

    audit = {
        "files": {"input_file": str(TCGA_INPUT_FILE)},
        "n_samples": int(len(data)),
        "n_features": int(len(feature_columns)),
        "domain_counts": data["domain"].value_counts().sort_index().to_dict(),
        "class_counts": data["label"].value_counts().sort_index().to_dict(),
        "fraction_non_integer_values": frac_non_integer,
        "input_is_raw_counts": input_is_raw_counts,
        "selected_domains": selected_domains,
        "sample_type_counts": metadata["sample_type_code"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }

    return DatasetBundle(
        dataset_id="tcga_brca",
        display_name="TCGA-BRCA RNA-seq",
        data=data,
        sample_id_col="sample_id",
        domain_col="domain",
        label_col="label",
        feature_columns=feature_columns,
        harmonizable_features=feature_columns,
        passthrough_features=[],
        positive_label_name="tumor",
        domain_kind="tissue_source_site",
        domain_real=True,
        harmonization_method="ComBat_seq",
        harmonization_allowed=harmonization_allowed,
        harmonization_reason=harmonization_reason,
        interpretability_note="The domain is the real TCGA Tissue Source Site (TSS) extracted from the barcode. This is interpretable as cross-center transportability, not as sequencing-batch transportability. Exact ComBat-seq is only valid if the input matrix contains raw integer counts.",
        audit=audit,
        extra={
            "tcga_input_is_raw_counts": input_is_raw_counts,
            "tcga_frac_non_integer": frac_non_integer,
        },
    )


def load_requested_datasets(dataset_ids: list[str]) -> list[DatasetBundle]:
    loaders = {
        "abide2": load_abide2_dataset,
        "uci_heart": load_uci_dataset,
        "tcga_brca": load_tcga_dataset,
    }
    bundles: list[DatasetBundle] = []
    for dataset_id in dataset_ids:
        if dataset_id not in loaders:
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        bundles.append(loaders[dataset_id]())
    return bundles


def save_audit(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(bundle.audit, indent=2), encoding="utf-8"
    )
