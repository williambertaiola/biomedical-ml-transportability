"""Transportability experiment: main."""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

from src.io_load import discover_input_paths, load_tables, summarize_tables
from src.preprocess import (
    build_phenotypic_base,
    build_subject_level_dataset,
    compute_missingness_table,
    deduplicate_qap,
    infer_schema,
    select_model_features,
)
from src.report import generate_report
from src.splits import add_domain_columns, generate_splits
from src.utils import dump_json, ensure_dir, log, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABIDE II multi-site DG-ready pipeline"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config.yaml"
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text)
    else:
        # Fallback: support JSON-formatted YAML (valid YAML subset) when PyYAML is unavailable.
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must define a dictionary at top level.")
    return cfg


def save_dataframe_with_parquet_fallback(
    df: pd.DataFrame, parquet_path: Path, csv_path: Path
) -> None:
    try:
        df.to_parquet(parquet_path, index=False)
        log(f"Saved parquet: {parquet_path}")
    except Exception as exc:
        log(f"Parquet save failed ({exc}). Writing CSV only: {csv_path}")
        note_path = parquet_path.with_suffix(".parquet.unavailable.txt")
        note_path.write_text(
            "Parquet export unavailable in current environment. "
            "Install pyarrow or fastparquet and rerun.\n",
            encoding="utf-8",
        )
    df.to_csv(csv_path, index=False)
    log(f"Saved csv: {csv_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("project", {}).get("random_seed", 42))
    set_global_seed(seed)

    project_root = args.config.parent.resolve()
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = ensure_dir(output_dir)
    splits_dir = ensure_dir(output_dir / "splits")

    log(f"Project root: {project_root}")
    log(f"Output dir: {output_dir}")

    input_paths = discover_input_paths(config.get("paths", {}))
    tables = load_tables(input_paths)
    tables_renamed = {
        "phenotypic": tables["phenotypic"],
        "anat_qap": tables["anat_qap"],
        "dti_qap": tables["dti_qap"],
        "functional_qap": tables["functional_qap"],
    }

    schema = infer_schema(tables_renamed, config)

    file_overview = summarize_tables(input_paths, tables_renamed)
    file_overview.to_csv(output_dir / "file_overview.csv", index=False)

    phen_base = build_phenotypic_base(tables_renamed["phenotypic"], schema, config)
    phen_base = phen_base.dropna(subset=["subject_id"]).copy()

    dedup_mode = config.get("dedup", {}).get("mode", "keep_best_scan")
    dedup_mode = dedup_mode.lower().strip()
    if dedup_mode not in {"keep_best_scan", "aggregate"}:
        raise ValueError("dedup.mode must be one of: keep_best_scan, aggregate")

    anat_dedup, anat_audit = deduplicate_qap(
        qap_df=tables_renamed["anat_qap"],
        modality_key="anat_qap",
        modality_prefix="anat",
        mapping=schema.qap_mapping["anat_qap"],
        mode=dedup_mode,
    )
    dti_dedup, dti_audit = deduplicate_qap(
        qap_df=tables_renamed["dti_qap"],
        modality_key="dti_qap",
        modality_prefix="dti",
        mapping=schema.qap_mapping["dti_qap"],
        mode=dedup_mode,
    )
    func_dedup, func_audit = deduplicate_qap(
        qap_df=tables_renamed["functional_qap"],
        modality_key="functional_qap",
        modality_prefix="func",
        mapping=schema.qap_mapping["functional_qap"],
        mode=dedup_mode,
    )
    audit_df = pd.concat([anat_audit, dti_audit, func_audit], ignore_index=True)
    audit_df.to_csv(output_dir / "audit_duplicates.csv", index=False)

    subject_df = build_subject_level_dataset(
        phen_base=phen_base,
        dedup_tables={"anat": anat_dedup, "dti": dti_dedup, "func": func_dedup},
    )
    subject_df = subject_df.dropna(subset=["subject_id"]).copy()
    subject_df = subject_df.drop_duplicates(
        subset=["subject_id"], keep="first"
    ).reset_index(drop=True)

    missingness = compute_missingness_table(subject_df)
    missingness.to_csv(output_dir / "missingness_by_column.csv", index=False)

    model_cols, feature_selection = select_model_features(subject_df, config)
    feature_selection.to_csv(output_dir / "feature_selection.csv", index=False)

    subject_df = add_domain_columns(subject_df, config)
    model_cols_plus = [c for c in model_cols if c in subject_df.columns]
    for c in [
        "subject_id",
        "site_id",
        "diagnosis",
        "diagnosis_label",
        "domain_site",
        "domain_lodo",
    ]:
        if c in subject_df.columns and c not in model_cols_plus:
            model_cols_plus.append(c)
    model_df = subject_df[model_cols_plus].copy()

    save_dataframe_with_parquet_fallback(
        subject_df,
        parquet_path=output_dir / "subject_level_merged.parquet",
        csv_path=output_dir / "subject_level_merged.csv",
    )
    save_dataframe_with_parquet_fallback(
        model_df,
        parquet_path=output_dir / "cleaned_dataset.parquet",
        csv_path=output_dir / "cleaned_dataset.csv",
    )

    split_summary, split_files = generate_splits(
        dataset_df=model_df,
        model_feature_cols=model_cols_plus,
        config=config,
        output_splits_dir=splits_dir,
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    report_paths = generate_report(
        output_dir=output_dir,
        full_df=subject_df,
        model_df=model_df,
        tables={
            "file_overview": file_overview,
            "missingness": missingness,
            "feature_selection": feature_selection,
            "audit_duplicates": audit_df,
            "split_summary": split_summary,
        },
        config=config,
    )

    run_metadata = {
        "config_path": str(args.config.resolve()),
        "input_paths": {k: str(v) for k, v in input_paths.items()},
        "schema_mapping": {
            "phen_subject": schema.phen_subject,
            "phen_site": schema.phen_site,
            "phen_diagnosis": schema.phen_diagnosis,
            "phen_age": schema.phen_age,
            "phen_sex": schema.phen_sex,
            "qap_mapping": schema.qap_mapping,
        },
        "dedup_mode": dedup_mode,
        "n_subjects_final": int(len(model_df)),
        "n_sites_final": int(model_df["domain_site"].nunique())
        if "domain_site" in model_df.columns
        else 0,
        "split_files_n": len(split_files),
        "report_md": str(report_paths["report_md"]),
    }
    dump_json(output_dir / "run_metadata.json", run_metadata)

    log("Pipeline completed successfully.")
    log(f"Cleaned dataset: {output_dir / 'cleaned_dataset.parquet'}")
    log(f"Audit file: {output_dir / 'audit_duplicates.csv'}")
    log(f"Splits folder: {splits_dir}")
    log(f"Report: {report_paths['report_md']}")


if __name__ == "__main__":
    main()
