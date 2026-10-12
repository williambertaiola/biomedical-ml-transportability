"""Transportability experiment: run dashi tcga brca rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run rebuilt DASHI pipeline on TCGA-BRCA and save outputs."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to TCGA-BRCA root folder containing TCGA-BRCA.star_counts.tsv",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/tcga_brca_dashi_rebuild",
        help="Directory where outputs will be saved",
    )
    args = parser.parse_args()

    from src.dashi_rmd_backend_tcga_brca import (
        dashi_generate_and_store,
        resolve_data_root,
    )

    data_root = resolve_data_root(args.data_root)
    output_root = Path(args.output_root)

    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Output root: {output_root}")

    ok = dashi_generate_and_store(data_root=data_root, output_root=output_root)
    if not ok:
        print("[ERROR] Pipeline did not complete successfully.")
        return 1

    tables_dir = output_root / "tables"
    data_dir = output_root / "data"

    artifacts_path = tables_dir / "artifacts.json"
    summary_path = tables_dir / "preprocessing_summary.csv"
    dashi_ready_path = data_dir / "tcga_brca_dashi_ready.csv"

    print("[INFO] Pipeline completed.")
    if summary_path.exists():
        print(f"[INFO] Summary table: {summary_path}")
    if dashi_ready_path.exists():
        print(f"[INFO] DASHI-ready dataset: {dashi_ready_path}")

    if artifacts_path.exists():
        with artifacts_path.open("r", encoding="utf-8") as f:
            artifacts = json.load(f)
        print("[INFO] Key artifacts:")
        for k, v in artifacts.items():
            print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
