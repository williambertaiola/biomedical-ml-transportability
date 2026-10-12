#!/usr/bin/env bash
# Preserve the experiment order of the original preprocessing launcher.
set -euo pipefail
REPOSITORY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPOSITORY_DIR"
PYTHON_EXECUTABLE="${THESIS_PYTHON:-python3}"
"$PYTHON_EXECUTABLE" preprocessing/abide2/run_abide2_composite_unified.py
"$PYTHON_EXECUTABLE" preprocessing/abide2/run_abide2_transportability_rich.py
"$PYTHON_EXECUTABLE" preprocessing/abide2/run_standard_plots.py
"$PYTHON_EXECUTABLE" preprocessing/uci_heart/run_pipeline.py
"$PYTHON_EXECUTABLE" preprocessing/uci_heart/run_standard_plots.py
"$PYTHON_EXECUTABLE" preprocessing/tcga_brca/run_tcga_brca_initial_audit.py
"$PYTHON_EXECUTABLE" preprocessing/tcga_brca/run_tcga_brca_stage_ready_preprocessing.py
