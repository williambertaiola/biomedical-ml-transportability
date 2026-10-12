# Cross-Site Transportability in Biomedical Machine Learning: Evaluation and Adaptation Strategies

## Overview

Research code from William Bertaiola's Master's Thesis in Bioengineering at the University of Padua, developed in collaboration with the Universitat Politècnica de València. The project evaluates how biomedical classifiers transfer between institutions and how model adaptation changes external performance.

The code covers UCI Heart Disease, ABIDE II tabular phenotype data, and TCGA-BRCA expression and clinical stage data. It contains multiple experimental generations. The organized M0–M4 pipelines are the starting point; earlier CORAL, pseudo-labeling, alternative TCGA targets, and DASHI workflows remain available under `experiments/legacy/` and `analysis/`.

## Research objectives

- Compare in-distribution (ID) validation with out-of-distribution (OOD) target-site evaluation.
- Evaluate baseline, ComBat, source-domain weighting, and labeled target adaptation strategies.
- Examine site differences, metadata completeness, sample size, entropy, and their associations with performance.
- Preserve and inspect repeated seeded evaluations, method comparisons, and SHAP analyses.

## Datasets

| Dataset | Organized experiment | Required inputs |
| --- | --- | --- |
| UCI Heart Disease | Binary disease classification across Cleveland, Hungarian, Long Beach VA, and Switzerland domains | Analysis-ready CSV and feature schema JSON |
| ABIDE II | ASD/control prediction from the strict tabular phenotype variant | Strict subject-level CSV and preprocessing blueprint JSON |
| TCGA-BRCA | Stage I–II versus Stage III–IV prediction across tissue source sites | Stage transportability CSV with `ENSG` gene columns |

No subject-level data, fitted models, thesis PDF, or generated study figures are distributed here. [Dataset instructions](docs/DATASETS.md) document the raw and processed formats, official sources, and known provenance gaps. Public availability is not treated as blanket redistribution permission.

## Methodology

The organized pipelines implement:

| Method | Implementation |
| --- | --- |
| M0 | Baseline source model |
| M1 | ComBat adjustment of training features |
| M2 | Source-domain weighting from pairwise source transfer performance |
| M3 | Blend source-model probabilities with an adapter trained on the labeled target training partition |
| M4 | ComBat plus the target adapter used in M3 |

The estimators are logistic regression, random forest, and histogram gradient boosting. Evaluation includes ROC-AUC, F1, balanced accuracy, expected calibration error (ECE), and ID–OOD gaps. TCGA additionally selects operating thresholds from ID predictions and applies training-only gene selection. TCGA and phenotype pipelines retain different model settings.

M3 in these pipelines uses target labels. The legacy CORAL and pseudo-labeling implementations are separate experiments. The preserved ComBat implementation adjusts training matrices while leaving held-out matrices unharmonized before ordinary model preprocessing; this behavior is intentional preservation of the thesis code, not a claim of conventional out-of-sample ComBat. See [methodology and execution details](docs/EXPERIMENTS.md).

## Repository structure

```text
TFM_Bertaiola/
├── README.md
├── REFACTORING_REPORT.md
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
├── constraints-original.txt
├── constraints-tested.txt
├── pyproject.toml
├── .gitignore
├── configs/              # Portable paths, schemas, and sanitized recorded runs
├── src/
│   ├── uci_transportability/
│   ├── abide_transportability/
│   ├── tcga_transportability/
│   └── thesis_tools/     # Shared metrics and path configuration
├── preprocessing/        # Separate ABIDE, UCI, and TCGA preprocessing packages
├── experiments/
│   ├── uci/              # Organized runners and previous baseline implementation
│   ├── abide/            # Organized runners and previous pipeline
│   ├── tcga/             # Organized runners and reporting variants
│   └── legacy/           # Distinct earlier experiments, DASHI modules, and R code
├── analysis/             # Dataset-specific and comparative post-experiment analyses
├── scripts/              # Preprocessing launcher and archive comparison tool
├── tests/                # Scientific invariants, paths, provenance and hygiene
├── docs/                 # Architecture, dependencies, inventories and validation
├── data/README.md        # Instructions only; actual inputs remain ignored
└── results/.gitkeep      # Generated results remain ignored
```

The [code index](docs/CODE_INDEX.md) describes every original Python script and its final location. The [full tree](docs/REPOSITORY_TREE.md) lists all final files. No notebooks were present in the archive.

## Installation

After publishing your reviewed copy, clone your chosen repository URL:

```bash
git clone "$REPOSITORY_URL" TFM_Bertaiola
cd TFM_Bertaiola
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m pip install -e . --no-deps
```

Set `REPOSITORY_URL` to the URL you create yourself. This project has no configured remote. Python syntax requires 3.10 or later; validation was performed on Python 3.12.14. The observed validation constraints reproduce the tested dependency versions. `constraints-original.txt` preserves the four versions found in archived run metadata, but is not a complete original environment lockfile.

For tests, install `requirements-dev.txt` with the same constraints. For SHAP, extra estimators, and archived analyses, install `requirements-optional.txt`. DASHI requires identifying the original compatible distribution or source revision; it is deliberately not resolved to an arbitrary similarly named package. See [dependencies](docs/DEPENDENCIES.md).

## Configuring dataset paths

Run the examples below from the repository root. Place authorized raw inputs under `data/raw/uci/`, `data/raw/abide2/`, and `data/raw/tcga_brca/`, following [DATASETS.md](docs/DATASETS.md). Generated preprocessing bundles go to `results/Data/` using the original bundle names.

`configs/paths.json` contains portable repository-relative defaults. To keep data and outputs elsewhere:

```bash
export THESIS_DATA_ROOT="$PRIVATE_DATA_DIRECTORY"
export THESIS_RESULTS_ROOT="$PRIVATE_RESULTS_DIRECTORY"
```

Define those two directory variables first. The override roots replace `data/` and `results/` prefixes. A complete custom path registry can be selected with `THESIS_PATHS_FILE`; use `configs/paths.local.json` for private overrides. Copy the full default registry before editing individual entries. Preprocessing JSON templates resolve relative paths against the repository, even when invoked from another working directory. Experiment CLI input/output flags override their defaults.

## Usage

Prepare data with the appropriate pipeline:

```bash
python preprocessing/uci_heart/run_pipeline.py
python preprocessing/abide2/run_abide2_composite_unified.py
python preprocessing/abide2/run_abide2_transportability_rich.py
python preprocessing/tcga_brca/run_tcga_brca_initial_audit.py
python preprocessing/tcga_brca/run_tcga_brca_stage_ready_preprocessing.py
```

The ABIDE rich-processing runner writes both strict and rich variants. Each runner supports its own `--config` flag. `bash scripts/run_all_preprocessing.sh` preserves the original sequence when all raw inputs are available.

Run the organized pipelines on locally obtained or regenerated processed data:

```bash
python experiments/uci/bootstrap/run_m0_baseline.py \
  --methods M0,M1,M2,M3,M4 --seed 42 \
  --output-dir results/UCI/Methods/UCI_Heart_M4_ComBat_bootstrap_outputs

python experiments/abide/bootstrap/run_abide_baseline.py \
  --methods M0,M1,M2,M3,M4 --seed 42 \
  --output-dir results/ABIDE/Methods/ABIDE2_M4_ComBat_transductive_bootstrap10_outputs

python experiments/tcga/bootstrap/run_tcga_baseline.py \
  --methods M0,M1,M2,M3,M4 --seed 42 \
  --output-dir results/TCGA/Methods/TCGA_BRCA_M4_ComBat_transductive_bootstrap10_outputs
```

These historical output names are retained for downstream compatibility; a directory name containing `bootstrap10` does not establish that ten iterations ran. The baseline examples above run a single seeded evaluation. `run_m0_baseline.py` is the historical UCI filename and supports all five methods. ABIDE and TCGA baseline CLI defaults enable only M0; UCI enables M0–M4. Explicit method lists avoid ambiguity.

For repeated evaluations, use a bootstrap runner and state the count explicitly:

```bash
python experiments/uci/bootstrap/run_uci_bootstrap_resampling.py \
  --methods M0,M1,M2,M3,M4 --n-bootstrap 10 --seed 42 --seed-step 1 \
  --output-dir results/UCI/Methods/UCI_Heart_M4_ComBat_bootstrap_outputs
```

The analogous ABIDE and TCGA runners are documented in [EXPERIMENTS.md](docs/EXPERIMENTS.md). They repeat seeded partitions and evaluations; do not interpret them as case-resampling bootstrap confidence intervals. All runners expose `--help`, and processed inputs can be supplied through `--input` and, where applicable, `--schema`.

After results exist:

```bash
python analysis/uci/entropy/generate_entropy_analysis.py \
  --input-root results/UCI --output-dir results/UCI/Entropy
python analysis/uci/site_distance/generate_site_distance_analysis.py \
  --input-root results/UCI --output-dir results/UCI/Site_distance
```

Some later thesis analyses require several method bundles and historical summary tables. Their requirements are listed in the code index and execution guide. They have not all been rerun.

## Reproducibility

Core experiment seed defaults remain 42; the UCI preprocessing seed remains 20260327. Split ratios, method weights, estimators, feature selection, and metric implementations retain their original values. Saved run snapshots record selected methods and seeds. Configured filesystem locations have changed; numerical changes have not been introduced into the core calculations.

Feasible checks are reproducible with:

```bash
python -m pytest -q
ruff check .
ruff format --check .
python scripts/compare_archive.py --archive "$THESIS_ARCHIVE" \
  --dataset uci --methods M0,M1,M2,M3,M4
```

Define `THESIS_ARCHIVE` as the path to your read-only original ZIP. The comparison extracts private inputs into a temporary directory and compares original/refactored DataFrames exactly. It prints aggregate table sizes and status, without retaining patient records in the repository. [VALIDATION.md](docs/VALIDATION.md) states exactly which executions passed and which remain unverified.

## Results

The runners generate performance tables, prediction records, split manifests, configuration snapshots, figures, and Markdown reports in the selected output directory. These files can contain subject identifiers and are ignored by Git. Keep them local. No numerical study conclusions or performance values are asserted in this README.

Validation compared the full UCI M0–M4 run and selected ABIDE/TCGA target-site runs against the original implementations in the same software environment. Exact agreement for those runs is a regression check, not a guarantee that every archived experiment or another software environment reproduces the thesis.

## Academic context and author

**William Bertaiola** — Master's Thesis in Bioengineering, University of Padua, in collaboration with the Universitat Politècnica de València.

Thesis: *Cross-Site Transportability in Biomedical Machine Learning: Evaluation and Adaptation Strategies*.

## License

No project license was found or selected. No license grant has been added. Before publication, select an appropriate code license and review provenance/permissions for the preserved DASHI-derived modules and other contributions. Dataset terms are separate from the code license.
