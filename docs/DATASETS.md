# Dataset acquisition and input contracts

No biomedical records are bundled. The original archive remains the private source for existing preprocessed inputs; raw source provenance is incomplete for some workflows. Preserve the original data ordering and version when reproducing a thesis run.

## UCI Heart Disease

The official [UCI Heart Disease record](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease) identifies four databases and specifies CC BY 4.0. Obtain the four **raw 76-field** files `cleveland.data`, `hungarian.data`, `long-beach-va.data`, and `switzerland.data`, plus `heart-disease.names`, from the dataset download. The default preprocessing loader distinguishes the raw format from 14-field processed exports; do not replace raw inputs with only the Cleveland export.

```text
data/raw/uci/
  cleveland.data
  hungarian.data
  long-beach-va.data
  switzerland.data
  heart-disease.names
```

`preprocessing/uci_heart/src/ingestion/specs.py` defines field positions and domain aliases. Missing-value interpretation, duplicate checks, cleaning, and disease target binarization remain in `preprocessing/audit.py`. The preprocessing seed is 20260327. Input records with disease codes 1–4 are collapsed to target 1; 0 is target 0. Domain labels are `cleveland`, `hungarian`, `long_beach_va`, and `switzerland`.

Generated bundle:

```text
results/Data/UCI_Heart_preprocessing/
  data/processed/uci_heart_analysis_ready.csv
  data/processed/uci_heart_feature_schema.json
  data/processed/uci_heart_quality_checks.json
  data/splits/lodo_domains.json
```

The analysis CSV includes `subject_id`, `domain_id`, `target_original`, `target`, and the 13 predictive fields: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`. The original schema is also provided under `configs/schemas/`; it contains column definitions only. The four raw files were present in the archive and were used privately to validate preprocessing. They are intentionally excluded from this repository despite UCI's stated license, following the project requirement to exclude patient-level biomedical data.

## ABIDE II

Use the [official ABIDE II release page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html) for composite phenotype files, longitudinal phenotype files, MRI quality metric tables, and site-specific data. The [ABIDE access information](https://fcon_1000.projects.nitrc.org/indi/abide/databases.html) describes provider registration and usage agreements. These pages do not establish unrestricted redistribution rights for every derived subject-level table. Verify the terms of the actual provider/release you obtain; records remain excluded here.

```text
data/raw/abide2/
  ABIDEII_Composite_Phenotypic.csv
  ABIDEII_Long_Composite_Phenotypic.csv
  ABIDEII_MRI_Quality_Metrics/
  # Optional site-specific phenotype/source folders used by the chosen runner
```

The composite runner merges phenotype and quality-metric sources using the existing subject/site normalization and collapse rules. The rich-processing runner builds both strict and rich cross-sectional variants. The strict model inputs are `age_at_scan`, `sex_code`, `handedness_category_code`, `eye_status_at_scan_code`, and `fiq`. Identifier, site, diagnosis labels, and high target-proximity clinical blocks are kept out of strict predictive features. Legacy experiments may use richer variables; do not conflate their results with strict phenotype experiments.

```text
results/Data/ABIDE2_preprocessing/data/processed/
  abide2_composite_subject_level_unified_database.csv
  abide2_core_multisite_transportability_dataset_strict.csv
  abide2_core_multisite_transportability_dataset.csv
  abide2_core_multisite_preprocessing_blueprint.json
```

The core loader expects `subject_id`, `site`, and diagnosis information and constructs `target` through its original mappings. The blueprint template under `configs/schemas/` retains strict and rich feature lists. This is a tabular phenotype pipeline; it does not establish an MRI image-processing implementation. Full raw phenotype/QAP collections are not available in the archive, so raw ABIDE preprocessing was not rerun.

## TCGA-BRCA

TCGA inputs must be obtained under the applicable [GDC access procedures](https://gdc.cancer.gov/access-data/data-access-processes-and-tools) and [data analysis policies](https://gdc.cancer.gov/analyze-data/data-analysis-policies). GDC distinguishes open from controlled access; controlled files require authorization, and all users agree not to reidentify participants. Access to a dataset does not establish provenance or redistribution permission for arbitrary local derivatives.

```text
data/raw/tcga_brca/
  TCGA-BRCA.star_counts.tsv
  clinical/
    clinical.tsv
    # Additional clinical TSVs accepted by the original clinical loader
```

The expression loader expects a gene-by-sample or sample-by-gene wide table with recognizable TCGA sample barcodes, not individual GDC STAR files placed unchanged into a directory. Its orientation/cleaning behavior is in `preprocessing/tcga_brca/src/ingestion/expression.py`. The precise expression-table exporter, upstream release, and expression-unit provenance could not be verified from the archived code. Do not assume that downloading current per-case GDC files reproduces this matrix.

The stage-ready pipeline retains primary tumors (sample type `01`), ranks duplicate primary samples by its existing quality/signal/barcode rules, merges clinical stage, maps Stage I–II to 0 and Stage III–IV to 1, and filters sites according to its recorded class-count requirement. These selection rules were preserved.

```text
results/Data/TCGA_BRCA_preprocessing/data/processed/
  tcga_brca_stage_transportability_dataset.csv
  tcga_brca_stage_annotated_primary_tumors.csv
  tcga_brca_stage_ready_summary.json
```

The organized model loader requires `subject_id`, `site`, `target`, `target_label`, and `ENSG`-prefixed gene features. Its site eligibility rules and training-only feature selection remain in the core package. Raw expression and clinical files are absent from the archive; the archived stage-ready matrix was available for private comparisons.

Legacy TCGA scripts additionally implement tumor/normal, sample-type 01/06, and artificial PCA/KMeans targets. They require their own original input formats and are not interchangeable with stage transportability.

## Local paths and output privacy

Portable defaults are repository-relative. `THESIS_DATA_ROOT` and `THESIS_RESULTS_ROOT` override the two root prefixes; `THESIS_PATHS_FILE` selects a complete path registry. Explicit CLI inputs can point to existing private processed bundles without moving them into the repository. Output tables and snapshots may contain subject identifiers or local paths. Keep `data/` and `results/` ignored and inspect publication contents before uploading.

Official access/terms pages were checked on 12 October 2026. No data redistribution is performed by this repository.
