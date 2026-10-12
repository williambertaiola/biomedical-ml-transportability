# Validation performed on 12 October 2026

Validation used Python 3.12.14 on macOS ARM64 in a fresh temporary virtual environment. Exact installed package versions are in `constraints-tested.txt`. Both the original and refactored implementations ran in this same environment; this does not establish agreement with an unrecorded historical environment.

| Check | Actual outcome |
| --- | --- |
| Complete ZIP CRC verification | All 18,318 outer entries passed; nested UCI SHAP ZIP also passed |
| Source archive integrity | SHA-256 before/after unchanged (see final report) |
| Python syntax | All published `.py` files compiled and parsed with Python 3.10 grammar |
| Formatting | Ruff formatter check passed |
| Static correctness | Ruff F821/F822/F823 checks passed |
| Core imports | All 40 organized package and shared helper modules imported |
| Git ignore behavior | All 412 publication files visible; five private/build artifact examples ignored |
| Dependencies | Core and optional requirements installed; `pip check` passed |
| Packaging | Editable installation from `pyproject.toml` passed |
| Automated tests | 12 passed; one expected sklearn warning for a single-class metric edge case |
| Entry-point help | 94 inspected entry points passed using their supported file/module invocation |
| UCI raw preprocessing | Exact CSV and scientific JSON agreement with the original |
| UCI M0–M4 | All four target sites; all three estimators; eight output DataFrames matched exactly |
| ABIDE M0–M4 | Target `ABIDEII-BNI_1`; all three estimators; eight output DataFrames matched exactly |
| TCGA M0–M4 | Target `AN`; all three estimators; eight output DataFrames matched exactly |
| UCI bootstrap CLI | One M0 iteration across all four sites completed, including tables, plots and Markdown report |
| Archive comparison utility | Executed on UCI M0; eight DataFrames matched exactly |
| Preprocessing shell | Bash syntax check passed; full combined launcher not executed |

The automated tests cover metric boundary/empty/single-class cases, target partition identity and disjointness, estimator settings, training-only preprocessing, threshold selection, configurable paths, source syntax, publication hygiene and original-code coverage. Core top-level scientific definitions are protected by portable AST fingerprints computed from the original source, excluding docstrings. Shared metrics retain the original implementations.

The UCI preprocessing comparison checked the unified and analysis-ready CSVs, feature schema, LODO manifest, quality checks and ingestion summary. The final 920-row analysis-ready table matched exactly. Reports/manifests containing execution timestamps and filesystem locations were not compared byte-for-byte.

The scientific comparisons used seed 42 and unchanged model, feature selection and adaptation configuration. Dataset sizes entering the tested organized loaders were 920 UCI rows, 1114 ABIDE rows, and 829 eligible TCGA rows. These are validation-input counts, not study-performance claims. ABIDE and TCGA were restricted to the stated target sites for computational practicality; their source data and source validation retained the original rules. Clinical records and prediction tables stayed in temporary private files and were not copied into this repository. `validation_results.json` contains aggregate table sizes and status only.

Across the 495 original Python files, 5796 top-level function/class definitions were compared including duplicate snapshots: 5692 matched structurally after docstring removal; 104 differences are declared in `semantic_changes.json`. These differences cover filesystem/default configuration, environment lookup, reporting prose, one archived report crash fix, and lazy optional imports. The core model/split/adaptation function fingerprints passed.

Initial help inspection invoked two package modules directly; their relative-import failures were resolved by checking `python -m combat_transportability.<module>` with the local package source root. The DASHI rebuild wrapper initially imported its unavailable optional dependency before parsing help; moving that import after argument parsing made help available. Actual DASHI execution remains unvalidated and requires the correct package.

During test development, the TCGA-specific 300-tree forest setting was distinguished from the 500-tree phenotype setting, and AST fingerprints were made independent of interpreter-specific empty-field display. The final tests protect the original settings. No model hyperparameter was changed to make a test pass.

## Validation limits

- Full raw ABIDE and TCGA preprocessing was not executed: complete raw phenotype/QAP collections and the original wide expression/clinical inputs are absent.
- Only one ABIDE and one TCGA target were evaluated in the M0–M4 comparisons; all-target and ten-iteration thesis reruns remain unverified.
- The one-iteration UCI bootstrap CLI verifies execution/reporting, not repeated-run uncertainty claims.
- Later SHAP, metadata, distance, entropy, pattern-discovery and comparative result chains were inspected and their available help/import interfaces checked, but their full analyses were not executed.
- Earlier standalone/legacy variants were preserved and statically inspected; no numerical-equivalence claim is made for unexecuted variants.
- The compatible upstream DASHI distribution/revision and license provenance remain unresolved. Local overridden modules do not replace the missing package.
- Optional packages were installed but not all corresponding model/analysis algorithms were exercised. R scripts and RMarkdown rendering were not run.
- ComBat backend choice, library version differences, site eligibility fallbacks and missing-value behavior can affect cross-environment results. The original scientific behavior was preserved, not independently redesigned or statistically endorsed.
