# Dependencies and software environment

`requirements.txt` covers the organized experiment packages and preprocessing/report scripts: NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, PyYAML, joblib, openpyxl, statsmodels and neuroCombat. `requirements-optional.txt` adds Plotly, SHAP, XGBoost, LightGBM, prince, umap-learn, inmoose, tqdm and pyarrow for archived analyses. Standard-library modules are not listed as pip dependencies. `requirements-dev.txt` adds pytest and Ruff.

Lower bounds for NumPy/pandas/scikit-learn/matplotlib/seaborn/SciPy/PyYAML/pyarrow originate in the archived ABIDE DG requirements. They describe recorded requirements, not a tested compatibility matrix for every historical version. Python syntax parses as 3.10; the validation interpreter was Python 3.12.14 on macOS ARM64.

The four archived weighting `run_config.json` records identify **numpy 2.4.2, pandas 3.0.0, matplotlib 3.10.8, scikit-learn 1.8.0**. Those exact observed versions are preserved in `constraints-original.txt`. Other historical package versions and the original Python version could not be established. This partial record is not represented as a complete environment lockfile.

`constraints-tested.txt` records actual installed versions in the fresh isolated validation environment. Core comparisons and tests used these versions. It contains constraints for optional dependencies as well, because they were installed for import/entry-point inspection. Optional package installation and `pip check` succeeded. Native-library/platform behavior outside this validation platform is not guaranteed.

Example installations:

```bash
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -m pip install -r requirements-optional.txt -c constraints-tested.txt
# Partial historical environment, not separately validated in this task:
python -m pip install -r requirements.txt -c constraints-original.txt
```

The source imports `dashi`, including `dashi._constants`, supervised characterization, source/temporal maps, and variability metrics. The archive supplies local overrides but no verified upstream distribution/version or complete installation metadata. Do not install a similarly named package and assume it provides this API. This dependency remains unresolved. Its preserved local code may also need attribution/license review before release.

R scripts require combinations of tidyverse, tidymodels, pROC, ranger, xgboost, jsonlite, glmnet, readr, dplyr, tibble, knitr and an RMarkdown renderer. The exact original R versions are unknown and the R workflows were not executed. Shell preprocessing uses Bash and a selected Python executable (`THESIS_PYTHON`, otherwise `python3`).

The core neuroCombat backend was installed and available for the comparison runs. UCI/ABIDE contain a local parametric fallback if that backend is unavailable; TCGA requests neuroCombat explicitly. Keep backend availability consistent when comparing results.
