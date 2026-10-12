# Refactoring report

Prepared on 12 October 2026 for William Bertaiola's Master's Thesis in Bioengineering at the University of Padua, in collaboration with the Universitat Politècnica de València.

## A. Original project analysis

The original ZIP contained 18,318 entries and 9,432,805,399 expanded bytes, including macOS metadata. There were 8,344 substantive files, 495 Python files (255 byte-distinct contents), three R scripts, two RMarkdown templates, two shell launchers and no notebooks. The original archive contains code mixed with biomedical inputs, processed tables, serialized models and generated figures/reports. A nested UCI SHAP archive contained an identical snapshot of an unpacked source script.

The main workflow is dataset preprocessing → source/target site partitions → M0–M4 models → ID/OOD metrics and repeated seeded evaluations → metadata, distance, entropy, sample-size, pattern and SHAP analyses. Multiple experimental generations were verified, including separate legacy CORAL/pseudo-labeling and alternative TCGA targets. Details, all original directories and per-file roles are in [ORIGINAL_ARCHITECTURE.md](docs/ORIGINAL_ARCHITECTURE.md), [CODE_INDEX.md](docs/CODE_INDEX.md) and the JSON inventories.

The original ZIP was read only. Its full CRC check passed and its SHA-256 is:

```text
ce8495bff5ce97d75e52b8ea49d03303330995f849f975fe0ee2ab52654a2bda
```

The before/after checksum was verified before delivery. No original dataset, output or archive member was modified or deleted.

## B. Final repository structure

The repository contains 412 publication files, including 331 Python sources. The full tree follows; a separate copy is in [REPOSITORY_TREE.md](docs/REPOSITORY_TREE.md).

```text
TFM_Bertaiola/
├── analysis/
│   ├── abide/
│   │   ├── analysis_final/
│   │   │   ├── scripts/
│   │   │   │   └── run_interaction_analysis_final_reporting.py
│   │   │   └── shap_interactions/
│   │   │       └── reports/
│   │   │           └── run_shap_interaction_values_analysis.py
│   │   ├── dashi/
│   │   │   └── run_abide_dashi.py
│   │   ├── entropy/
│   │   │   └── generate_entropy_analysis.py
│   │   ├── heatmap_cluster/
│   │   │   └── generate_clustered_heatmap_analysis.py
│   │   ├── metadata_discussion/
│   │   │   └── generate_metadata_analysis.py
│   │   ├── pattern_discovery/
│   │   │   └── generate_pattern_discovery_analysis.py
│   │   ├── quality_report/
│   │   │   └── generate_abide_quality_report.py
│   │   ├── sample_size/
│   │   │   └── generate_sample_size_tables.py
│   │   ├── shap/
│   │   │   ├── interactions/
│   │   │   │   └── scripts/
│   │   │   │       └── run_meta_learning_interactions.py
│   │   │   ├── scripts/
│   │   │   │   └── run_meta_learning_analysis.py
│   │   │   └── run_shap_by_method_abide.py
│   │   └── site_distance/
│   │       └── generate_site_distance_analysis.py
│   ├── comparative/
│   │   ├── 2_discussion_best_method/
│   │   │   ├── generate_dataset_goal_interpretation.py
│   │   │   └── generate_method_robustness_condition_analysis.py
│   │   └── 3_dataset_overview_and_metadata/
│   │       ├── abide_generate_metadata_analysis.py
│   │       ├── make_chapter4_dataset_figures.py
│   │       ├── tcga_generate_metadata_analysis.py
│   │       └── uci_generate_metadata_analysis.py
│   ├── repository/
│   │   └── generate_output_analysis.py
│   ├── tcga/
│   │   ├── analysis_final/
│   │   │   └── interaction_analysis_final_reporting/
│   │   │       ├── scripts/
│   │   │       │   └── run_interaction_analysis_final_reporting.py
│   │   │       └── shap_interactions/
│   │   │           └── reports/
│   │   │               └── run_shap_interaction_values_analysis.py
│   │   ├── dashi/
│   │   │   └── run_tcga_dashi.py
│   │   ├── entropy/
│   │   │   └── generate_entropy_analysis.py
│   │   ├── heatmap_cluster/
│   │   │   └── generate_clustered_heatmap_analysis.py
│   │   ├── metadata_discussion/
│   │   │   └── generate_metadata_analysis.py
│   │   ├── pattern_discovery/
│   │   │   └── generate_pattern_discovery_analysis.py
│   │   ├── quality_report/
│   │   │   └── generate_tcga_quality_report.py
│   │   ├── sample_size/
│   │   │   └── generate_sample_size_tables.py
│   │   ├── shap/
│   │   │   ├── interactions/
│   │   │   │   └── scripts/
│   │   │   │       └── run_meta_learning_interactions.py
│   │   │   ├── scripts/
│   │   │   │   └── run_meta_learning_analysis.py
│   │   │   └── run_shap_by_method_tcga.py
│   │   └── site_distance/
│   │       └── generate_site_distance_analysis.py
│   └── uci/
│       ├── analysis_final/
│       │   ├── interaction_analysis_extended/
│       │   │   └── scripts/
│       │   │       └── run_interaction_analysis_extended.py
│       │   └── interaction_analysis_final_reporting/
│       │       ├── scripts/
│       │       │   └── run_interaction_analysis_final_reporting.py
│       │       └── shap_interactions/
│       │           └── reports/
│       │               └── run_shap_interaction_values_analysis.py
│       ├── dashi/
│       │   └── run_uci_dashi.py
│       ├── entropy/
│       │   └── generate_entropy_analysis.py
│       ├── heatmap_cluster/
│       │   └── generate_clustered_heatmap_analysis.py
│       ├── metadata_discussion/
│       │   └── generate_metadata_analysis.py
│       ├── pattern_discovery/
│       │   └── generate_pattern_discovery_analysis.py
│       ├── quality_report/
│       │   └── generate_uci_quality_report.py
│       ├── sample_size/
│       │   └── generate_sample_size_tables.py
│       ├── shap/
│       │   ├── interactions/
│       │   │   └── scripts/
│       │   │       └── run_meta_learning_interactions.py
│       │   ├── scripts/
│       │   │   └── run_meta_learning_analysis.py
│       │   └── run_shap_by_method_uci.py
│       └── site_distance/
│           └── generate_site_distance_analysis.py
├── configs/
│   ├── recorded_runs/
│   │   ├── run_01.json
│   │   ├── run_02.json
│   │   ├── run_03.json
│   │   ├── run_04.json
│   │   ├── run_05.json
│   │   ├── run_06.json
│   │   ├── run_07.json
│   │   ├── run_08.json
│   │   ├── run_09.json
│   │   ├── run_10.json
│   │   ├── run_11.json
│   │   ├── run_12.json
│   │   ├── run_13.json
│   │   ├── run_14.json
│   │   ├── run_15.json
│   │   ├── run_16.json
│   │   ├── run_17.json
│   │   ├── run_18.json
│   │   ├── run_19.json
│   │   ├── run_20.json
│   │   ├── run_21.json
│   │   ├── run_22.json
│   │   ├── run_23.json
│   │   ├── run_24.json
│   │   ├── run_25.json
│   │   ├── run_26.json
│   │   ├── run_27.json
│   │   ├── run_28.json
│   │   ├── run_29.json
│   │   └── run_30.json
│   ├── schemas/
│   │   ├── abide2_core_multisite_preprocessing_blueprint.json
│   │   └── uci_heart_feature_schema.json
│   └── paths.json
├── data/
│   └── README.md
├── docs/
│   ├── archive_classification_summary.json
│   ├── archive_inventory.json
│   ├── CODE_INDEX.md
│   ├── code_provenance.json
│   ├── DATASETS.md
│   ├── DEPENDENCIES.md
│   ├── entrypoint_validation.json
│   ├── EXPERIMENTS.md
│   ├── ORIGINAL_ARCHITECTURE.md
│   ├── recorded_run_provenance.json
│   ├── REPOSITORY_TREE.md
│   ├── semantic_changes.json
│   ├── VALIDATION.md
│   └── validation_results.json
├── experiments/
│   ├── abide/
│   │   ├── bootstrap/
│   │   │   ├── merge_abide_bootstrap_batch_outputs.py
│   │   │   ├── run_abide_baseline.py
│   │   │   ├── run_abide_bootstrap_resampling.py
│   │   │   └── run_abide_fast_bootstrap10.py
│   │   └── transportability_previous/
│   │       ├── src/
│   │       │   └── abide_transportability/
│   │       │       ├── __init__.py
│   │       │       ├── config.py
│   │       │       ├── data.py
│   │       │       ├── experiment.py
│   │       │       ├── harmonization.py
│   │       │       ├── metrics.py
│   │       │       ├── models.py
│   │       │       ├── plots.py
│   │       │       ├── reporting.py
│   │       │       ├── transductive.py
│   │       │       └── weighting.py
│   │       └── run_abide_baseline.py
│   ├── legacy/
│   │   ├── abide2_dg_pipeline/
│   │   │   ├── src/
│   │   │   │   ├── init__.py
│   │   │   │   ├── io_load.py
│   │   │   │   ├── preprocess.py
│   │   │   │   ├── report.py
│   │   │   │   ├── splits.py
│   │   │   │   └── utils.py
│   │   │   ├── config.yaml
│   │   │   ├── main.py
│   │   │   └── requirements.txt
│   │   ├── abide2_m0_uniform/
│   │   │   └── abide2_m0_uniform.py
│   │   ├── abide2_transductive_learning_new/
│   │   │   ├── transductive_abide/
│   │   │   │   ├── analysis.py
│   │   │   │   ├── config.py
│   │   │   │   ├── data_utils.py
│   │   │   │   ├── init__.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── models.py
│   │   │   │   ├── plotting.py
│   │   │   │   ├── preprocessing.py
│   │   │   │   └── reporting.py
│   │   │   └── run_abide2_transductive_learning.py
│   │   ├── abide2_weighting_output/
│   │   │   ├── weighting_pipeline.py
│   │   │   └── weighting_utils.py
│   │   ├── baseline_m0/
│   │   │   ├── configs/
│   │   │   │   └── uci_m0_config.json
│   │   │   ├── src/
│   │   │   │   └── baseline_m0.py
│   │   │   └── run_baseline_m0.py
│   │   ├── baseline_m0_uci_sitepair/
│   │   │   ├── configs/
│   │   │   │   └── uci_m0_config.json
│   │   │   ├── src/
│   │   │   │   └── baseline_m0.py
│   │   │   ├── regenerate_uci_figures.py
│   │   │   └── run_baseline_m0.py
│   │   ├── combat_transportability_project/
│   │   │   ├── src/
│   │   │   │   └── combat_transportability/
│   │   │   │       ├── bootstrap.py
│   │   │   │       ├── datasets.py
│   │   │   │       ├── init__.py
│   │   │   │       ├── pairwise.py
│   │   │   │       └── pipeline.py
│   │   │   ├── requirements.txt
│   │   │   ├── run_all.py
│   │   │   └── run_pairwise_heatmaps.py
│   │   ├── dashi_abide_preprocessed/
│   │   │   └── abide2_dashi_preprocessed_pipeline.py
│   │   ├── dashi_python_improved/
│   │   │   ├── delivery_bundle/
│   │   │   │   └── code/
│   │   │   │       └── supervised_abide2.py
│   │   │   ├── supervised_characterization/
│   │   │   │   ├── arrange_metrics.py
│   │   │   │   ├── estimate_models.py
│   │   │   │   ├── init__.py
│   │   │   │   └── plot_performance.py
│   │   │   ├── unsupervised_characterization/
│   │   │   │   ├── data_source_map/
│   │   │   │   │   ├── data_source_map.py
│   │   │   │   │   ├── data_source_map_plotter.py
│   │   │   │   │   └── init__.py
│   │   │   │   ├── data_temporal_map/
│   │   │   │   │   ├── data_temporal_map.py
│   │   │   │   │   ├── data_temporal_map_plotter.py
│   │   │   │   │   └── init__.py
│   │   │   │   └── variability_metrics/
│   │   │   │       ├── estimate_msv_metrics.py
│   │   │   │       ├── igt_plotting.py
│   │   │   │       ├── igt_projection.py
│   │   │   │       ├── igt_projection_estimator.py
│   │   │   │       ├── igt_trajectory_estimator.py
│   │   │   │       ├── init__.py
│   │   │   │       └── plot_msv.py
│   │   │   ├── common_pipeline.py
│   │   │   ├── dataset_loaders.py
│   │   │   ├── run_all_improved.py
│   │   │   ├── supervised_abide2.py
│   │   │   ├── supervised_tcga.py
│   │   │   ├── supervised_uci.py
│   │   │   ├── unsupervised_abide2.py
│   │   │   ├── unsupervised_tcga.py
│   │   │   └── unsupervised_uci.py
│   │   ├── dashi_tcga_preprocessed/
│   │   │   └── tcga_dashi_preprocessed_pipeline.py
│   │   ├── dashi_uci_preprocessed/
│   │   │   └── uci_dashi_preprocessed_pipeline.py
│   │   ├── final_work_build/
│   │   │   └── 01_data_preprocessing/
│   │   │       └── shared/
│   │   │           └── thesis_plot_style.py
│   │   ├── r/
│   │   │   ├── abide/
│   │   │   │   └── abide2_lodo_transportability_m0.R
│   │   │   ├── tcga/
│   │   │   │   └── tcga_brca_lodo_transportability_m0.R
│   │   │   ├── templates/
│   │   │   │   ├── abide2_dg_report.rmd
│   │   │   │   └── mlomics_report.rmd
│   │   │   └── uci/
│   │   │       └── merge_uci_datasets.R
│   │   ├── report_tcga_transd_learning_smoke/
│   │   │   └── tcga_brca_transductive_learning_preprocessed.py
│   │   ├── src/
│   │   │   ├── dashi_rmd_backend.py
│   │   │   ├── dashi_rmd_backend_abide2.py
│   │   │   ├── dashi_rmd_backend_tcga_brca.py
│   │   │   ├── dashi_uci_pipeline.py
│   │   │   └── init__.py
│   │   ├── tcga_brca_initial_audit/
│   │   │   ├── configs/
│   │   │   │   ├── tcga_brca_audit_config.json
│   │   │   │   └── tcga_brca_site_filtered_process_config.json
│   │   │   ├── src/
│   │   │   │   ├── audit/
│   │   │   │   │   ├── barcodes.py
│   │   │   │   │   ├── clinical.py
│   │   │   │   │   ├── init__.py
│   │   │   │   │   ├── plots.py
│   │   │   │   │   ├── reporting.py
│   │   │   │   │   └── summaries.py
│   │   │   │   ├── ingestion/
│   │   │   │   │   ├── expression.py
│   │   │   │   │   └── init__.py
│   │   │   │   ├── preprocessing/
│   │   │   │   │   ├── blueprint.py
│   │   │   │   │   ├── init__.py
│   │   │   │   │   └── site_filtered_database.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── init__.py
│   │   │   │   │   └── io.py
│   │   │   │   └── init__.py
│   │   │   ├── run_tcga_brca_initial_audit.py
│   │   │   └── run_tcga_brca_site_filtered_database.py
│   │   ├── tcga_m0_from_scratch/
│   │   │   └── tcga_m0_from_scratch.py
│   │   ├── tcga_m0_uniform_single/
│   │   │   └── tcga_m0_uniform.py
│   │   ├── tcga_stage_reimagined_m0/
│   │   │   └── tcga_stage_reimagined_m0.py
│   │   ├── tcga_transductive_learning_new/
│   │   │   ├── transductive_tcga/
│   │   │   │   ├── analysis.py
│   │   │   │   ├── config.py
│   │   │   │   ├── data_utils.py
│   │   │   │   ├── init__.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── models.py
│   │   │   │   ├── plotting.py
│   │   │   │   ├── preprocessing.py
│   │   │   │   └── reporting.py
│   │   │   └── run_tcga_transductive_learning.py
│   │   ├── uci_combat_output_staging_run2/
│   │   │   └── uci_combat_lodo_pipeline.py
│   │   ├── uci_m0_uniform_single/
│   │   │   └── uci_m0_uniform.py
│   │   ├── uci_originals_build/
│   │   │   └── weighting/
│   │   │       ├── weighting_pipeline.py
│   │   │       └── weighting_utils.py
│   │   ├── uci_transductive_learning_new/
│   │   │   ├── output_local/
│   │   │   │   └── code/
│   │   │   │       ├── transductive_uci/
│   │   │   │       │   ├── adaptation.py
│   │   │   │       │   ├── adaptation_3.py
│   │   │   │       │   ├── analysis.py
│   │   │   │       │   ├── analysis_3.py
│   │   │   │       │   ├── config.py
│   │   │   │       │   ├── data_utils.py
│   │   │   │       │   ├── init__.py
│   │   │   │       │   ├── init___3.py
│   │   │   │       │   ├── metrics.py
│   │   │   │       │   ├── metrics_3.py
│   │   │   │       │   ├── models.py
│   │   │   │       │   ├── models_3.py
│   │   │   │       │   ├── plotting.py
│   │   │   │       │   └── reporting.py
│   │   │   │       └── run_uci_transductive_learning.py
│   │   │   ├── transductive_uci/
│   │   │   │   ├── adaptation.py
│   │   │   │   ├── analysis.py
│   │   │   │   ├── config.py
│   │   │   │   ├── data_utils.py
│   │   │   │   ├── init__.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── models.py
│   │   │   │   ├── plotting.py
│   │   │   │   └── reporting.py
│   │   │   └── run_uci_transductive_learning.py
│   │   ├── uci_weighting_output/
│   │   │   ├── weighting_pipeline.py
│   │   │   └── weighting_utils.py
│   │   ├── abide2_combat_lodo_pipeline.py
│   │   ├── abide2_transductive_learning.py
│   │   ├── compare_uci_methods_to_m0.py
│   │   ├── mloomics_complete_analysis.py
│   │   ├── package_dataset_original_outputs.py
│   │   ├── run_dashi_tcga_brca_rebuild.py
│   │   ├── run_dashi_uci.py
│   │   ├── run_dashi_uci_multisource_2.py
│   │   ├── tcga_brca_prepare_by_site.py
│   │   ├── tcga_brca_transductive_learning_binary.py
│   │   ├── tcga_brca_transductive_learning_fixed.py
│   │   ├── tcga_brca_transductive_learning_preprocessed.py
│   │   ├── tcga_combat_lodo_pipeline.py
│   │   ├── tcga_pca_transductive_learning.py
│   │   ├── tcga_separati_transductive_learning.py
│   │   ├── tcga_separati_transductive_learning_01_vs_06.py
│   │   ├── tcga_weighting_runner.py
│   │   ├── uci_combat_lodo_pipeline.py
│   │   ├── uci_finalize_results.py
│   │   ├── uci_heart_domain_workflow.py
│   │   ├── uci_heart_transductive_learning_sites.py
│   │   ├── weighting_pipeline.py
│   │   ├── weighting_utils.py
│   │   └── whiten_zero_heatmaps_uci.py
│   ├── tcga/
│   │   └── bootstrap/
│   │       ├── bootstrap_reporting.py
│   │       ├── merge_m0_m1_m2_m3_with_m4_outputs.py
│   │       ├── merge_m0_m1_m2_with_m3_outputs.py
│   │       ├── merge_m0_m1_outputs.py
│   │       ├── merge_m0_m1_with_m2_outputs.py
│   │       ├── merge_tcga_bootstrap_batch_outputs.py
│   │       ├── run_tcga_baseline.py
│   │       ├── run_tcga_bootstrap_resampling.py
│   │       └── run_tcga_fast_bootstrap10.py
│   └── uci/
│       ├── baseline_previous/
│       │   ├── src/
│       │   │   └── uci_transportability/
│       │   │       ├── __init__.py
│       │   │       ├── config.py
│       │   │       ├── data.py
│       │   │       ├── experiment.py
│       │   │       ├── metrics.py
│       │   │       ├── models.py
│       │   │       ├── plots.py
│       │   │       └── reporting.py
│       │   └── run_m0_baseline.py
│       └── bootstrap/
│           ├── generate_best_method_table.py
│           ├── run_m0_baseline.py
│           ├── run_m0_m1_combat.py
│           ├── run_m0_m1_m2.py
│           ├── run_m0_m1_m2_m3.py
│           └── run_uci_bootstrap_resampling.py
├── preprocessing/
│   ├── abide2/
│   │   ├── configs/
│   │   │   ├── abide2_composite_unified_config.json
│   │   │   ├── abide2_core_transportability_config.json
│   │   │   ├── abide2_folder_unified_config.json
│   │   │   ├── abide2_processing_config.json
│   │   │   └── abide2_transportability_rich_config.json
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── abide2_composite_unified.py
│   │   │   ├── abide2_core_transportability.py
│   │   │   ├── abide2_folder_unified.py
│   │   │   ├── abide2_processing.py
│   │   │   └── abide2_transportability_rich.py
│   │   ├── run_abide2_composite_unified.py
│   │   ├── run_abide2_core_transportability.py
│   │   ├── run_abide2_folder_unified.py
│   │   ├── run_abide2_processing.py
│   │   ├── run_abide2_transportability_rich.py
│   │   └── run_standard_plots.py
│   ├── shared/
│   │   └── thesis_plot_style.py
│   ├── tcga_brca/
│   │   ├── configs/
│   │   │   ├── tcga_brca_audit_config.json
│   │   │   ├── tcga_brca_site_filtered_process_config.json
│   │   │   └── tcga_brca_stage_ready_config.json
│   │   ├── src/
│   │   │   ├── audit/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── barcodes.py
│   │   │   │   ├── clinical.py
│   │   │   │   ├── plots.py
│   │   │   │   ├── reporting.py
│   │   │   │   └── summaries.py
│   │   │   ├── ingestion/
│   │   │   │   ├── __init__.py
│   │   │   │   └── expression.py
│   │   │   ├── preprocessing/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── blueprint.py
│   │   │   │   ├── site_filtered_database.py
│   │   │   │   └── stage_ready_dataset.py
│   │   │   ├── utils/
│   │   │   │   ├── __init__.py
│   │   │   │   └── io.py
│   │   │   └── __init__.py
│   │   ├── run_tcga_brca_initial_audit.py
│   │   ├── run_tcga_brca_site_filtered_database.py
│   │   └── run_tcga_brca_stage_ready_preprocessing.py
│   └── uci_heart/
│       ├── configs/
│       │   └── default_config.json
│       ├── src/
│       │   ├── ingestion/
│       │   │   ├── __init__.py
│       │   │   ├── loaders.py
│       │   │   └── specs.py
│       │   ├── preprocessing/
│       │   │   ├── __init__.py
│       │   │   ├── audit.py
│       │   │   ├── example_lodo_usage.py
│       │   │   ├── reporting.py
│       │   │   ├── schema.py
│       │   │   ├── splits.py
│       │   │   └── train_only.py
│       │   ├── utils/
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   ├── filesystem.py
│       │   │   └── logging_utils.py
│       │   └── __init__.py
│       ├── run_pipeline.py
│       └── run_standard_plots.py
├── results/
│   └── .gitkeep
├── scripts/
│   ├── compare_archive.py
│   └── run_all_preprocessing.sh
├── src/
│   ├── abide_transportability/
│   │   ├── __init__.py
│   │   ├── bootstrap_reporting.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── experiment.py
│   │   ├── harmonization.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── plots.py
│   │   ├── reporting.py
│   │   ├── transductive.py
│   │   └── weighting.py
│   ├── tcga_transportability/
│   │   ├── __init__.py
│   │   ├── bootstrap_reporting.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── experiment.py
│   │   ├── feature_selection.py
│   │   ├── harmonization.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── plots.py
│   │   ├── reporting.py
│   │   ├── transductive.py
│   │   └── weighting.py
│   ├── thesis_tools/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── paths.py
│   └── uci_transportability/
│       ├── __init__.py
│       ├── bootstrap_reporting.py
│       ├── config.py
│       ├── data.py
│       ├── experiment.py
│       ├── harmonization.py
│       ├── metrics.py
│       ├── models.py
│       ├── plots.py
│       ├── reporting.py
│       ├── transductive.py
│       └── weighting.py
├── tests/
│   ├── test_repository.py
│   └── test_scientific_invariants.py
├── .gitignore
├── .thesis-root
├── constraints-original.txt
├── constraints-tested.txt
├── pyproject.toml
├── README.md
├── REFACTORING_REPORT.md
├── requirements-dev.txt
├── requirements-optional.txt
└── requirements.txt
```

## C. Changes performed

- Moved the organized UCI, ABIDE and TCGA bootstrap scientific packages into separate root `src/` packages. Moved their runners into `experiments/<dataset>/bootstrap/` and retained earlier baseline/pipeline generations separately.
- Moved preprocessing into `preprocessing/abide2`, `preprocessing/uci_heart` and `preprocessing/tcga_brca`, preserving local import context. Recovered the shared plot style from the archived working-build snapshot.
- Moved post-experiment scripts into `analysis/<dataset>` and `analysis/comparative`; preserved the output inventory script in `analysis/repository`.
- Preserved distinct standalone, staging and historical source versions under `experiments/legacy`. Normalized legacy source folder names and filenames, including names containing spaces. Preserved R baselines and moved report templates out of output directories.
- Redirected 172 verified exact Python snapshots to canonical source locations. All 495 original Python files map to existing final sources in `code_provenance.json`. Identical package-local copies remain where merging import contexts could change behavior. No byte-distinct scientific implementation was excluded merely for looking redundant.
- Consolidated the three identical binary-metric implementations into `thesis_tools.metrics`; dataset modules re-export them. TCGA threshold-selection logic remains dataset-specific.
- Added portable root/path configuration, environment root overrides, JSON path resolution and separate ignored output directories. Preserved original configuration values while replacing personal input/output paths. Deferred one optional DASHI import until after CLI parsing.
- Applied consistent Ruff formatting, removed 48 reviewed unused imports, added module/core-function docstrings, and retained diagnostics/dependency probes whose behavior could matter. Import ordering around plotting/backend configuration was preserved.
- Fixed an undefined `directories` variable in the separate archived TCGA reporting module by deriving the figure directory from `_ensure_dirs(config.output_dir)`. The organized imported reporting implementation was already distinct and remains numerically unchanged.
- Replaced patient-like barcode examples in prose with synthetic examples. Excluded all real biomedical tables, trained models, generated study figures, logs, thesis manuscript and macOS metadata from the publication copy.
- Preserved 30 sanitized recorded-run configuration records and four observed dependency versions. Omitted inferred dataset-schema payloads from weighting run records; the inference implementations remain in source. Definitions-only UCI/ABIDE schema templates are included.
- Created the English README, acquisition/input documentation, execution guide, dependency guide, complete code/directory inventories, per-symbol change record, validation documentation and tests. Added packaging, editable installation support, a portable shell launcher and a private archive comparison utility.

Every move and rename is listed in [CODE_INDEX.md](docs/CODE_INDEX.md) and [code_provenance.json](docs/code_provenance.json). Original source hashes identify snapshot provenance. The final file inventory classifies every outer ZIP member; patient-barcode-looking inventory paths are masked.

## D. Scientific integrity

Core data-selection, preprocessing, partitioning, seeds, estimator settings, feature selection, adaptation algorithms, metrics, threshold selection and result calculations were retained. Shared metric consolidation was preceded by exact structural comparison. Original source fingerprints are used in the regression tests for every core top-level scientific function/class; those tests passed.

The UCI all-site M0–M4 comparison and selected ABIDE/TCGA target comparisons matched the originals exactly in all eight exported DataFrames per dataset, including predictions, assignments, pairwise weighting, metrics and input tables. No methodological improvement or numerical bug fix was introduced into these core implementations.

The changes that can affect behavior are filesystem/default input/output routing, JSON relative-path interpretation, optional dependency discovery, reporting text and the archived report crash fix. Users must configure the same original data rather than substitute a different input with a similar filename. Available ComBat backends and library versions can change numerical results without source changes. These limits are explicit; no cross-environment identity guarantee is made.

The original ComBat training-only behavior, labeled target adaptation, TCGA ID-derived operating thresholds, and repeated-seed evaluation semantics were documented. They were not relabeled as universal leakage-free procedures or conventional patient bootstrap confidence intervals.

## E. Validation results

See [VALIDATION.md](docs/VALIDATION.md) and [validation_results.json](docs/validation_results.json) for exact scope and aggregate evidence.

- ZIP CRC: all outer entries passed; nested ZIP passed; source SHA-256 unchanged.
- Syntax: every published Python file compiled and parsed as Python 3.10 grammar.
- Static correctness and formatting: Ruff checks passed.
- Imports: all organized scientific packages and shared helpers imported.
- Environment: declared core/optional packages installed; `pip check` passed; editable installation passed.
- Automated tests: **12 passed**, with one expected sklearn single-class edge-case warning.
- Entry-point interfaces: **94 help checks passed**, including corrected package-module invocations. Help success is not full experimental execution.
- UCI preprocessing: two CSVs and four scientific JSON outputs matched the originals exactly.
- UCI M0–M4: all four target sites and three models; exact DataFrame agreement.
- ABIDE M0–M4: target `ABIDEII-BNI_1` and three models; exact DataFrame agreement.
- TCGA M0–M4: target `AN` and three models; exact DataFrame agreement.
- UCI bootstrap CLI: one M0 iteration completed, including plots/tables/report.
- Archive comparison utility: executed successfully on UCI M0 with exact agreement.
- Preprocessing launcher: Bash syntax passed; full mixed-dataset launcher not executed.

All biomedical validation inputs and raw comparison tables were kept in a private temporary workspace and excluded from the deliverable. Generated public validation metadata contains status and table sizes only. No claim is made that unexecuted historical experiments passed.

## F. Remaining limitations

1. Raw ABIDE composite/site phenotype and complete QAP source collections are absent; the original raw TCGA expression/clinical source inputs are absent. Processed versions in the archive enabled the comparisons but cannot establish full acquisition provenance.
2. The exact wide TCGA expression exporter, units/release lineage and original complete environment could not be verified. Four package versions were preserved; the recorded environment is partial.
3. Matching upstream DASHI installation and source/license provenance remain unresolved. DASHI workflows and their preserved local overrides need review before publication or execution.
4. ABIDE/TCGA all-target and ten-iteration thesis reruns, older standalone variants, and full SHAP/metadata/comparative analysis chains were not executed. Optional dependency installation is not algorithmic validation. R scripts/RMarkdown were not executed.
5. Existing missing-feature/single-class warnings, fallback splits, train-only ComBat behavior and backend differences remain part of the original methodology. This refactor did not independently establish statistical validity of all study comparisons.
6. Some downstream analyses depend on historical method folder names or pre-existing summary tables. The documentation states those conventions; reproducing every historical result chain requires its original prerequisites.
7. Legacy runners sometimes rebuild/clear designated output folders. Use dedicated destinations. Their inherited behavior was preserved, and defaults now avoid the user's general Desktop folders.
8. No code license was selected or applied. Attribution and permissions for preserved third-party-derived code remain an author decision.

## G. GitHub publication checklist

- [x] Local repository created in the requested destination; original destination contents preserved.
- [x] Original ZIP unchanged and checked for integrity.
- [x] All Python sources accounted for; distinct scientific variants retained.
- [x] README, data contracts, dependencies, execution documentation and refactoring report provided.
- [x] Portable filesystem defaults; no personal machine paths in code/configuration.
- [x] No patient-level input, predictions, trained models, generated biomedical output or credentials copied into the publication set.
- [x] `.gitignore` excludes data, results, snapshots, environment files and caches.
- [x] Feasible syntax/import/static/entry-point/tests and representative original/refactored numerical comparisons passed.
- [x] No remote repository created, no remote configured, no commit, push or upload performed.
- [ ] **Author review:** choose an appropriate code license if intended and verify contributor/DASHI attribution and redistribution rights.
- [ ] **Research review:** confirm original input provenance and dataset provider terms for your intended use.
- [ ] **Optional reproduction:** supply missing raw data/DASHI revision and execute any remaining historical analyses or full study reruns needed for your publication claims.

The local refactor is complete for review. Public release as a licensed open-source project still requires the stated author/provenance decisions; full-thesis numerical reproducibility is not claimed.
