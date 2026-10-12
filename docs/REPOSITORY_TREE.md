# Complete final repository tree

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
