"""Regression tests for calculations, partitioning and train-only transforms."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from thesis_tools.metrics import compute_binary_metrics, compute_ece, id_ood_gap


def test_binary_metrics_and_bin_boundaries():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 0.2, 0.8, 1.0])
    metrics = compute_binary_metrics(y, p)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["ece"] == pytest.approx(0.4)
    assert compute_ece(np.array([]), np.array([])) != compute_ece(
        np.array([]), np.array([])
    )
    assert np.isnan(
        compute_binary_metrics(np.array([1, 1]), np.array([0.3, 0.7]))["roc_auc"]
    )
    assert np.isnan(id_ood_gap(np.nan, 0.5))


def test_target_partition_preserves_original_stratified_assignment(tmp_path):
    from uci_transportability.config import ExperimentConfig
    from uci_transportability.data import build_target_site_split

    frame = pd.DataFrame(
        {
            "subject_id": [f"synthetic_{i}" for i in range(48)],
            "domain_id": ["cleveland"] * 24 + ["hungarian"] * 24,
            "target": [0, 1] * 24,
        }
    )
    config = ExperimentConfig(Path("unused.csv"), Path("unused.json"), tmp_path)
    split = build_target_site_split(
        frame, target_site="cleveland", config=config, logger=logging.getLogger("test")
    )
    target = frame.loc[frame.domain_id == "cleveland"].reset_index(drop=True)
    train, test = train_test_split(
        target, test_size=0.5, random_state=42, shuffle=True, stratify=target.target
    )
    assert split.target_train_df.subject_id.tolist() == train.subject_id.tolist()
    assert split.target_test_df.subject_id.tolist() == test.subject_id.tolist()
    groups = [
        set(part.subject_id)
        for part in (split.source_df, split.target_train_df, split.target_test_df)
    ]
    assert (
        not groups[0] & groups[1]
        and not groups[0] & groups[2]
        and not groups[1] & groups[2]
    )
    assert set.union(*groups) == set(frame.subject_id)


@pytest.mark.parametrize("dataset", ["uci", "abide", "tcga"])
def test_model_configuration_retains_seed_and_hyperparameters(dataset, tmp_path):
    module = importlib.import_module(dataset + "_transportability.config")
    kwargs = dict(input_path=Path("unused.csv"), output_dir=tmp_path)
    if dataset != "tcga":
        kwargs["schema_path"] = Path("unused.json")
    config = module.ExperimentConfig(**kwargs)
    model_module = importlib.import_module(dataset + "_transportability.models")
    specs = model_module.build_model_specs(config)
    assert tuple(specs) == (
        "LogisticRegression",
        "RandomForest",
        "HistGradientBoosting",
    )
    for spec in specs.values():
        assert spec.estimator.random_state == 42
    assert specs["RandomForest"].estimator.n_estimators == (
        300 if dataset == "tcga" else 500
    )
    assert specs["HistGradientBoosting"].estimator.learning_rate == 0.05
    assert specs["HistGradientBoosting"].estimator.max_depth == 4
    assert specs["HistGradientBoosting"].estimator.max_iter == 300


def test_preprocessor_fits_training_values_only():
    from uci_transportability.models import build_preprocessor

    schema = {"numeric_features": ["age"], "categorical_features": ["sex"]}
    preprocessor = build_preprocessor(schema, scale_numeric=True)
    train = pd.DataFrame({"age": [10.0, 20.0, np.nan], "sex": [0, 1, 0]})
    test = pd.DataFrame({"age": [999.0], "sex": [2]})
    preprocessor.fit(train)
    before = (
        preprocessor.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_.copy()
    )
    transformed = preprocessor.transform(test)
    np.testing.assert_array_equal(before, [15.0])
    np.testing.assert_array_equal(
        before,
        preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_,
    )
    assert transformed.shape == (1, 3)
    np.testing.assert_array_equal(transformed[0, 1:], [0.0, 0.0])


def test_tcga_threshold_selection_original_tiebreak_and_empty_input():
    from tcga_transportability.metrics import select_operating_threshold

    result = select_operating_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert result["decision_threshold"] == 0.6
    assert result["threshold_selection_value"] == 1.0
    assert (
        select_operating_threshold(np.array([]), np.array([]))["decision_threshold"]
        == 0.5
    )
