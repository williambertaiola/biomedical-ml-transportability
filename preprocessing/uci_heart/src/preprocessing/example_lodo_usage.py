"""Preprocessing: example lodo usage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.train_only import build_train_only_preprocessor  # noqa: E402


def run_example() -> None:
    schema_path = PROJECT_ROOT / "data/processed/uci_heart_feature_schema.json"
    data_path = PROJECT_ROOT / "data/processed/uci_heart_analysis_ready.csv"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    df = pd.read_csv(data_path)

    for held_out_domain in sorted(df["domain_id"].dropna().unique().tolist()):
        train_df = df.loc[df["domain_id"] != held_out_domain].copy()
        test_df = df.loc[df["domain_id"] == held_out_domain].copy()

        x_train = train_df[schema["predictive_feature_columns"]]
        y_train = train_df[schema["target_column"]]
        x_test = test_df[schema["predictive_feature_columns"]]

        preprocessor = build_train_only_preprocessor(
            schema=schema, drop_zero_variance=True
        )
        x_train_transformed = preprocessor.fit_transform(x_train, y_train)
        x_test_transformed = preprocessor.transform(x_test)

        model = LogisticRegression(max_iter=1000, random_state=20260327)
        model.fit(x_train_transformed, y_train)
        print(
            f"Held-out domain: {held_out_domain} | "
            f"train shape after preprocessing: {x_train_transformed.shape} | "
            f"test shape after preprocessing: {x_test_transformed.shape}"
        )


if __name__ == "__main__":
    run_example()
