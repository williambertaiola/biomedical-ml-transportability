"""Transportability pipeline module: harmonization."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from typing import Any, Callable

import numpy as np
import pandas as pd

from uci_transportability.config import ExperimentConfig


EPSILON = 1e-8


@dataclass(frozen=True)
class CombatRoles:
    combat_features: list[str]
    numeric_passthrough: list[str]
    categorical_passthrough: list[str]
    covariate_continuous: list[str]
    covariate_categorical: list[str]

    @property
    def all_model_features(self) -> list[str]:
        ordered = [
            *self.combat_features,
            *self.numeric_passthrough,
            *self.categorical_passthrough,
        ]
        return list(dict.fromkeys(ordered))


@lru_cache(maxsize=1)
def _load_neurocombat() -> Callable[..., dict[str, Any]] | None:
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        return None
    return neuroCombat


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return False
    return bool(np.allclose(numeric, np.round(numeric), atol=1e-8))


def infer_combat_roles(
    *,
    df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> CombatRoles:
    predictive_columns = schema["predictive_feature_columns"]
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    covariate_continuous = [
        column
        for column in config.combat_continuous_covariates
        if column in predictive_columns
    ]
    covariate_categorical = [
        column
        for column in config.combat_categorical_covariates
        if column in predictive_columns
    ]

    combat_features: list[str] = []
    numeric_passthrough: list[str] = []
    categorical_passthrough: list[str] = []

    for column in predictive_columns:
        if column in covariate_continuous:
            numeric_passthrough.append(column)
            continue
        if column in covariate_categorical:
            categorical_passthrough.append(column)
            continue
        if column in categorical_features:
            categorical_passthrough.append(column)
            continue
        if column in numeric_features:
            series = pd.to_numeric(df[column], errors="coerce")
            if (
                _is_integer_like(series)
                and int(series.nunique(dropna=True))
                <= config.combat_low_cardinality_threshold
            ):
                categorical_passthrough.append(column)
            else:
                combat_features.append(column)
            continue
        categorical_passthrough.append(column)

    return CombatRoles(
        combat_features=list(dict.fromkeys(combat_features)),
        numeric_passthrough=list(dict.fromkeys(numeric_passthrough)),
        categorical_passthrough=list(dict.fromkeys(categorical_passthrough)),
        covariate_continuous=list(dict.fromkeys(covariate_continuous)),
        covariate_categorical=list(dict.fromkeys(covariate_categorical)),
    )


def _build_fill_values(
    train_df: pd.DataFrame,
    roles: CombatRoles,
) -> tuple[dict[str, float], dict[str, str]]:
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}

    for column in dict.fromkeys([*roles.combat_features, *roles.numeric_passthrough]):
        numeric = pd.to_numeric(train_df[column], errors="coerce")
        numeric_fill[column] = float(numeric.median()) if numeric.notna().any() else 0.0

    for column in dict.fromkeys(
        [*roles.covariate_categorical, *roles.categorical_passthrough]
    ):
        series = train_df[column].dropna()
        categorical_fill[column] = (
            str(series.mode().iloc[0]) if not series.empty else "missing"
        )

    return numeric_fill, categorical_fill


def _filled_covariates(
    *,
    train_df: pd.DataFrame,
    roles: CombatRoles,
    numeric_fill: dict[str, float],
    categorical_fill: dict[str, str],
) -> pd.DataFrame:
    covars = pd.DataFrame(index=train_df.index)
    for column in roles.covariate_continuous:
        covars[column] = (
            pd.to_numeric(train_df[column], errors="coerce")
            .fillna(numeric_fill[column])
            .astype(float)
        )
    for column in roles.covariate_categorical:
        covars[column] = (
            train_df[column]
            .astype("object")
            .where(train_df[column].notna(), categorical_fill[column])
            .astype(str)
        )
    return covars


def _build_covariate_design(
    covariates: pd.DataFrame,
    roles: CombatRoles,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in roles.covariate_continuous:
        if column in covariates.columns:
            frames.append(
                pd.DataFrame(
                    {
                        column: pd.to_numeric(
                            covariates[column], errors="coerce"
                        ).astype(float)
                    }
                )
            )
    for column in roles.covariate_categorical:
        if column in covariates.columns:
            dummies = pd.get_dummies(
                covariates[column].astype(str),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            if not dummies.empty:
                frames.append(dummies)

    if not frames:
        return pd.DataFrame(index=covariates.index)
    return pd.concat(frames, axis=1)


def _aprior(values: np.ndarray) -> float:
    mean_value = float(np.mean(values))
    var_value = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    if var_value < EPSILON:
        return 1e6
    return float((2.0 * var_value + mean_value**2) / var_value)


def _bprior(values: np.ndarray) -> float:
    mean_value = float(np.mean(values))
    var_value = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    if var_value < EPSILON:
        return mean_value * 1e6
    return float((mean_value * var_value + mean_value**3) / var_value)


def _it_sol(
    standardized_batch_data: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_bar: float,
    t2: float,
    a_prior: float,
    b_prior: float,
    convergence: float = 1e-4,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    n_batch_samples = standardized_batch_data.shape[1]
    gamma_old = gamma_hat.copy()
    delta_old = np.clip(delta_hat.copy(), EPSILON, None)

    for _ in range(max_iter):
        gamma_new = (t2 * n_batch_samples * gamma_hat + delta_old * gamma_bar) / (
            t2 * n_batch_samples + delta_old
        )
        sum_squared = ((standardized_batch_data - gamma_new[:, None]) ** 2).sum(axis=1)
        delta_new = (0.5 * sum_squared + b_prior) / (
            n_batch_samples / 2.0 + a_prior - 1.0
        )
        delta_new = np.clip(delta_new, EPSILON, None)

        change_gamma = np.max(
            np.abs(gamma_new - gamma_old) / (np.abs(gamma_old) + EPSILON)
        )
        change_delta = np.max(
            np.abs(delta_new - delta_old) / (np.abs(delta_old) + EPSILON)
        )
        gamma_old = gamma_new
        delta_old = delta_new
        if max(change_gamma, change_delta) < convergence:
            break

    return gamma_old, delta_old


def _fit_local_parametric_combat_train_only(
    *,
    train_matrix: pd.DataFrame,
    batch_train: pd.Series,
    covariates_train: pd.DataFrame,
    roles: CombatRoles,
) -> pd.DataFrame:
    if train_matrix.empty:
        return train_matrix.copy()

    batch_series = batch_train.astype(str).reset_index(drop=True)
    batch_design = pd.get_dummies(batch_series, drop_first=False, dtype=float)
    covariate_design = _build_covariate_design(
        covariates_train.reset_index(drop=True), roles
    )
    design = pd.concat([batch_design, covariate_design], axis=1)

    data = train_matrix.reset_index(drop=True).to_numpy(dtype=float).T
    design_matrix = design.to_numpy(dtype=float)
    b_hat = np.linalg.pinv(design_matrix) @ train_matrix.reset_index(
        drop=True
    ).to_numpy(dtype=float)

    n_batch_levels = batch_design.shape[1]
    batch_coefficients = b_hat[:n_batch_levels, :]
    batch_sizes = batch_design.sum(axis=0).to_numpy(dtype=float)
    grand_mean = (batch_sizes / batch_sizes.sum()) @ batch_coefficients

    if covariate_design.shape[1] > 0:
        stand_mean = (
            np.tile(grand_mean[:, None], (1, data.shape[1]))
            + (covariate_design.to_numpy(dtype=float) @ b_hat[n_batch_levels:, :]).T
        )
    else:
        stand_mean = np.tile(grand_mean[:, None], (1, data.shape[1]))

    fitted_values = (design_matrix @ b_hat).T
    residuals = data - fitted_values
    var_pooled = residuals.var(axis=1, ddof=0)
    var_pooled = np.clip(var_pooled, EPSILON, None)
    standardized_data = (data - stand_mean) / np.sqrt(var_pooled[:, None])

    batch_masks = [
        batch_series.to_numpy() == level for level in batch_design.columns.tolist()
    ]
    gamma_hat = np.zeros((len(batch_masks), standardized_data.shape[0]), dtype=float)
    delta_hat = np.zeros((len(batch_masks), standardized_data.shape[0]), dtype=float)

    for batch_index, mask in enumerate(batch_masks):
        batch_block = standardized_data[:, mask]
        gamma_hat[batch_index, :] = batch_block.mean(axis=1)
        if batch_block.shape[1] > 1:
            delta_hat[batch_index, :] = batch_block.var(axis=1, ddof=1)
        else:
            delta_hat[batch_index, :] = 1.0

    delta_hat = np.clip(delta_hat, EPSILON, None)
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    t2 = np.clip(t2, EPSILON, None)
    a_prior = np.apply_along_axis(_aprior, 1, delta_hat)
    b_prior = np.apply_along_axis(_bprior, 1, delta_hat)

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    for batch_index, mask in enumerate(batch_masks):
        gamma_star[batch_index, :], delta_star[batch_index, :] = _it_sol(
            standardized_batch_data=standardized_data[:, mask],
            gamma_hat=gamma_hat[batch_index, :],
            delta_hat=delta_hat[batch_index, :],
            gamma_bar=float(gamma_bar[batch_index]),
            t2=float(t2[batch_index]),
            a_prior=float(a_prior[batch_index]),
            b_prior=float(b_prior[batch_index]),
        )

    harmonized = standardized_data.copy()
    for batch_index, mask in enumerate(batch_masks):
        harmonized[:, mask] = (
            harmonized[:, mask] - gamma_star[batch_index, :][:, None]
        ) / np.sqrt(delta_star[batch_index, :][:, None])

    adjusted = harmonized * np.sqrt(var_pooled[:, None]) + stand_mean
    return pd.DataFrame(
        adjusted.T, index=train_matrix.index, columns=train_matrix.columns
    )


def _split_constant_and_variable_features(
    train_matrix: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    variable_features: list[str] = []
    constant_features: list[str] = []
    for column in train_matrix.columns:
        values = pd.to_numeric(train_matrix[column], errors="coerce")
        if values.nunique(dropna=False) <= 1:
            constant_features.append(column)
            continue
        if float(values.var(ddof=0)) < EPSILON:
            constant_features.append(column)
            continue
        variable_features.append(column)
    return variable_features, constant_features


def _choose_reference_batch(batch_train: pd.Series) -> str:
    return str(batch_train.astype(str).value_counts().idxmax())


def _fit_neurocombat_train_only(
    *,
    train_matrix: pd.DataFrame,
    batch_train: pd.Series,
    covariates_train: pd.DataFrame,
    roles: CombatRoles,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    neurocombat = _load_neurocombat()
    if neurocombat is None:
        raise ImportError(
            "neuroCombat is not available in the current Python environment."
        )

    batch_series = batch_train.astype(str).reset_index(drop=True)
    covars = covariates_train.reset_index(drop=True).copy()
    covars["batch"] = batch_series.to_numpy()
    covars = covars[["batch", *covariates_train.columns.tolist()]]
    reference_batch = _choose_reference_batch(batch_series)

    with redirect_stdout(StringIO()):
        result = neurocombat(
            dat=train_matrix.reset_index(drop=True).to_numpy(dtype=float).T,
            covars=covars,
            batch_col="batch",
            categorical_cols=roles.covariate_categorical or None,
            continuous_cols=roles.covariate_continuous or None,
            ref_batch=reference_batch,
        )

    harmonized = pd.DataFrame(
        result["data"].T, index=train_matrix.index, columns=train_matrix.columns
    )
    return harmonized, {
        "combat_backend": "neuroCombat",
        "combat_reference_batch": reference_batch,
    }


def _fit_best_available_combat_train_only(
    *,
    train_matrix: pd.DataFrame,
    batch_train: pd.Series,
    covariates_train: pd.DataFrame,
    roles: CombatRoles,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    neurocombat = _load_neurocombat()
    if neurocombat is not None:
        return _fit_neurocombat_train_only(
            train_matrix=train_matrix,
            batch_train=batch_train,
            covariates_train=covariates_train,
            roles=roles,
        )

    harmonized = _fit_local_parametric_combat_train_only(
        train_matrix=train_matrix,
        batch_train=batch_train,
        covariates_train=covariates_train,
        roles=roles,
    )
    return harmonized, {
        "combat_backend": "local_parametric_combat",
        "combat_reference_batch": _choose_reference_batch(batch_train),
    }


def prepare_method_train_test_frames(
    *,
    method_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    if method_name == "M0":
        return (
            train_out,
            test_out,
            {
                "harmonization": "none",
                "combat_backend": "none",
                "combat_reference_batch": "",
                "n_combat_features": 0,
                "combat_features": "",
                "combat_skipped_features": "",
                "covariate_continuous": "",
                "covariate_categorical": "",
            },
        )

    if method_name in {"M2", "M3"}:
        return (
            train_out,
            test_out,
            {
                "harmonization": "none",
                "combat_backend": "none",
                "combat_reference_batch": "",
                "n_combat_features": 0,
                "combat_features": "",
                "combat_skipped_features": "",
                "covariate_continuous": "",
                "covariate_categorical": "",
            },
        )

    if method_name not in {"M1", "M4"}:
        raise ValueError(
            f"Unsupported method requested in this pipeline: {method_name}"
        )

    roles = infer_combat_roles(df=train_df, schema=schema, config=config)
    if not roles.combat_features:
        return (
            train_out,
            test_out,
            {
                "harmonization": "combat_skipped_no_features",
                "combat_backend": "none",
                "combat_reference_batch": "",
                "n_combat_features": 0,
                "combat_features": "",
                "combat_skipped_features": "",
                "covariate_continuous": "|".join(roles.covariate_continuous),
                "covariate_categorical": "|".join(roles.covariate_categorical),
            },
        )

    numeric_fill, categorical_fill = _build_fill_values(train_df, roles)
    filled_train_matrix = pd.DataFrame(index=train_df.index)
    for column in roles.combat_features:
        filled_train_matrix[column] = (
            pd.to_numeric(train_df[column], errors="coerce")
            .fillna(numeric_fill[column])
            .astype(float)
        )

    active_combat_features, skipped_combat_features = (
        _split_constant_and_variable_features(filled_train_matrix)
    )
    if not active_combat_features:
        return (
            train_out,
            test_out,
            {
                "harmonization": "combat_skipped_constant_features",
                "combat_backend": "none",
                "combat_reference_batch": "",
                "n_combat_features": 0,
                "combat_features": "",
                "combat_skipped_features": "|".join(skipped_combat_features),
                "covariate_continuous": "|".join(roles.covariate_continuous),
                "covariate_categorical": "|".join(roles.covariate_categorical),
            },
        )

    covariates_train = _filled_covariates(
        train_df=train_df,
        roles=roles,
        numeric_fill=numeric_fill,
        categorical_fill=categorical_fill,
    )
    harmonized_train_matrix, runtime_details = _fit_best_available_combat_train_only(
        train_matrix=filled_train_matrix[active_combat_features],
        batch_train=train_df[config.domain_column],
        covariates_train=covariates_train,
        roles=roles,
    )

    for column in active_combat_features:
        train_out[column] = harmonized_train_matrix[column].astype(float).to_numpy()
        train_out.loc[train_df[column].isna(), column] = np.nan

    return (
        train_out,
        test_out,
        {
            "harmonization": "combat_train_only",
            "combat_backend": runtime_details.get("combat_backend", "unknown"),
            "combat_reference_batch": runtime_details.get("combat_reference_batch", ""),
            "n_combat_features": int(len(active_combat_features)),
            "combat_features": "|".join(active_combat_features),
            "combat_skipped_features": "|".join(skipped_combat_features),
            "covariate_continuous": "|".join(roles.covariate_continuous),
            "covariate_categorical": "|".join(roles.covariate_categorical),
        },
    )
