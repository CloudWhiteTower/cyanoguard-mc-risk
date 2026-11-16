"""Tune XGBoost with expanding-year CV and reserve 2021 for validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from mod_mc.data.stonelab_osugi_dataset import MODEL_FEATURE_COLUMNS
from mod_mc.training.dataset_metrics import (
    _load_dataset,
    classification_metrics,
    filter_data_by_year,
)


def expanding_year_folds(
    data: pd.DataFrame,
    available_years: list[int],
    validation_years: list[int],
    target: str,
) -> list[tuple[list[int], int, pd.DataFrame, pd.DataFrame]]:
    """Construct leakage-free expanding-window folds grouped by calendar year."""
    folds = []
    for validation_year in validation_years:
        train_years = [year for year in available_years if year < validation_year]
        train = data.loc[data["year"].isin(train_years)].copy()
        validation = data.loc[data["year"].eq(validation_year)].copy()
        if train.empty or validation.empty:
            raise ValueError(f"Cannot construct temporal fold for {validation_year}")
        if train["target_date"].max() >= validation["target_date"].min():
            raise ValueError(f"Temporal leakage detected in fold {validation_year}")
        if train[target].nunique() < 2 or validation[target].nunique() < 2:
            raise ValueError(f"Fold {validation_year} must contain both classes")
        folds.append((train_years, validation_year, train, validation))
    return folds


def choose_rounded_mcc_threshold(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
    decimals: int = 3,
) -> tuple[float, pd.DataFrame]:
    """Choose a quantized threshold by MCC, with balanced metrics as tie-breakers."""
    labels_array = np.asarray(labels, dtype=int)
    if np.unique(labels_array).size < 2:
        raise ValueError("Threshold selection requires positive and negative observations")
    step = 10.0 ** (-decimals)
    candidates = np.round(np.arange(step, 1.0, step), decimals)
    rows = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels_array,
            predictions,
            labels=[0, 1],
        ).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "mcc": float(matthews_corrcoef(labels_array, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels_array, predictions)
                ),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    curve = pd.DataFrame(rows)
    ranked = curve.sort_values(
        ["mcc", "balanced_accuracy", "specificity", "sensitivity", "threshold"],
        ascending=[False, False, False, False, False],
    )
    return float(ranked.iloc[0]["threshold"]), curve


def derive_rounded_tier_thresholds(
    scores: np.ndarray,
    warning_threshold: float,
    decimals: int = 3,
) -> tuple[float, float]:
    """Derive lower response boundaries from non-warning validation scores."""
    non_warning = scores[scores < warning_threshold]
    if non_warning.size == 0:
        raise ValueError("No validation score is below the warning threshold")
    low = round(float(np.quantile(non_warning, 0.75)), decimals)
    medium = round(float(np.quantile(non_warning, 0.95)), decimals)
    step = 10.0 ** (-decimals)
    medium = min(medium, warning_threshold - step)
    low = min(low, medium - step)
    if not 0 <= low < medium < warning_threshold:
        raise ValueError("Rounded response boundaries are not strictly ordered")
    return low, medium


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_metrics(
    candidate_id: int,
    parameters: dict[str, Any],
    fixed_parameters: dict[str, Any],
    folds: list[tuple[list[int], int, pd.DataFrame, pd.DataFrame]],
    features: list[str],
    target: str,
    random_state: int,
    baseline: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows = []
    for train_years, validation_year, train, validation in folds:
        model = XGBClassifier(
            **fixed_parameters,
            **parameters,
            random_state=random_state,
        )
        model.fit(train[features], train[target], verbose=False)
        scores = model.predict_proba(validation[features])[:, 1]
        fold_rows.append(
            {
                "candidate": candidate_id,
                "train_years": ",".join(str(year) for year in train_years),
                "validation_year": validation_year,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "validation_positive_rows": int(validation[target].sum()),
                "roc_auc": float(roc_auc_score(validation[target], scores)),
                "pr_auc": float(average_precision_score(validation[target], scores)),
                "brier_score": float(brier_score_loss(validation[target], scores)),
                "log_loss": float(log_loss(validation[target], scores, labels=[0, 1])),
            }
        )
    table = pd.DataFrame(fold_rows)
    mean_roc_auc = float(table["roc_auc"].mean())
    mean_pr_auc = float(table["pr_auc"].mean())
    mean_brier = float(table["brier_score"].mean())
    min_roc_auc = float(table["roc_auc"].min())
    selection_score = (
        0.50 * mean_roc_auc
        + 0.25 * mean_pr_auc
        + 0.15 * (1.0 - mean_brier)
        + 0.10 * min_roc_auc
    )
    summary = {
        "candidate": candidate_id,
        "is_original_baseline": baseline,
        **{key: _python_value(value) for key, value in parameters.items()},
        "selection_score": float(selection_score),
        "mean_roc_auc": mean_roc_auc,
        "minimum_year_roc_auc": min_roc_auc,
        "std_roc_auc": float(table["roc_auc"].std(ddof=0)),
        "mean_pr_auc": mean_pr_auc,
        "mean_brier_score": mean_brier,
        "mean_log_loss": float(table["log_loss"].mean()),
    }
    return summary, fold_rows


def run_temporal_cv_experiment(config_path: Path, project_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    target = str(config["target_column"])
    data = _load_dataset(project_root / config["data_path"], target)
    data = filter_data_by_year(
        data,
        int(config["data_scope"]["start_year"]),
        int(config["data_scope"]["end_year"]),
    )
    development_years = [int(year) for year in config["development_years"]]
    validation_year = int(config["validation_year"])
    development = data.loc[data["year"].isin(development_years)].copy()
    validation = data.loc[data["year"].eq(validation_year)].copy()
    if development["target_date"].max() >= validation["target_date"].min():
        raise ValueError("Development data must precede the held-out validation year")

    features = list(MODEL_FEATURE_COLUMNS)
    random_state = int(config["random_state"])
    folds = expanding_year_folds(
        development,
        development_years,
        [int(year) for year in config["cv_validation_years"]],
        target,
    )
    sampled_parameters = list(
        ParameterSampler(
            config["search_space"],
            n_iter=int(config["search_iterations"]),
            random_state=random_state,
        )
    )
    baseline_parameters = dict(config["original_baseline_parameters"])
    candidates = [baseline_parameters, *sampled_parameters]
    fixed_parameters = dict(config["model_fixed"])
    summaries = []
    all_fold_rows = []
    for candidate_id, parameters in enumerate(candidates):
        summary, fold_rows = _candidate_metrics(
            candidate_id,
            parameters,
            fixed_parameters,
            folds,
            features,
            target,
            random_state,
            baseline=candidate_id == 0,
        )
        summaries.append(summary)
        all_fold_rows.extend(fold_rows)

    search_table = pd.DataFrame(summaries).sort_values(
        [
            "selection_score",
            "minimum_year_roc_auc",
            "mean_roc_auc",
            "mean_pr_auc",
            "mean_brier_score",
        ],
        ascending=[False, False, False, False, True],
    )
    search_table["rank"] = np.arange(1, len(search_table) + 1)
    minimum_acceptable_auc = float(config["minimum_acceptable_year_roc_auc"])
    search_table["passes_minimum_year_auc"] = search_table[
        "minimum_year_roc_auc"
    ].ge(minimum_acceptable_auc)
    eligible = search_table.loc[search_table["passes_minimum_year_auc"]]
    if eligible.empty:
        raise ValueError("No candidate meets the minimum yearly ROC-AUC requirement")
    best_row = eligible.iloc[0]
    best_candidate = int(best_row["candidate"])
    best_parameters = candidates[best_candidate]

    model = XGBClassifier(
        **fixed_parameters,
        **best_parameters,
        random_state=random_state,
    )
    model.fit(development[features], development[target], verbose=False)
    validation_scores = model.predict_proba(validation[features])[:, 1]
    threshold_decimals = int(config["threshold_decimals"])
    threshold, threshold_curve = choose_rounded_mcc_threshold(
        validation[target],
        validation_scores,
        decimals=threshold_decimals,
    )
    low_threshold, medium_threshold = derive_rounded_tier_thresholds(
        validation_scores,
        threshold,
        decimals=threshold_decimals,
    )
    validation_metrics = classification_metrics(
        validation[target],
        validation_scores,
        threshold,
    )

    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.json")
    predictions = validation[
        ["target_date", "year", "MC_day_max_ug_L", target]
    ].copy()
    predictions["risk_score"] = validation_scores
    predictions["predicted_label"] = (validation_scores >= threshold).astype(int)
    predictions["risk_tier"] = np.select(
        [
            validation_scores < low_threshold,
            validation_scores < medium_threshold,
            validation_scores < threshold,
        ],
        ["normal", "low", "medium"],
        default="warning",
    )
    predictions.to_csv(
        output_dir / "validation_2021_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    threshold_curve.to_csv(output_dir / "validation_2021_threshold_curve.csv", index=False)
    search_table.to_csv(output_dir / "cv_hyperparameter_search.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    baseline_row = search_table.loc[search_table["is_original_baseline"]].iloc[0]
    metrics = {
        "experiment_name": config["experiment_name"],
        "task": "past 7 days -> target-day Total Microcystins >= 1 ug/L",
        "data_scope": {
            "start_year": int(config["data_scope"]["start_year"]),
            "end_year": int(config["data_scope"]["end_year"]),
            "rows": int(len(data)),
            "positive_rows": int(data[target].sum()),
        },
        "development": {
            "years": development_years,
            "rows": int(len(development)),
            "positive_rows": int(development[target].sum()),
        },
        "held_out_validation": {
            "year": validation_year,
            "rows": int(len(validation)),
            "positive_rows": int(validation[target].sum()),
        },
        "cross_validation": {
            "method": "four-fold expanding-window validation grouped by year",
            "validation_years": config["cv_validation_years"],
            "candidate_count": len(candidates),
            "selection_rule": (
                "require every yearly ROC-AUC >= 0.50, then maximize "
                "0.50*mean_ROC_AUC + 0.25*mean_PR_AUC + "
                "0.15*(1-mean_Brier) + 0.10*minimum_year_ROC_AUC"
            ),
            "minimum_acceptable_year_roc_auc": minimum_acceptable_auc,
            "best_candidate": best_candidate,
            "best_parameters": {
                **fixed_parameters,
                **{key: _python_value(value) for key, value in best_parameters.items()},
            },
            "best_summary": {
                key: float(best_row[key])
                for key in (
                    "selection_score",
                    "mean_roc_auc",
                    "minimum_year_roc_auc",
                    "std_roc_auc",
                    "mean_pr_auc",
                    "mean_brier_score",
                    "mean_log_loss",
                )
            },
            "original_baseline": {
                "rank": int(baseline_row["rank"]),
                "selection_score": float(baseline_row["selection_score"]),
                "mean_roc_auc": float(baseline_row["mean_roc_auc"]),
                "minimum_year_roc_auc": float(baseline_row["minimum_year_roc_auc"]),
                "mean_pr_auc": float(baseline_row["mean_pr_auc"]),
                "mean_brier_score": float(baseline_row["mean_brier_score"]),
            },
        },
        "threshold_selection": (
            "2021 held-out validation; 0.001 grid; maximize MCC, then balanced accuracy, "
            "specificity, and sensitivity"
        ),
        "decision_threshold": threshold,
        "threshold_decimals": threshold_decimals,
        "operational_thresholds": {
            "normal_to_low": low_threshold,
            "low_to_medium": medium_threshold,
            "medium_to_warning": threshold,
            "method": (
                "75th and 95th percentiles of non-warning 2021 validation scores; "
                "all boundaries rounded to three decimals"
            ),
        },
        "validation_2021": validation_metrics,
        "validation_2021_risk_tier_counts": {
            str(name): int(count)
            for name, count in predictions["risk_tier"].value_counts().items()
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return output_dir
