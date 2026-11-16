"""Shared input and metric functions from the original project."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from mod_mc.data.stonelab_osugi_dataset import MODEL_FEATURE_COLUMNS

def _load_dataset(path: Path, target_column: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["target_date"])
    required = {"target_date", "year", target_column, *MODEL_FEATURE_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"训练数据缺少字段：{', '.join(missing)}")
    if data["target_date"].duplicated().any():
        raise ValueError("target_date 存在重复值")
    if not set(data[target_column].unique()).issubset({0, 1}):
        raise ValueError("目标标签必须仅包含 0 和 1")
    return data.sort_values("target_date").reset_index(drop=True)


def filter_data_by_year(
    data: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """将一次实验可见的数据严格限制在配置指定的年份范围内。"""
    if start_year > end_year:
        raise ValueError("数据起始年份不能晚于结束年份")
    selected = data.loc[data["year"].between(start_year, end_year)].copy()
    if selected.empty:
        raise ValueError("指定年份范围内没有可用样本")
    return selected.sort_values("target_date").reset_index(drop=True)


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    has_both_classes = y_true.nunique() == 2
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predicted)) if has_both_classes else None
        ),
        "precision": (
            float(precision_score(y_true, predicted, zero_division=0))
            if int(predicted.sum()) > 0
            else None
        ),
        "recall_sensitivity": (
            float(recall_score(y_true, predicted, zero_division=0))
            if int(y_true.sum()) > 0
            else None
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "f1": (
            float(f1_score(y_true, predicted, zero_division=0))
            if int(y_true.sum()) > 0 and int(predicted.sum()) > 0
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_true, probabilities)) if has_both_classes else None
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "mean_risk_score": float(np.mean(probabilities)),
        "false_alarms": int(fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
