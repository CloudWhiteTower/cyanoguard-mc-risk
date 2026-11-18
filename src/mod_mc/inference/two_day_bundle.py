"""Inference-only composition of two frozen XGBoost calendar-horizon models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from mod_mc.data.stonelab_osugi_dataset import (
    MODEL_FEATURE_COLUMNS, OSUGI_VARIABLES, VARIABLES, _window_statistics,
)
from mod_mc.data.mc_history_dataset import MC_HISTORY_FEATURES

FEATURE_COLUMNS = [*MODEL_FEATURE_COLUMNS, *MC_HISTORY_FEATURES]


def calendar_dates(values, name: str) -> pd.Series:
    dates = pd.to_datetime(pd.Series(values), errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError(f"{name} must contain nonmissing timezone-naive calendar dates")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{name} must use dates at midnight, not intraday timestamps")
    return dates.reset_index(drop=True)


def _daily(data: pd.DataFrame, date_column: str, source_name: str) -> pd.DataFrame:
    if date_column not in data:
        raise ValueError(f"{source_name} is missing {date_column}")
    result = data.copy()
    result[date_column] = calendar_dates(result[date_column], source_name).to_numpy()
    if result[date_column].duplicated().any():
        raise ValueError(f"{source_name} must already be aggregated to unique dates")
    return result.sort_values(date_column).reset_index(drop=True)


def build_shared_features(
    stonelab_daily: pd.DataFrame,
    osugi_daily: pd.DataFrame,
    mc_daily: pd.DataFrame,
    window_end_dates,
) -> pd.DataFrame:
    """One feature vector for [end-6, end]; no future-day labels are required."""
    ends = calendar_dates(window_end_dates, "window_end_date")
    if ends.empty or ends.duplicated().any():
        raise ValueError("At least one unique window end date is required")
    stone = _daily(stonelab_daily, "date", "Stone Lab daily")
    osugi = _daily(osugi_daily, "date", "OSUGI daily")
    mc = _daily(mc_daily, "target_date", "MC daily")
    if "MC_day_max_ug_L" not in mc:
        raise ValueError("MC daily requires MC_day_max_ug_L in ug/L")
    mc["MC_day_max_ug_L"] = pd.to_numeric(mc.MC_day_max_ug_L, errors="raise")
    if "MC_all_below_0_15" in mc:
        mc["MC_all_below_0_15"] = pd.to_numeric(mc.MC_all_below_0_15, errors="raise")
        if not mc.MC_all_below_0_15.isin([0, 1]).all():
            raise ValueError("MC censoring flags must be 0 or 1")
        if mc.loc[mc.MC_all_below_0_15.eq(1), "MC_day_max_ug_L"].notna().any():
            raise ValueError("Censored MC must remain missing; no fabricated concentration")
    sources = {}
    for variable in VARIABLES:
        source = osugi if variable in OSUGI_VARIABLES else stone
        values = pd.to_numeric(source[variable], errors="raise") if variable in source else pd.Series(np.nan, index=source.index)
        sources[variable] = (source.date, values)
    rows = []
    for end in ends:
        anchor = end + pd.Timedelta(days=1)
        row = {"window_start_date": end - pd.Timedelta(days=6), "window_end_date": end}
        for variable, (dates, values) in sources.items():
            mean, latest, trend, _, _ = _window_statistics(dates, values, anchor)
            for statistic, value in zip(("7d_mean", "latest", "trend"), (mean, latest, trend)):
                row[f"{variable}_{statistic}"] = value
        mean, latest, trend, _, _ = _window_statistics(mc.target_date, mc.MC_day_max_ug_L, anchor)
        row.update(dict(zip(MC_HISTORY_FEATURES, (mean, latest, trend))))
        rows.append(row)
    return pd.DataFrame(rows)


class TwoDayPredictor:
    """Load two existing models, pass identical X to both, keep separate thresholds."""

    def __init__(self, bundle_root: str | Path):
        self.root = Path(bundle_root).resolve()
        self.spec = json.loads((self.root / "bundle.json").read_text(encoding="utf-8"))
        if self.spec["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Bundle feature schema differs from the supported 42-feature schema")
        if [head["horizon_days"] for head in self.spec["heads"]] != [1, 2]:
            raise ValueError("Bundle must define horizons 1 and 2 in order")
        self.models = []
        for head in self.spec["heads"]:
            path = (self.root / head["model_path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Model path must stay inside the inference bundle")
            if hashlib.sha256(path.read_bytes()).hexdigest() != head["model_sha256"]:
                raise ValueError("Model hash does not match the frozen inference bundle")
            low, medium, warning = head["response_thresholds"]
            if not 0 <= low < medium < warning <= 1 or warning != head["decision_threshold"]:
                raise ValueError("Invalid response thresholds")
            model = XGBClassifier()
            model.load_model(path)
            if model.get_booster().feature_names != FEATURE_COLUMNS:
                raise ValueError("Model input names/order do not match the shared schema")
            self.models.append(model)

    def predict_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(["window_end_date", *FEATURE_COLUMNS]).difference(frame.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate input column names are not allowed")
        ends = calendar_dates(frame.window_end_date, "window_end_date")
        if ends.empty or ends.duplicated().any():
            raise ValueError("At least one unique window end date is required")
        if "window_start_date" in frame:
            starts = calendar_dates(frame.window_start_date, "window_start_date")
            if not starts.eq(ends - pd.Timedelta(days=6)).all():
                raise ValueError("Input must span exactly seven calendar days")
        features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError("Infinite inputs are invalid; missing values must be NaN")
        available = features.notna().sum(axis=1).to_numpy()
        result = pd.DataFrame({"model_version": "v3", "window_start_date": ends - pd.Timedelta(days=6),
            "window_end_date": ends, "available_feature_count": available,
            "input_status": np.where(available == 0, "all_missing", np.where(available < 42, "partially_missing", "complete"))})
        for head, model in zip(self.spec["heads"], self.models):
            prefix = f"day{head['horizon_days']}"
            scores = model.predict_proba(features)[:, 1]
            low, medium, warning = head["response_thresholds"]
            result[f"{prefix}_target_date"] = ends + pd.Timedelta(days=head["horizon_days"])
            result[f"{prefix}_risk_score"] = scores
            result[f"{prefix}_predicted_label"] = (scores >= warning).astype(int)
            result[f"{prefix}_response_level"] = np.select(
                [scores < low, scores < medium, scores < warning], ["normal", "low", "medium"], default="warning")
        return result

    def predict_daily(self, stonelab_daily, osugi_daily, mc_daily, window_end_dates):
        return self.predict_features(build_shared_features(stonelab_daily, osugi_daily, mc_daily, window_end_dates))
