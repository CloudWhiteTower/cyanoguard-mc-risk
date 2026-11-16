"""Validate a feature table and apply the bundled model without fitting it."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def risk_tiers(scores: np.ndarray, thresholds: dict) -> np.ndarray:
    """Apply the original left-closed response intervals to unrounded scores."""
    return np.select(
        [scores < thresholds["normal_to_low"],
         scores < thresholds["low_to_medium"],
         scores < thresholds["medium_to_warning"]],
        ["normal", "low", "medium"], default="warning",
    )


class FrozenPredictor:
    def __init__(self) -> None:
        resources = files("mod_mc.resources")
        self.metadata = json.loads(resources.joinpath("metadata.json").read_text("utf-8"))
        model_bytes = resources.joinpath("model.json").read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != self.metadata["model_sha256"]:
            raise ValueError("Bundled model checksum does not match metadata")
        self.model = XGBClassifier(n_jobs=1)
        self.model.load_model(bytearray(model_bytes))
        self.features = list(self.model.get_booster().feature_names or [])
        if self.features != self.metadata["feature_names"] or len(self.features) != 39:
            raise ValueError("Bundled model feature schema does not match metadata")
        self.thresholds = self.metadata["operational_thresholds"]
        low = self.thresholds["normal_to_low"]
        medium = self.thresholds["low_to_medium"]
        warning = self.thresholds["medium_to_warning"]
        if not 0 <= low < medium < warning <= 1:
            raise ValueError("Bundled thresholds must be ordered within [0, 1]")
        if warning != self.metadata["decision_threshold"]:
            raise ValueError("Warning threshold does not match decision threshold")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Input must contain at least one data row")
        if data.columns.duplicated().any():
            raise ValueError("Input contains duplicate column names")
        missing = sorted(set(self.features) - set(data.columns))
        if missing:
            raise ValueError("Missing feature columns: " + ", ".join(missing))
        numeric = pd.DataFrame(index=data.index)
        for name in self.features:
            values = data[name]
            if pd.api.types.is_bool_dtype(values):
                raise ValueError(f"Feature {name} must be numeric, not boolean")
            text = values.astype("string").str.strip()
            absent = values.isna() | text.eq("") | text.str.casefold().eq("nan")
            try:
                parsed = pd.to_numeric(values.where(~absent, np.nan), errors="raise")
                array = parsed.to_numpy(dtype=float, na_value=np.nan)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Feature {name} contains a nonnumeric value") from error
            if np.isinf(array).any() or (np.abs(array[np.isfinite(array)]) >
                                        np.finfo(np.float32).max).any():
                raise ValueError(f"Feature {name} contains infinity or exceeds float32 range")
            numeric[name] = array
        result = pd.DataFrame({"row_number": np.arange(1, len(data) + 1)})
        if "target_date" in data:
            text_dates = data["target_date"].astype(str)
            try:
                for value in text_dates:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                        raise ValueError(value)
                    date.fromisoformat(value)
            except ValueError as error:
                raise ValueError("target_date must contain ISO dates (YYYY-MM-DD)") from error
            if text_dates.duplicated().any():
                raise ValueError("target_date contains duplicate dates")
            result["target_date"] = text_dates.to_numpy()
        scores = self.model.predict_proba(numeric[self.features])[:, 1]
        result["risk_score"] = scores
        result["predicted_label"] = (scores >= self.metadata["decision_threshold"]).astype(int)
        result["risk_tier"] = risk_tiers(scores, self.thresholds)
        result["missing_feature_count"] = numeric.isna().sum(axis=1).to_numpy()
        return result


def read_feature_csv(path: Path) -> pd.DataFrame:
    """Check CSV structure before pandas can rename duplicate headers."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if not header or any(not item.strip() for item in header):
            raise ValueError("CSV needs a header with nonempty column names")
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names")
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(header)}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def predict_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Never overwrite an existing file, including input or frozen resources."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output already exists: {output_path}")
    result = FrozenPredictor().predict(read_feature_csv(input_path))
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False, float_format="%.9g", lineterminator="\n")
    return result
