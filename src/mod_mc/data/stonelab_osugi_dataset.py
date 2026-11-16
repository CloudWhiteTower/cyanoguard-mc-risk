"""构建 Stone Lab + OSUGI 七日前窗 Total Microcystin 分类数据集。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd

VARIABLES = (
    "WaterTemperature",
    "DissolvedOxygen",
    "pH",
    "Turbidity",
    "Conductivity",
    "ChlorophyllA",
    "TP",
    "TN",
    "NO3NO2",
    "Secchi",
    "Phycocyanin",
    "DRP",
    "NH4",
)
WINDOW_STATISTICS = ("7d_mean", "latest", "trend")
MODEL_FEATURE_COLUMNS = tuple(
    f"{variable}_{statistic}"
    for variable in VARIABLES
    for statistic in WINDOW_STATISTICS
)
AUXILIARY_FEATURE_COLUMNS = tuple(
    column
    for variable in VARIABLES
    for column in (f"{variable}_n_obs_7d", f"{variable}_days_since_last")
)

STONELAB_COLUMNS = {
    "ChlorophyllA": ("Chlorophyll (µg/L)", 1.0),
    "TP": ("TP (µg P/L)", 0.001),
    "TN": ("TN (µg N/L)", 0.001),
    "NO3NO2": ("Nitrate+NO2 (µg N/L)", 0.001),
    "Secchi": ("Secchi (m)", 1.0),
    "DRP": ("DRP (µg P/L)", 0.001),
    "NH4": ("Ammonium (µg N/L)", 0.001),
}

OSUGI_VARIABLES = {
    "WaterTemperature",
    "DissolvedOxygen",
    "pH",
    "Turbidity",
    "Conductivity",
    "Phycocyanin",
}

# 下载到的 NCEI 文件没有携带 ERDDAP schema 中列出的 QC flag。
# 这里只删除违反变量基本物理定义的值，不使用经验分位数或主观“正常范围”。
PHYSICAL_BOUNDS = {
    "DissolvedOxygen": (0.0, None),
    "pH": (0.0, 14.0),
    "Turbidity": (0.0, None),
    "Conductivity": (0.0, None),
    "Phycocyanin": (0.0, None),
}


@dataclass(frozen=True)
class BuildOutputs:
    dataset_path: Path
    labels_path: Path
    stonelab_daily_path: Path
    osugi_daily_path: Path
    summary_path: Path


def _classify_osugi_file(path: Path) -> str | None:
    name = path.name.lower()
    if "mass-concentration-of-oxygen" in name:
        return "DissolvedOxygen"
    if "sea-water-ph" in name or name == "osugi_ph.nc":
        return "pH"
    if "sea-water-temperature" in name or "sea_water_temperature" in name:
        return "WaterTemperature"
    if "turbidity" in name:
        return "Turbidity"
    if "conductivity" in name:
        return "Conductivity"
    if "blue_green_algae" in name:
        return "Phycocyanin"
    return None


def _measurement_variable(dataset: netCDF4.Dataset) -> netCDF4.Variable:
    if "obs" in dataset.variables:
        return dataset.variables["obs"]
    excluded = {
        "time",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "z",
        "height",
        "crs",
        "timeseries",
    }
    candidates = [
        variable
        for name, variable in dataset.variables.items()
        if name not in excluded and "time" in variable.dimensions and variable.dtype.kind in "fiu"
    ]
    if len(candidates) != 1:
        names = [variable.name for variable in candidates]
        raise ValueError(f"无法唯一确定 NetCDF 测量变量：{names}")
    return candidates[0]


def _explicit_invalid_mask(
    dataset: netCDF4.Dataset,
    measurement: netCDF4.Variable,
) -> np.ndarray:
    """仅依据文件 metadata 中明确标为坏/失败/无效的 QC 值过滤。"""
    shape = np.asarray(measurement[:]).shape
    invalid = np.zeros(shape, dtype=bool)
    ancillary_names = str(getattr(measurement, "ancillary_variables", "")).split()
    for name in ancillary_names:
        if name not in dataset.variables:
            continue
        qc = dataset.variables[name]
        if not hasattr(qc, "flag_values") or not hasattr(qc, "flag_meanings"):
            continue
        values = np.asarray(qc[:])
        meanings = str(qc.flag_meanings).split()
        flags = np.asarray(qc.flag_values).reshape(-1)
        for flag, meaning in zip(flags, meanings, strict=False):
            normalized = meaning.lower()
            if any(word in normalized for word in ("bad", "fail", "invalid")):
                invalid |= np.broadcast_to(values == flag, shape)
    return invalid


def _convert_osugi_units(values: np.ndarray, variable: str, units: str) -> np.ndarray:
    normalized = units.strip().lower().replace(" ", "")
    converted = values.astype(float, copy=True)
    if variable == "WaterTemperature":
        if normalized in {"k", "kelvin"}:
            converted -= 273.15
        elif normalized not in {"degree_celsius", "degrees_celsius", "degc", "c", "°c"}:
            raise ValueError(f"不支持的水温单位：{units!r}")
    elif variable == "DissolvedOxygen":
        if normalized in {"kg/m³", "kg/m3"}:
            converted *= 1000.0
        elif normalized not in {"mg/l", "mg/liter", "mg/litre"}:
            raise ValueError(f"不支持的 DO 单位：{units!r}")
    elif variable == "Turbidity" and normalized not in {"ntu"}:
        raise ValueError(f"不支持的浊度单位：{units!r}")
    elif variable == "Conductivity" and normalized not in {"us/cm", "µs/cm", "μs/cm"}:
        raise ValueError(f"不支持的电导率单位：{units!r}")
    elif variable == "Phycocyanin" and normalized not in {"rfu"}:
        raise ValueError(f"不支持的藻蓝蛋白单位：{units!r}")
    return converted


def _apply_physical_bounds(values: np.ndarray, variable: str) -> tuple[np.ndarray, int]:
    filtered = values.astype(float, copy=True)
    invalid = np.zeros(filtered.shape, dtype=bool)
    if variable in PHYSICAL_BOUNDS:
        lower, upper = PHYSICAL_BOUNDS[variable]
        if lower is not None:
            invalid |= filtered < lower
        if upper is not None:
            invalid |= filtered > upper
        filtered[invalid] = np.nan
    return filtered, int(invalid.sum())


def _netcdf_file_daily(path: Path, variable: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    with netCDF4.Dataset(path) as dataset:
        measurement = _measurement_variable(dataset)
        if "time" not in measurement.dimensions:
            raise ValueError(f"测量变量没有 time 维：{path}")
        raw = np.ma.asarray(measurement[:])
        values = np.asarray(raw.filled(np.nan), dtype=float)
        values[_explicit_invalid_mask(dataset, measurement)] = np.nan
        if hasattr(measurement, "valid_min"):
            values[values < float(measurement.valid_min)] = np.nan
        if hasattr(measurement, "valid_max"):
            values[values > float(measurement.valid_max)] = np.nan

        units = str(getattr(measurement, "units", ""))
        values = _convert_osugi_units(values, variable, units)
        values, physical_invalid_count = _apply_physical_bounds(values, variable)

        time_axis = measurement.dimensions.index("time")
        time_length = values.shape[time_axis]
        values = np.moveaxis(values, time_axis, 0).reshape(time_length, -1)
        valid_counts = np.isfinite(values).sum(axis=1)
        per_timestamp = np.divide(
            np.nansum(values, axis=1),
            valid_counts,
            out=np.full(time_length, np.nan),
            where=valid_counts > 0,
        )

        time_variable = dataset.variables["time"]
        calendar = getattr(time_variable, "calendar", "standard")
        decoded = netCDF4.num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=calendar,
            only_use_cftime_datetimes=True,
        )
        dates = pd.to_datetime(
            [f"{item.year:04d}-{item.month:02d}-{item.day:02d}" for item in decoded]
        )
        frame = pd.DataFrame({"date": dates, "value": per_timestamp}).dropna(subset=["value"])
        daily = frame.groupby("date", as_index=False)["value"].mean()
        audit = {
            "source_file": str(path),
            "variable": variable,
            "measurement_name": measurement.name,
            "source_units": units,
            "raw_time_points": int(len(dates)),
            "valid_time_points": int(frame.shape[0]),
            "physical_invalid_values": physical_invalid_count,
            "daily_rows": int(daily.shape[0]),
            "explicit_qc_variables": str(
                getattr(measurement, "ancillary_variables", "")
            ).split(),
        }
    return daily, audit


def build_osugi_daily(osugi_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    per_file: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for path in sorted(osugi_root.rglob("*.nc")):
        variable = _classify_osugi_file(path)
        if variable is None:
            continue
        daily, audit = _netcdf_file_daily(path, variable)
        audit["source_file"] = str(path.relative_to(osugi_root))
        daily["variable"] = variable
        daily["source_file"] = str(path.relative_to(osugi_root))
        per_file.append(daily)
        audit_rows.append(audit)
    if not per_file:
        raise ValueError("没有找到可用于模型的 OSUGI NetCDF")

    long = pd.concat(per_file, ignore_index=True)
    # 先得到每个传感器文件的日均值，再对同一变量的多个传感器日均值等权平均。
    combined = long.groupby(["date", "variable"], as_index=False)["value"].mean()
    daily = combined.pivot(index="date", columns="variable", values="value").reset_index()
    daily.columns.name = None
    return daily.sort_values("date").reset_index(drop=True), audit_rows


def parse_microcystin(value: object) -> tuple[bool, float | None, bool]:
    """返回（是否测量、精确数值、是否低于 0.15 的删失观测）。"""
    if pd.isna(value) or not str(value).strip():
        return False, None, False
    text = str(value).strip()
    if text.startswith("<"):
        return True, None, True
    try:
        return True, float(text), False
    except ValueError as error:
        raise ValueError(f"无法解析 Total Microcystins：{value!r}") from error


def build_stonelab_daily(
    source_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(source_path)
    source["date"] = pd.to_datetime(source["Date"], errors="raise")

    daily_features = pd.DataFrame({"date": sorted(source["date"].unique())})
    for output, (column, factor) in STONELAB_COLUMNS.items():
        numeric = pd.to_numeric(source[column], errors="coerce") * factor
        grouped = (
            pd.DataFrame({"date": source["date"], output: numeric})
            .groupby("date", as_index=False)[output]
            .mean()
        )
        daily_features = daily_features.merge(grouped, on="date", how="left")

    parsed = source["Total Microcystins (µg/L)"].map(parse_microcystin)
    source["mc_measured"] = parsed.map(lambda item: item[0])
    source["mc_exact"] = parsed.map(lambda item: item[1])
    source["mc_below_0_15"] = parsed.map(lambda item: item[2])

    label_rows: list[dict[str, Any]] = []
    for date, group in source.groupby("date"):
        measured = group.loc[group["mc_measured"]]
        if measured.empty:
            continue
        exact = pd.to_numeric(measured["mc_exact"], errors="coerce")
        day_max = float(exact.max()) if exact.notna().any() else np.nan
        label_rows.append(
            {
                "target_date": date,
                "year": int(date.year),
                "MC_day_max_ug_L": day_max,
                "MC_all_below_0_15": int(exact.isna().all()),
                "MC_ge_1ug_L": int(exact.ge(1.0).any()),
                "MC_measurement_rows": int(len(measured)),
            }
        )
    labels = pd.DataFrame(label_rows).sort_values("target_date").reset_index(drop=True)
    year_counts = (
        labels.groupby("year")["MC_ge_1ug_L"]
        .agg(label_days="count", positive_days="sum")
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "source_rows": int(len(source)),
        "source_unique_dates": int(source["date"].nunique()),
        "label_days": int(len(labels)),
        "positive_days": int(labels["MC_ge_1ug_L"].sum()),
        "all_below_0_15_days": int(labels["MC_all_below_0_15"].sum()),
        "year_counts": year_counts,
    }
    return daily_features, labels, summary


def _window_statistics(
    dates: pd.Series,
    values: pd.Series,
    target_date: pd.Timestamp,
) -> tuple[float, float, float, int, float]:
    start = target_date - pd.Timedelta(days=7)
    end = target_date - pd.Timedelta(days=1)
    mask = dates.between(start, end) & values.notna()
    window_dates = dates.loc[mask]
    window_values = values.loc[mask].astype(float)
    count = int(len(window_values))
    if count == 0:
        return np.nan, np.nan, np.nan, 0, np.nan

    mean = float(window_values.mean())
    latest_index = window_dates.idxmax()
    latest = float(values.loc[latest_index])
    days_since_last = float((target_date - dates.loc[latest_index]).days)
    trend = np.nan
    if count >= 2:
        offsets = (window_dates - start).dt.days.to_numpy(dtype=float)
        centered = offsets - offsets.mean()
        denominator = float(np.square(centered).sum())
        if denominator > 0:
            trend = float(
                (centered * (window_values.to_numpy() - window_values.mean())).sum()
                / denominator
            )
    return mean, latest, trend, count, days_since_last


def build_window_dataset(
    labels: pd.DataFrame,
    stonelab_daily: pd.DataFrame,
    osugi_daily: pd.DataFrame,
) -> pd.DataFrame:
    sources: dict[str, pd.DataFrame] = {}
    for variable in VARIABLES:
        source = osugi_daily if variable in OSUGI_VARIABLES else stonelab_daily
        if variable not in source.columns:
            source = source.copy()
            source[variable] = np.nan
        sources[variable] = source[["date", variable]].sort_values("date").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for label in labels.itertuples(index=False):
        target_date = pd.Timestamp(label.target_date)
        row: dict[str, Any] = {
            "target_date": target_date,
            "feature_window_start": target_date - pd.Timedelta(days=7),
            "feature_window_end": target_date - pd.Timedelta(days=1),
            "year": int(label.year),
            "MC_day_max_ug_L": label.MC_day_max_ug_L,
            "MC_all_below_0_15": int(label.MC_all_below_0_15),
            "MC_ge_1ug_L": int(label.MC_ge_1ug_L),
        }
        for variable in VARIABLES:
            source = sources[variable]
            mean, latest, trend, count, days_since = _window_statistics(
                source["date"], source[variable], target_date
            )
            row[f"{variable}_7d_mean"] = mean
            row[f"{variable}_latest"] = latest
            row[f"{variable}_trend"] = trend
            row[f"{variable}_n_obs_7d"] = count
            row[f"{variable}_days_since_last"] = days_since
        rows.append(row)
    dataset = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    ordered = [
        "target_date",
        "feature_window_start",
        "feature_window_end",
        "year",
        *MODEL_FEATURE_COLUMNS,
        *AUXILIARY_FEATURE_COLUMNS,
        "MC_day_max_ug_L",
        "MC_all_below_0_15",
        "MC_ge_1ug_L",
    ]
    return dataset.loc[:, ordered]


def build_dataset(
    stonelab_path: Path,
    osugi_root: Path,
    processed_root: Path,
) -> BuildOutputs:
    features_dir = processed_root / "features"
    labels_dir = processed_root / "labels"
    model_input_dir = processed_root / "model_input"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    model_input_dir.mkdir(parents=True, exist_ok=True)

    stonelab_daily, labels, stonelab_summary = build_stonelab_daily(stonelab_path)
    osugi_daily, netcdf_audit = build_osugi_daily(osugi_root)
    dataset = build_window_dataset(labels, stonelab_daily, osugi_daily)

    stonelab_daily_path = features_dir / "stonelab_daily_features.csv"
    osugi_daily_path = features_dir / "osugi_daily_features.csv"
    labels_path = labels_dir / "stonelab_mc_daily_labels.csv"
    dataset_path = model_input_dir / "stonelab_osugi_7d_39features.csv"
    summary_path = model_input_dir / "build_summary.json"

    stonelab_daily.to_csv(stonelab_daily_path, index=False, date_format="%Y-%m-%d")
    osugi_daily.to_csv(osugi_daily_path, index=False, date_format="%Y-%m-%d")
    labels.to_csv(labels_path, index=False, date_format="%Y-%m-%d")
    dataset.to_csv(dataset_path, index=False, date_format="%Y-%m-%d")

    coverage = {}
    for variable in VARIABLES:
        columns = [f"{variable}_{statistic}" for statistic in WINDOW_STATISTICS]
        coverage[variable] = {
            "label_days_with_any_window_observation": int(
                dataset[f"{variable}_n_obs_7d"].gt(0).sum()
            ),
            "missing_fraction_by_feature": {
                column: float(dataset[column].isna().mean()) for column in columns
            },
        }
    summary = {
        "task": "past 7 calendar days -> target-day Total Microcystins >= 1 ug/L",
        "window": "t-7 through t-1; target day excluded",
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "auxiliary_feature_count": len(AUXILIARY_FEATURE_COLUMNS),
        "dataset_rows": int(len(dataset)),
        "stonelab": stonelab_summary,
        "osugi": {
            "model_netcdf_files": len(netcdf_audit),
            "explicit_qc_variable_count": int(
                sum(bool(row["explicit_qc_variables"]) for row in netcdf_audit)
            ),
            "daily_start": osugi_daily["date"].min().date().isoformat(),
            "daily_end": osugi_daily["date"].max().date().isoformat(),
            "daily_rows": int(len(osugi_daily)),
            "file_count_by_variable": {
                variable: int(sum(row["variable"] == variable for row in netcdf_audit))
                for variable in sorted(OSUGI_VARIABLES)
            },
            "physical_invalid_values_by_variable": {
                variable: int(
                    sum(
                        row["physical_invalid_values"]
                        for row in netcdf_audit
                        if row["variable"] == variable
                    )
                )
                for variable in sorted(OSUGI_VARIABLES)
            },
            "netcdf_audit": netcdf_audit,
        },
        "coverage": coverage,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return BuildOutputs(
        dataset_path=dataset_path,
        labels_path=labels_path,
        stonelab_daily_path=stonelab_daily_path,
        osugi_daily_path=osugi_daily_path,
        summary_path=summary_path,
    )
