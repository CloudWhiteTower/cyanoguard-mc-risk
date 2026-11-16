"""Restrict target dates before calling the original seven-day feature builder."""
import json
from pathlib import Path

import pandas as pd

from mod_mc.data.stonelab_osugi_dataset import (
    BuildOutputs, build_osugi_daily, build_stonelab_daily, build_window_dataset,
)

def build_dataset_for_years(stonelab_path: Path, osugi_root: Path, output: Path,
                          start_year: int = 2015, end_year: int = 2021) -> BuildOutputs:
    if not 1900 <= start_year <= end_year <= 2200:
        raise ValueError('Target years must be ordered within 1900-2200')
    laboratory, labels, _ = build_stonelab_daily(stonelab_path)
    sensor, audit = build_osugi_daily(osugi_root)
    labels = labels.loc[labels['year'].between(start_year, end_year)].copy()
    if labels.empty:
        raise ValueError('No measured target days in the requested years')
    start = pd.Timestamp(f'{start_year}-01-01') - pd.Timedelta(days=7)
    end = pd.Timestamp(f'{end_year}-12-31')
    laboratory = laboratory.loc[laboratory['date'].between(start, end)].copy()
    sensor = sensor.loc[sensor['date'].between(start, end)].copy()
    features = build_window_dataset(labels, laboratory, sensor)
    for folder in ['features', 'labels', 'model_input']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    paths = BuildOutputs(
        output / 'model_input/stonelab_osugi_7d_39features.csv',
        output / 'labels/stonelab_mc_daily_labels.csv',
        output / 'features/stonelab_daily_features.csv',
        output / 'features/osugi_daily_features.csv',
        output / 'model_input/build_summary.json',
    )
    for frame, path in [(features, paths.dataset_path), (labels, paths.labels_path),
                        (laboratory, paths.stonelab_daily_path), (sensor, paths.osugi_daily_path)]:
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
    summary = {'target_years': [start_year, end_year], 'dataset_rows': len(features),
        'positive_rows': int(features['MC_ge_1ug_L'].sum()),
        'feature_count': 39, 'window': 't-7 through t-1; target day excluded',
        'lookback_source_start': str(start.date()), 'source_end': str(end.date()),
        'netcdf_source_files': len(audit),
        'missing_features': {name: int(features[name].isna().sum())
                             for name in features if name.endswith(('_7d_mean', '_latest', '_trend'))}}
    paths.summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return paths
