from pathlib import Path
import subprocess
import sys

import netCDF4
import numpy as np
import pandas as pd

from mod_mc.data.prepare_source import CONVERSIONS, prepare_stonelab
from mod_mc.data.stonelab_osugi_dataset import build_dataset, MODEL_FEATURE_COLUMNS
from mod_mc.data.final_dataset import build_dataset_for_years

def test_synthetic_workbook_to_daily_to_features_to_prediction(tmp_path):
    # Seven preceding days plus a target day whose large reading must be excluded.
    dates = pd.date_range('2030-07-01', periods=8)
    frame = pd.DataFrame({'Program': ['SL Buoy'] * 7 + ['SL Bouy'], 'Date': dates,
        'Total Microcystins (µg/L)': ['< 0.15'] * 7 + [1.2],
        'Chlorophyll (µg/L)': [2.0] * 7 + [9999.0], 'Secchi (m)': [2.0] * 8,
        **{name: [1.0] * 8 for name in CONVERSIONS}})
    workbook = tmp_path / 'source.xlsx'
    frame.to_excel(workbook, sheet_name='data', index=False)
    normalized = tmp_path / 'source.csv'
    assert prepare_stonelab(workbook, normalized) == 8
    prepared = pd.read_csv(normalized)
    assert prepared['TP (µg P/L)'].iloc[0] == 30.974
    assert prepared['TN (µg N/L)'].iloc[0] == 14.007
    raw = tmp_path / 'raw'
    raw.mkdir()
    with netCDF4.Dataset(raw / 'sea-water-temperature.nc', 'w') as dataset:
        dataset.createDimension('time', 8)
        time = dataset.createVariable('time', 'f8', ('time',))
        time.units = 'days since 2030-07-01 00:00:00'
        time[:] = np.arange(8)
        obs = dataset.createVariable('obs', 'f8', ('time',))
        obs.units = 'degree_celsius'
        obs[:] = [20.0] * 7 + [9999.0]
    outputs = build_dataset(normalized, raw, tmp_path / 'built')
    data = pd.read_csv(outputs.dataset_path)
    target = data.iloc[-1]
    assert target['ChlorophyllA_7d_mean'] == 2
    assert target['WaterTemperature_latest'] == 20
    assert target['TP_7d_mean'] == 0.030974
    assert target['MC_ge_1ug_L'] == 1
    assert target['Phycocyanin_n_obs_7d'] == 0
    assert np.isnan(target['Phycocyanin_latest'])
    assert len([name for name in data if name in MODEL_FEATURE_COLUMNS]) == 39
    result = subprocess.run([sys.executable, '-m', 'mod_mc.cli', '--input',
        str(outputs.dataset_path), '--output', str(tmp_path / 'predicted.csv')],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(tmp_path / 'predicted.csv')) == 8
    # New orchestration uses the same features but restricts all output periods.
    final_outputs = build_dataset_for_years(normalized, raw, tmp_path / 'scoped', 2030, 2030)
    pd.testing.assert_frame_equal(data, pd.read_csv(final_outputs.dataset_path))


def test_default_builder_excludes_2022_targets_and_observations(tmp_path):
    dates = ['2015-07-01', '2021-07-01', '2022-07-01']
    frame = pd.DataFrame({'Date': dates, 'Total Microcystins (µg/L)': [0.2, 1.5, 999],
        'Chlorophyll (µg/L)': [2, 3, 999], 'Secchi (m)': [2, 2, 999],
        **{destination: [1.0, 1.0, 999.0] for destination, _ in CONVERSIONS.values()}})
    source = tmp_path / 'laboratory.csv'
    frame.to_csv(source, index=False)
    raw = tmp_path / 'raw'
    raw.mkdir()
    with netCDF4.Dataset(raw / 'sea-water-temperature.nc', 'w') as dataset:
        dataset.createDimension('time', 3)
        time = dataset.createVariable('time', 'f8', ('time',))
        time.units = 'days since 2015-01-01 00:00:00'
        time[:] = [(pd.Timestamp(day)-pd.Timestamp('2015-01-01')).days for day in dates]
        obs = dataset.createVariable('obs', 'f8', ('time',))
        obs.units = 'degree_celsius'
        obs[:] = [20, 21, 999]
    result = build_dataset_for_years(source, raw, tmp_path / 'final')
    targets = pd.read_csv(result.dataset_path)
    assert targets.year.tolist() == [2015, 2021]
    assert pd.read_csv(result.labels_path).year.tolist() == [2015, 2021]
    for path in [result.stonelab_daily_path, result.osugi_daily_path]:
        assert pd.to_datetime(pd.read_csv(path)['date']).dt.year.max() == 2021
