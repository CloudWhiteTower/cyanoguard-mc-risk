"""Prepare the source workbook using the project's original subset and conversion factors."""
from pathlib import Path

import pandas as pd

CONVERSIONS = {
    'TP (µmol/L)': ('TP (µg P/L)', 30.974),
    'TN (µmol/L)': ('TN (µg N/L)', 14.007),
    'Nitrate+NO2 (µmol/L)': ('Nitrate+NO2 (µg N/L)', 14.007),
    'DRP (µmol/L)': ('DRP (µg P/L)', 30.974),
    'Ammonium (µmol/L)': ('Ammonium (µg N/L)', 14.007),
}
PASSTHROUGH = ['Date', 'Total Microcystins (µg/L)', 'Chlorophyll (µg/L)', 'Secchi (m)']

def prepare_stonelab(input_path: Path, output_path: Path) -> int:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f'Output already exists: {output_path}')
    data = pd.read_excel(input_path, sheet_name='data', engine='openpyxl')
    required = {'Program', *PASSTHROUGH, *CONVERSIONS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError('Workbook missing columns: ' + ', '.join(missing))
    data = data.loc[data['Program'].astype(str).str.strip().isin(['SL Buoy', 'SL Bouy'])]
    if data.empty:
        raise ValueError('Workbook contains no SL Buoy/SL Bouy records')
    prepared = data[PASSTHROUGH].copy()
    prepared['Date'] = pd.to_datetime(prepared['Date'], errors='raise').dt.strftime('%Y-%m-%d')
    for source, (destination, factor) in CONVERSIONS.items():
        prepared[destination] = pd.to_numeric(data[source], errors='coerce') * factor
    with output_path.open('x', encoding='utf-8', newline='') as handle:
        prepared.to_csv(handle, index=False, lineterminator='\n')
    return len(prepared)
