"""Write deterministic synthetic features; never read measured observations."""
from pathlib import Path
from importlib.resources import files
import json
import pandas as pd

def make_synthetic() -> pd.DataFrame:
    features = json.loads(files('mod_mc.resources').joinpath('metadata.json').read_text('utf-8'))['feature_names']
    values = dict(WaterTemperature=20, DissolvedOxygen=8, pH=8, Turbidity=1,
                  Conductivity=300, ChlorophyllA=5, TP=0.02, TN=1,
                  NO3NO2=0.2, Secchi=2, Phycocyanin=5, DRP=0.01, NH4=0.02)
    rows = []
    for day, pigment in enumerate([0.1, 1.0, 4.0, 15.0, 40.0], start=1):
        row = {'target_date': f'2030-07-{day:02d}'}
        for feature in features:
            variable, statistic = feature.rsplit('_', 1)
            if feature.endswith('_7d_mean'):
                variable, statistic = feature[:-8], 'mean'
            base = pigment if variable in {'ChlorophyllA', 'Phycocyanin'} else values[variable]
            row[feature] = 0.0 if statistic == 'trend' else base
        rows.append(row)
    rows.append({'target_date': '2030-07-06', **{name: float('nan') for name in features}})
    return pd.DataFrame(rows)[['target_date', *features]]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New synthetic CSV')
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8', newline='') as handle:
        make_synthetic().to_csv(handle, index=False, lineterminator='\n')
