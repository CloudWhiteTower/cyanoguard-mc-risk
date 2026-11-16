"""Check a prediction output against the frozen synthetic reference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def check(actual: Path, expected: Path) -> None:
    result = pd.read_csv(actual)
    reference = pd.read_csv(expected)
    assert result.columns.tolist() == reference.columns.tolist(), 'Output columns differ'
    assert len(result) == len(reference), 'Output row count differs'
    np.testing.assert_allclose(result.risk_score, reference.risk_score, rtol=0, atol=1e-7)
    for column in reference.columns.drop('risk_score'):
        assert result[column].tolist() == reference[column].tolist(), f'{column} differs'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actual', type=Path)
    args = parser.parse_args()
    check(args.actual, Path(__file__).resolve().parents[1] / 'examples/expected_predictions.csv')
    print('All six synthetic predictions match the frozen reference.')
