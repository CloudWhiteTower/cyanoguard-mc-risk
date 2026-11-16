from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mod_mc.inference.predict import FrozenPredictor, predict_csv, read_feature_csv, risk_tiers

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def sample():
    return read_feature_csv(ROOT / 'examples/synthetic_features.csv')

def test_synthetic_predictions_match_independent_frozen_reference(sample):
    result = FrozenPredictor().predict(sample)
    expected = pd.read_csv(ROOT / 'examples/expected_predictions.csv')
    np.testing.assert_allclose(result.risk_score, expected.risk_score, rtol=0, atol=1e-7)
    for column in ['row_number', 'target_date', 'predicted_label', 'risk_tier', 'missing_feature_count']:
        assert result[column].tolist() == expected[column].tolist()

def test_reordered_columns_and_extra_metadata_are_safe(sample):
    predictor = FrozenPredictor()
    expected = predictor.predict(sample)
    reordered = sample[sample.columns[::-1]].assign(ignored_metadata='anything')
    pd.testing.assert_frame_equal(expected, predictor.predict(reordered))

def test_missing_values_follow_model_paths_and_are_counted(sample):
    result = FrozenPredictor().predict(sample)
    assert result.missing_feature_count.iloc[-1] == 39
    assert np.isfinite(result.risk_score).all()

@pytest.mark.parametrize('bad', ['abc', 'null', 'inf', '-inf', '1e40'])
def test_rejects_bad_numeric_values(sample, bad):
    sample.loc[0, 'ChlorophyllA_latest'] = bad
    with pytest.raises(ValueError, match='nonnumeric|infinity|float32'):
        FrozenPredictor().predict(sample)

def test_rejects_missing_column(sample):
    with pytest.raises(ValueError, match='Missing feature columns'):
        FrozenPredictor().predict(sample.drop(columns='TP_trend'))

@pytest.mark.parametrize('text', ['', 'a,a\n1,2\n', 'a,b\n1\n', 'a,b\n1,2,3\n', ',b\n1,2\n'])
def test_rejects_malformed_csv(tmp_path, text):
    path = tmp_path / 'bad.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_feature_csv(path)

def test_rejects_empty_feature_table(sample):
    with pytest.raises(ValueError, match='at least one'):
        FrozenPredictor().predict(sample.iloc[:0])

@pytest.mark.parametrize('bad', ['not-a-date', '2030-02-30', '2030-7-1', ''])
def test_rejects_invalid_target_date(sample, bad):
    sample.loc[0, 'target_date'] = bad
    with pytest.raises(ValueError, match='ISO dates'):
        FrozenPredictor().predict(sample)

def test_rejects_duplicate_dates(sample):
    sample.loc[1, 'target_date'] = sample.loc[0, 'target_date']
    with pytest.raises(ValueError, match='duplicate dates'):
        FrozenPredictor().predict(sample)

def test_date_is_optional_and_input_order_is_retained(sample):
    result = FrozenPredictor().predict(sample.drop(columns='target_date').iloc[::-1])
    assert 'target_date' not in result
    assert result.row_number.tolist() == list(range(1, len(sample) + 1))
    assert result.missing_feature_count.iloc[0] == 39

def test_exact_tier_boundaries():
    thresholds = json.loads(files('mod_mc.resources').joinpath('metadata.json').read_text('utf-8'))['operational_thresholds']
    low, medium, warning = (thresholds[name] for name in ['normal_to_low', 'low_to_medium', 'medium_to_warning'])
    scores = np.array([np.nextafter(low, 0), low, np.nextafter(medium, 0), medium,
                       np.nextafter(warning, 0), warning, 1.0])
    assert risk_tiers(scores, thresholds).tolist() == ['normal', 'low', 'low', 'medium', 'medium', 'warning', 'warning']

def test_prediction_never_overwrites_files(tmp_path):
    input_file = ROOT / 'examples/synthetic_features.csv'
    original = input_file.read_bytes()
    with pytest.raises(FileExistsError):
        predict_csv(input_file, input_file)
    assert input_file.read_bytes() == original
    output = tmp_path / 'existing.csv'
    output.write_text('keep this')
    with pytest.raises(FileExistsError):
        predict_csv(input_file, output)
    assert output.read_text() == 'keep this'
    model = Path(str(files('mod_mc.resources').joinpath('model.json')))
    before = model.read_bytes()
    with pytest.raises(FileExistsError):
        predict_csv(input_file, model)
    assert model.read_bytes() == before

def test_installed_cli_from_unrelated_directory(tmp_path):
    output = tmp_path / 'result.csv'
    process = subprocess.run([sys.executable, '-m', 'mod_mc.cli', '--input',
        str(ROOT / 'examples/synthetic_features.csv'), '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert len(pd.read_csv(output)) == 6
    repeated = subprocess.run([sys.executable, '-m', 'mod_mc.cli', '--input',
        str(ROOT / 'examples/synthetic_features.csv'), '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert 'already exists' in repeated.stderr


def test_reproduction_command_refuses_existing_directory_without_training(tmp_path):
    protected = tmp_path / 'protected'
    protected.mkdir()
    marker = protected / 'model.json'
    marker.write_text('preserve frozen result')
    result = subprocess.run([sys.executable, '-c',
        'from mod_mc.cli import reproduce_main; reproduce_main()',
        '--dataset', str(ROOT / 'examples/synthetic_features.csv'),
        '--output', str(protected)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'already exists' in result.stderr
    assert marker.read_text() == 'preserve frozen result'


def test_schema_documents_the_saved_feature_order():
    predictor = FrozenPredictor()
    schema = json.loads((ROOT / 'docs/feature-schema.json').read_text('utf-8'))
    assert [feature['name'] for feature in schema['features']] == predictor.features
    assert len(schema['features']) == 39
