# Synthetic example

`synthetic_features.csv` contains six invented rows dated in 2030. Five use simple
constant feature values with different pigment inputs; the sixth has all 39
features missing. These rows do not come from measured data and do not demonstrate
model accuracy. They test installation, CSV handling and missing-value inference.

`expected_predictions.csv` was computed using the original frozen XGBoost model,
its feature order and the published thresholds, without fitting the model.
Scores may differ by small floating-point amounts across platforms; the check
uses an absolute tolerance of `1e-7`. Labels, tiers and missing counts must match.

To recreate the synthetic input in a new file:

```bash
python scripts/generate_examples.py --output synthetic-new.csv
```
