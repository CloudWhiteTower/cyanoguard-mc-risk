# CyanoGuard · MC Risk Predictor

MC Risk Predictor estimates the risk that **total microcystins (Total MC) reach
1 µg/L** on a future calendar day. The latest model bundle uses a shared
seven-day window of water-quality observations and historical Total MC to
produce separate **Day 1** and **Day 2** predictions.

## Project and contribution

I contributed to the early-warning model's mathematical design and implementation
for SCAU-China's 2026 iGEM project, CyanoGuard, and to writing the Dry Lab and
Engineering pages. This portfolio repository carries the team's published
Wiki baseline and our subsequent dual-day inference bundle. It retains the
upstream license and team credits.

- [Official iGEM GitLab v1.0.0](https://gitlab.igem.org/2026/software/scau-china/mc-risk-prediction/-/tree/v1.0.0)
- [Team contribution record](https://2026.igem.wiki/scau-china/team/attributions)
- Upstream commit `330014260fcac30e81ccfc1c3973fbf056abf5fb`

## Run v3 with historical Total MC

Use Python 3.12. After activating a new virtual environment, run:

```bash
python -m pip install -r requirements.lock
python -m pip install .
mc-risk-predict-v3 --features-csv examples/v3_synthetic_features.csv --output-csv v3_predictions.csv
```

The command accepts 42 features for days $d-6$ through $d$, including the mean,
latest value and linear trend of observed Total MC. The two XGBoost models
independently score days $d+1$ and $d+2$ using that same input. A score is
an uncalibrated per-day risk estimate. [Daily table input, version history and evaluation](docs/dual-day-v3.md)

## Wiki baseline with 39 features

The sections below document the original single-day model. Its command
`mc-risk-predict` still takes the 39-feature CSV, and its data preparation and
four-fold reproduction commands still follow the original Wiki configuration.

## Install and run

Use **Python 3.12 on 64-bit Windows or Linux**. Create an isolated environment
from this repository's root:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` in a Linux shell. Then run:

```bash
python -m pip install -r requirements.lock
python -m pip install .
mc-risk-predict --input examples/synthetic_features.csv --output predictions.csv
python scripts/check_example.py predictions.csv
```

The six example rows are **synthetic**. They exercise the interface and missing
values; they are not environmental observations or evidence of model accuracy.
The expected outputs were calculated directly from the original frozen model.
Prediction runs locally without network access or training. Existing output
files are refused, so use a new filename for another run.

On Linux, XGBoost requires the OpenMP runtime (`libgomp1` on Debian/Ubuntu).
Windows requires Microsoft's Visual C++ runtime. The pinned
[`xgboost-cpu` distribution](https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only)
runs this model on the CPU without CUDA dependencies.

## Input and output

Supply a UTF-8 CSV with the [39 feature columns and units](docs/input-schema.md).
The model uses 13 variables, each summarized as its seven-day mean, most recent
observed value and linear trend. For target day *t*, every feature must use only
**t−7 through t−1**. Raw daily measurements must first pass through the feature
builder; the prediction command expects the completed feature table.

Column order may vary. Extra metadata columns are ignored. `target_date` is
optional; if present, use unique `YYYY-MM-DD` dates. Missing values may be empty
or `NaN`; they remain missing for XGBoost. The command validates column names,
CSV structure and numeric values, but cannot verify the units or time window
of a feature supplied by the caller.

The output preserves row order and contains `row_number`, optional `target_date`,
`risk_score`, `predicted_label`, `risk_tier` and `missing_feature_count`.
`predicted_label=1` when the unrounded score is at least **0.255**.

| Response tier | Score interval |
| --- | --- |
| `normal` | < 0.152 |
| `low` | ≥ 0.152 and < 0.184 |
| `medium` | ≥ 0.184 and < 0.255 |
| `warning` | ≥ 0.255 |

Missing inputs follow the model's learned branches, so even an entirely missing
row returns a score. Check `missing_feature_count` before interpreting it.

## Model and evaluation

Release **1.0.0** contains the final model described on the
[SCAU-China Model page](https://2026.igem.wiki/scau-china/dry-lab/model): 20 trees of maximum
depth 1, saved by XGBoost 3.2.0. Its SHA-256 is
`c621d77b8cdbd7a82ef6b97cd0cea8429110cf6e5539cf82052674974b785608`.

The model was fitted on 241 labeled target days from 2015–2020, including 53
positive days. Hyperparameters were selected using four expanding-window folds:
2015–2016 → 2017, 2015–2017 → 2018, 2015–2018 → 2019 and 2015–2019 → 2020.
The [four-fold results](results/cv_fold_metrics.csv) and
[aggregate evaluation](results/validation_metrics.json) accompany the source.

The 2021 validation set contains 35 measured target days, including 7 positive
days. It produced ROC-AUC **0.9362**, average precision **0.6774**, Brier score
**0.1130**, and TN/FP/FN/TP **25/3/0/7** at the warning threshold.
The same 2021 data were used to choose the threshold by Matthews correlation
coefficient and to derive the lower response boundaries. These figures therefore
do not constitute an independent test of the selected thresholds.

The model has not been independently validated for other stations or years.
Sparse observations and changes in sampling, instruments or bloom conditions can
affect its predictions. Scores are not calibrated probabilities.

## Data and reproduction

Training data came from Ohio State University's Stone Laboratory laboratory/grab
samples and the OSUGI/Gibraltar Island Buoy sensor archive. Original observations
and row-level derivatives are not distributed here. See
[data sources, citations and download links](docs/data-sources.md), then follow
the [feature-building and four-fold reproduction steps](docs/reproduction.md).
Only synthetic inputs are needed for the quick start.

## Tests and build

```bash
python -m pip install -r requirements-dev.lock
python -m pip install .
python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

The GitHub workflow builds and installs the wheel, tests input validation and
window construction, and checks the synthetic predictions from a separate
working directory. It does not retrain the model.

Team code, documentation, synthetic examples and the team-generated model use the
[MIT license](LICENSE). Dependencies and source datasets retain their own terms;
see [attribution](NOTICE.md).
