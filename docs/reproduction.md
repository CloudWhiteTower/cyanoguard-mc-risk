# Reproduction

## Frozen prediction

The README quick start reproduces the included synthetic predictions with the
bundled model. The model is never fitted by `mc-risk-predict`. Package resources
load from the installed Python package, independently of the working directory.

The input schema is available as [Markdown](input-schema.md) and
[JSON](feature-schema.json). Do not round model scores before applying thresholds.

## Obtain the original observations

1. Download the exact [2013–2022 Stone Lab workbook](https://ohioseagrant.osu.edu/download/h6hnx)
   as `user-data/waterqualitydata2013-2022.xlsx`. Check SHA-256 against
   `07475cd1600cf169f1c9e35fd066f73938b0abf76f75448041b7f8c98715cdb6`.
2. Download the OSUGI source files using their recorded URLs and checksums:

```bash
python scripts/fetch_osugi.py --output user-data/osugi
```

The downloader checks every file's size and SHA-256, reuses matching local files,
and refuses changed files. If an upstream file changes or becomes unavailable,
resolve the source difference before claiming an exact reproduction. Downloads
are explicit; prediction does not download observations.

## Construct the labeled feature table

```bash
mc-risk-prepare-stonelab --input user-data/waterqualitydata2013-2022.xlsx --output user-data/stonelab.csv
mc-risk-build --stonelab user-data/stonelab.csv --osugi-root user-data/osugi --output user-data/features
```

Every output file/directory must be new. Workbook preparation selects `SL Buoy`
and `SL Bouy` records. It retains the measured microcystin values and censoring
strings. It converts nutrient columns from µmol/L to µg elemental N or P/L using
the original project's factors, **14.007** for N and **30.974** for P. The feature
builder converts these nutrient values to mg N or P/L by multiplying by 0.001.

The original data functions retain their source-specific unit conversions and
quality filtering. Sensor observations are averaged within each day and file,
then sensor-file daily means are averaged equally for the same variable. Laboratory
replicates are averaged by date for input features. The daily target is 1 if any
measured total microcystin value is ≥1 µg/L; source values below the reporting
limit are treated as measured negatives without assigning an invented concentration.

The default builder output is restricted to target years **2015–2021**.
Lookback observations may start seven days before the first target year. Later
years are excluded. Target-day observations never enter their own feature window.
Missing features and trends remain missing. The expected final table has **276
target days**, with **241** development days and **35** validation days.

The resulting prediction input is:
`user-data/features/model_input/stonelab_osugi_7d_39features.csv`.
For a new research dataset, the same builder accepts explicit `--start-year` and
`--end-year` values. Doing so does not validate the frozen model for those years.

## Reproduce the four-fold selection

The following command **runs training**. It is separate from frozen prediction
and writes only to a new directory:

```bash
mc-risk-reproduce --dataset user-data/features/model_input/stonelab_osugi_7d_39features.csv --output reproduced
```

The saved configuration fixes random seed **42**, one CPU worker, 256 sampled
parameter combinations plus the original reference combination, and the four
expanding-year folds described in the README. It performs **1,028 CV fits**,
followed by one final fit on 2015–2020. Any input rows outside 2015–2021 are
excluded by the original training routine.

A candidate must reach ROC-AUC ≥0.50 in every fold. Eligible candidates are
ranked by:

`0.50 × mean ROC-AUC + 0.25 × mean average precision + 0.15 × (1 − mean Brier) + 0.10 × minimum yearly ROC-AUC`.

The selected model has 20 trees, depth 1, learning rate 0.1, minimum child weight
3, subsample 0.7, column subsample 0.8, L2 regularization 10, L1 regularization 0,
gamma 0.1 and positive-class weight 1.5. The complete settings are packaged in
`mod_mc/resources/training.yaml`.

Threshold selection uses a 0.001 grid on 2021, maximizing Matthews correlation
coefficient, then balanced accuracy, specificity, sensitivity and threshold in
that order. The 75th and 95th percentiles of non-warning 2021 scores define lower
response boundaries, rounded to three decimals. See the accompanying
[aggregate metrics](../results/validation_metrics.json).

Training can depend on native-library/platform details. Use the bundled frozen
model to reproduce this release's predictions; report any differences from a
new training run. The packaging checks replay inference and feature construction
without retraining or replacing the frozen model.
