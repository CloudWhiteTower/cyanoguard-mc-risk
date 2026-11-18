# v3 · Historical Total MC and two-day inference

The original Wiki baseline used 13 water-quality variables and 39 derived
features. Day 1 v2 added historical total microcystins (Total MC) as a 14th
variable. Each variable contributes its seven-day mean, latest observed value
and time trend. Day 2 was trained separately against the second future calendar
day. The v3 bundle uses these two frozen models with a shared input builder.

## Window and fields

A window ending on date $d$ covers $d-6$ through $d$, inclusive. It produces
separate outputs for $d+1$ and $d+2$. The total microcystin measurements for
either output day fall outside the input window. When replaying a forecast,
include only measurements that would have been available at the cutoff,
accounting for any sampling and reporting delay.

The original 39 features retain their [column names and units](input-schema.md).
The three history features are `MC_history_7d_mean` and `MC_history_latest`
in μg/L, and `MC_history_trend` in μg/L/day. Trends use elapsed calendar days
and require at least two observations. Missing inputs remain missing. The
`window_end_date` column is required, and `window_start_date` can be supplied
to check the window length. [Complete feature order and model checksums](../src/mod_mc/resources/v3/bundle.json)

## Build features from daily tables

The bundled CSVs are synthetic. They have unique dates per source and start
from an invented seven-day window ending on 2030-01-07.

```bash
mc-risk-predict-v3 --stonelab-daily examples/v3_synthetic_stonelab_daily.csv --osugi-daily examples/v3_synthetic_osugi_daily.csv --mc-daily examples/v3_synthetic_mc_daily.csv --window-end 2030-01-07 --output-csv v3_daily_predictions.csv
```

Stone Lab and OSUGI tables use `date` plus the corresponding daily water-quality
columns. Use the source assignments and units from the 39-feature schema.
The historical MC table uses `target_date` (the date of the historical
observation) and `MC_day_max_ug_L` (the daily observed maximum). If
`MC_all_below_0_15` is provided with value 1, leave the concentration blank.
The feature builder excludes dates beyond the window. The synthetic daily
tables reproduce the supplied synthetic feature CSV.

Each output includes the input window, feature availability, both target dates,
and a separate score, label and response level for each horizon. The warning
thresholds are 0.409 for Day 1 and 0.667 for Day 2.

## Evaluation

| Frozen component | Four-fold temporal mean ROC-AUC | 2021 ROC-AUC | 2021 TN / FP / FN / TP |
| --- | ---: | ---: | ---: |
| Day 1 with MC history | 0.8960 | 0.9796 | 26 / 2 / 0 / 7 |
| Day 2 with MC history | 0.8863 | 0.9566 | 26 / 2 / 1 / 6 |

The components were developed on 2015–2020 data. The 2021 data provided
35 labeled dates per horizon and were also used for threshold selection.
The 515-window historical engineering replay checked that the bundle
preserved each model's saved outputs. There were 37 fully paired windows in
that replay, with only two fully paired windows in 2021, both negative on
both days. These counts support per-horizon checks and leave joint positive
performance undetermined. The v3 release packages inference with frozen weights
and thresholds. [Original data providers](data-sources.md)
