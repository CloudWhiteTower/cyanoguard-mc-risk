# Input features

The prediction command accepts one row per target date, with all 39 named columns
below. Column order may vary: the command restores the model's saved order.
Extra columns are ignored. Optional `target_date` must be a unique ISO date
(`YYYY-MM-DD`); otherwise `row_number` identifies each prediction.

For target day *t*, use only the seven complete calendar days **t−7 through t−1**.
`7d_mean` is the arithmetic mean of available daily observations; `latest` is the
most recent available daily value within that window; `trend` is the ordinary
least-squares slope against the actual day offsets. Missing days are not compressed
into consecutive observations. No target-day observation belongs in a feature.

Leave a missing value empty or write `NaN`. An empty window produces three missing
features; fewer than two distinct observed dates produces a missing trend. The
model follows its learned missing-value branches. It does not interpolate,
forward-fill or substitute zero. `missing_feature_count` in the output makes this
visible; even an entirely missing row yields a model score and needs particular care.

The CLI rejects missing columns, duplicate headers, malformed row widths,
nonnumeric values, infinity and numbers outside float32 range. It cannot establish
whether caller-supplied units or observation dates are correct.

| Order | Column | Unit |
| --- | --- | --- |
| 1 | `WaterTemperature_7d_mean` | degree Celsius |
| 2 | `WaterTemperature_latest` | degree Celsius |
| 3 | `WaterTemperature_trend` | degree Celsius per day |
| 4 | `DissolvedOxygen_7d_mean` | mg/L |
| 5 | `DissolvedOxygen_latest` | mg/L |
| 6 | `DissolvedOxygen_trend` | mg/L per day |
| 7 | `pH_7d_mean` | dimensionless |
| 8 | `pH_latest` | dimensionless |
| 9 | `pH_trend` | dimensionless per day |
| 10 | `Turbidity_7d_mean` | NTU |
| 11 | `Turbidity_latest` | NTU |
| 12 | `Turbidity_trend` | NTU per day |
| 13 | `Conductivity_7d_mean` | µS/cm |
| 14 | `Conductivity_latest` | µS/cm |
| 15 | `Conductivity_trend` | µS/cm per day |
| 16 | `ChlorophyllA_7d_mean` | µg/L |
| 17 | `ChlorophyllA_latest` | µg/L |
| 18 | `ChlorophyllA_trend` | µg/L per day |
| 19 | `TP_7d_mean` | mg P/L |
| 20 | `TP_latest` | mg P/L |
| 21 | `TP_trend` | mg P/L per day |
| 22 | `TN_7d_mean` | mg N/L |
| 23 | `TN_latest` | mg N/L |
| 24 | `TN_trend` | mg N/L per day |
| 25 | `NO3NO2_7d_mean` | mg N/L |
| 26 | `NO3NO2_latest` | mg N/L |
| 27 | `NO3NO2_trend` | mg N/L per day |
| 28 | `Secchi_7d_mean` | m |
| 29 | `Secchi_latest` | m |
| 30 | `Secchi_trend` | m per day |
| 31 | `Phycocyanin_7d_mean` | RFU |
| 32 | `Phycocyanin_latest` | RFU |
| 33 | `Phycocyanin_trend` | RFU per day |
| 34 | `DRP_7d_mean` | mg P/L |
| 35 | `DRP_latest` | mg P/L |
| 36 | `DRP_trend` | mg P/L per day |
| 37 | `NH4_7d_mean` | mg N/L |
| 38 | `NH4_latest` | mg N/L |
| 39 | `NH4_trend` | mg N/L per day |

Phycocyanin is instrument fluorescence in relative fluorescence units (RFU). It is not a mass concentration. ChlorophyllA is laboratory chlorophyll a. Nutrient features use elemental nitrogen or phosphorus as shown.
