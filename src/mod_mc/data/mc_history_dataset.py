"""Add quantified Total MC history using the v1 seven-calendar-day statistics."""
from __future__ import annotations

import pandas as pd

from mod_mc.data.stonelab_osugi_dataset import _window_statistics

MC_HISTORY_FEATURES = ("MC_history_7d_mean", "MC_history_latest", "MC_history_trend")


def add_mc_history(base: pd.DataFrame, daily_labels: pd.DataFrame) -> pd.DataFrame:
    history = daily_labels.sort_values("target_date").reset_index(drop=True)
    if base.target_date.duplicated().any() or history.target_date.duplicated().any():
        raise ValueError("Target dates and daily MC dates must be unique")
    censored = history.MC_all_below_0_15.eq(1)
    if history.loc[censored, "MC_day_max_ug_L"].notna().any():
        raise ValueError("Censored concentrations must remain missing")
    rows = []
    for date in base.target_date:
        mean, latest, trend, count, age = _window_statistics(
            history.target_date, history.MC_day_max_ug_L, date)
        rows.append([mean, latest, trend, count, age])
    result = base.copy()
    columns = [*MC_HISTORY_FEATURES, "MC_history_quantified_days_7d",
               "MC_history_days_since_last_quantified"]
    for index, column in enumerate(columns):
        result[column] = [row[index] for row in rows]
    return result
