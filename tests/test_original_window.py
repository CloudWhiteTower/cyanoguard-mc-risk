import unittest

import numpy as np
import pandas as pd

from mod_mc.data.stonelab_osugi_dataset import (
    _apply_physical_bounds,
    _window_statistics,
    parse_microcystin,
)


class StoneLabOsugiDatasetTests(unittest.TestCase):
    def test_microcystin_censor_is_measured_negative_without_imputation(self) -> None:
        measured, exact, censored = parse_microcystin("< 0.15")
        self.assertTrue(measured)
        self.assertIsNone(exact)
        self.assertTrue(censored)

    def test_window_excludes_target_day_and_ignores_missing(self) -> None:
        dates = pd.Series(pd.date_range("2022-01-01", "2022-01-08"))
        values = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 7.0, 8.0, 1000.0])
        mean, latest, trend, count, days_since = _window_statistics(
            dates, values, pd.Timestamp("2022-01-08")
        )
        self.assertAlmostEqual(mean, 4.75)
        self.assertEqual(latest, 8.0)
        self.assertEqual(count, 4)
        self.assertEqual(days_since, 1.0)
        self.assertTrue(np.isfinite(trend))

    def test_empty_window_stays_nan(self) -> None:
        dates = pd.Series(pd.date_range("2022-01-01", periods=3))
        values = pd.Series([np.nan, np.nan, np.nan])
        result = _window_statistics(dates, values, pd.Timestamp("2022-01-08"))
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], 0)

    def test_only_basic_physical_invalid_values_are_removed(self) -> None:
        filtered, count = _apply_physical_bounds(
            np.array([-998.0, 7.8, 15.0]), "pH"
        )
        self.assertEqual(count, 2)
        self.assertTrue(np.isnan(filtered[0]))
        self.assertEqual(filtered[1], 7.8)
        self.assertTrue(np.isnan(filtered[2]))


if __name__ == "__main__":
    unittest.main()
