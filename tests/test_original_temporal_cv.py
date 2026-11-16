import unittest

import numpy as np
import pandas as pd

from mod_mc.training.tune_temporal_cv_2021 import (
    choose_rounded_mcc_threshold,
    derive_rounded_tier_thresholds,
    expanding_year_folds,
)


class TemporalCrossValidationTests(unittest.TestCase):
    def test_expanding_folds_never_use_future_years(self) -> None:
        data = pd.DataFrame(
            {
                "target_date": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2015-01-02",
                        "2016-01-01",
                        "2016-01-02",
                        "2017-01-01",
                        "2017-01-02",
                    ]
                ),
                "year": [2015, 2015, 2016, 2016, 2017, 2017],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        folds = expanding_year_folds(data, [2015, 2016, 2017], [2016, 2017], "target")
        for _, _, train, validation in folds:
            self.assertLess(train.target_date.max(), validation.target_date.min())

    def test_mcc_threshold_is_limited_to_three_decimals(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1004, 0.3996, 0.6004, 0.9002])
        threshold, curve = choose_rounded_mcc_threshold(labels, scores, decimals=3)
        self.assertEqual(threshold, 0.600)
        self.assertTrue(
            np.allclose(curve["threshold"] * 1000, np.round(curve["threshold"] * 1000))
        )

    def test_tier_thresholds_are_rounded_and_ordered(self) -> None:
        scores = np.array([0.101, 0.121, 0.151, 0.191, 0.301])
        low, medium = derive_rounded_tier_thresholds(scores, 0.300, decimals=3)
        self.assertLess(low, medium)
        self.assertLess(medium, 0.300)
        self.assertEqual(low, round(low, 3))
        self.assertEqual(medium, round(medium, 3))


if __name__ == "__main__":
    unittest.main()
