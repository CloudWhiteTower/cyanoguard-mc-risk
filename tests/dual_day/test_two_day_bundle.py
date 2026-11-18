import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from mod_mc.inference.two_day_bundle import FEATURE_COLUMNS, TwoDayPredictor, build_shared_features
from mod_mc.cli_v3 import DEFAULT_BUNDLE

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = DEFAULT_BUNDLE


class SharedWindowTests(unittest.TestCase):
    def test_shared_features_ignore_both_future_days_and_outside_window(self):
        dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-08", "2021-01-09", "2021-01-10"])
        stone = pd.DataFrame({"date": dates, "ChlorophyllA": [999., 1., 7., 999., 999.]})
        osugi = pd.DataFrame({"date": dates, "WaterTemperature": [999., 1., 7., 999., 999.]})
        mc = pd.DataFrame({"target_date": dates, "MC_day_max_ug_L": [999., 1., 7., 999., 999.], "MC_all_below_0_15": [0]*5})
        row = build_shared_features(stone, osugi, mc, ["2021-01-08"]).iloc[0]
        self.assertEqual(row.window_start_date, pd.Timestamp("2021-01-02"))
        for name in ["ChlorophyllA", "WaterTemperature", "MC_history"]:
            self.assertEqual(row[name+"_7d_mean"], 4.)
            self.assertEqual(row[name+"_latest"], 7.)
            self.assertAlmostEqual(row[name+"_trend"], 1.)

    def test_censored_mc_cannot_have_a_fabricated_value(self):
        empty = pd.DataFrame({"date": pd.to_datetime([])})
        mc = pd.DataFrame({"target_date": ["2021-01-08"], "MC_day_max_ug_L": [0.075], "MC_all_below_0_15": [1]})
        with self.assertRaises(ValueError):
            build_shared_features(empty, empty, mc, ["2021-01-08"])

    def test_duplicate_daily_dates_rejected(self):
        stone = pd.DataFrame({"date": ["2021-01-01", "2021-01-01"]})
        empty = pd.DataFrame({"date": pd.to_datetime([])})
        mc = pd.DataFrame({"target_date": [], "MC_day_max_ug_L": []})
        with self.assertRaises(ValueError):
            build_shared_features(stone, empty, mc, ["2021-01-08"])


class TwoDayPredictorTests(unittest.TestCase):
    def setUp(self):
        self.predictor = TwoDayPredictor(BUNDLE)
        self.example = pd.read_csv(ROOT / "examples/v3_synthetic_features.csv")

    def test_both_models_receive_same_matrix_and_keep_separate_thresholds(self):
        received = []
        def fake(frame):
            received.append(frame)
            return np.tile([0.5, 0.5], (len(frame), 1))
        with patch.object(XGBClassifier, "fit", side_effect=AssertionError("Fitting forbidden")):
            with patch.object(self.predictor.models[0], "predict_proba", side_effect=fake):
                with patch.object(self.predictor.models[1], "predict_proba", side_effect=fake):
                    row = self.predictor.predict_features(self.example).iloc[0]
        self.assertIs(received[0], received[1])
        self.assertEqual(list(received[0].columns), FEATURE_COLUMNS)
        self.assertEqual(row.day1_predicted_label, 1)
        self.assertEqual(row.day2_predicted_label, 0)
        self.assertEqual(row.day1_response_level, "warning")
        self.assertEqual(row.day2_response_level, "low")

    def test_target_dates_and_feature_column_reordering(self):
        a = self.predictor.predict_features(self.example)
        b = self.predictor.predict_features(self.example[self.example.columns[::-1]])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.day1_target_date.iloc[0], pd.Timestamp("2030-01-08"))
        self.assertEqual(a.day2_target_date.iloc[0], pd.Timestamp("2030-01-09"))

    def test_absent_field_and_infinite_input_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_features(self.example.drop(columns=FEATURE_COLUMNS[0]))
        bad = self.example.copy(); bad[FEATURE_COLUMNS[0]] = np.inf
        with self.assertRaises(ValueError):
            self.predictor.predict_features(bad)

    def test_nonseven_day_window_rejected(self):
        bad = self.example.copy(); bad["window_start_date"] = "2029-12-31"
        with self.assertRaises(ValueError):
            self.predictor.predict_features(bad)

    def test_extra_target_columns_are_not_consumed(self):
        a = self.predictor.predict_features(self.example)
        frame = self.example.copy()
        frame["MC_ge_1ug_L"] = 1; frame["MC_day_max_ug_L"] = 9999.
        pd.testing.assert_frame_equal(a, self.predictor.predict_features(frame))

    def test_all_missing_inputs_are_flagged_and_not_imputed_or_dropped(self):
        frame = self.example.copy(); frame.loc[:, FEATURE_COLUMNS] = np.nan
        row = self.predictor.predict_features(frame).iloc[0]
        self.assertEqual(row.available_feature_count, 0)
        self.assertEqual(row.input_status, "all_missing")
        self.assertTrue(0 <= row.day1_risk_score <= 1)
        self.assertTrue(frame[FEATURE_COLUMNS].isna().all().all())

    def test_altered_model_hash_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="v3_hash_test_") as temporary:
            root = Path(temporary)
            spec = json.loads((BUNDLE / "bundle.json").read_text(encoding="utf-8"))
            (root / "bundle.json").write_text(json.dumps(spec), encoding="utf-8")
            path = root / spec["heads"][0]["model_path"]
            path.parent.mkdir(parents=True); path.write_bytes(b"altered-model")
            with self.assertRaises(ValueError):
                TwoDayPredictor(root)

    def test_installed_command_loads_models_from_another_directory(self):
        command = shutil.which("mc-risk-predict-v3")
        self.assertIsNotNone(command, "Install the project before running this test")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            subprocess.run(
                [command, "--features-csv", str(ROOT / "examples/v3_synthetic_features.csv"),
                 "--output-csv", str(output)],
                cwd=directory, check=True, capture_output=True,
            )
            actual = pd.read_csv(output, parse_dates=[
                "window_start_date", "window_end_date", "day1_target_date", "day2_target_date"
            ])
        expected = self.predictor.predict_features(self.example)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
