from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("merei_features", Path(__file__).resolve().parents[1] / "experiment" / "features.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(values, hours=None, user="opaque_a"):
    n = len(values)
    if hours is None:
        hours = np.arange(n) * 3
    return pd.DataFrame({"row_id": np.arange(n), "user_id": user,
                         "timestamp": pd.Timestamp("2023-01-01") + pd.to_timedelta(hours, unit="h"),
                         "label": np.zeros(n), "x": values, "constant": np.full(n, 5.0)})


class FeatureTests(unittest.TestCase):
    def test_strict_current_gap_and_greedy_source_thinning_exclude_exact_two_hours(self):
        seconds = np.array([0, 7200, 7201, 14401, 14402, 21602, 21603])
        frame = sample([0, 999, 10, 999, 20, 999, 30], hours=seconds / 3600)
        result, metadata = module.build_features(frame, ["x"], {})
        self.assertEqual(result["meta__accepted_source_indicator"].tolist(), [1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(result["meta__eligible_history_count"].tolist(), [0, 0, 1, 1, 2, 2, 3])
        self.assertTrue(np.isnan(result.loc[4, "static_abs_z__x"]))
        self.assertAlmostEqual(float(result.loc[6, "static_abs_z__x"]), 20 / (math.sqrt(200 / 3) + 1e-6), places=6)
        self.assertEqual(metadata["accepted_source_rows"], 4)

    def test_static_first_three_and_ewma_population_variance_are_past_only(self):
        frame = sample([0, 10, 20, 30, 100, 200, 30])
        result, metadata = module.build_features(frame, ["x"], {})
        self.assertAlmostEqual(float(result.loc[3, "history_mean__x"]), 10.0)
        self.assertAlmostEqual(float(result.loc[3, "adaptive_abs_z__x"]), 27.1 / (math.sqrt(40.59) + 1e-6), places=5)
        self.assertAlmostEqual(float(result.loc[3, "adaptive_movement_abs_z__x"]), 7.1 / (math.sqrt(200 / 3) + 1e-6), places=6)
        self.assertEqual(result.loc[3, "static_abs_z__x"], result.loc[6, "static_abs_z__x"])
        self.assertFalse(metadata["source_label_used"])
        self.assertFalse(metadata["reference_is_assumed_safe"])

    def test_drift_recent_six_older_six_and_trajectory_use_history_only(self):
        frame = sample(np.arange(15, dtype=float))
        result, _ = module.build_features(frame, ["x"], {})
        expected = 6.0 / (math.sqrt(35 / 12) + 1e-6)
        self.assertAlmostEqual(float(result.loc[12, "drift_abs_z__x"]), expected, places=5)
        self.assertTrue(np.isnan(result.loc[12, "drift_change_abs__x"]))
        self.assertAlmostEqual(float(result.loc[13, "drift_change_abs__x"]), 0.0, places=5)
        changed = frame.copy(deep=True)
        changed.loc[12, "x"] = 1e6
        changed_result, _ = module.build_features(changed, ["x"], {})
        self.assertEqual(result.loc[12, "drift_abs_z__x"], changed_result.loc[12, "drift_abs_z__x"])

    def test_future_perturbation_and_labels_do_not_change_past_features(self):
        frame = sample(np.arange(25, dtype=float))
        baseline, metadata = module.build_features(frame, ["x", "constant"], {})
        changed = frame.copy(deep=True)
        changed.loc[changed.row_id > 15, "x"] = -9999
        changed["label"] = np.arange(len(changed)) % 2
        actual, _ = module.build_features(changed, ["x", "constant"], {})
        pd.testing.assert_frame_equal(baseline.loc[:15], actual.loc[:15])
        without_label, _ = module.build_features(frame.drop(columns="label"), ["x", "constant"], {})
        pd.testing.assert_frame_equal(baseline, without_label)
        self.assertEqual(len(metadata["representations"]["R0"]), 2)
        self.assertEqual(len(metadata["representations"]["R1"]), 7)

    def test_other_user_values_and_dataframe_precision_do_not_leak(self):
        a = sample(np.arange(18, dtype=float), user="opaque_a")
        b = sample(np.arange(18, dtype=float), user="opaque_b")
        b["row_id"] += len(a)
        frame = pd.concat([a, b], ignore_index=True)
        expected, _ = module.build_features(frame, ["x"], {})
        changed = frame.copy(deep=True)
        changed.loc[changed.user_id.eq("opaque_b"), "x"] = 9999
        changed["timestamp"] = changed["timestamp"].dt.as_unit("us")
        actual, _ = module.build_features(changed, ["x"], {})
        pd.testing.assert_frame_equal(expected.loc[:len(a) - 1], actual.loc[:len(a) - 1])

    def test_missing_values_stay_missing_constants_finite_and_cohort_not_filtered(self):
        frame = sample([1, np.nan, 3, 4] + list(range(4, 18)), hours=np.arange(18) * 25)
        snapshot = frame.copy(deep=True)
        result, metadata = module.build_features(frame, ["x", "constant"], {})
        self.assertTrue(np.isnan(result.loc[1, "current__x"]))
        self.assertAlmostEqual(float(result.loc[3, "static_abs_z__x"]), 2 / (1 + 1e-6), places=5)
        self.assertEqual(float(result.loc[12, "static_abs_z__constant"]), 0)
        self.assertFalse(np.isinf(result.to_numpy()).any())
        self.assertEqual(float(result.loc[11, "meta__common_cohort_eligible"]), 0)
        self.assertEqual(float(result.loc[12, "meta__common_cohort_eligible"]), 1)
        self.assertEqual(len(result), len(frame))
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in result.dtypes))
        self.assertEqual(list(result.index), list(frame.row_id))
        pd.testing.assert_frame_equal(frame, snapshot)
        json.dumps(metadata, allow_nan=False)

    def test_bad_time_order_reserved_predictors_and_short_gaps_are_rejected(self):
        frame = sample([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            module.build_features(frame.iloc[::-1], ["x"], {})
        with self.assertRaises(ValueError):
            module.build_features(frame, ["label"], {})
        with self.assertRaises(ValueError):
            module.build_features(frame, ["x"], {"history_gap_hours": 1})
        with self.assertRaises(ValueError):
            module.build_features(frame, ["x"], {"source_thinning_gap_hours": 1})


if __name__ == "__main__":
    unittest.main()
