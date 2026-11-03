"""Hand-calculated ranking, pairing and user-cluster inference checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merei_evaluation", HERE / "experiment/evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def predictions(users=24):
    rows = []
    for representation in ["R0_CURRENT", "R1_HISTORY", "R2_STATIC", "R3_ADAPTIVE", "R4_DRIFT"]:
        for index in range(users):
            label = index % 2
            rows.append(dict(dataset="toy", split="FINAL", cohort="UNSEEN", user_id=f"u{index:02}",
                             row_id=f"r{index:02}", label=label, model="LR", representation=representation,
                             score=0.8 if label else 0.2, available_at=pd.Timestamp("2026-01-01")))
    return pd.DataFrame(rows)


def config(**kwargs):
    return dict(datasets=["toy"], primary_models=["LR"], bootstrap_repeats=128, seed=42, **kwargs)


class RankingTests(unittest.TestCase):
    def test_tied_scores_use_threshold_ap_not_arbitrary_label_order(self):
        y = np.array([1., 0., 1., 0.])
        scores = np.array([0.9, 0.9, 0.5, 0.1])
        ap, auc = evaluation._rank_metrics(y, scores)
        self.assertAlmostEqual(ap, 7 / 12)
        self.assertAlmostEqual(auc, 0.625)
        self.assertEqual((ap, auc), evaluation._rank_metrics(y[[1, 0, 2, 3]], scores[[1, 0, 2, 3]]))

    def test_capacity_is_day_specific_with_row_id_tie_break(self):
        frame = pd.DataFrame(dict(label=[1, 0, 1, 0], score=[0.9, 0.9, 0.8, 0.1],
                                  row_id=["z", "a", "c", "d"], _prediction_day=["day1", "day1", "day2", "day2"]))
        result = evaluation._capacity_metrics(frame, 0.25, True)
        self.assertEqual(result["selected_rows"], 2)
        self.assertEqual(result["selected_positive_rows"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_one_class_metrics_are_not_reported_as_zero_or_one(self):
        for y in [np.array([0., 0.]), np.array([1., 1.])]:
            ap, auc = evaluation._rank_metrics(y, np.array([0.2, 0.8]))
            self.assertTrue(np.isnan(ap) and np.isnan(auc))

    def test_if_brier_excluded_even_when_scores_happen_to_be_bounded(self):
        frame = predictions().query("representation == 'R0_CURRENT'").copy()
        frame["model"] = "IF"
        metrics, paired, _ = evaluation.evaluate_predictions(frame, config())
        self.assertEqual(metrics.query("metric == 'BRIER'").iloc[0].status, "NOT_APPLICABLE")
        self.assertTrue(paired.status.eq("NOT_ESTIMABLE").all())


class PairingTests(unittest.TestCase):
    def test_identical_predictions_produce_zero_paired_effect_and_intervals(self):
        frame = predictions()
        before = frame.copy(deep=True)
        _, paired, meta = evaluation.evaluate_predictions(frame, config())
        self.assertEqual(len(paired), 4)
        self.assertTrue(paired.status.eq("ESTIMABLE").all())
        for field in ["effect", "ci95_lower", "ci95_upper", "bonferroni_lower", "bonferroni_upper"]:
            self.assertTrue(paired[field].eq(0).all())
        self.assertEqual(meta["primary_family_size"], 4)
        pd.testing.assert_frame_equal(frame, before)

    def test_mismatched_cohort_rows_are_not_silently_intersected(self):
        frame = predictions()
        frame = frame.loc[~(frame.representation.eq("R2_STATIC") & frame.row_id.eq("r00"))]
        with self.assertRaisesRegex(ValueError, "identical eligible rows"):
            evaluation.evaluate_predictions(frame, config())

    def test_changed_ground_truth_and_nonfinite_scores_rejected(self):
        frame = predictions()
        frame.loc[frame.index[0], "label"] = 1
        with self.assertRaisesRegex(ValueError, "ground truth"):
            evaluation.evaluate_predictions(frame, config())
        frame = predictions()
        frame.loc[0, "score"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluation.evaluate_predictions(frame, config())

    def test_same_user_cannot_enter_known_and_unseen_cohorts(self):
        frame = predictions()
        frame.loc[0, "cohort"] = "KNOWN"
        with self.assertRaisesRegex(ValueError, "both evaluation cohorts"):
            evaluation.evaluate_predictions(frame, config())

    def test_control_contrasts_are_descriptive_without_expanding_primary_family(self):
        frame = predictions()
        activity = frame.query("representation == 'R0_CURRENT'").copy()
        activity["representation"] = "C_ACTIVITY"
        _, paired, meta = evaluation.evaluate_predictions(pd.concat([frame, activity]), config())
        controls = paired.loc[paired.lower_representation.eq("C_ACTIVITY")]
        self.assertEqual(len(controls), 1)
        self.assertFalse(controls.iloc[0].primary)
        self.assertEqual(controls.iloc[0].status, "POINT_ESTIMATE_ONLY")
        self.assertEqual(meta["primary_family_size"], 4)

    def test_daily_primary_rejects_multiple_same_user_day_observations(self):
        frame = predictions()
        extra = frame.iloc[[0]].copy()
        extra["row_id"] = "another_row"
        extra["available_at"] += pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "multiple observations per user/day"):
            evaluation.evaluate_predictions(pd.concat([frame, extra]), config())

    def test_latest_daily_selection_uses_time_and_id_not_labels(self):
        frame = predictions().iloc[[0]].copy()
        other = frame.copy(); other["row_id"] = "later"; other["available_at"] += pd.Timedelta(hours=1)
        both = pd.concat([frame, other], ignore_index=True)
        selected = evaluation.latest_user_day_rows(both)
        both["label"] = 1 - both.label
        changed = evaluation.latest_user_day_rows(both)
        self.assertEqual(selected.row_id.tolist(), ["later"])
        self.assertEqual(selected.row_id.tolist(), changed.row_id.tolist())


class BootstrapTests(unittest.TestCase):
    def test_weighted_ap_matches_explicit_repeated_user_observations(self):
        frame = pd.DataFrame(dict(label=[1, 0, 1, 0], score=[0.9, 0.9, 0.5, 0.1], user_id=["a", "b", "a", "c"]))
        weights = np.array([[2, 1, 3], [0, 1, 1], [1, 0, 0]], dtype=np.uint16)
        actual = evaluation._bootstrap_ap(frame, weights, {"a": 0, "b": 1, "c": 2}, chunk_size=2)
        multiplicities = [2, 1, 2, 3]
        expected = evaluation._rank_metrics(np.repeat(frame.label.to_numpy(float), multiplicities),
                                             np.repeat(frame.score.to_numpy(float), multiplicities))[0]
        self.assertAlmostEqual(actual[0], expected)
        self.assertTrue(np.isnan(actual[1]) and np.isnan(actual[2]))

    def test_global_draws_reproducible_and_family_does_not_shrink(self):
        frame = predictions()
        first = evaluation.evaluate_predictions(frame, config())
        second = evaluation.evaluate_predictions(frame.sample(frac=1, random_state=2), config())
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assertEqual(first[2]["shared_draw_matrix_sha256"], second[2]["shared_draw_matrix_sha256"])
        settings = config(); settings["datasets"] = ["toy", "missing"]
        _, paired, meta = evaluation.evaluate_predictions(frame, settings)
        self.assertEqual(meta["primary_family_size"], 8)
        self.assertEqual(len(paired), 8)
        self.assertTrue(paired.query("dataset == 'missing'").status.eq("NOT_ESTIMABLE").all())

    def test_repeated_global_users_share_exact_draws_across_datasets(self):
        frame = predictions()
        baseline = frame.representation.eq("R1_HISTORY")
        frame.loc[baseline, "score"] = 0.5
        other = frame.copy(); other["dataset"] = "second"
        settings = config(); settings["datasets"] = ["toy", "second"]
        _, paired, meta = evaluation.evaluate_predictions(pd.concat([frame, other]), settings)
        one = paired.loc[paired.dataset.eq("toy")].reset_index(drop=True)
        two = paired.loc[paired.dataset.eq("second")].reset_index(drop=True)
        pd.testing.assert_frame_equal(one.drop(columns="dataset"), two.drop(columns="dataset"))
        self.assertEqual(meta["global_user_count"], 24)

    def test_too_few_two_class_draws_reported_without_fake_interval(self):
        _, paired, _ = evaluation.evaluate_predictions(predictions(users=2), config())
        self.assertTrue(paired.status.eq("NOT_ESTIMABLE_UNCERTAINTY").all())
        self.assertTrue(paired.ci95_lower.isna().all())
        self.assertTrue(paired.effect.eq(0).all())


if __name__ == "__main__":
    unittest.main()
