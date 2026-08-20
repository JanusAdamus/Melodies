from __future__ import annotations

import math
import unittest

from Comparacion.decision import pareto_front
from Comparacion.structural_metrics import (
    adjusted_rand_index,
    boundary_f1,
    normalized_mutual_information,
)
from Comparacion.statistics import pairwise_model_comparisons


class StructuralMetricTests(unittest.TestCase):
    def test_boundary_f1_counts_exact_and_tolerant_matches(self) -> None:
        """Catches ignoring tolerance or counting an out-of-range boundary."""

        exact = boundary_f1([2, 6, 10], [2, 6, 11], tolerance=0)
        tolerant = boundary_f1([2, 6, 10], [2, 6, 11], tolerance=1)

        self.assertEqual(exact["n_matches"], 2)
        self.assertAlmostEqual(exact["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(exact["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(exact["f1"], 2.0 / 3.0)
        self.assertEqual(tolerant["n_matches"], 3)
        self.assertEqual(tolerant["precision"], 1.0)
        self.assertEqual(tolerant["recall"], 1.0)
        self.assertEqual(tolerant["f1"], 1.0)

    def test_boundary_f1_uses_each_prediction_at_most_once(self) -> None:
        """Catches reusing one prediction for two nearby references."""

        result = boundary_f1([10, 11], [10], tolerance=1)

        self.assertEqual(result["n_matches"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2.0 / 3.0)

    def test_boundary_f1_defines_empty_set_scores(self) -> None:
        """Catches NaN or division-by-zero behavior for empty boundary sets."""

        self.assertEqual(
            boundary_f1([], [], tolerance=0),
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "n_matches": 0,
                "n_reference": 0,
                "n_predicted": 0,
                "tolerance": 0,
            },
        )
        self.assertEqual(boundary_f1([], [3], tolerance=0)["f1"], 0.0)
        self.assertEqual(boundary_f1([3], [], tolerance=0)["f1"], 0.0)

    def test_boundary_f1_rejects_nonintegral_or_negative_inputs(self) -> None:
        """Catches lossy coercion and invalid coordinate domains."""

        invalid_calls = (
            ([1.5], [1], 0),
            ([-1], [1], 0),
            ([True], [1], 0),
            ([1], [1], -1),
            ([1], [1], 0.5),
        )
        for reference, predicted, tolerance in invalid_calls:
            with self.subTest(reference=reference, predicted=predicted, tolerance=tolerance):
                with self.assertRaises((TypeError, ValueError)):
                    boundary_f1(reference, predicted, tolerance)

    def test_partition_scores_are_one_for_identical_one_cluster_partitions(self) -> None:
        """Catches zero-entropy and zero-denominator degeneracies."""

        self.assertEqual(normalized_mutual_information(["a", "a", "a"], [7, 7, 7]), 1.0)
        self.assertEqual(adjusted_rand_index(["a", "a", "a"], [7, 7, 7]), 1.0)

    def test_partition_scores_ignore_permuted_hashable_label_values(self) -> None:
        """Catches comparing label values instead of partition membership."""

        reference = [("verse",), ("verse",), frozenset({"chorus"}), frozenset({"chorus"})]
        predicted = [9, 9, 4, 4]

        self.assertAlmostEqual(normalized_mutual_information(reference, predicted), 1.0)
        self.assertAlmostEqual(adjusted_rand_index(reference, predicted), 1.0)

    def test_partition_scores_consume_label_iterables_once(self) -> None:
        """Catches a contingency implementation that exhausts generators."""

        nmi = normalized_mutual_information(
            (label for label in ["a", "a", "b", "b"]),
            (label for label in [2, 2, 1, 1]),
        )
        ari = adjusted_rand_index(
            (label for label in ["a", "a", "b", "b"]),
            (label for label in [2, 2, 1, 1]),
        )

        self.assertEqual(nmi, 1.0)
        self.assertEqual(ari, 1.0)

    def test_partition_scores_match_literal_crossed_contingency_table(self) -> None:
        """Catches incorrect contingency marginals or ARI chance adjustment."""

        reference = ["a", "a", "b", "b"]
        predicted = ["x", "y", "x", "y"]

        self.assertAlmostEqual(normalized_mutual_information(reference, predicted), 0.0)
        self.assertAlmostEqual(adjusted_rand_index(reference, predicted), -0.5)

    def test_partition_scores_require_equal_nonempty_sequences(self) -> None:
        """Catches vacuous scores and silent truncation of unequal inputs."""

        for reference, predicted in (([], []), (["a"], []), (["a"], ["x", "y"])):
            with self.subTest(reference=reference, predicted=predicted):
                with self.assertRaises(ValueError):
                    normalized_mutual_information(reference, predicted)
                with self.assertRaises(ValueError):
                    adjusted_rand_index(reference, predicted)


class PairedInferenceTests(unittest.TestCase):
    def test_three_models_average_repeats_and_report_oriented_pairwise_differences(self) -> None:
        """Catches seed-level pairing, fraction leakage, and reversed NLL differences."""

        rows = [
            {"model": "alpha", "piece_id": "p1", "frac": 1.0, "test_nll": -1.0},
            {"model": "alpha", "piece_id": "p1", "frac": 1.0, "test_nll": 1.0},
            {"model": "alpha", "piece_id": "p2", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "p3", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "p4", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "p5", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "p2", "frac": 0.5, "test_nll": -100.0},
            {"model": "beta", "piece_id": "p1", "frac": 1.0, "test_nll": 0.0},
            {"model": "beta", "piece_id": "p1", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "piece_id": "p2", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "piece_id": "p3", "frac": 1.0, "test_nll": 3.0},
            {"model": "beta", "piece_id": "p4", "frac": 1.0, "test_nll": 4.0},
            {"model": "beta", "piece_id": "p5", "frac": 1.0, "test_nll": 5.0},
            {"model": "gamma", "piece_id": "p1", "frac": 1.0, "test_nll": -2.0},
            {"model": "gamma", "piece_id": "p1", "frac": 1.0, "test_nll": 0.0},
            {"model": "gamma", "piece_id": "p2", "frac": 1.0, "test_nll": 1.0},
            {"model": "gamma", "piece_id": "p3", "frac": 1.0, "test_nll": 2.0},
            {"model": "gamma", "piece_id": "p4", "frac": 1.0, "test_nll": 3.0},
            {"model": "gamma", "piece_id": "p5", "frac": 1.0, "test_nll": 4.0},
        ]

        result = pairwise_model_comparisons(rows, bootstrap_samples=512, seed=23)
        repeated = pairwise_model_comparisons(rows, bootstrap_samples=512, seed=23)
        comparisons = {
            (item["model_a"], item["model_b"]): item
            for item in result["comparisons"]
        }

        self.assertEqual(result, repeated)
        self.assertEqual(result["metric"], "test_nll")
        self.assertEqual(result["difference_orientation"], "model_a_minus_model_b")
        self.assertEqual(list(comparisons), [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")])
        self.assertEqual(comparisons[("alpha", "beta")]["n_pairs"], 5)
        self.assertEqual(comparisons[("alpha", "beta")]["piece_ids"], ["p1", "p2", "p3", "p4", "p5"])
        self.assertEqual(comparisons[("alpha", "beta")]["mean_difference"], -3.0)
        self.assertEqual(comparisons[("alpha", "beta")]["median_difference"], -3.0)
        self.assertEqual(comparisons[("alpha", "beta")]["difference_orientation"], "model_a_minus_model_b")
        self.assertEqual(comparisons[("alpha", "beta")]["interpretation"], "negative_favors_model_a")
        self.assertEqual(comparisons[("alpha", "gamma")]["mean_difference"], -1.8)
        self.assertEqual(comparisons[("alpha", "gamma")]["median_difference"], -2.0)
        self.assertEqual(comparisons[("beta", "gamma")]["mean_difference"], 1.2)
        self.assertEqual(comparisons[("beta", "gamma")]["median_difference"], 1.0)
        self.assertGreaterEqual(comparisons[("alpha", "beta")]["bootstrap_95_ci"][0], -5.0)
        self.assertLessEqual(comparisons[("alpha", "beta")]["bootstrap_95_ci"][1], -1.0)

        valid = sorted(
            (
                item["p_value"],
                item["p_value_holm"],
            )
            for item in result["comparisons"]
            if item["wilcoxon_status"] == "ok"
        )
        self.assertEqual(len(valid), 3)
        self.assertEqual([adjusted for _, adjusted in valid], sorted(adjusted for _, adjusted in valid))
        for raw, adjusted in valid:
            self.assertGreaterEqual(adjusted, raw)
            self.assertLessEqual(adjusted, 1.0)

    def test_missing_and_nonfinite_rows_do_not_fabricate_pairs(self) -> None:
        """Catches imputing absent work metrics or treating partial fractions as full data."""

        rows = [
            {"model": "alpha", "piece_id": "p1", "frac": 1.0, "test_nll": 1.0},
            {"model": "alpha", "piece_id": "p2", "frac": 1.0, "test_nll": math.nan},
            {"model": "alpha", "piece_id": "p3", "frac": 1.0},
            {"model": "beta", "piece_id": "p1", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "piece_id": "p2", "frac": 1.0, "test_nll": 3.0},
            {"model": "beta", "piece_id": "p3", "frac": 1.0, "test_nll": 4.0},
            {"model": "gamma", "piece_id": "p1", "frac": 0.5, "test_nll": 0.0},
            {"model": "gamma", "piece_id": "p2", "frac": 1.0, "test_nll": math.inf},
        ]

        result = pairwise_model_comparisons(rows, bootstrap_samples=128, seed=7)
        comparisons = {
            (item["model_a"], item["model_b"]): item
            for item in result["comparisons"]
        }

        self.assertEqual(list(comparisons), [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")])
        alpha_beta = comparisons[("alpha", "beta")]
        self.assertEqual(alpha_beta["n_pairs"], 1)
        self.assertEqual(alpha_beta["piece_ids"], ["p1"])
        self.assertEqual(alpha_beta["mean_difference"], -1.0)
        self.assertEqual(alpha_beta["bootstrap_95_ci"], [-1.0, -1.0])
        self.assertEqual(alpha_beta["wilcoxon_status"], "unavailable")
        self.assertEqual(alpha_beta["wilcoxon_reason"], "fewer_than_two_nonzero_differences")
        self.assertIsNone(alpha_beta["p_value"])
        self.assertIsNone(alpha_beta["p_value_holm"])

        for pair in (("alpha", "gamma"), ("beta", "gamma")):
            with self.subTest(pair=pair):
                comparison = comparisons[pair]
                self.assertEqual(comparison["status"], "unavailable")
                self.assertEqual(comparison["reason"], "no_paired_finite_works")
                self.assertEqual(comparison["n_pairs"], 0)
                self.assertIsNone(comparison["mean_difference"])
                self.assertIsNone(comparison["bootstrap_95_ci"])

    def test_paired_inference_validates_bootstrap_configuration(self) -> None:
        """Catches vacuous resampling and non-reproducible seed coercion."""

        for bootstrap_samples, seed in ((0, 1), (-1, 1), (1.5, 1), (True, 1), (10, -1), (10, 1.5)):
            with self.subTest(bootstrap_samples=bootstrap_samples, seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    pairwise_model_comparisons([], bootstrap_samples=bootstrap_samples, seed=seed)


class ParetoDecisionTests(unittest.TestCase):
    def test_pareto_front_retains_tradeoffs_and_equal_duplicates(self) -> None:
        """Catches single-axis ranking, weak dominance, and duplicate removal."""

        rows = [
            {"model": "balanced", "test_nll": 1.0, "train_time": 10.0, "boundary_f1": 0.8},
            {"model": "dominated", "test_nll": 2.0, "train_time": 10.0, "boundary_f1": 0.7},
            {"model": "predictive", "test_nll": 0.5, "train_time": 20.0, "boundary_f1": 0.6},
            {"model": "duplicate", "test_nll": 1.0, "train_time": 10.0, "boundary_f1": 0.8},
        ]

        front = pareto_front(
            rows,
            minimize=("test_nll", "train_time"),
            maximize=("boundary_f1",),
        )

        self.assertEqual(
            [row["model"] for row in front],
            ["balanced", "predictive", "duplicate"],
        )

    def test_missing_or_nonfinite_required_axis_makes_pair_incomparable(self) -> None:
        """Catches treating unavailable metrics as best, worst, or ignorable."""

        rows = [
            {"model": "complete", "test_nll": 1.0, "train_time": 1.0, "boundary_f1": 0.9},
            {"model": "weak_complete", "test_nll": 2.0, "train_time": 2.0, "boundary_f1": 0.8},
            {"model": "missing_structure", "test_nll": 3.0, "train_time": 3.0},
            {"model": "nonfinite_cost", "test_nll": 3.0, "train_time": math.inf, "boundary_f1": 0.2},
        ]
        original = [dict(row) for row in rows]

        front = pareto_front(
            rows,
            minimize=("test_nll", "train_time"),
            maximize=("boundary_f1",),
        )

        self.assertEqual(
            [row["model"] for row in front],
            ["complete", "missing_structure", "nonfinite_cost"],
        )
        self.assertEqual(rows, original)

    def test_pareto_front_rejects_an_axis_with_conflicting_direction(self) -> None:
        """Catches contradictory dominance declarations for one metric."""

        with self.assertRaises(ValueError):
            pareto_front(
                [{"model": "a", "score": 1.0}],
                minimize=("score",),
                maximize=("score",),
            )


if __name__ == "__main__":
    unittest.main()
