from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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

    def test_boundary_f1_rejects_duplicate_annotation_coordinates(self) -> None:
        """Catches treating repeated annotation rows as distinct boundaries."""

        for reference, predicted in (([2, 2, 8], [2, 8]), ([2, 8], [2, 8, 8])):
            with self.subTest(reference=reference, predicted=predicted):
                with self.assertRaises(ValueError):
                    boundary_f1(reference, predicted, tolerance=0)

    def test_partition_scores_are_one_for_identical_one_cluster_partitions(self) -> None:
        """Catches zero-entropy and zero-denominator degeneracies."""

        self.assertEqual(normalized_mutual_information(["a", "a", "a"], [7, 7, 7]), 1.0)
        self.assertEqual(adjusted_rand_index(["a", "a", "a"], [7, 7, 7]), 1.0)

    def test_partition_scores_are_one_for_permuted_all_singleton_partitions(self) -> None:
        """Catches mishandling the zero-denominator all-singleton ARI case."""

        reference = ["a", "b", "c", "d"]
        predicted = [40, 30, 20, 10]

        self.assertEqual(normalized_mutual_information(reference, predicted), 1.0)
        self.assertEqual(adjusted_rand_index(reference, predicted), 1.0)

    def test_partition_scores_are_symmetric_and_invariant_to_label_permutation(self) -> None:
        """Catches asymmetric marginals or dependence on cluster label names."""

        reference = ["a", "a", "a", "b", "b", "c"]
        predicted = ["x", "x", "y", "y", "z", "z"]
        permuted_prediction = [3, 3, 1, 1, 2, 2]

        for left, right in (
            (reference, predicted),
            (predicted, reference),
            (reference, permuted_prediction),
        ):
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(
                    normalized_mutual_information(left, right),
                    0.5206652463984818,
                )
                self.assertAlmostEqual(
                    adjusted_rand_index(left, right),
                    0.07407407407407407,
                )

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
    def test_canonical_variants_and_repeated_seeds_are_averaged_before_inference(self) -> None:
        """Catches file-level pairing or using first, last, or seed-level values."""

        rows = [
            {"model": "alpha", "piece_id": "one-a", "canonical_work_id": "work-one", "frac": 1.0, "data_seed": 1, "model_seed": 1, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "one-b", "canonical_work_id": "work-one", "frac": 1.0, "data_seed": 1, "model_seed": 2, "test_nll": 9.0},
            {"model": "alpha", "piece_id": "one-a", "canonical_work_id": "work-one", "frac": 1.0, "data_seed": 2, "model_seed": 1, "test_nll": 12.0},
            {"model": "beta", "piece_id": "one-c", "canonical_work_id": "work-one", "frac": 1.0, "data_seed": 1, "model_seed": 1, "test_nll": 2.0},
            {"model": "beta", "piece_id": "one-d", "canonical_work_id": "work-one", "frac": 1.0, "data_seed": 2, "model_seed": 2, "test_nll": 20.0},
            {"model": "alpha", "piece_id": "two-a", "canonical_work_id": "work-two", "frac": 1.0, "data_seed": 1, "model_seed": 1, "test_nll": 30.0},
            {"model": "alpha", "piece_id": "two-b", "canonical_work_id": "work-two", "frac": 1.0, "data_seed": 2, "model_seed": 2, "test_nll": 0.0},
            {"model": "beta", "piece_id": "two-c", "canonical_work_id": "work-two", "frac": 1.0, "data_seed": 1, "model_seed": 1, "test_nll": 1.0},
            {"model": "beta", "piece_id": "two-d", "canonical_work_id": "work-two", "frac": 1.0, "data_seed": 1, "model_seed": 2, "test_nll": 5.0},
            {"model": "beta", "piece_id": "two-c", "canonical_work_id": "work-two", "frac": 1.0, "data_seed": 2, "model_seed": 1, "test_nll": 12.0},
            {"model": "alpha", "piece_id": "ignored", "canonical_work_id": "work-two", "frac": 0.5, "data_seed": 9, "model_seed": 9, "test_nll": -100.0},
        ]

        def checked_wilcoxon(differences: object) -> SimpleNamespace:
            self.assertEqual(tuple(float(value) for value in differences), (-4.0, 9.0))
            return SimpleNamespace(statistic=1.0, pvalue=0.5)

        with patch("Comparacion.statistics.wilcoxon", side_effect=checked_wilcoxon):
            result = pairwise_model_comparisons(rows, bootstrap_samples=4, seed=2)

        comparison = result["comparisons"][0]
        self.assertEqual(comparison["n_pairs"], 2)
        self.assertEqual(comparison["canonical_work_ids"], ["work-one", "work-two"])
        self.assertEqual(comparison["mean_difference"], 2.5)
        self.assertEqual(comparison["median_difference"], 2.5)
        self.assertEqual(comparison["bootstrap_95_ci"], [-4.0, 2.5])
        self.assertNotEqual(comparison["bootstrap_95_ci"], [0.0, 4.8125])
        self.assertEqual(comparison["difference_orientation"], "model_a_minus_model_b")
        self.assertEqual(comparison["interpretation"], "negative_favors_model_a")

    def test_missing_canonical_ids_are_not_replaced_with_piece_ids(self) -> None:
        """Catches silently pairing file variants whose canonical identity is absent."""

        rows = [
            {"model": "alpha", "piece_id": "valid-a", "canonical_work_id": "real-work", "frac": 1.0, "test_nll": 1.0},
            {"model": "beta", "piece_id": "valid-b", "canonical_work_id": "real-work", "frac": 1.0, "test_nll": 2.0},
            {"model": "alpha", "piece_id": "shared-file", "frac": 1.0, "test_nll": 100.0},
            {"model": "beta", "piece_id": "shared-file", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "piece_id": "empty-file", "canonical_work_id": "", "frac": 1.0, "test_nll": 50.0},
            {"model": "beta", "piece_id": "empty-file", "canonical_work_id": "   ", "frac": 1.0, "test_nll": 0.0},
        ]

        result = pairwise_model_comparisons(rows, bootstrap_samples=8, seed=4)
        comparison = result["comparisons"][0]

        self.assertEqual(comparison["n_pairs"], 1)
        self.assertEqual(comparison["canonical_work_ids"], ["real-work"])
        self.assertEqual(comparison["mean_difference"], -1.0)
        self.assertEqual(comparison["bootstrap_95_ci"], [-1.0, -1.0])

    def test_holm_adjustment_has_exact_values_for_three_distinct_raw_p_values(self) -> None:
        """Catches wrong rank factors, pair mapping, monotonicity, or capping logic."""

        rows = [
            {"model": "alpha", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 1.0},
            {"model": "alpha", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 1.0},
            {"model": "alpha", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 1.0},
            {"model": "beta", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 3.0},
            {"model": "beta", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 4.0},
            {"model": "gamma", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 4.0},
            {"model": "gamma", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 6.0},
            {"model": "gamma", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 8.0},
        ]
        scripted_results = [
            SimpleNamespace(statistic=4.0, pvalue=0.8),
            SimpleNamespace(statistic=1.0, pvalue=0.01),
            SimpleNamespace(statistic=3.0, pvalue=0.6),
        ]

        with patch("Comparacion.statistics.wilcoxon", side_effect=scripted_results):
            result = pairwise_model_comparisons(rows, bootstrap_samples=8, seed=11)
        comparisons = {
            (item["model_a"], item["model_b"]): item
            for item in result["comparisons"]
        }

        self.assertEqual(result["n_valid_tests"], 3)
        self.assertAlmostEqual(comparisons[("alpha", "beta")]["p_value"], 0.8)
        self.assertAlmostEqual(comparisons[("alpha", "beta")]["p_value_holm"], 1.0)
        self.assertAlmostEqual(comparisons[("alpha", "gamma")]["p_value"], 0.01)
        self.assertAlmostEqual(comparisons[("alpha", "gamma")]["p_value_holm"], 0.03)
        self.assertAlmostEqual(comparisons[("beta", "gamma")]["p_value"], 0.6)
        self.assertAlmostEqual(comparisons[("beta", "gamma")]["p_value_holm"], 1.0)

    def test_actual_wilcoxon_is_used_for_nonzero_canonical_work_differences(self) -> None:
        """Catches replacing the SciPy integration with only scripted test behavior."""

        rows = [
            {"model": "alpha", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 0.0},
            {"model": "alpha", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 0.0},
            {"model": "beta", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 1.0},
            {"model": "beta", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 3.0},
        ]

        result = pairwise_model_comparisons(rows, bootstrap_samples=8, seed=5)
        comparison = result["comparisons"][0]

        self.assertEqual(comparison["wilcoxon_status"], "ok")
        self.assertEqual(comparison["wilcoxon_statistic"], 0.0)
        self.assertEqual(comparison["p_value"], 0.25)
        self.assertEqual(comparison["p_value_holm"], 0.25)

    def test_missing_and_nonfinite_rows_do_not_fabricate_pairs(self) -> None:
        """Catches imputing absent work metrics or treating partial fractions as full data."""

        rows = [
            {"model": "alpha", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 1.0},
            {"model": "alpha", "canonical_work_id": "w2", "frac": 1.0, "test_nll": math.nan},
            {"model": "alpha", "canonical_work_id": "w3", "frac": 1.0},
            {"model": "beta", "canonical_work_id": "w1", "frac": 1.0, "test_nll": 2.0},
            {"model": "beta", "canonical_work_id": "w2", "frac": 1.0, "test_nll": 3.0},
            {"model": "beta", "canonical_work_id": "w3", "frac": 1.0, "test_nll": 4.0},
            {"model": "gamma", "canonical_work_id": "w1", "frac": 0.5, "test_nll": 0.0},
            {"model": "gamma", "canonical_work_id": "w2", "frac": 1.0, "test_nll": math.inf},
        ]

        result = pairwise_model_comparisons(rows, bootstrap_samples=128, seed=7)
        comparisons = {
            (item["model_a"], item["model_b"]): item
            for item in result["comparisons"]
        }

        self.assertEqual(list(comparisons), [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")])
        alpha_beta = comparisons[("alpha", "beta")]
        self.assertEqual(alpha_beta["n_pairs"], 1)
        self.assertEqual(alpha_beta["canonical_work_ids"], ["w1"])
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

    def test_incomplete_candidate_cannot_dominate_a_complete_row(self) -> None:
        """Catches treating a missing maximize value as an artificial best score."""

        rows = [
            {"model": "complete", "test_nll": 2.0, "train_time": 2.0, "boundary_f1": 0.5},
            {"model": "incomplete_challenger", "test_nll": 1.0, "train_time": 1.0},
        ]

        front = pareto_front(
            rows,
            minimize=("test_nll", "train_time"),
            maximize=("boundary_f1",),
        )

        self.assertEqual([row["model"] for row in front], ["complete", "incomplete_challenger"])

    def test_pareto_front_sanitizes_declared_nonfinite_axes_for_strict_json(self) -> None:
        """Catches returning NaN or infinity while preserving incomparable rows."""

        rows = [
            {"model": "finite", "test_nll": 2.0, "train_time": 2.0, "boundary_f1": 0.5},
            {"model": "nan_loss", "test_nll": math.nan, "train_time": 1.0, "boundary_f1": 0.9},
            {"model": "infinite_axes", "test_nll": 1.0, "train_time": math.inf, "boundary_f1": -math.inf},
            {"model": "missing_structure", "test_nll": 1.0, "train_time": 1.0},
        ]
        original = [dict(row) for row in rows]

        front = pareto_front(
            rows,
            minimize=("test_nll", "train_time"),
            maximize=("boundary_f1",),
        )

        self.assertEqual(
            [row["model"] for row in front],
            ["finite", "nan_loss", "infinite_axes", "missing_structure"],
        )
        self.assertIsNone(front[1]["test_nll"])
        self.assertIsNone(front[2]["train_time"])
        self.assertIsNone(front[2]["boundary_f1"])
        self.assertNotIn("boundary_f1", front[3])
        self.assertEqual(rows, original)
        self.assertTrue(math.isnan(rows[1]["test_nll"]))
        self.assertTrue(math.isinf(rows[2]["train_time"]))
        json.dumps(front, allow_nan=False)

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
