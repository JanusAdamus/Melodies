from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from Comparacion.vomm import VariableOrderMarkovModel, select_vomm_by_validation
from next_token_experiment.schemas import PreparedPiece


def _piece(piece_id: str, tokens: list[int]) -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=f"/tmp/{piece_id}.musicxml",
        title=piece_id,
        composer="composer",
        canonical_work_id=piece_id,
        representation="pitch_class",
        vocabulary=["0", "1", "<BOS>"],
        tokens=tokens,
        n_events=len(tokens),
        metadata={},
    )


class VariableOrderMarkovModelTests(unittest.TestCase):
    def test_constructor_rejects_fractional_and_nonfinite_inputs(self) -> None:
        """Catches lossy integer casts and non-finite smoothing parameters."""

        invalid_parameters = (
            {"max_order": 1.5, "vocabulary_size": 3},
            {"max_order": math.inf, "vocabulary_size": 3},
            {"max_order": math.nan, "vocabulary_size": 3},
            {"max_order": 1, "vocabulary_size": 2.5},
            {"max_order": 1, "vocabulary_size": math.nan},
            {"max_order": 1, "vocabulary_size": math.inf},
            {"max_order": 1, "vocabulary_size": 3, "alpha": math.nan},
            {"max_order": 1, "vocabulary_size": 3, "alpha": math.inf},
            {"max_order": 1, "vocabulary_size": 3, "backoff_strength": math.nan},
            {"max_order": 1, "vocabulary_size": 3, "backoff_strength": math.inf},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    VariableOrderMarkovModel(**parameters)

    def test_training_rejects_noninteger_and_bos_targets(self) -> None:
        """Catches silently coercing data or counting BOS as a musical target."""

        for invalid_target in (0.5, True, 2):
            with self.subTest(invalid_target=invalid_target):
                model = VariableOrderMarkovModel(max_order=1, vocabulary_size=3)
                with self.assertRaises(ValueError):
                    model.fit([[invalid_target]])

    def test_evaluation_rejects_noninteger_and_bos_targets(self) -> None:
        """Catches silently coercing evaluation data or scoring BOS as music."""

        model = VariableOrderMarkovModel(max_order=1, vocabulary_size=3).fit([[0, 1]])
        for invalid_target in (0.5, True, 2):
            with self.subTest(invalid_target=invalid_target):
                with self.assertRaises(ValueError):
                    model.evaluate([_piece("invalid", [invalid_target])], max_context_length=2)

    def test_prediction_rejects_invalid_tokens_and_bos_placement(self) -> None:
        """Catches accepting malformed external prediction contexts."""

        model = VariableOrderMarkovModel(max_order=2, vocabulary_size=3).fit([[0, 1]])
        invalid_contexts = ([0.5], [True], [-1], [3], [0, 2], [2, 2])
        for context in invalid_contexts:
            with self.subTest(context=context):
                with self.assertRaises(ValueError):
                    model.predict_distribution(context)

        distribution = model.predict_distribution([2, 0, 1])
        self.assertAlmostEqual(float(distribution.sum()), 1.0)

    def test_distribution_is_normalized_and_strictly_positive(self) -> None:
        """Catches a missing smoothing or defensive normalization branch."""

        model = VariableOrderMarkovModel(max_order=2, vocabulary_size=3)
        model.fit([[0, 1, 0], [0, 1, 0]])

        distribution = model.predict_distribution([2, 0, 1])

        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        self.assertTrue((distribution > 0.0).all())

    def test_unseen_long_context_uses_observed_shorter_suffix(self) -> None:
        """Catches skipping an available suffix after a long-context miss."""

        model = VariableOrderMarkovModel(max_order=2, vocabulary_size=3)
        model.fit([[0, 1, 0], [0, 1, 0]])

        shorter_suffix = model.predict_distribution([1])
        unseen_long_context = model.predict_distribution([1, 1])

        self.assertAlmostEqual(float(shorter_suffix[0]), 71.0 / 105.0)
        self.assertAlmostEqual(float(unseen_long_context[0]), 71.0 / 105.0)
        self.assertAlmostEqual(float(unseen_long_context[1]), 13.0 / 63.0)
        self.assertAlmostEqual(float(unseen_long_context[2]), 37.0 / 315.0)

    def test_repeated_long_context_outweighs_unigram_probability(self) -> None:
        """Catches ignoring an observed higher-order context during interpolation."""

        model = VariableOrderMarkovModel(max_order=2, vocabulary_size=3)
        model.fit([[0, 1, 0], [0, 1, 0]])

        unigram = model.predict_distribution([])
        repeated_context = model.predict_distribution([0, 1])

        self.assertAlmostEqual(float(unigram[0]), 3.0 / 5.0)
        self.assertAlmostEqual(float(repeated_context[0]), 221.0 / 315.0)
        self.assertGreater(float(repeated_context[0]), float(unigram[0]))

    def test_evaluation_scores_each_asymmetric_slice_target_exactly_once(self) -> None:
        """Catches duplicating a slice head or omitting the short tail."""

        class InstrumentedVOMM(VariableOrderMarkovModel):
            def __init__(self) -> None:
                super().__init__(max_order=1, vocabulary_size=3)
                self.contexts: list[list[int]] = []

            def predict_distribution(self, context: list[int]) -> np.ndarray:
                self.contexts.append(list(context))
                return super().predict_distribution(context)

        model = InstrumentedVOMM().fit([[0, 1, 0, 1, 1]])
        evaluation = model.evaluate([_piece("tail", [0, 1, 0, 1, 1])], max_context_length=3)

        expected_log_likelihood = math.log(
            (32.0 / 65.0) * (179.0 / 273.0) * (113.0 / 273.0) * (24.0 / 65.0) * (127.0 / 273.0)
        )
        self.assertEqual(
            model.contexts,
            [[2], [2, 0], [2, 0, 1], [2], [2, 1]],
        )
        self.assertEqual(evaluation["summary"]["n_tokens"], 5)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_log_likelihood)
        self.assertAlmostEqual(
            evaluation["summary"]["test_nll_per_token"],
            -expected_log_likelihood / 5.0,
        )

    def test_evaluation_accumulates_literal_nll_accuracy_and_brier_score(self) -> None:
        """Catches deriving aggregate metrics from anything but every full distribution."""

        class ScriptedVOMM(VariableOrderMarkovModel):
            def __init__(self) -> None:
                super().__init__(max_order=0, vocabulary_size=3)
                self.distributions = iter(
                    (
                        np.array([0.7, 0.2, 0.1]),
                        np.array([0.1, 0.6, 0.3]),
                        np.array([0.8, 0.1, 0.1]),
                    )
                )

            def predict_distribution(self, context: list[int]) -> np.ndarray:
                return next(self.distributions)

        model = ScriptedVOMM().fit([[]])
        evaluation = model.evaluate(
            [_piece("mostly-correct", [0, 1]), _piece("incorrect", [1])],
            max_context_length=2,
        )

        summary = evaluation["summary"]
        first_piece, second_piece = evaluation["piece_metrics"]
        self.assertIn("accuracy", summary)
        self.assertIn("brier_score", summary)
        self.assertIn("accuracy", first_piece)
        self.assertIn("brier_score", first_piece)
        self.assertIn("accuracy", second_piece)
        self.assertIn("brier_score", second_piece)
        self.assertAlmostEqual(summary["test_nll_per_token"], -math.log(0.7 * 0.6 * 0.1) / 3.0)
        self.assertAlmostEqual(summary["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["brier_score"], 0.62)
        self.assertAlmostEqual(first_piece["nll_per_token"], -math.log(0.7 * 0.6) / 2.0)
        self.assertAlmostEqual(first_piece["accuracy"], 1.0)
        self.assertAlmostEqual(first_piece["brier_score"], 0.20)
        self.assertAlmostEqual(second_piece["nll_per_token"], -math.log(0.1))
        self.assertAlmostEqual(second_piece["accuracy"], 0.0)
        self.assertAlmostEqual(second_piece["brier_score"], 1.46)

    def test_validation_selection_rejects_zero_token_sets(self) -> None:
        """Catches selecting an order from a vacuous zero-token NLL."""

        for validation_pieces in ([], [_piece("empty", [])]):
            with self.subTest(validation_pieces=validation_pieces):
                with self.assertRaises(ValueError):
                    select_vomm_by_validation(
                        train_sequences=[[0, 1]],
                        validation_pieces=validation_pieces,
                        candidate_orders=(0, 1),
                        vocabulary_size=3,
                    )

    def test_validation_selection_retains_the_lowest_nll_order(self) -> None:
        """Catches returning a non-selected candidate after validation scoring."""

        with patch("Comparacion.vomm.time.perf_counter", side_effect=range(10)):
            model = select_vomm_by_validation(
                train_sequences=[[0, 1, 0], [0, 1, 0]],
                validation_pieces=[_piece("validation", [0, 1, 0])],
                candidate_orders=(0, 1),
                vocabulary_size=3,
                max_context_length=3,
            )

        expected_validation_nll = -math.log((71.0 / 105.0) * (37.0 / 63.0) * (71.0 / 105.0)) / 3.0
        self.assertEqual(model.selected_order, 1)
        self.assertEqual(model.max_order, 1)
        self.assertAlmostEqual(model.validation_nll_per_token, expected_validation_nll)
        self.assertAlmostEqual(float(model.predict_distribution([0])[1]), 37.0 / 63.0)

        evaluation = model.evaluate([_piece("test", [0, 1, 0])], max_context_length=3)
        summary = evaluation["summary"]
        self.assertEqual(summary["model"], "vomm")
        self.assertEqual(summary["selected_order"], 1)
        self.assertEqual(summary["n_params"], 5)
        self.assertEqual(summary["count_table_size"], 5)
        self.assertIn("selection_wall_clock_s", summary)
        self.assertIn("selected_fit_wall_clock_s", summary)
        self.assertIn("selected_validation_evaluation_wall_clock_s", summary)
        self.assertEqual(summary["selection_wall_clock_s"], 9.0)
        self.assertEqual(summary["fit_wall_clock_s"], 9.0)
        self.assertEqual(summary["train_time_sec"], 9.0)
        self.assertEqual(summary["selected_fit_wall_clock_s"], 1.0)
        self.assertEqual(summary["selected_validation_evaluation_wall_clock_s"], 1.0)
        self.assertGreaterEqual(summary["evaluation_wall_clock_s"], 0.0)
        self.assertIn("accuracy", summary)
        self.assertIn("brier_score", summary)


if __name__ == "__main__":
    unittest.main()
