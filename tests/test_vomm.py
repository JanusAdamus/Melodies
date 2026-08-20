from __future__ import annotations

import math
import unittest

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

    def test_evaluation_scores_the_one_token_tail_after_a_context_reset(self) -> None:
        """Catches dropping a final evaluation slice shorter than max_context_length."""

        model = VariableOrderMarkovModel(max_order=1, vocabulary_size=3)
        model.fit([[0, 1, 0]])

        evaluation = model.evaluate([_piece("tail", [0, 1, 0])], max_context_length=2)

        expected_log_likelihood = math.log(26.0 / 45.0) + math.log(7.0 / 15.0) + math.log(26.0 / 45.0)
        self.assertEqual(evaluation["summary"]["n_tokens"], 3)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_log_likelihood)
        self.assertAlmostEqual(
            evaluation["summary"]["test_nll_per_token"],
            -expected_log_likelihood / 3.0,
        )

    def test_validation_selection_retains_the_lowest_nll_order(self) -> None:
        """Catches returning a non-selected candidate after validation scoring."""

        model = select_vomm_by_validation(
            train_sequences=[[0, 1, 0], [0, 1, 0]],
            validation_pieces=[_piece("validation", [0, 1, 0])],
            candidate_orders=(0, 1),
            vocabulary_size=3,
            max_context_length=3,
        )

        expected_validation_nll = -math.log((71.0 / 105.0) * (37.0 / 63.0) * (71.0 / 105.0)) / 3.0
        self.assertEqual(model.selected_order, 1)
        self.assertAlmostEqual(model.validation_nll_per_token, expected_validation_nll)


if __name__ == "__main__":
    unittest.main()
