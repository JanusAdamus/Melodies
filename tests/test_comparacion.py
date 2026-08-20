from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Comparacion.classical_models import FiniteGlobalHMM, FiniteHMMFitResult, GlobalHDPHMM
from Comparacion.splits import build_fixed_splits, build_nested_training_subsets
from next_token_experiment.schemas import PreparedPiece
from src.models.inference import forward_log_likelihood


def _piece(piece_id: str, token: int, length: int, source: str) -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=f"/tmp/{source}/{piece_id}.musicxml",
        title=piece_id,
        composer="composer",
        canonical_work_id=piece_id,
        representation="pitch_class",
        vocabulary=[str(index) for index in range(12)],
        tokens=[token for _ in range(length)],
        n_events=length,
        metadata={},
    )


class ComparisonSplitTests(unittest.TestCase):
    def test_fixed_splits_and_nested_subsets_are_deterministic(self) -> None:
        pieces = []
        for index in range(6):
            pieces.append(_piece(f"mt_short_{index}", index % 12, 40, "MuseTrainer"))
            pieces.append(_piece(f"symbtr_long_{index}", (index + 1) % 12, 140, "SymbTr"))

        splits = build_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
        self.assertTrue(splits.test_pieces)
        self.assertTrue(splits.validation_pieces)
        self.assertTrue(splits.train_pool_pieces)

        nested = build_nested_training_subsets(splits.train_pool_pieces, fractions=(0.25, 0.50, 1.0), data_seed=3)
        subset_ids = [set(piece.piece_id for piece in subset) for _, subset in nested]
        self.assertTrue(subset_ids[0].issubset(subset_ids[1]))
        self.assertTrue(subset_ids[1].issubset(subset_ids[2]))


class ClassicalModelTests(unittest.TestCase):
    def test_classical_score_conditions_on_bos_without_counting_it(self) -> None:
        train_pieces = [_piece("train", 0, 4, "MuseTrainer")]
        validation_pieces = [_piece("validation", 0, 2, "MuseTrainer")]
        test_pieces = [_piece("test", 0, 2, "SymbTr")]
        model = FiniteGlobalHMM(candidate_num_states=(1,), max_iterations=2, tolerance=1e-4, seed=1)

        try:
            model.fit(train_pieces, validation_pieces, bos_token_id=12, max_context_length=128)
            evaluation = model.evaluate(test_pieces, bos_token_id=12, max_context_length=128)
        except TypeError as error:
            self.fail(str(error))

        expected_score = 2 * math.log(float(model.emission_matrix[0, 0]) + 1e-12)
        self.assertEqual(evaluation["summary"]["n_tokens"], 2)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_score)

    def test_classical_score_resets_bos_at_evaluation_slice_boundaries(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(2,), max_iterations=1, tolerance=1e-4, seed=1)
        model.initial_probs = np.array([0.6, 0.4])
        model.transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.emission_matrix = np.array(
            [
                [0.4, 0.01, *([0.019] * 10), 0.4],
                [0.01, 0.6, *([0.029] * 10), 0.1],
            ]
        )
        model.selected_states = 2
        model.fit_result = FiniteHMMFitResult(
            selected_states=2,
            validation_nll=0.0,
            train_log=[],
            train_wall_clock_s=0.0,
        )
        piece = _piece("test", 0, 3, "SymbTr")
        piece.tokens[1:] = [1, 1]

        def joint_log_likelihood(tokens: list[int]) -> float:
            value, _ = forward_log_likelihood(
                initial_probs=model.initial_probs,
                transition_matrix=model.transition_matrix,
                emission_matrix=model.emission_matrix,
                observations=np.array(tokens),
            )
            return value

        context_score = joint_log_likelihood([12])
        expected_score = (
            joint_log_likelihood([12, 0, 1])
            - context_score
            + joint_log_likelihood([12, 1])
            - context_score
        )
        evaluation = model.evaluate([piece], bos_token_id=12, max_context_length=2)

        self.assertEqual(evaluation["summary"]["n_tokens"], 3)
        self.assertAlmostEqual(evaluation["piece_metrics"][0]["log_likelihood"], expected_score)

    def test_finite_hmm_fits_and_scores_synthetic_sequences(self) -> None:
        train_pieces = [_piece("a", 0, 12, "MuseTrainer"), _piece("b", 0, 12, "SymbTr")]
        validation_pieces = [_piece("c", 0, 10, "MuseTrainer")]
        test_pieces = [_piece("d", 0, 10, "SymbTr")]

        model = FiniteGlobalHMM(candidate_num_states=(2, 3), max_iterations=10, tolerance=1e-4, seed=1)
        fit_result = model.fit(train_pieces, validation_pieces, bos_token_id=12)
        evaluation = model.evaluate(test_pieces, bos_token_id=12)

        self.assertIn(fit_result.selected_states, {2, 3})
        self.assertLess(evaluation["summary"]["test_perplexity"], 2.0)
        self.assertEqual(len(evaluation["piece_metrics"]), 1)

    def test_hdp_hmm_fits_smoke_sequence(self) -> None:
        train_pieces = [_piece("a", 4, 14, "MuseTrainer"), _piece("b", 4, 14, "SymbTr")]
        validation_pieces = [_piece("c", 4, 10, "MuseTrainer")]
        test_pieces = [_piece("d", 4, 10, "SymbTr")]

        model = GlobalHDPHMM(
            truncation_level=4,
            n_iters=8,
            burn_in=4,
            hyperparameter_grid=((8.0, 4.0, 2.0),),
            seed=2,
        )
        fit_result = model.fit(train_pieces, validation_pieces, bos_token_id=12)
        evaluation = model.evaluate(test_pieces, bos_token_id=12)

        self.assertIn("validation_nll_per_token", fit_result)
        self.assertLess(evaluation["summary"]["test_perplexity"], 3.0)

    def test_hdp_hmm_retains_result_for_selected_hyperparameters(self) -> None:
        class DeterministicHDPHMM:
            def __init__(self, *, alpha: float, **kwargs) -> None:
                self.alpha = alpha

            def fit_sequences(self, observations):
                token_probability = 0.8 if self.alpha == 1.0 else 0.2
                emission = np.full((1, 13), (1.0 - token_probability) / 12.0)
                emission[0, 0] = token_probability
                return SimpleNamespace(
                    posterior_initial_mean=np.array([1.0]),
                    posterior_transition_mean=np.array([[1.0]]),
                    posterior_emission_mean=emission,
                    log_likelihood=-self.alpha,
                    effective_states=1,
                )

        model = GlobalHDPHMM(
            truncation_level=1,
            n_iters=1,
            burn_in=0,
            hyperparameter_grid=((1.0, 4.0, 2.0), (2.0, 4.0, 2.0)),
            seed=2,
        )
        with patch("Comparacion.classical_models.TruncatedHDPHMM", DeterministicHDPHMM):
            fit_result = model.fit(
                [_piece("train", 0, 2, "MuseTrainer")],
                [_piece("validation", 0, 2, "SymbTr")],
                bos_token_id=12,
                max_context_length=128,
            )

        self.assertEqual(fit_result["selected_hyperparameters"]["alpha"], 1.0)
        self.assertAlmostEqual(model.best_result.posterior_emission_mean[0, 0], 0.8)


if __name__ == "__main__":
    unittest.main()
