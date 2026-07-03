from __future__ import annotations

import unittest

from Comparacion.classical_models import FiniteGlobalHMM, GlobalHDPHMM
from Comparacion.splits import build_fixed_splits, build_nested_training_subsets
from next_token_experiment.schemas import PreparedPiece


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


if __name__ == "__main__":
    unittest.main()
