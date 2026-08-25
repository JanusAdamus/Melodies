from __future__ import annotations

import unittest

from next_token_experiment.data import dataset as dataset_module
from next_token_experiment.data.dataset import WindowedSequenceDataset
from next_token_experiment.data.dataset import build_training_exposure_audit
from next_token_experiment.data.dataset import build_window_slices
from next_token_experiment.data.tokenizer import SequenceTokenizer
from next_token_experiment.experiment.splits import canonicalize_work_label
from next_token_experiment.experiment.splits import deterministic_group_split
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.protocol import estimate_transformer_parameter_count
from next_token_experiment.protocol import validate_experiment_scope
from next_token_experiment.schemas import PreparedPiece


class ProtocolTests(unittest.TestCase):
    def test_default_protocol_stays_within_scope(self) -> None:
        config = build_default_experiment_config()
        self.assertEqual(validate_experiment_scope(config), [])

    def test_parameter_count_stays_small(self) -> None:
        config = build_default_experiment_config()
        estimate = estimate_transformer_parameter_count(
            vocab_size=12,
            d_model=config.transformer.d_model,
            n_layers=config.transformer.n_layers,
            ff_dim=config.transformer.ff_dim,
            max_positions=config.windows.max_context_length,
            tie_embeddings=config.transformer.tie_input_output_embeddings,
        )
        self.assertLessEqual(estimate, 500_000)

    def test_window_builder_uses_tail_window_when_large_enough(self) -> None:
        windows = build_window_slices(
            sequence_length=150,
            max_context_length=128,
            stride=64,
            min_window_length=32,
        )
        self.assertEqual(windows, [(0, 128), (64, 150)])

    def test_evaluation_slices_keep_short_tail_exactly_once(self) -> None:
        self.assertTrue(
            hasattr(dataset_module, "build_evaluation_slices"),
            "build_evaluation_slices must define the shared evaluation protocol",
        )
        build_evaluation_slices = dataset_module.build_evaluation_slices
        self.assertEqual(build_evaluation_slices(150, 128), [(0, 128), (128, 150)])
        self.assertEqual(build_evaluation_slices(129, 128), [(0, 128), (128, 129)])
        with self.assertRaises(ValueError):
            build_evaluation_slices(1, 1)
        with self.assertRaises(ValueError):
            build_evaluation_slices(0, 1)

    def test_validation_dataset_scores_every_token_once(self) -> None:
        piece = PreparedPiece(
            piece_id="piece",
            source_path="/tmp/piece.musicxml",
            title="piece",
            composer="composer",
            canonical_work_id="piece",
            representation="pitch_class",
            vocabulary=[str(index) for index in range(12)],
            tokens=[0] * 129,
            n_events=129,
            metadata={},
        )
        tokenizer = SequenceTokenizer(
            representation="pitch_class",
            musical_vocabulary=tuple(piece.vocabulary),
        )

        dataset = WindowedSequenceDataset(
            pieces=[piece],
            tokenizer=tokenizer,
            split="validation",
            max_context_length=128,
            stride=64,
            min_window_length=32,
            max_windows=1,
        )

        self.assertEqual(
            [(example.start_index, example.stop_index) for example in dataset.examples],
            [(0, 128), (128, 129)],
        )
        self.assertEqual(sum(len(example.target_tokens) for example in dataset.examples), 129)

    def test_canonicalize_work_label_removes_arrangement_noise(self) -> None:
        label = canonicalize_work_label("Canon_in_D_easy_piano.mxl")
        self.assertEqual(label, "canon in d")

    def test_group_split_is_deterministic(self) -> None:
        group_ids = ["canon in d", "fur elise", "clair de lune", "moonlight"]
        first = deterministic_group_split(group_ids, 0.5, 0.25, 0.25, seed=7)
        second = deterministic_group_split(group_ids, 0.5, 0.25, 0.25, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()


class TrainingExposureAuditTests(unittest.TestCase):
    def _piece(self, piece_id: str, length: int) -> PreparedPiece:
        return PreparedPiece(
            piece_id=piece_id,
            source_path=f"/tmp/{piece_id}.musicxml",
            title=piece_id,
            composer="composer",
            canonical_work_id=piece_id,
            representation="pitch_class",
            vocabulary=[str(index) for index in range(12)],
            tokens=[index % 12 for index in range(length)],
            n_events=length,
            metadata={},
        )

    def _dataset(self, split: str, stride: int) -> WindowedSequenceDataset:
        piece = self._piece("piece", 320)
        tokenizer = SequenceTokenizer(
            representation="pitch_class",
            musical_vocabulary=tuple(piece.vocabulary),
        )
        return WindowedSequenceDataset(
            pieces=[piece],
            tokenizer=tokenizer,
            split=split,
            max_context_length=128,
            stride=stride,
            min_window_length=32,
        )

    def test_stride_64_exposes_targets_more_than_once(self) -> None:
        audit = build_training_exposure_audit(self._dataset("train", 64), stride=64)

        self.assertEqual(audit["split"], "train")
        self.assertEqual(audit["stride"], 64)
        self.assertEqual(audit["n_unique_targets"], 320)
        self.assertGreater(audit["n_target_exposures"], audit["n_unique_targets"])
        self.assertGreater(audit["max_exposure"], 1)
        self.assertEqual(audit["n_bos_restarts"], audit["n_windows"])
        self.assertEqual(
            sum(audit["exposure_multiplicity"].values()), audit["n_unique_targets"]
        )

    def test_stride_128_exposes_each_target_once(self) -> None:
        audit = build_training_exposure_audit(self._dataset("train", 128), stride=128)

        self.assertEqual(audit["n_target_exposures"], audit["n_unique_targets"])
        self.assertEqual(audit["max_exposure"], 1)
        self.assertEqual(audit["mean_exposure"], 1.0)

    def test_validation_and_test_never_overlap(self) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split):
                audit = build_training_exposure_audit(self._dataset(split, 64), stride=64)

                self.assertEqual(audit["max_exposure"], 1)
                self.assertEqual(audit["n_target_exposures"], audit["n_unique_targets"])
                self.assertTrue(audit["non_overlapping"])
