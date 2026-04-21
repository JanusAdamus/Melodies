from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import music21

from next_token_experiment.config import (
    CorpusConfig,
    ExperimentConfig,
    FiniteHMMConfig,
    HDPHMMConfig,
    HardwareConfig,
    MetricsConfig,
    PreprocessingConfig,
    RepresentationConfig,
    SplitConfig,
    StorageConfig,
    TaskConfig,
    TransformerConfig,
    WindowConfig,
)
from next_token_experiment.data.dataset import build_dataset_bundle
from next_token_experiment.data.preprocess import prepare_corpus
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.data.validation import validate_dataset_bundle, validate_prepared_pieces
from next_token_experiment.experiment.splits import assign_piece_splits


def write_scale_score(path: Path, title: str, composer: str, midi_values: list[int]) -> None:
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=title, composer=composer)
    part = music21.stream.Part()
    for value in midi_values:
        part.append(music21.note.Note(midi=value, quarterLength=1.0))
    score.insert(0, part)
    score.write("musicxml", fp=str(path))


def build_test_config(root_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(),
        corpus=CorpusConfig(
            name="tmp_corpus",
            root_dir=root_dir,
            min_events_per_piece=16,
        ),
        preprocessing=PreprocessingConfig(),
        representation=RepresentationConfig(primary="pitch_class", alternative=None),
        windows=WindowConfig(max_context_length=16, min_window_length=8, train_stride=8, eval_stride=16),
        split=SplitConfig(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, seed=7),
        metrics=MetricsConfig(),
        hardware=HardwareConfig(cpu_threads=1),
        finite_hmm=FiniteHMMConfig(),
        hdp_hmm=HDPHMMConfig(),
        transformer=TransformerConfig(batch_size=4, max_epochs=2, early_stopping_patience=2),
        storage=StorageConfig(results_root=str(Path(root_dir) / "results")),
    )


class PipelineTests(unittest.TestCase):
    def test_prepare_corpus_and_split_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_scale_score(root / "canon_standard.musicxml", "Canon in D", "Johann Pachelbel", [60, 62, 64, 65] * 8)
            write_scale_score(root / "canon_easy.musicxml", "Canon in D", "Johann Pachelbel", [60, 60, 62, 64] * 8)
            write_scale_score(root / "fur_elise.musicxml", "Fur Elise", "Beethoven", [69, 68, 69, 68, 69, 64, 67, 65] * 4)
            write_scale_score(root / "clair_de_lune.musicxml", "Clair de Lune", "Debussy", [60, 63, 67, 70] * 8)
            write_scale_score(root / "waltz.musicxml", "Waltz in A Minor", "Chopin", [57, 60, 64, 69] * 8)
            write_scale_score(root / "too_short.musicxml", "Mini Piece", "Anon", [60, 62, 64, 65] * 2)

            config = build_test_config(str(root))
            preparation = prepare_corpus(config)
            self.assertEqual(len(preparation.pieces), 5)
            self.assertEqual(len(preparation.exclusions), 1)
            self.assertEqual(preparation.exclusions[0].reason, "too_short")
            self.assertEqual(validate_prepared_pieces(preparation.pieces), [])

            tokenizer = build_tokenizer(config.representation)
            split_assignments = assign_piece_splits(preparation.pieces, config.split)
            bundle = build_dataset_bundle(
                prepared_pieces=preparation.pieces,
                exclusions=preparation.exclusions,
                split_assignments=split_assignments,
                tokenizer=tokenizer,
                config=config,
            )
            self.assertEqual(validate_dataset_bundle(bundle), [])
            self.assertGreater(bundle.train_dataset_size, 0)
            self.assertGreater(bundle.validation_dataset_size, 0)
            self.assertGreater(bundle.test_dataset_size, 0)

            canon_records = [record for record in bundle.manifest if record.title == "Canon in D"]
            self.assertEqual(len(canon_records), 2)
            self.assertEqual(len({record.split for record in canon_records}), 1)


if __name__ == "__main__":
    unittest.main()
