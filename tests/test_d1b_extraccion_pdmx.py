"""Hallazgo D1b: el preprocesamiento de PDMX usaba la extraccion antigua.

`next_token_experiment/data/preprocess.py` debe extraer con
`extract_events_for_mode`, con `melody` por omision y `legacy_all_parts` solo para
reproducir agosto. El cache guarda el modo y nadie usa un cache de otro modo.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import music21

from Comparacion.cli import build_parser
from next_token_experiment.config import PreprocessingConfig
from next_token_experiment.data.preprocess import (
    CorpusCacheModeError,
    default_corpus_cache_name,
    prepare_corpus,
    preprocess_score_file,
)
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.schemas import PreparedPiece
from src.data.parsing import extract_events, extract_events_for_mode, parse_score


def _write_two_part_score(path: Path) -> None:
    """Sol: acorde Do-Mi-Sol y luego Re5; fa: Do3 y Sol2 en los mismos ataques."""

    treble = music21.stream.Part()
    treble_measure = music21.stream.Measure(number=1)
    treble_measure.append(music21.clef.TrebleClef())
    treble_measure.append(music21.meter.TimeSignature("2/4"))
    treble_measure.append(music21.chord.Chord(["C4", "E4", "G4"], quarterLength=1.0))
    treble_measure.append(music21.note.Note("D5", quarterLength=1.0))
    treble.append(treble_measure)

    bass = music21.stream.Part()
    bass_measure = music21.stream.Measure(number=1)
    bass_measure.append(music21.clef.BassClef())
    bass_measure.append(music21.meter.TimeSignature("2/4"))
    bass_measure.append(music21.note.Note("C3", quarterLength=1.0))
    bass_measure.append(music21.note.Note("G2", quarterLength=1.0))
    bass.append(bass_measure)

    score = music21.stream.Score()
    score.insert(0, treble)
    score.insert(0, bass)
    score.write("musicxml", fp=str(path))


def _config(root: Path, **preprocessing):
    config = build_default_experiment_config()
    return replace(
        config,
        corpus=replace(config.corpus, root_dir=str(root), min_events_per_piece=1),
        preprocessing=replace(config.preprocessing, **preprocessing),
    )


class ExtractionModeTests(unittest.TestCase):
    def test_default_extraction_mode_is_melody(self) -> None:
        self.assertEqual(PreprocessingConfig().extraction_mode, "melody")
        self.assertEqual(
            build_default_experiment_config().preprocessing.extraction_mode, "melody"
        )

    def test_preprocess_uses_the_top_voice_of_the_treble_part_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dos_partes.musicxml"
            _write_two_part_score(path)
            config = _config(root)

            prepared = preprocess_score_file(
                path, str(root), config.preprocessing, config.representation, 1
            )

            self.assertIsInstance(prepared, PreparedPiece)
            # Nota superior del acorde (Sol = 7), luego Re (2); el bajo no entra.
            self.assertEqual(prepared.tokens, [7, 2])
            self.assertEqual(prepared.metadata["extraction_mode"], "melody")

    def test_legacy_mode_reproduces_the_august_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dos_partes.musicxml"
            _write_two_part_score(path)
            config = _config(root, extraction_mode="legacy_all_parts")

            prepared = preprocess_score_file(
                path, str(root), config.preprocessing, config.representation, 1
            )

            august = extract_events(
                parse_score(path),
                include_rests=config.preprocessing.include_rests,
                prefer_treble=config.preprocessing.prefer_treble,
            )
            self.assertEqual(
                prepared.tokens,
                [event.representative_pitch_class for event in august],
            )
            self.assertNotEqual(prepared.tokens, [7, 2])
            self.assertEqual(prepared.metadata["extraction_mode"], "legacy_all_parts")

    def test_melody_mode_refuses_rests_instead_of_ignoring_them(self) -> None:
        score = music21.stream.Score()
        with self.assertRaises(ValueError):
            extract_events_for_mode(score, "melody", include_rests=True)

    def test_profile_with_rests_keeps_the_extraction_that_has_rests(self) -> None:
        from next_token_experiment.profiles import build_profile_config

        config = build_profile_config("research_richer_events")
        self.assertTrue(config.preprocessing.include_rests)
        self.assertEqual(config.preprocessing.extraction_mode, "legacy_all_parts")

    def test_unknown_mode_fails_instead_of_excluding_every_piece(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dos_partes.musicxml"
            _write_two_part_score(path)
            config = _config(root, extraction_mode="otra")
            with self.assertRaises(ValueError):
                preprocess_score_file(
                    path, str(root), config.preprocessing, config.representation, 1
                )


class CorpusCacheModeTests(unittest.TestCase):
    def test_default_cache_name_carries_the_mode(self) -> None:
        self.assertEqual(
            default_corpus_cache_name(3000, "melody"), "corpus_cache_3000_melody.jsonl"
        )
        # El cache de agosto conserva su nombre para poder reproducirlo.
        self.assertEqual(
            default_corpus_cache_name(3000, "legacy_all_parts"), "corpus_cache_3000.jsonl"
        )

    def test_cache_entries_record_the_extraction_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scores"
            root.mkdir()
            _write_two_part_score(root / "a.musicxml")
            cache = Path(tmp) / "cache.jsonl"

            prepare_corpus(_config(root), n_workers=1, cache_path=cache, progress_every=0)

            rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["extraction_mode"], "melody")

    def test_cache_from_another_mode_is_refused_and_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scores"
            root.mkdir()
            _write_two_part_score(root / "a.musicxml")
            cache = Path(tmp) / "cache.jsonl"
            prepare_corpus(
                _config(root, extraction_mode="legacy_all_parts"),
                n_workers=1,
                cache_path=cache,
                progress_every=0,
            )
            before = cache.read_text(encoding="utf-8")

            with self.assertRaises(CorpusCacheModeError):
                prepare_corpus(_config(root), n_workers=1, cache_path=cache, progress_every=0)
            self.assertEqual(cache.read_text(encoding="utf-8"), before)

    def test_cache_without_mode_is_the_legacy_extraction(self) -> None:
        """Los caches anteriores a D1b no guardan modo; todos salieron de la extraccion antigua."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scores"
            root.mkdir()
            _write_two_part_score(root / "a.musicxml")
            cache = Path(tmp) / "cache.jsonl"
            prepare_corpus(
                _config(root, extraction_mode="legacy_all_parts"),
                n_workers=1,
                cache_path=cache,
                progress_every=0,
            )
            rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row.pop("extraction_mode")
            cache.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaises(CorpusCacheModeError):
                prepare_corpus(_config(root), n_workers=1, cache_path=cache, progress_every=0)
            legacy = prepare_corpus(
                _config(root, extraction_mode="legacy_all_parts"),
                n_workers=1,
                cache_path=cache,
                progress_every=0,
            )
            self.assertEqual(len(legacy.pieces), 1)


class RunnerExtractionTests(unittest.TestCase):
    def test_cli_extraction_defaults_to_melody(self) -> None:
        self.assertEqual(build_parser().parse_args([]).extraction, "melody")
        self.assertEqual(
            build_parser().parse_args(["--extraction", "legacy_all_parts"]).extraction,
            "legacy_all_parts",
        )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--extraction", "otra"])

    def test_runner_refuses_a_cache_of_another_mode_before_writing(self) -> None:
        from Comparacion.config import build_default_learning_curve_config
        from Comparacion.runner import run_learning_curve_experiment

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "corpus_cache_3000.jsonl"
            cache.write_text(
                json.dumps({"kind": "exclusion", "piece_id": "x", "path": "x",
                            "reason": "too_short", "detail": "",
                            "extraction_mode": "legacy_all_parts"}) + "\n",
                encoding="utf-8",
            )
            config = build_default_learning_curve_config(
                corpus_root="synthetic-corpus", results_root=tmp
            )
            with patch("Comparacion.runner.prepare_corpus") as prepare:
                with self.assertRaises(CorpusCacheModeError):
                    run_learning_curve_experiment(
                        config, run_name="corrida", corpus_cache_path=cache
                    )
            prepare.assert_not_called()
            self.assertFalse((Path(tmp) / "corrida").exists())


if __name__ == "__main__":
    unittest.main()
