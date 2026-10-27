"""Pruebas de la tarea 1.4 (datos): D3, D4, D2 y compases reales."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import music21
import pandas as pd

from src.analysis.library_batch import prepare_corpus
from src.analysis.library_visualization import build_analysis_explanation, generate_analysis_figures
from src.data.dedup import content_fingerprint, deduplicate_by_fingerprint
from src.data.library_catalog import count_measures
from src.data.multicorpus import stratified_symbtr_sample
from src.data.observations import build_observation_sequence
from src.data.parsing import MusicalEvent

from tests.test_multicorpus import SIMPLE_MUSICXML


def _event(pitch_class: int | None) -> MusicalEvent:
    return MusicalEvent(0, 0.0, 1.0, "chord", "NC", (), (), pitch_class, None, 1, 1.0)


class NoneEventTests(unittest.TestCase):
    def test_pitch_class_without_pitch_raises_explicit_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "sin pitch representativo"):
            build_observation_sequence([_event(0), _event(None)], "pitch_class")


class MeasureCountTests(unittest.TestCase):
    def test_counts_real_measures_of_one_part_not_max_number(self) -> None:
        score = music21.stream.Score()
        for _ in range(2):
            part = music21.stream.Part()
            for number in (0, 1, 2):  # anacrusa numerada 0
                measure = music21.stream.Measure(number=number)
                measure.append(music21.note.Note("C4"))
                part.append(measure)
            score.insert(0, part)
        self.assertEqual(count_measures(score), 3)


class StratifiedSampleTests(unittest.TestCase):
    def _paths(self, sizes: dict[str, int]) -> list[Path]:
        return [Path(f"{makam}--sarki--duyek--t{i}--c.xml") for makam, n in sizes.items() for i in range(n)]

    def test_minimum_one_per_makam_then_proportional_remainder(self) -> None:
        sample = stratified_symbtr_sample(self._paths({"a": 6, "b": 3, "c": 1}), 5, seed=7)
        counts = Counter(path.name.split("--")[0] for path in sample)
        self.assertEqual(counts, Counter({"a": 2, "b": 2, "c": 1}))

    def test_without_room_for_every_makam_is_purely_proportional(self) -> None:
        sample = stratified_symbtr_sample(self._paths({"a": 8, "b": 4, "c": 1, "d": 1}), 3, seed=7)
        counts = Counter(path.name.split("--")[0] for path in sample)
        self.assertEqual(counts, Counter({"a": 2, "b": 1}))

    def test_is_deterministic_and_not_alphabetical(self) -> None:
        paths = self._paths({"a": 50, "b": 50})
        first = stratified_symbtr_sample(paths, 10, seed=7)
        self.assertEqual(first, stratified_symbtr_sample(paths, 10, seed=7))
        self.assertNotEqual(sorted(first), sorted(paths)[:5] + sorted(paths)[50:55])
        self.assertEqual(len(set(first)), 10)

    def test_limit_larger_than_corpus_returns_everything(self) -> None:
        paths = self._paths({"a": 2, "b": 1})
        self.assertEqual(sorted(stratified_symbtr_sample(paths, 10, seed=7)), sorted(paths))


class DedupTests(unittest.TestCase):
    def test_fingerprint_depends_only_on_pitch_class_sequence(self) -> None:
        self.assertEqual(content_fingerprint([0, 4, 7]), content_fingerprint((0, 4, 7)))
        self.assertNotEqual(content_fingerprint([0, 4, 7]), content_fingerprint([0, 7, 4]))

    def test_keeps_first_path_and_records_what_was_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "path": ["z/dup.mxl", "a/orig.mxl", "m/other.mxl"],
                "content_fingerprint": ["h1", "h1", "h2"],
            }
        )
        kept, removed = deduplicate_by_fingerprint(frame)
        self.assertEqual(sorted(kept["path"]), ["a/orig.mxl", "m/other.mxl"])
        self.assertEqual(removed[["path", "duplicate_of"]].values.tolist(), [["z/dup.mxl", "a/orig.mxl"]])


class PrepareCorpusTests(unittest.TestCase):
    def test_catalog_and_extraction_errors_are_counted_together_and_duplicates_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.musicxml").write_text(SIMPLE_MUSICXML, encoding="utf-8")
            (root / "b.musicxml").write_text(SIMPLE_MUSICXML, encoding="utf-8")
            (root / "broken.musicxml").write_text("<score-partwise>", encoding="utf-8")
            catalog = pd.DataFrame(
                {
                    "path": [str(root / name) for name in ("a.musicxml", "b.musicxml", "broken.musicxml", "missing.mxl")],
                    "error": ["", "", "", "FileNotFoundError: missing"],
                }
            )
            prepared = prepare_corpus(catalog, extraction="melody")
        self.assertEqual([Path(p).name for p in prepared.pieces["path"]], ["a.musicxml"])
        self.assertEqual([Path(p).name for p in prepared.duplicates["path"]], ["b.musicxml"])
        self.assertEqual(sorted(prepared.failures["error_stage"]), ["catalogo", "extraccion"])
        self.assertIn(str(root / "a.musicxml"), prepared.events)

    def test_report_counts_errors_of_every_stage(self) -> None:
        analysis = pd.DataFrame(
            {
                "title": ["ok", "x", "y", "z"],
                "finite_log_likelihood": [-1.0, None, None, None],
                "hdp_log_likelihood": [-0.5, None, None, None],
                "error": ["", "Cat", "Ext", "Fit"],
                "error_stage": ["", "catalogo", "extraccion", "analisis"],
            }
        )
        text = build_analysis_explanation(analysis)
        self.assertIn("Obras con error: 3 (catalogo: 1, extraccion: 1, analisis: 1)", text)
        with tempfile.TemporaryDirectory() as tmp:
            generate_analysis_figures(analysis, tmp)  # las filas con error no rompen las figuras


if __name__ == "__main__":
    unittest.main()
