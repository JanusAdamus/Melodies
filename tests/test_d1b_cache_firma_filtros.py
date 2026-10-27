"""Hallazgo D1b (auditoria): la firma del cache no incluia los filtros del corpus.

Un cache hecho con ``min_events_per_piece=1`` y reutilizado con ``min_events_per_piece=32``
devolvia la pieza corta como pieza, donde una corrida limpia la excluye como
``too_short``, sin ``CorpusCacheModeError``. La firma debe cubrir todo lo que cambia el
contenido de una fila: extraccion, representacion, umbral de eventos, version del
extractor; y el ``piece_id`` debe ser el de la raiz de la corrida.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from next_token_experiment.data.preprocess import (
    CorpusCacheModeError,
    prepare_corpus,
)
from tests.test_d1b_extraccion_pdmx import _config, _write_two_part_score


def _with_min_events(config, min_events: int):
    return replace(config, corpus=replace(config.corpus, min_events_per_piece=min_events))


def _rows(cache: Path) -> list[dict]:
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]


def _rewrite(cache: Path, rows: list[dict]) -> None:
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MinEventsSignatureTests(unittest.TestCase):
    def _corpus(self, tmp: str) -> tuple[Path, Path]:
        root = Path(tmp) / "scores"
        root.mkdir()
        # En melodia son 2 eventos: pasa con umbral 1 y es too_short con umbral 32.
        _write_two_part_score(root / "a.musicxml")
        return root, Path(tmp) / "cache.jsonl"

    def test_cache_with_another_threshold_is_refused_and_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = self._corpus(tmp)
            loose = prepare_corpus(_with_min_events(_config(root), 1), n_workers=1,
                                   cache_path=cache, progress_every=0)
            self.assertEqual(len(loose.pieces), 1)
            before = cache.read_text(encoding="utf-8")

            strict = _with_min_events(_config(root), 32)
            clean = prepare_corpus(strict, n_workers=1, progress_every=0)
            self.assertEqual([item.reason for item in clean.exclusions], ["too_short"])
            self.assertEqual(clean.pieces, [])

            with self.assertRaises(CorpusCacheModeError):
                prepare_corpus(strict, n_workers=1, cache_path=cache, progress_every=0)
            self.assertEqual(cache.read_text(encoding="utf-8"), before)

    def test_cache_rows_record_the_threshold_and_extractor_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = self._corpus(tmp)
            prepare_corpus(_with_min_events(_config(root), 1), n_workers=1,
                           cache_path=cache, progress_every=0)
            row = _rows(cache)[0]
            self.assertEqual(row["min_events_per_piece"], 1)
            self.assertIn("extractor_version", row)

    def test_cache_without_threshold_is_august_32(self) -> None:
        """Todas las filas de los caches de artifacts/ dicen min_events_per_piece=32."""

        with tempfile.TemporaryDirectory() as tmp:
            root, cache = self._corpus(tmp)
            prepare_corpus(_with_min_events(_config(root), 32), n_workers=1,
                           cache_path=cache, progress_every=0)
            rows = _rows(cache)
            for row in rows:
                row.pop("min_events_per_piece")
                row.pop("extractor_version")
            _rewrite(cache, rows)

            with self.assertRaises(CorpusCacheModeError):
                prepare_corpus(_with_min_events(_config(root), 1), n_workers=1,
                               cache_path=cache, progress_every=0)
            reused = prepare_corpus(_with_min_events(_config(root), 32), n_workers=1,
                                    cache_path=cache, progress_every=0)
            self.assertEqual([item.reason for item in reused.exclusions], ["too_short"])

    def test_cache_of_another_extractor_version_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = self._corpus(tmp)
            config = _with_min_events(_config(root), 1)
            prepare_corpus(config, n_workers=1, cache_path=cache, progress_every=0)
            rows = _rows(cache)
            for row in rows:
                row["extractor_version"] = row["extractor_version"] - 1
            _rewrite(cache, rows)

            with self.assertRaises(CorpusCacheModeError):
                prepare_corpus(config, n_workers=1, cache_path=cache, progress_every=0)


class PieceIdFromRunRootTests(unittest.TestCase):
    def test_cached_piece_id_matches_a_clean_run_with_another_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "scores"
            inner = outer / "sub"
            inner.mkdir(parents=True)
            _write_two_part_score(inner / "a.musicxml")
            cache = Path(tmp) / "cache.jsonl"
            prepare_corpus(_config(outer), n_workers=1, cache_path=cache, progress_every=0)

            clean = prepare_corpus(_config(inner), n_workers=1, progress_every=0)
            reused = prepare_corpus(_config(inner), n_workers=1, cache_path=cache,
                                    progress_every=0)
            self.assertEqual(
                [piece.piece_id for piece in reused.pieces],
                [piece.piece_id for piece in clean.pieces],
            )
            self.assertEqual(
                reused.pieces[0].metadata["relative_path"],
                clean.pieces[0].metadata["relative_path"],
            )


class RunnerCacheSignatureTests(unittest.TestCase):
    def _config(self, results_root: str, min_events: int):
        from Comparacion.config import build_default_learning_curve_config

        config = build_default_learning_curve_config(
            corpus_root="synthetic-corpus", results_root=results_root
        )
        return replace(config, experiment=_with_min_events(config.experiment, min_events))

    def test_runner_refuses_a_cache_with_another_threshold_before_writing(self) -> None:
        from Comparacion.runner import run_learning_curve_experiment

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, 32)
            cache = Path(tmp) / "corpus_cache_3000_melody.jsonl"
            cache.write_text(
                json.dumps({"kind": "exclusion", "piece_id": "x", "path": "x",
                            "reason": "too_short", "detail": "",
                            "extraction_mode": "melody", "min_events_per_piece": 1,
                            "representation": "pitch_class"}) + "\n",
                encoding="utf-8",
            )
            with patch("Comparacion.runner.prepare_corpus") as prepare:
                with self.assertRaises(CorpusCacheModeError):
                    run_learning_curve_experiment(
                        config, run_name="corrida", corpus_cache_path=cache
                    )
            prepare.assert_not_called()
            self.assertFalse((Path(tmp) / "corrida").exists())

    def test_resume_with_another_threshold_is_refused(self) -> None:
        from Comparacion.runner import (
            ResumeExtractionMismatchError,
            run_learning_curve_experiment,
        )

        with tempfile.TemporaryDirectory() as tmp:
            previous = self._config(tmp, 32)
            run = Path(tmp) / "corrida"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(asdict(previous)), encoding="utf-8")
            with patch("Comparacion.runner.prepare_corpus") as prepare:
                with self.assertRaises(ResumeExtractionMismatchError):
                    run_learning_curve_experiment(
                        self._config(tmp, 1), run_name="corrida", resume=True
                    )
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
