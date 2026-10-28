"""prepare_corpus sobrevive a la caída del pool de procesos.

Con la máquina ocupada el pool puede morir (BrokenProcessPool, u OSError como
«[WinError 6] Controlador no válido»). Las partituras sin resultado se preparan en
secuencia, con RuntimeWarning, y el resultado y el caché son los mismos que en serie.
Un error de la propia preparación no se traga.
"""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

from next_token_experiment.data import preprocess
from next_token_experiment.tests.test_pipeline import build_test_config, write_scale_score


def _dying_pool(error: BaseException, after: int):
    """Un ProcessPoolExecutor que entrega `after` resultados y luego lanza `error`."""

    class DyingPool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def map(self, function, iterable, chunksize=1):
            for index, item in enumerate(iterable):
                if index == after:
                    raise error
                yield function(item)

    return DyingPool


def _write_corpus(root: Path) -> None:
    root.mkdir(parents=True)
    write_scale_score(root / "a.musicxml", "Canon in D", "Johann Pachelbel", [60, 62, 64, 65] * 8)
    write_scale_score(root / "b.musicxml", "Fur Elise", "Beethoven", [69, 68, 69, 68, 69, 64, 67, 65] * 4)
    write_scale_score(root / "c.musicxml", "Waltz in A Minor", "Chopin", [57, 60, 64, 69] * 8)
    write_scale_score(root / "d.musicxml", "Mini Piece", "Anon", [60, 62, 64, 65] * 2)


def _cache_paths(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(row.get("source_path") or row.get("path") for row in rows)


class PreparePoolFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _write_corpus(cls.root / "corpus")
        cls.config = build_test_config(str(cls.root / "corpus"))
        cls.serial = preprocess.prepare_corpus(
            cls.config, n_workers=1, cache_path=cls.root / "serial.jsonl", progress_every=0
        )
        cls.serial_cache = _cache_paths(cls.root / "serial.jsonl")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _run_with(self, error: BaseException, after: int, cache_name: str):
        cache = self.root / cache_name
        with patch.object(preprocess, "ProcessPoolExecutor", _dying_pool(error, after)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = preprocess.prepare_corpus(self.config, n_workers=2, cache_path=cache, progress_every=0)
        return result, caught, _cache_paths(cache)

    def test_broken_pool_falls_back_to_serial_with_identical_result(self) -> None:
        for error, name in (
            (BrokenProcessPool("A process in the process pool was terminated abruptly"), "broken.jsonl"),
            (OSError(6, "Controlador no válido"), "oserror.jsonl"),
        ):
            with self.subTest(error=type(error).__name__):
                result, caught, cache = self._run_with(error, after=2, cache_name=name)
                self.assertEqual(result, self.serial)
                self.assertEqual(cache, self.serial_cache)
                runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
                self.assertEqual(len(runtime), 1)
                self.assertIn("2/4", str(runtime[0].message))

    def test_errors_of_the_preparation_itself_are_not_swallowed(self) -> None:
        with self.assertRaises(ValueError):
            self._run_with(ValueError("fallo de la preparación"), after=1, cache_name="error.jsonl")


if __name__ == "__main__":
    unittest.main()
