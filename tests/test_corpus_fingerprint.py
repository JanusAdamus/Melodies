from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.corpus_fingerprint import fingerprint_corpus_cache


def _write_cache(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


class CorpusFingerprintTests(unittest.TestCase):
    def test_ignores_local_paths_and_jsonl_order(self) -> None:
        piece = {
            "kind": "piece",
            "piece_id": "scores/example.mxl",
            "source_path": r"C:\\datasets\\PDMX\\scores\\example.mxl",
            "title": "Example",
            "composer": "Composer",
            "canonical_work_id": "composer-example",
            "representation": "pitch_class",
            "tokens": [0, 4, 7],
            "n_events": 3,
            "metadata": {"relative_path": "scores/example.mxl", "part_count": 1},
        }
        exclusion = {
            "kind": "exclusion",
            "piece_id": "scores/broken.mxl",
            "path": r"C:\\datasets\\PDMX\\scores\\broken.mxl",
            "reason": "parse_error",
            "detail": r"Failure while reading C:\\datasets\\PDMX\\scores\\broken.mxl",
        }
        moved_piece = {**piece, "source_path": "/mnt/pdmx/scores/example.mxl"}
        moved_exclusion = {
            **exclusion,
            "path": "/mnt/pdmx/scores/broken.mxl",
            "detail": "Failure while reading /mnt/pdmx/scores/broken.mxl",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            _write_cache(first, [piece, exclusion])
            _write_cache(second, [moved_exclusion, moved_piece])

            first_fingerprint = fingerprint_corpus_cache(first)
            second_fingerprint = fingerprint_corpus_cache(second)

        self.assertEqual(first_fingerprint["sha256"], second_fingerprint["sha256"])
        self.assertEqual(first_fingerprint["counts"], second_fingerprint["counts"])
        self.assertEqual(first_fingerprint["counts"]["entries"], 2)
        self.assertEqual(first_fingerprint["counts"]["events"], 3)

    def test_changes_when_tokens_or_exclusion_reason_change(self) -> None:
        entries = [
            {
                "kind": "piece",
                "piece_id": "piece",
                "source_path": "A:/piece.mxl",
                "title": "Piece",
                "composer": "Composer",
                "canonical_work_id": "piece",
                "representation": "pitch_class",
                "tokens": [1, 2, 3],
                "n_events": 3,
                "metadata": {},
            },
            {
                "kind": "exclusion",
                "piece_id": "excluded",
                "path": "A:/excluded.mxl",
                "reason": "parse_error",
                "detail": "local detail",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = root / "original.jsonl"
            token_change = root / "token-change.jsonl"
            reason_change = root / "reason-change.jsonl"
            _write_cache(original, entries)
            _write_cache(
                token_change,
                [{**entries[0], "tokens": [1, 2, 4]}, entries[1]],
            )
            _write_cache(
                reason_change,
                [entries[0], {**entries[1], "reason": "event_extraction_error"}],
            )

            hashes = {
                fingerprint_corpus_cache(path)["sha256"]
                for path in (original, token_change, reason_change)
            }

        self.assertEqual(len(hashes), 3)

    def test_rejects_duplicate_piece_ids(self) -> None:
        duplicate = {
            "kind": "exclusion",
            "piece_id": "duplicate",
            "path": "C:/duplicate.mxl",
            "reason": "parse_error",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache.jsonl"
            _write_cache(cache, [duplicate, duplicate])

            with self.assertRaisesRegex(ValueError, "duplicate piece_id"):
                fingerprint_corpus_cache(cache)

    def test_reports_invalid_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache.jsonl"
            valid = {
                "kind": "exclusion",
                "piece_id": "valid",
                "path": "C:/valid.mxl",
                "reason": "parse_error",
            }
            cache.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, ":2"):
                fingerprint_corpus_cache(cache)


if __name__ == "__main__":
    unittest.main()
