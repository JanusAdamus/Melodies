"""Cachés escritas antes del 2026-10-26 traen la obra canónica truncada (auditoría C3)."""

from __future__ import annotations

import unittest

from next_token_experiment.data.preprocess import _decode_cache_entry


class CacheCanonicalIdTests(unittest.TestCase):
    def test_decoding_an_old_cache_recomputes_the_canonical_id(self) -> None:
        payload = {
            "kind": "piece",
            "piece_id": "0\\0\\Qm1.mxl",
            "source_path": "C:\\PDMX\\0\\0\\Qm1.mxl",
            "title": "Scatter the Mud",
            "composer": "anon.",
            "canonical_work_id": "anon",
            "representation": "pitch_class",
            "tokens": [0, 2, 4],
            "n_events": 3,
            "metadata": {},
        }

        piece = _decode_cache_entry(payload, ["0", "2", "4"], "C:\\PDMX")

        self.assertEqual(piece.canonical_work_id, "anon scatter the mud")


if __name__ == "__main__":
    unittest.main()
