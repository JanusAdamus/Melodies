from __future__ import annotations

import unittest

from src.data.library_catalog import infer_arrangement_tag, infer_composer, infer_difficulty_bucket, infer_form


class LibraryCatalogTests(unittest.TestCase):
    def test_infer_composer_from_filename(self) -> None:
        self.assertEqual(infer_composer("Chopin_-_Nocturne_Op._9_No._1.mxl"), "Chopin")
        self.assertEqual(infer_composer("Bach_Minuet_in_G_Major_BWV_Anh._114.mxl"), "Bach")

    def test_infer_form_from_title(self) -> None:
        self.assertEqual(infer_form("Nocturne No. 20 in C Minor"), "nocturne")
        self.assertEqual(infer_form("Piano Sonata No. 11"), "sonata")

    def test_infer_arrangement_and_difficulty(self) -> None:
        self.assertEqual(infer_arrangement_tag("Fur Elise Easy Piano"), "pedagogical_easy")
        self.assertEqual(infer_arrangement_tag("WA Mozart Marche Turque fingered"), "annotated_fingering")
        self.assertEqual(infer_difficulty_bucket("Fur Elise Easy Piano", 80, 2.0), "easy")
        self.assertEqual(infer_difficulty_bucket("Ballade no. 1", 1200, 8.4), "advanced")


if __name__ == "__main__":
    unittest.main()
