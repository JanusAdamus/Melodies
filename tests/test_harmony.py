from __future__ import annotations

import unittest

from src.models.harmony import (
    DIATONIC_MODES,
    build_harmonic_state_space,
    infer_chord_candidates_from_pitch_classes,
    infer_mode_candidates,
)


class HarmonyVocabularyTests(unittest.TestCase):
    def test_extended_vocabulary_contains_expected_labels(self) -> None:
        labels = {state.label for state in build_harmonic_state_space()}
        self.assertIn("C:maj7", labels)
        self.assertIn("G:9", labels)
        self.assertIn("D:sus4", labels)
        self.assertIn("A:m9", labels)
        self.assertLessEqual(len(labels), 12 * 24)

    def test_mode_inventory_contains_all_diatonic_modes(self) -> None:
        self.assertEqual(set(DIATONIC_MODES.keys()), {"ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"})

    def test_chord_candidate_matching_prefers_extended_chord(self) -> None:
        candidates = infer_chord_candidates_from_pitch_classes([0, 4, 7, 11, 2], top_n=3)
        self.assertEqual(candidates[0][0], "C:maj9")

    def test_mode_candidate_matching_detects_dorian(self) -> None:
        candidates = infer_mode_candidates([2, 4, 5, 7, 9, 11, 0], tonic=2, top_n=3)
        self.assertEqual(candidates[0][0], "D:dorian")


if __name__ == "__main__":
    unittest.main()
