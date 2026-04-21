from __future__ import annotations

import unittest

import music21
import numpy as np

from src.data.observations import build_observation_sequence
from src.data.parsing import extract_events
from src.models.finite_hmm import FiniteChordHMM


def build_score() -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    for pitch_name in ["C4", "E4", "G4", "C5", "E5", "G5"]:
        part.append(music21.note.Note(pitch_name, quarterLength=1.0))
    score.insert(0, part)
    return score


class FiniteHMMTests(unittest.TestCase):
    def test_baseline_runs_and_returns_valid_distributions(self) -> None:
        events = extract_events(build_score())
        observations = build_observation_sequence(events, "pitch_class")
        result = FiniteChordHMM().fit_predict(observations)
        self.assertEqual(len(result.latent_states), observations.size)
        self.assertGreater(len(result.state_space), 24)
        self.assertLessEqual(len(result.active_states), len(result.state_space))
        self.assertTrue(np.allclose(result.transition_matrix.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(result.emission_matrix.sum(axis=1), 1.0))
        self.assertEqual(len(result.modal_labels), observations.size)
        self.assertTrue(any(":" in label for label in result.latent_labels))


if __name__ == "__main__":
    unittest.main()
