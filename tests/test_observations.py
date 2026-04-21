from __future__ import annotations

import unittest

import music21

from src.data.observations import build_observation_sequence
from src.data.parsing import extract_events


def build_score() -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    part.append(music21.note.Note("C4", quarterLength=1.0))
    part.append(music21.note.Note("D4", quarterLength=0.5))
    part.append(music21.note.Note("E4", quarterLength=2.0))
    score.insert(0, part)
    return score


class ObservationTests(unittest.TestCase):
    def test_pitch_class_observations(self) -> None:
        events = extract_events(build_score())
        sequence = build_observation_sequence(events, "pitch_class")
        self.assertEqual(sequence.vocab_size, 12)
        self.assertListEqual(sequence.tokens.tolist(), [0, 2, 4])

    def test_interval_observations(self) -> None:
        events = extract_events(build_score())
        sequence = build_observation_sequence(events, "interval", max_interval=12)
        self.assertEqual(sequence.decoded[0], "START")
        self.assertEqual(sequence.decoded[1], "INT_+2")
        self.assertEqual(sequence.decoded[2], "INT_+2")

    def test_pitch_class_duration_observations(self) -> None:
        events = extract_events(build_score())
        sequence = build_observation_sequence(events, "pitch_class_duration", duration_bins=(0.5, 1.0, 2.0))
        self.assertEqual(sequence.size, 3)
        self.assertTrue(all("|dur=" in value for value in sequence.vocabulary))


if __name__ == "__main__":
    unittest.main()

