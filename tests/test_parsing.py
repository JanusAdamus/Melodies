from __future__ import annotations

import unittest

import music21

from src.data.parsing import extract_events


def build_test_score() -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    part.append(music21.meter.TimeSignature("4/4"))
    note_c = music21.note.Note("C4", quarterLength=1.0)
    chord_g = music21.chord.Chord(["G3", "B3", "D4"], quarterLength=2.0)
    part.append(note_c)
    part.append(chord_g)
    score.insert(0, part)
    return score


class ParsingTests(unittest.TestCase):
    def test_extract_events_returns_ordered_note_and_chord(self) -> None:
        events = extract_events(build_test_score())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].kind, "note")
        self.assertEqual(events[0].representative_pitch_class, 0)
        self.assertEqual(events[1].kind, "chord")
        self.assertEqual(events[1].representative_pitch_class, 7)
        self.assertGreater(events[1].representative_midi, 0)


if __name__ == "__main__":
    unittest.main()

