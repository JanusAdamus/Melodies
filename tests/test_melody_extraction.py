from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import music21

from src.data.parsing import (
    EXTRACTION_MODES,
    extract_events,
    extract_events_for_mode,
    extract_melody_events,
    melody_midi,
    parse_score,
    select_melody_part,
)

LIBRARY = Path(__file__).resolve().parents[2] / "Melodies" / "external" / "library" / "scores"

# Minueto en sol, BWV Anh. 114, compases 1-16 de la mano derecha, leidos de la partitura.
MINUET_GOLD = [
    2, 7, 9, 11, 0, 2, 7, 7, 4, 0, 2, 4, 6, 7, 7, 7, 0, 2, 0, 11, 9, 11, 0, 11, 9, 7, 6, 7, 9, 11, 7, 11,
    9, 2, 7, 9, 11, 0, 2, 7, 7, 4, 0, 2, 4, 6, 7, 7, 7, 0, 2, 0, 11, 9, 11, 0, 11, 9, 7, 9, 11, 9, 7, 6, 7,
]
HAPPY_BIRTHDAY_GOLD = [7, 7, 9, 7, 0, 11, 7, 7, 9, 7, 2, 0, 7, 7, 7, 4, 0, 11, 9, 5, 5, 4, 0, 2, 0]


def _part(clef: music21.clef.Clef, elements: list) -> music21.stream.Part:
    part = music21.stream.Part()
    measure = music21.stream.Measure(number=1)
    measure.append(clef)
    measure.append(music21.meter.TimeSignature("4/4"))
    for element in elements:
        measure.append(element)
    part.append(measure)
    return part


def _score(*parts: music21.stream.Part) -> music21.stream.Score:
    score = music21.stream.Score()
    for part in parts:
        score.insert(0, part)
    return score


def _pcs(events) -> list[int]:
    return [event.representative_pitch_class for event in events]


def _library(name: str) -> Path:
    path = LIBRARY / name
    if not path.exists():
        raise unittest.SkipTest(f"partitura no disponible: {path}")
    return path


class MelodySelectionTests(unittest.TestCase):
    def test_uses_part_whose_first_clef_is_treble_even_if_listed_second(self) -> None:
        bass = _part(music21.clef.BassClef(), [music21.note.Note("C3", quarterLength=4.0)])
        treble = _part(music21.clef.TrebleClef(), [music21.note.Note("E5", quarterLength=4.0)])
        self.assertEqual(_pcs(extract_melody_events(_score(bass, treble))), [4])

    def test_left_hand_switching_to_treble_is_not_melody(self) -> None:
        right = _part(music21.clef.TrebleClef(), [music21.note.Note("G4", quarterLength=4.0)])
        left = _part(music21.clef.BassClef(), [music21.note.Note("C3"), music21.clef.TrebleClef(), music21.note.Note("A5", quarterLength=3.0)])
        self.assertIs(select_melody_part(_score(right, left)), right)

    def test_no_treble_part_falls_back_to_highest_mean_pitch(self) -> None:
        low = _part(music21.clef.BassClef(), [music21.note.Note("C2", quarterLength=4.0)])
        high = _part(music21.clef.BassClef(), [music21.note.Note("C4", quarterLength=4.0)])
        self.assertIs(select_melody_part(_score(low, high)), high)


class MelodyEventTests(unittest.TestCase):
    def test_ties_are_merged(self) -> None:
        first = music21.note.Note("C5", quarterLength=2.0)
        first.tie = music21.tie.Tie("start")
        second = music21.note.Note("C5", quarterLength=2.0)
        second.tie = music21.tie.Tie("stop")
        events = extract_melody_events(_score(_part(music21.clef.TrebleClef(), [first, second])))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].duration, 4.0)

    def test_grace_notes_and_chord_symbols_are_dropped(self) -> None:
        grace = music21.note.Note("F5").getGrace()
        symbol = music21.harmony.ChordSymbol("C")
        no_chord = music21.harmony.NoChord()
        main = music21.note.Note("E5", quarterLength=4.0)
        part = _part(music21.clef.TrebleClef(), [])
        measure = part.getElementsByClass(music21.stream.Measure).first()
        measure.insert(0, symbol)
        measure.insert(0, no_chord)
        measure.insert(0, grace)
        measure.insert(0, main)
        self.assertEqual(_pcs(extract_melody_events(_score(part))), [4])

    def test_chord_in_melody_contributes_its_top_note_not_its_root(self) -> None:
        chord = music21.chord.Chord(["C4", "E4", "G4"], quarterLength=4.0)
        events = extract_melody_events(_score(_part(music21.clef.TrebleClef(), [chord])))
        self.assertEqual(_pcs(events), [7])

    def test_simultaneous_voices_keep_the_highest_onset(self) -> None:
        part = _part(music21.clef.TrebleClef(), [])
        measure = part.getElementsByClass(music21.stream.Measure).first()
        upper, lower = music21.stream.Voice(), music21.stream.Voice()
        upper.append(music21.note.Note("D5", quarterLength=4.0))
        lower.append(music21.note.Note("B4", quarterLength=4.0))
        measure.insert(0, upper)
        measure.insert(0, lower)
        self.assertEqual(_pcs(extract_melody_events(_score(part))), [2])


class MicrotoneRuleTests(unittest.TestCase):
    def _pitch(self, name: str, accidental: str) -> music21.pitch.Pitch:
        pitch = music21.pitch.Pitch(name)
        pitch.accidental = music21.pitch.Accidental(accidental)
        return pitch

    def test_koma_accidentals_round_to_written_natural(self) -> None:
        self.assertEqual(melody_midi(self._pitch("B4", "half-flat")) % 12, 11)
        self.assertEqual(melody_midi(self._pitch("C4", "half-sharp")) % 12, 0)

    def test_standard_accidentals_match_music21(self) -> None:
        for name in ("C#4", "B-3", "F##5", "E--4", "G4"):
            pitch = music21.pitch.Pitch(name)
            self.assertEqual(melody_midi(pitch), pitch.midi)
            self.assertEqual(melody_midi(pitch) % 12, pitch.pitchClass)

    def test_symbtr_file_with_slash_flat_key_signature_parses(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>M</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions>
<key><key-step>B</key-step><key-alter>-1</key-alter><key-accidental>slash-flat</key-accidental></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration><type>whole</type><accidental>slash-flat</accidental></note>
</measure></part></score-partwise>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hicaz--sarki--duyek--x--y.xml"
            path.write_text(xml, encoding="utf-8")
            events = extract_melody_events(parse_score(path))
        self.assertEqual(_pcs(events), [10])


    def test_symbtr_key_signature_in_commas_parses(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>M</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions>
<key><key-step>F</key-step><key-alter>+5</key-alter><key-accidental>slash-quarter-sharp</key-accidental></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>2</duration><type>half</type><accidental>slash-quarter-sharp</accidental></note>
<note><pitch><step>G</step><octave>5</octave></pitch><duration>2</duration><type>half</type></note>
</measure></part></score-partwise>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mahur--sarki--aksak--x--y.xml"
            path.write_text(xml, encoding="utf-8")
            events = extract_melody_events(parse_score(path))
        self.assertEqual(_pcs(events), [6, 7])


class ModeTests(unittest.TestCase):
    def test_modes_are_explicit_and_legacy_is_preserved(self) -> None:
        self.assertEqual(EXTRACTION_MODES, ("melody", "legacy_all_parts"))
        bass = _part(music21.clef.BassClef(), [music21.note.Note("C3", quarterLength=4.0)])
        treble = _part(music21.clef.TrebleClef(), [music21.note.Note("E5", quarterLength=4.0)])
        score = _score(treble, bass)
        legacy = extract_events_for_mode(score, "legacy_all_parts")
        self.assertEqual(_pcs(legacy), _pcs(extract_events(score, prefer_treble=False)))
        self.assertEqual(_pcs(extract_events_for_mode(score, "melody")), [4])
        with self.assertRaises(ValueError):
            extract_events_for_mode(score, "otra")


class GoldenScoreTests(unittest.TestCase):
    def test_minuet_measures_1_to_16(self) -> None:
        events = extract_melody_events(parse_score(_library("Bach_Minuet_in_G_Major_BWV_Anh._114.mxl")))
        first16 = [event for event in events if event.measure is not None and event.measure <= 16]
        self.assertEqual(_pcs(first16), MINUET_GOLD)

    def test_moonlight_third_movement_has_no_left_hand_events_at_start(self) -> None:
        events = extract_melody_events(parse_score(_library("Sonate_No._14_Moonlight_3rd_Movement.mxl")))
        # La mano derecha entra en el tercer tiempo del compas 1 (C#4); el C#2 y el arpegio grave son de la izquierda.
        self.assertEqual(events[0].offset, 2.0)
        self.assertEqual(events[0].label, "C#4")
        self.assertTrue(all(event.representative_midi >= 55 for event in events[:20]))

    def test_happy_birthday_has_no_duplicated_chord_symbols(self) -> None:
        events = extract_melody_events(parse_score(_library("Happy_Birthday_To_You_C_Major.mxl")))
        self.assertEqual(_pcs(events), HAPPY_BIRTHDAY_GOLD)


if __name__ == "__main__":
    unittest.main()
