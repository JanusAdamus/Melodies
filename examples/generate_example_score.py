from __future__ import annotations

from pathlib import Path

import music21


def build_demo_score() -> music21.stream.Score:
    """Construye una mini pieza simbolica con armonia mas rica para demos."""

    score = music21.stream.Score(id="demo_hdp_hmm")
    part = music21.stream.Part(id="melody")
    part.append(music21.tempo.MetronomeMark(number=92))
    part.append(music21.meter.TimeSignature("4/4"))
    part.append(music21.key.Key("C"))

    # La secuencia arpegia sonoridades C:maj9, A:m7, D:m9, G:9 y C:add9.
    pitches = [
        ("C4", 1.0),
        ("E4", 1.0),
        ("G4", 1.0),
        ("B4", 0.5),
        ("D5", 0.5),
        ("A4", 1.0),
        ("C5", 1.0),
        ("E5", 1.0),
        ("G5", 1.0),
        ("D4", 1.0),
        ("F4", 1.0),
        ("A4", 1.0),
        ("C5", 0.5),
        ("E5", 0.5),
        ("G4", 1.0),
        ("C5", 1.0),
        ("D5", 1.0),
        ("F5", 0.5),
        ("A5", 0.5),
        ("C5", 1.0),
        ("D5", 0.5),
        ("E5", 0.5),
        ("G5", 1.0),
        ("A5", 1.0),
    ]

    for pitch_name, duration in pitches:
        note = music21.note.Note(pitch_name)
        note.quarterLength = duration
        part.append(note)

    score.insert(0, part)
    return score


def main() -> None:
    output_path = Path("examples/example_score.musicxml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score = build_demo_score()
    score.write("musicxml", fp=str(output_path))
    print(f"Partitura de ejemplo generada en: {output_path}")


if __name__ == "__main__":
    main()
