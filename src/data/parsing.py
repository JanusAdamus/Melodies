from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover - depende del entorno local.
    pass

import music21


@dataclass(frozen=True)
class MusicalEvent:
    """Representa un evento musical atomico ordenado en el tiempo."""

    index: int
    offset: float
    duration: float
    kind: str
    label: str
    pitch_classes: tuple[int, ...]
    midi_pitches: tuple[int, ...]
    representative_pitch_class: int | None
    representative_midi: int | None
    measure: int | None
    beat: float | None


def parse_score(path: str | Path) -> music21.stream.Score:
    """Carga una partitura MusicXML o MIDI con music21."""

    return music21.converter.parse(str(path))


def select_parts(score: music21.stream.Score, prefer_treble: bool = True) -> list[music21.stream.Part]:
    """Selecciona partes relevantes; prioriza claves agudas si existen."""

    parts = list(score.parts)
    if not parts:
        return [score]
    if not prefer_treble:
        return parts

    treble_parts = []
    for part in parts:
        if part.recurse().getElementsByClass(music21.clef.TrebleClef):
            treble_parts.append(part)
    return treble_parts or parts


def merge_parts(
    score: music21.stream.Score, prefer_treble: bool = True
) -> music21.stream.Stream:
    """Fusiona partes en un flujo temporal comun ordenado por offset."""

    merged = music21.stream.Stream()
    for part in select_parts(score, prefer_treble=prefer_treble):
        for element in part.flatten().notesAndRests:
            merged.insert(element.offset, element)
    return merged


def _representative_pitch_class_for_chord(chord: music21.chord.Chord) -> int | None:
    root = chord.root()
    if root is not None:
        return int(root.pitchClass)
    if chord.pitches:
        return int(chord.bass().pitchClass)
    return None


def _representative_midi_for_chord(chord: music21.chord.Chord) -> int | None:
    if not chord.pitches:
        return None
    root_pitch_class = _representative_pitch_class_for_chord(chord)
    if root_pitch_class is None:
        return int(chord.bass().midi)
    for pitch in sorted(chord.pitches, key=lambda item: item.midi):
        if pitch.pitchClass == root_pitch_class:
            return int(pitch.midi)
    return int(chord.bass().midi)


def extract_events(
    score_or_stream: music21.stream.Score | music21.stream.Stream,
    include_rests: bool = False,
    prefer_treble: bool = True,
) -> list[MusicalEvent]:
    """Convierte una partitura en una secuencia ordenada de eventos musicales."""

    if isinstance(score_or_stream, music21.stream.Score):
        stream = merge_parts(score_or_stream, prefer_treble=prefer_treble)
    else:
        stream = score_or_stream

    ordered = sorted(stream.flatten().notesAndRests, key=lambda item: (item.offset, item.priority))
    events: list[MusicalEvent] = []

    for index, element in enumerate(ordered):
        if isinstance(element, music21.note.Rest) and not include_rests:
            continue

        if isinstance(element, music21.note.Note):
            event = MusicalEvent(
                index=len(events),
                offset=float(element.offset),
                duration=float(element.quarterLength),
                kind="note",
                label=element.pitch.nameWithOctave,
                pitch_classes=(int(element.pitch.pitchClass),),
                midi_pitches=(int(element.pitch.midi),),
                representative_pitch_class=int(element.pitch.pitchClass),
                representative_midi=int(element.pitch.midi),
                measure=element.measureNumber,
                beat=float(element.beat) if element.beat is not None else None,
            )
            events.append(event)
            continue

        if isinstance(element, music21.chord.Chord):
            pitches = tuple(sorted(int(p.midi) for p in element.pitches))
            pitch_classes = tuple(sorted(int(p.pitchClass) for p in element.pitches))
            event = MusicalEvent(
                index=len(events),
                offset=float(element.offset),
                duration=float(element.quarterLength),
                kind="chord",
                label=".".join(p.nameWithOctave for p in sorted(element.pitches, key=lambda item: item.midi)),
                pitch_classes=pitch_classes,
                midi_pitches=pitches,
                representative_pitch_class=_representative_pitch_class_for_chord(element),
                representative_midi=_representative_midi_for_chord(element),
                measure=element.measureNumber,
                beat=float(element.beat) if element.beat is not None else None,
            )
            events.append(event)
            continue

        if isinstance(element, music21.note.Rest):
            event = MusicalEvent(
                index=len(events),
                offset=float(element.offset),
                duration=float(element.quarterLength),
                kind="rest",
                label="Rest",
                pitch_classes=tuple(),
                midi_pitches=tuple(),
                representative_pitch_class=None,
                representative_midi=None,
                measure=element.measureNumber,
                beat=float(element.beat) if element.beat is not None else None,
            )
            events.append(event)

    return events


def events_to_dataframe(events: Iterable[MusicalEvent]) -> pd.DataFrame:
    """Convierte eventos musicales en una tabla util para exportacion."""

    return pd.DataFrame(asdict(event) for event in events)
