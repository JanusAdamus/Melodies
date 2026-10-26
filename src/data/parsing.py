from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
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


EXTRACTION_MODES = ("melody", "legacy_all_parts")

_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Regla unica de microtonos: cada altura se lleva al semitono temperado mas cercano de la
# altura que el nombre de la alteracion denota en la notacion turca Arel-Ezgi-Uzdilek
# (1 coma = 1/9 de tono ~ 22.6 cents), que es la que usa SymbTr. Verificado contra la
# columna Koma53 de los .txt de SymbTr: koma (quarter/half) = +-1 coma -> natural escrito;
# bakiye (slash-flat, sharp) = 4 comas y kucuk mucenneb (flat, slash-quarter-sharp) = 5
# comas -> +-1 semitono; buyuk mucenneb (double-slash-flat, slash-sharp) = 8 -> +-2.
# music21 da alter=0 a slash-flat y redondea half-sharp distinto en pitchClass y en midi;
# por eso la clase de altura y el MIDI salen del mismo entero: pc == midi % 12 siempre.
ACCIDENTAL_SEMITONES = {
    "natural": 0,
    "sharp": 1,
    "flat": -1,
    "double-sharp": 2,
    "double-flat": -2,
    "half-flat": 0,
    "quarter-flat": 0,
    "half-sharp": 0,
    "quarter-sharp": 0,
    "slash-flat": -1,
    "double-slash-flat": -2,
    "slash-quarter-sharp": 1,
    "slash-sharp": 2,
}


def parse_score(path: str | Path) -> music21.stream.Score:
    """Carga una partitura MusicXML o MIDI con music21.

    music21 no acepta las armaduras no tradicionales de SymbTr (``key-accidental`` turco,
    ``key-alter`` en comas, p. ej. +5) y falla en ~28 % de los archivos. En SymbTr cada nota
    alterada lleva su ``<accidental>`` explicito (verificado contra los .txt), asi que la
    armadura no aporta altura y se reintenta sin los ``<key>`` que tienen ``<key-step>``.
    """

    try:
        return music21.converter.parse(str(path))
    except music21.pitch.AccidentalException:
        text = Path(path).read_text(encoding="utf-8")
        cleaned = re.sub(r"<key>(?:(?!</key>).)*<key-step>.*?</key>", "", text, flags=re.DOTALL)
        return music21.converter.parseData(cleaned, format="musicxml")


def melody_midi(pitch: music21.pitch.Pitch) -> int:
    """Altura MIDI entera de ``pitch`` segun la regla unica de microtonos (ver arriba)."""

    accidental = pitch.accidental
    offset = 0 if accidental is None else ACCIDENTAL_SEMITONES.get(accidental.name, int(accidental.alter))
    return 12 * (pitch.implicitOctave + 1) + _STEP_SEMITONES[pitch.step] + offset


def _first_clef_is_treble(part: music21.stream.Stream) -> bool:
    first = part.recurse().getElementsByClass(music21.clef.Clef).first()
    return first is not None and first.sign == "G"


def _mean_midi(part: music21.stream.Stream) -> float:
    values = [pitch.midi for element in part.recurse().notes for pitch in element.pitches]
    return sum(values) / len(values) if values else float("-inf")


def select_melody_part(score: music21.stream.Score | music21.stream.Stream) -> music21.stream.Stream:
    """Elige la parte que lleva la melodia.

    Criterio: la primera parte, en orden de partitura, cuya PRIMERA clave es de sol. Se mira
    la clave inicial y no cualquier clave porque un pentagrama conserva su identidad: la mano
    izquierda que pasa un momento a sol sigue siendo acompanamiento, y la derecha que baja a fa
    sigue siendo la melodia. Con varias partes en sol (piano a cuatro manos, conjunto) gana la
    primera, que por convencion de partitura es la voz superior. Si ninguna empieza en sol,
    la de mayor altura media.
    """

    parts = list(score.parts) if isinstance(score, music21.stream.Score) else []
    if not parts:
        return score
    treble = [part for part in parts if _first_clef_is_treble(part)]
    if treble:
        return treble[0]
    return max(parts, key=_mean_midi)


def extract_melody_events(score: music21.stream.Score | music21.stream.Stream) -> list[MusicalEvent]:
    """Melodia = voz superior de la parte en clave de sol (decision 1 del preregistro).

    Ligaduras unidas con ``stripTies``; sin notas de adorno (duracion 0) ni cifrados
    (``harmony.Harmony``, que incluye ``ChordSymbol`` y ``NoChord``). En cada ataque se toma
    la altura mas aguda entre voces y acordes (la nota superior del acorde, no su raiz).
    """

    part = select_melody_part(score).stripTies(inPlace=False)
    top_by_offset: dict[float, tuple[int, music21.pitch.Pitch, music21.base.Music21Object]] = {}
    # shortcut: un ataque grave bajo una nota aguda aun sonando se conserva (skyline por ataque).
    for element in part.flatten().notes:
        if isinstance(element, music21.harmony.Harmony) or element.duration.isGrace or element.quarterLength == 0:
            continue
        if not element.pitches:
            continue
        midi, pitch = max(((melody_midi(p), p) for p in element.pitches), key=lambda item: item[0])
        offset = float(element.offset)
        if offset not in top_by_offset or midi > top_by_offset[offset][0]:
            top_by_offset[offset] = (midi, pitch, element)

    events: list[MusicalEvent] = []
    for offset in sorted(top_by_offset):
        midi, pitch, element = top_by_offset[offset]
        events.append(
            MusicalEvent(
                index=len(events),
                offset=offset,
                duration=float(element.quarterLength),
                kind="note",
                label=pitch.nameWithOctave,
                pitch_classes=(midi % 12,),
                midi_pitches=(midi,),
                representative_pitch_class=midi % 12,
                representative_midi=midi,
                measure=element.measureNumber,
                beat=float(element.beat) if element.beat is not None else None,
            )
        )
    return events


def extract_events_for_mode(
    score: music21.stream.Score | music21.stream.Stream,
    mode: str = "melody",
    prefer_treble: bool = False,
) -> list[MusicalEvent]:
    """Extrae eventos con un modo explicito.

    ``melody``: voz superior de la parte en sol (la definicion vigente).
    ``legacy_all_parts``: la extraccion anterior al 2026-10-26 (todas las partes fusionadas,
    acordes reducidos a su raiz, ligaduras y adornos como eventos); se conserva para citar
    la version antigua. ``prefer_treble`` solo aplica a este modo.
    """

    if mode == "melody":
        return extract_melody_events(score)
    if mode == "legacy_all_parts":
        return extract_events(score, prefer_treble=prefer_treble)
    raise ValueError(f"Modo de extraccion no soportado: {mode!r}; usa uno de {EXTRACTION_MODES}.")


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
