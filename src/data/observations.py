from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
import pandas as pd

try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover - depende del entorno local.
    pass

import music21

from .parsing import MusicalEvent

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class ObservationSequence:
    """Secuencia discreta con su vocabulario y metadatos temporales."""

    observation_type: str
    tokens: np.ndarray
    vocabulary: list[str]
    decoded: list[str]
    events: list[MusicalEvent]
    extra: dict

    @property
    def size(self) -> int:
        return int(len(self.tokens))

    @property
    def vocab_size(self) -> int:
        return int(len(self.vocabulary))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for index, (token, decoded, event) in enumerate(zip(self.tokens, self.decoded, self.events)):
            rows.append(
                {
                    "t": index,
                    "offset": event.offset,
                    "duration": event.duration,
                    "measure": event.measure,
                    "beat": event.beat,
                    "event_kind": event.kind,
                    "event_label": event.label,
                    "token": int(token),
                    "observation": decoded,
                }
            )
        return pd.DataFrame(rows)


def available_observation_types() -> tuple[str, ...]:
    return ("pitch_class", "midi_note", "interval", "pitch_class_duration")


def _duration_to_bin(duration: float, bins: Sequence[float]) -> float:
    return min(bins, key=lambda item: abs(item - duration))


def _pair_sort_key(item: tuple[int, float]) -> tuple[int, float]:
    return item[0], item[1]


def build_observation_sequence(
    events: list[MusicalEvent],
    observation_type: str,
    max_interval: int = 12,
    duration_bins: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> ObservationSequence:
    """Construye observaciones discretas configurables a partir de eventos."""

    if observation_type not in available_observation_types():
        raise ValueError(f"Tipo de observacion no soportado: {observation_type}")
    if not events:
        raise ValueError("La secuencia de eventos esta vacia.")

    if observation_type == "pitch_class":
        vocabulary = PITCH_CLASS_NAMES.copy()
        tokens = np.array(
            [event.representative_pitch_class for event in events],
            dtype=int,
        )
        decoded = [vocabulary[token] for token in tokens]
        return ObservationSequence(observation_type, tokens, vocabulary, decoded, events, extra={})

    if observation_type == "midi_note":
        midi_values = [event.representative_midi for event in events]
        if any(value is None for value in midi_values):
            raise ValueError("No se puede construir midi_note si hay eventos sin pitch representativo.")
        unique = sorted(set(int(value) for value in midi_values if value is not None))
        lookup = {value: index for index, value in enumerate(unique)}
        tokens = np.array([lookup[int(value)] for value in midi_values], dtype=int)
        vocabulary = [music21.pitch.Pitch(midi=value).nameWithOctave for value in unique]
        decoded = [vocabulary[token] for token in tokens]
        return ObservationSequence(
            observation_type,
            tokens,
            vocabulary,
            decoded,
            events,
            extra={"midi_values": unique},
        )

    if observation_type == "interval":
        midi_values = [event.representative_midi for event in events]
        if any(value is None for value in midi_values):
            raise ValueError("No se puede construir interval si hay eventos sin pitch representativo.")
        labels = ["START"] + [f"INT_{delta:+d}" for delta in range(-max_interval, max_interval + 1)]
        offset = max_interval + 1
        tokens = []
        decoded = []
        previous = None
        for midi_value in midi_values:
            midi_value = int(midi_value)
            if previous is None:
                tokens.append(0)
                decoded.append("START")
            else:
                interval = max(-max_interval, min(max_interval, midi_value - previous))
                token = interval + offset
                tokens.append(token)
                decoded.append(labels[token])
            previous = midi_value
        return ObservationSequence(
            observation_type,
            np.array(tokens, dtype=int),
            labels,
            decoded,
            events,
            extra={"max_interval": max_interval},
        )

    pairs = []
    for event in events:
        if event.representative_pitch_class is None:
            raise ValueError("No se puede construir pitch_class_duration con eventos sin pitch representativo.")
        duration_bin = _duration_to_bin(event.duration, duration_bins)
        pairs.append((int(event.representative_pitch_class), float(duration_bin)))

    unique_pairs = sorted(set(pairs), key=_pair_sort_key)
    lookup = {pair: index for index, pair in enumerate(unique_pairs)}
    tokens = np.array([lookup[pair] for pair in pairs], dtype=int)
    vocabulary = [f"{PITCH_CLASS_NAMES[pitch_class]}|dur={duration:g}" for pitch_class, duration in unique_pairs]
    decoded = [vocabulary[token] for token in tokens]
    return ObservationSequence(
        observation_type,
        tokens,
        vocabulary,
        decoded,
        events,
        extra={"pairs": unique_pairs, "duration_bins": list(duration_bins)},
    )
