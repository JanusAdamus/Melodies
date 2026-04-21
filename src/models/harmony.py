from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import EPSILON, normalize

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DIATONIC_MODES: dict[str, tuple[int, ...]] = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}


@dataclass(frozen=True)
class ChordTemplate:
    """Plantilla abstracta de acorde independiente de la fundamental."""

    symbol: str
    family: str
    level: str
    intervals: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class HarmonicState:
    """Estado armonico concreto con raiz y plantilla."""

    index: int
    root: int
    template: ChordTemplate

    @property
    def label(self) -> str:
        return f"{PITCH_CLASS_NAMES[self.root]}:{self.template.symbol}"

    @property
    def chord_tones(self) -> tuple[int, ...]:
        return tuple((self.root + interval) % 12 for interval in self.template.intervals)

    @property
    def compatible_modes(self) -> tuple[str, ...]:
        tones = set(self.template.intervals)
        modes = [mode for mode, scale in DIATONIC_MODES.items() if tones.issubset(scale)]
        return tuple(modes)


@dataclass(frozen=True)
class ModalContext:
    """Contexto modal local asociado a una tonalidad/modalidad base."""

    tonic: int
    mode: str

    @property
    def label(self) -> str:
        return f"{PITCH_CLASS_NAMES[self.tonic]}:{self.mode}"

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.tonic + degree) % 12 for degree in DIATONIC_MODES[self.mode])


def build_chord_templates(include_six_nine: bool = True) -> list[ChordTemplate]:
    """Vocabulario armonico controlado hasta novena."""

    templates = [
        ChordTemplate("maj", "triad", "triad", (0, 4, 7), "Triada mayor"),
        ChordTemplate("min", "triad", "triad", (0, 3, 7), "Triada menor"),
        ChordTemplate("dim", "triad", "triad", (0, 3, 6), "Triada disminuida"),
        ChordTemplate("aug", "triad", "triad", (0, 4, 8), "Triada aumentada"),
        ChordTemplate("maj7", "seventh", "seventh", (0, 4, 7, 11), "Acorde mayor con septima mayor"),
        ChordTemplate("7", "seventh", "seventh", (0, 4, 7, 10), "Septima dominante"),
        ChordTemplate("m7", "seventh", "seventh", (0, 3, 7, 10), "Septima menor"),
        ChordTemplate("m(maj7)", "seventh", "seventh", (0, 3, 7, 11), "Menor con septima mayor"),
        ChordTemplate("ø7", "seventh", "seventh", (0, 3, 6, 10), "Semidisminuido"),
        ChordTemplate("dim7", "seventh", "seventh", (0, 3, 6, 9), "Septima disminuida"),
        ChordTemplate("6", "added", "sixth", (0, 4, 7, 9), "Acorde mayor con sexta"),
        ChordTemplate("m6", "added", "sixth", (0, 3, 7, 9), "Acorde menor con sexta"),
        ChordTemplate("add9", "added", "added", (0, 2, 4, 7), "Triada mayor con novena anadida"),
        ChordTemplate("madd9", "added", "added", (0, 2, 3, 7), "Triada menor con novena anadida"),
        ChordTemplate("sus2", "suspended", "suspended", (0, 2, 7), "Acorde suspendido con segunda"),
        ChordTemplate("sus4", "suspended", "suspended", (0, 5, 7), "Acorde suspendido con cuarta"),
        ChordTemplate("7sus4", "suspended", "seventh", (0, 5, 7, 10), "Dominante suspendido"),
        ChordTemplate("maj9", "ninth", "ninth", (0, 2, 4, 7, 11), "Mayor con novena"),
        ChordTemplate("9", "ninth", "ninth", (0, 2, 4, 7, 10), "Dominante con novena"),
        ChordTemplate("m9", "ninth", "ninth", (0, 2, 3, 7, 10), "Menor con novena"),
    ]
    if include_six_nine:
        templates.append(ChordTemplate("6/9", "added", "ninth", (0, 2, 4, 7, 9), "Acorde seis-novena"))
    return templates


def build_harmonic_state_space(include_six_nine: bool = True) -> list[HarmonicState]:
    """Construye el espacio armonico completo por fundamental."""

    templates = build_chord_templates(include_six_nine=include_six_nine)
    states = []
    index = 0
    for root in range(12):
        for template in templates:
            states.append(HarmonicState(index=index, root=root, template=template))
            index += 1
    return states


def build_modal_contexts() -> list[ModalContext]:
    contexts = []
    for tonic in range(12):
        for mode in DIATONIC_MODES:
            contexts.append(ModalContext(tonic=tonic, mode=mode))
    return contexts


def chord_states_dataframe(states: list[HarmonicState]) -> pd.DataFrame:
    rows = []
    for state in states:
        rows.append(
            {
                "index": state.index,
                "label": state.label,
                "root_pc": state.root,
                "root_name": PITCH_CLASS_NAMES[state.root],
                "quality": state.template.symbol,
                "family": state.template.family,
                "level": state.template.level,
                "intervals": str(state.template.intervals),
                "pitch_classes": str(state.chord_tones),
                "compatible_modes": ", ".join(state.compatible_modes),
                "description": state.template.description,
            }
        )
    return pd.DataFrame(rows)


def modal_contexts_dataframe(contexts: list[ModalContext]) -> pd.DataFrame:
    rows = []
    for context in contexts:
        rows.append(
            {
                "label": context.label,
                "tonic_pc": context.tonic,
                "tonic_name": PITCH_CLASS_NAMES[context.tonic],
                "mode": context.mode,
                "pitch_classes": str(context.pitch_classes),
            }
        )
    return pd.DataFrame(rows)


def chord_emission_distribution(
    state: HarmonicState,
    chord_tone_mass: float = 0.72,
    modal_tone_mass: float = 0.20,
    outside_mass: float = 0.08,
) -> np.ndarray:
    """Distribucion heuristica de emisiones guiada por acorde y modos compatibles."""

    distribution = np.zeros(12, dtype=float)
    chord_tones = set(state.chord_tones)

    modal_support = set()
    for mode in state.compatible_modes:
        modal_support.update((state.root + degree) % 12 for degree in DIATONIC_MODES[mode])
    scale_tones = modal_support - chord_tones
    outside_tones = set(range(12)) - chord_tones - scale_tones

    if chord_tones:
        for pitch_class in chord_tones:
            distribution[pitch_class] += chord_tone_mass / len(chord_tones)
    if scale_tones:
        for pitch_class in scale_tones:
            distribution[pitch_class] += modal_tone_mass / len(scale_tones)
    else:
        outside_mass += modal_tone_mass
    if outside_tones:
        for pitch_class in outside_tones:
            distribution[pitch_class] += outside_mass / len(outside_tones)
    return normalize(distribution)


def _pitch_class_profile_from_set(pitch_classes: Iterable[int]) -> np.ndarray:
    profile = np.zeros(12, dtype=float)
    for pitch_class in pitch_classes:
        profile[int(pitch_class) % 12] += 1.0
    if profile.sum() <= EPSILON:
        return np.full(12, 1.0 / 12.0)
    return normalize(profile)


def score_state_against_profile(profile: np.ndarray, state: HarmonicState) -> float:
    emission = chord_emission_distribution(state)
    return float(np.sum(profile * np.log(emission + EPSILON)))


def infer_chord_candidates_from_pitch_classes(
    pitch_classes: Iterable[int],
    states: list[HarmonicState] | None = None,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Devuelve los acordes mas plausibles para un conjunto de pitch classes."""

    if states is None:
        states = build_harmonic_state_space()
    profile = _pitch_class_profile_from_set(pitch_classes)
    scored = sorted(
        ((state.label, score_state_against_profile(profile, state)) for state in states),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored[:top_n]


def score_modal_context(profile: np.ndarray, context: ModalContext) -> float:
    support = set(context.pitch_classes)
    inside = float(np.sum(profile[list(support)]))
    outside = 1.0 - inside
    return inside - 0.6 * outside


def infer_mode_candidates(
    pitch_classes: Iterable[int],
    tonic: int | None = None,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Infere contextos modales plausibles a partir de pitch classes."""

    profile = _pitch_class_profile_from_set(pitch_classes)
    contexts = build_modal_contexts()
    if tonic is not None:
        contexts = [context for context in contexts if context.tonic == tonic]
    scored = sorted(
        ((context.label, score_modal_context(profile, context)) for context in contexts),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored[:top_n]


def infer_local_mode_label(
    root: int,
    local_pitch_classes: Iterable[int],
) -> str:
    """Selecciona el modo local mas plausible condicionado a una raiz armonica."""

    candidates = infer_mode_candidates(local_pitch_classes, tonic=root, top_n=1)
    return candidates[0][0] if candidates else f"{PITCH_CLASS_NAMES[root]}:ionian"


def harmonic_profile_from_observations(
    observations: np.ndarray,
    vocabulary: list[str],
    observation_type: str,
) -> np.ndarray | None:
    """Proyecta observaciones categóricas a un perfil de pitch classes cuando es posible."""

    if observation_type == "pitch_class":
        return normalize(np.asarray(observations, dtype=float))

    if observation_type == "pitch_class_duration":
        profile = np.zeros(12, dtype=float)
        for label, probability in zip(vocabulary, observations):
            pitch_name = label.split("|", 1)[0]
            if pitch_name in PITCH_CLASS_NAMES:
                profile[PITCH_CLASS_NAMES.index(pitch_name)] += probability
        return normalize(profile)

    return None
