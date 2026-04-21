from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.observations import ObservationSequence

from .harmony import (
    PITCH_CLASS_NAMES,
    HarmonicState,
    build_harmonic_state_space,
    chord_emission_distribution,
    chord_states_dataframe,
    infer_local_mode_label,
)
from .inference import forward_log_likelihood, viterbi_decode
from .utils import count_transitions, normalize

HARMONIC_STATE_SPACE = build_harmonic_state_space()
CHORD_STATES = [state.label for state in HARMONIC_STATE_SPACE]


@dataclass
class FiniteHMMResult:
    """Resultado del HMM finito armonico extendido."""

    latent_states: np.ndarray
    latent_labels: list[str]
    modal_labels: list[str]
    active_states: list[int]
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray
    empirical_transition_matrix: np.ndarray
    log_likelihood: float
    viterbi_score: float
    observations: ObservationSequence
    state_space: list[HarmonicState]

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "log_likelihood",
                    "viterbi_score",
                    "n_active_states",
                    "n_observations",
                    "state_space_size",
                ],
                "value": [
                    self.log_likelihood,
                    self.viterbi_score,
                    len(self.active_states),
                    self.observations.size,
                    len(self.state_space),
                ],
            }
        )

    def vocabulary_dataframe(self) -> pd.DataFrame:
        return chord_states_dataframe(self.state_space)


class FiniteChordHMM:
    """Baseline armonico con acordes extendidos y contexto modal local."""

    def __init__(
        self,
        stay_bias: float = 2.5,
        same_root_bias: float = 0.8,
        related_root_bias: float = 0.4,
        shared_mode_bias: float = 0.3,
        state_space: list[HarmonicState] | None = None,
        initial_probs: np.ndarray | None = None,
        transition_matrix: np.ndarray | None = None,
        emission_matrix: np.ndarray | None = None,
    ) -> None:
        self.stay_bias = stay_bias
        self.same_root_bias = same_root_bias
        self.related_root_bias = related_root_bias
        self.shared_mode_bias = shared_mode_bias
        self.state_space = state_space if state_space is not None else HARMONIC_STATE_SPACE
        self.state_labels = [state.label for state in self.state_space]
        self.n_states = len(self.state_space)
        self.initial_probs = initial_probs if initial_probs is not None else self._build_initial_probs()
        self.transition_matrix = transition_matrix if transition_matrix is not None else self._build_transition_matrix()
        self.emission_matrix = emission_matrix if emission_matrix is not None else self._build_emission_matrix()

    def _build_initial_probs(self) -> np.ndarray:
        return np.full(self.n_states, 1.0 / self.n_states, dtype=float)

    def _transition_score(self, source: HarmonicState, target: HarmonicState) -> float:
        score = 1.0
        if source.index == target.index:
            score += self.stay_bias
        if source.root == target.root:
            score += self.same_root_bias
            if source.template.level == target.template.level:
                score += 0.3
        interval = (target.root - source.root) % 12
        if interval in {5, 7}:
            score += self.related_root_bias
        if set(source.compatible_modes).intersection(target.compatible_modes):
            score += self.shared_mode_bias
        return score

    def _build_transition_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_states, self.n_states), dtype=float)
        for row_index, source in enumerate(self.state_space):
            for col_index, target in enumerate(self.state_space):
                matrix[row_index, col_index] = self._transition_score(source, target)
        return normalize(matrix, axis=1)

    def _build_emission_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_states, 12), dtype=float)
        for state in self.state_space:
            matrix[state.index] = chord_emission_distribution(state)
        return normalize(matrix, axis=1)

    def _infer_modal_labels(self, states: np.ndarray, observations: np.ndarray, window_radius: int = 2) -> list[str]:
        modal_labels = []
        for index, state_index in enumerate(states):
            left = max(0, index - window_radius)
            right = min(len(observations), index + window_radius + 1)
            local_pitch_classes = observations[left:right]
            root = self.state_space[int(state_index)].root
            modal_labels.append(infer_local_mode_label(root, local_pitch_classes))
        return modal_labels

    def fit_predict(self, observations: ObservationSequence) -> FiniteHMMResult:
        """Decodifica una secuencia de pitch classes con un vocabulario armonico extendido."""

        if observations.observation_type != "pitch_class":
            raise ValueError("El baseline armonico extendido solo acepta observaciones de tipo pitch_class.")

        states, viterbi_score = viterbi_decode(
            initial_probs=self.initial_probs,
            transition_matrix=self.transition_matrix,
            emission_matrix=self.emission_matrix,
            observations=observations.tokens,
        )
        log_likelihood, _ = forward_log_likelihood(
            initial_probs=self.initial_probs,
            transition_matrix=self.transition_matrix,
            emission_matrix=self.emission_matrix,
            observations=observations.tokens,
        )
        counts = count_transitions(states, self.n_states)
        empirical = normalize(counts.astype(float), axis=1)
        active_states = sorted(set(int(state) for state in states))
        labels = [self.state_labels[int(state)] for state in states]
        modal_labels = self._infer_modal_labels(states, observations.tokens)
        return FiniteHMMResult(
            latent_states=states,
            latent_labels=labels,
            modal_labels=modal_labels,
            active_states=active_states,
            initial_probs=self.initial_probs.copy(),
            transition_matrix=self.transition_matrix.copy(),
            emission_matrix=self.emission_matrix.copy(),
            empirical_transition_matrix=empirical,
            log_likelihood=log_likelihood,
            viterbi_score=viterbi_score,
            observations=observations,
            state_space=self.state_space,
        )
