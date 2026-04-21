from __future__ import annotations

import numpy as np

from .utils import EPSILON, logsumexp, sample_categorical_from_log_probs


def compute_emission_log_probs(emission_matrix: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Devuelve log P(y_t | z_t = k) para cada estado y observacion."""

    return np.log(emission_matrix[:, observations] + EPSILON)


def forward_log_likelihood(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Algoritmo forward en log."""

    n_states = transition_matrix.shape[0]
    n_steps = len(observations)
    emission_log = compute_emission_log_probs(emission_matrix, observations)
    alpha = np.full((n_steps, n_states), -np.inf, dtype=float)

    alpha[0] = np.log(initial_probs + EPSILON) + emission_log[:, 0]
    for time_index in range(1, n_steps):
        trans_scores = alpha[time_index - 1][:, None] + np.log(transition_matrix + EPSILON)
        alpha[time_index] = emission_log[:, time_index] + logsumexp(trans_scores, axis=0)
    return float(logsumexp(alpha[-1])), alpha


def backward_sample_states(
    alpha: np.ndarray,
    transition_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Muestra una trayectoria latente completa via FFBS."""

    n_steps, n_states = alpha.shape
    states = np.zeros(n_steps, dtype=int)
    states[-1] = sample_categorical_from_log_probs(alpha[-1], rng)

    log_transition = np.log(transition_matrix + EPSILON)
    for time_index in range(n_steps - 2, -1, -1):
        scores = alpha[time_index] + log_transition[:, states[time_index + 1]]
        states[time_index] = sample_categorical_from_log_probs(scores, rng)
    return states


def ffbs_sample(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    observations: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Forward filtering backward sampling para HMM discretos."""

    log_likelihood, alpha = forward_log_likelihood(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        observations=observations,
    )
    return backward_sample_states(alpha, transition_matrix, rng), log_likelihood


def viterbi_decode(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Decodificacion Viterbi en espacio logaritmico."""

    n_states = transition_matrix.shape[0]
    n_steps = len(observations)
    emission_log = compute_emission_log_probs(emission_matrix, observations)
    delta = np.full((n_steps, n_states), -np.inf, dtype=float)
    psi = np.zeros((n_steps, n_states), dtype=int)

    delta[0] = np.log(initial_probs + EPSILON) + emission_log[:, 0]
    for time_index in range(1, n_steps):
        scores = delta[time_index - 1][:, None] + np.log(transition_matrix + EPSILON)
        psi[time_index] = np.argmax(scores, axis=0)
        delta[time_index] = emission_log[:, time_index] + np.max(scores, axis=0)

    states = np.zeros(n_steps, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for time_index in range(n_steps - 2, -1, -1):
        states[time_index] = psi[time_index + 1, states[time_index + 1]]
    return states, float(np.max(delta[-1]))
