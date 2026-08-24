from __future__ import annotations

import math

import numpy as np

from .utils import EPSILON, length_buckets, logsumexp, sample_categorical_from_log_probs


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


def scaled_forward_log_likelihood(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    observations: np.ndarray,
) -> float:
    """log P(y_1:T) con reescalado por paso en vez de logsumexp.

    Devuelve lo mismo que forward_log_likelihood, sin la matriz alpha.
    """

    emission = emission_matrix[:, observations]
    alpha = initial_probs * emission[:, 0]
    scale = max(float(alpha.sum()), EPSILON)
    alpha = alpha / scale
    log_likelihood = math.log(scale)
    for step in range(1, observations.size):
        alpha = (alpha @ transition_matrix) * emission[:, step]
        scale = max(float(alpha.sum()), EPSILON)
        alpha /= scale
        log_likelihood += math.log(scale)
    return log_likelihood


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


def ffbs_sample_batch(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    sequences: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """FFBS sobre todas las secuencias a la vez. Equivale a ffbs_sample en bucle.

    Dos cambios respecto de la version por secuencia, ambos por costo y no por metodo:
    el filtrado usa reescalado en dominio lineal en vez de logsumexp, y el muestreo hacia
    atras usa el truco de Gumbel-max en vez de una llamada a rng.choice por paso.
    """

    n_states = transition_matrix.shape[0]
    log_transition = np.log(transition_matrix + EPSILON)
    sampled: list[np.ndarray | None] = [None] * len(sequences)
    log_likelihood = np.zeros(len(sequences), dtype=float)

    for index, padded, lengths in length_buckets(sequences):
        n_rows, max_steps = padded.shape
        live = np.arange(max_steps)[None, :] < lengths[:, None]  # (B, T) pasos reales
        emission = emission_matrix[:, padded].transpose(1, 0, 2)  # (B, K, T)

        alpha = np.empty((n_rows, max_steps, n_states), dtype=float)
        current = initial_probs[None, :] * emission[:, :, 0]
        scale = np.maximum(current.sum(axis=1, keepdims=True), EPSILON)
        current = current / scale
        alpha[:, 0] = current
        bucket_log_likelihood = np.log(scale[:, 0])

        for step in range(1, max_steps):
            moved = (current @ transition_matrix) * emission[:, :, step]
            step_scale = np.maximum(moved.sum(axis=1, keepdims=True), EPSILON)
            moved /= step_scale
            mask = live[:, step][:, None]
            current = np.where(mask, moved, current)
            alpha[:, step] = current
            bucket_log_likelihood += np.where(live[:, step], np.log(step_scale[:, 0]), 0.0)

        # Muestreo hacia atras. Gumbel-max sobre los log-pesos es una muestra
        # categorica exacta, y no necesita normalizar ni acumular la distribucion.
        states = np.empty((n_rows, max_steps), dtype=int)
        rows = np.arange(n_rows)
        last = lengths - 1
        tail = np.log(alpha[rows, last] + EPSILON)
        states[rows, last] = np.argmax(tail + rng.gumbel(size=tail.shape), axis=1)

        for step in range(max_steps - 2, -1, -1):
            ahead = live[:, step + 1]
            if not ahead.any():
                continue
            nxt = np.where(ahead, states[:, step + 1], 0)
            scores = np.log(alpha[:, step] + EPSILON) + log_transition[:, nxt].T
            drawn = np.argmax(scores + rng.gumbel(size=scores.shape), axis=1)
            # Los pasos ya fijados por `last` (relleno) no se tocan.
            states[:, step] = np.where(ahead, drawn, states[:, step])

        for row, original in enumerate(index):
            sampled[original] = states[row, : lengths[row]].copy()
            log_likelihood[original] = bucket_log_likelihood[row]

    return sampled, log_likelihood


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
