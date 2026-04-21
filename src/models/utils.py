from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPSILON = 1e-12


def set_random_seed(seed: int | None) -> np.random.Generator:
    """Crea un generador reproducible."""

    return np.random.default_rng(seed)


def normalize(vector: np.ndarray, axis: int | None = None, epsilon: float = EPSILON) -> np.ndarray:
    """Normaliza arreglos evitando divisiones por cero."""

    array = np.asarray(vector, dtype=float)
    if axis is None:
        total = array.sum()
        if total <= epsilon:
            return np.full_like(array, 1.0 / array.size)
        return array / total

    total = array.sum(axis=axis, keepdims=True)
    safe = np.where(total <= epsilon, 1.0, total)
    normalized = array / safe
    if axis == 1:
        zero_rows = np.where(total.squeeze(axis=axis) <= epsilon)[0]
        if zero_rows.size:
            normalized[zero_rows] = 1.0 / array.shape[1]
    if axis == 0:
        zero_cols = np.where(total.squeeze(axis=axis) <= epsilon)[0]
        if zero_cols.size:
            normalized[:, zero_cols] = 1.0 / array.shape[0]
    return normalized


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Implementacion local para evitar depender de scipy en la base del proyecto."""

    values = np.asarray(values, dtype=float)
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = values - max_values
    summed = np.sum(np.exp(stable), axis=axis, keepdims=True)
    result = max_values + np.log(summed + EPSILON)
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = normalize(probabilities)
    return int(rng.choice(len(probabilities), p=probabilities))


def sample_categorical_from_log_probs(log_probabilities: np.ndarray, rng: np.random.Generator) -> int:
    normalized = np.exp(log_probabilities - logsumexp(log_probabilities))
    return int(rng.choice(len(normalized), p=normalized))


def stick_breaking_from_v(v: np.ndarray) -> np.ndarray:
    """Construye beta a partir de variables v truncadas."""

    v = np.asarray(v, dtype=float)
    beta = np.zeros(v.size + 1, dtype=float)
    remaining = 1.0
    for index, value in enumerate(v):
        beta[index] = remaining * value
        remaining *= 1.0 - value
    beta[-1] = remaining
    return normalize(beta)


def sample_truncated_stick_breaking(
    gamma: float,
    n_states: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Muestra pesos truncados beta via stick-breaking debil."""

    if n_states < 1:
        raise ValueError("n_states debe ser >= 1")
    if n_states == 1:
        return np.array([], dtype=float), np.array([1.0], dtype=float)
    v = rng.beta(1.0, gamma, size=n_states - 1)
    beta = stick_breaking_from_v(v)
    return v, beta


def dirichlet_logpdf(x: np.ndarray, alpha: np.ndarray) -> float:
    """Log densidad de Dirichlet usando math.lgamma."""

    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if np.any(x <= 0.0) or np.any(alpha <= 0.0):
        return -np.inf
    total = math.lgamma(float(np.sum(alpha))) - float(np.sum([math.lgamma(value) for value in alpha]))
    total += float(np.sum((alpha - 1.0) * np.log(x)))
    return total


def count_transitions(states: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=int)
    for left, right in zip(states[:-1], states[1:]):
        counts[int(left), int(right)] += 1
    return counts


def count_emissions(states: np.ndarray, observations: np.ndarray, n_states: int, vocab_size: int) -> np.ndarray:
    counts = np.zeros((n_states, vocab_size), dtype=int)
    for state, observation in zip(states, observations):
        counts[int(state), int(observation)] += 1
    return counts


def one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=float)
    vector[index] = 1.0
    return vector


def contiguous_segments(states: Iterable[int]) -> list[tuple[int, int, int]]:
    """Devuelve segmentos como (estado, inicio, fin_exclusivo)."""

    states = list(int(state) for state in states)
    if not states:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state != current:
            segments.append((current, start, index))
            current = state
            start = index
    segments.append((current, start, len(states)))
    return segments

