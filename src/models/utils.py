from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp as scipy_logsumexp


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


# ponytail: presupuesto en celdas con relleno, no en numero de secuencias. Las piezas van
# de decenas a miles de tokens, asi que un lote de tamano fijo deja que la mas larga fije T
# para todas y el relleno domina. 100k celdas ~ 115 MB con K=48.
CELL_BUDGET = 100_000


def length_buckets(
    sequences: list[np.ndarray],
    cell_budget: int = CELL_BUDGET,
) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    """Agrupa secuencias de longitud similar en lotes rectangulares.

    Devuelve (indices originales, matriz rellenada, longitudes) por lote. Ordenar por
    longitud y cerrar el lote cuando `n_secuencias * longitud_maxima` supera el
    presupuesto acota cuanto relleno se procesa.
    """

    order = sorted(range(len(sequences)), key=lambda i: sequences[i].size)
    buckets: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order):
            longest = sequences[order[stop]].size
            if (stop + 1 - start) * longest > cell_budget:
                break
            stop += 1
        index = order[start:stop]
        lengths = np.array([sequences[i].size for i in index], dtype=int)
        padded = np.zeros((len(index), int(lengths.max())), dtype=int)
        for row, i in enumerate(index):
            padded[row, : lengths[row]] = sequences[i]
        buckets.append((index, padded, lengths))
        start = stop
    return buckets


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
    remaining = np.concatenate(([1.0], np.cumprod(1.0 - v)))
    return normalize(np.concatenate((v * remaining[:-1], remaining[-1:])))


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


def sample_dirichlet(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Muestra Dirichlet por la última dimensión de `alpha`, sin ceros exactos.

    Con concentraciones pequeñas (p. ej. alpha*beta_k de un estado sin uso) los gammas
    subdesbordan y `rng.dirichlet` devuelve ceros exactos, que vuelven -inf cualquier
    log densidad posterior. Aquí se trabaja en log: Gamma(a) = Gamma(a+1) * U^(1/a),
    se normaliza con logsumexp y se aplica un piso explícito de `np.finfo(float).tiny`
    (~2.2e-308): cambia la masa total en menos de K * 2.2e-308.
    """

    alpha = np.asarray(alpha, dtype=float)
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("Las concentraciones de la Dirichlet deben ser finitas y > 0.")
    log_gamma = np.log(rng.gamma(alpha + 1.0)) + np.log(rng.random(alpha.shape)) / alpha
    log_probs = log_gamma - scipy_logsumexp(log_gamma, axis=-1, keepdims=True)
    probs = np.maximum(np.exp(log_probs), np.finfo(float).tiny)
    return probs / probs.sum(axis=-1, keepdims=True)


def dirichlet_logpdf(x: np.ndarray, alpha: np.ndarray) -> float:
    """Log densidad de Dirichlet en el interior del simplex.

    Un cero en `x` no es un punto donde la densidad valga -inf: casi siempre es un
    subdesbordamiento de una coordenada positiva, y la densidad ahí depende de cuán
    pequeña era. Por eso se rechaza en vez de devolver -inf en silencio.
    """

    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha <= 0.0):
        raise ValueError("Las concentraciones de la Dirichlet deben ser > 0.")
    if np.any(x <= 0.0):
        raise ValueError("dirichlet_logpdf recibió coordenadas <= 0 (¿subdesbordamiento?).")
    total = float(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)))
    total += float(np.sum((alpha - 1.0) * np.log(x)))
    return total


def count_transitions(states: np.ndarray, n_states: int) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    if states.size < 2:
        return np.zeros((n_states, n_states), dtype=int)
    flat = np.bincount(states[:-1] * n_states + states[1:], minlength=n_states * n_states)
    return flat.reshape(n_states, n_states)


def count_emissions(states: np.ndarray, observations: np.ndarray, n_states: int, vocab_size: int) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    observations = np.asarray(observations, dtype=int)
    flat = np.bincount(states * vocab_size + observations, minlength=n_states * vocab_size)
    return flat.reshape(n_states, vocab_size)


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

