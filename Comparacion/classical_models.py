from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from next_token_experiment.data.dataset import build_evaluation_slices
from next_token_experiment.schemas import PreparedPiece
from src.data.observations import PITCH_CLASS_NAMES, ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
from src.models.inference import scaled_forward_log_likelihood
from src.models.utils import length_buckets

EPSILON = 1e-12
BOS_TOKEN = "<BOS>"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return matrix / row_sums


def _augment_piece_tokens(piece: PreparedPiece, bos_token_id: int) -> list[int]:
    return [bos_token_id] + [int(token) for token in piece.tokens]


def _build_observation_sequence(tokens: list[int], vocabulary: list[str]) -> ObservationSequence:
    return ObservationSequence(
        observation_type="pitch_class",
        tokens=np.array(tokens, dtype=int),
        vocabulary=vocabulary,
        decoded=[vocabulary[token] for token in tokens],
        events=[],
        extra={},
    )


def conditional_log_likelihood(
    *,
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    context_token: int,
    target_tokens: list[int],
) -> float:
    """Score targets conditioned on a context token without scoring the context."""

    if not target_tokens:
        return 0.0
    context_log_likelihood = scaled_forward_log_likelihood(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        observations=np.array([context_token], dtype=int),
    )
    joint_log_likelihood = scaled_forward_log_likelihood(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        observations=np.array([context_token, *target_tokens], dtype=int),
    )
    return float(joint_log_likelihood - context_log_likelihood)


def _pad_sequences(sequences: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lotes rectangulares (padded, lengths) agrupados por longitud."""

    return [(padded, lengths) for _, padded, lengths in length_buckets(sequences)]


def _expectation_batch(
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    padded: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Paso E de Baum-Welch con reescalado (Rabiner) sobre un lote de secuencias.

    Equivale al recorrido por secuencia en dominio logaritmico, pero cada paso temporal
    es un producto matricial en vez de un logsumexp sobre un bloque K*K.
    """

    n_sequences, max_steps = padded.shape
    n_states = transition_matrix.shape[0]
    live = np.arange(max_steps)[None, :] < lengths[:, None]

    emission = emission_matrix[:, padded].transpose(1, 0, 2)  # (B, K, T)

    alpha = np.zeros((n_sequences, max_steps, n_states), dtype=float)
    scale = np.ones((n_sequences, max_steps), dtype=float)

    current = initial_probs[None, :] * emission[:, :, 0]
    scale[:, 0] = np.maximum(current.sum(axis=1), EPSILON)
    current = current / scale[:, 0][:, None]
    alpha[:, 0] = current
    for step in range(1, max_steps):
        moved = (current @ transition_matrix) * emission[:, :, step]
        step_scale = np.maximum(moved.sum(axis=1), EPSILON)
        moved = moved / step_scale[:, None]
        mask = live[:, step]
        current = np.where(mask[:, None], moved, current)
        scale[:, step] = np.where(mask, step_scale, 1.0)
        alpha[:, step] = current

    log_likelihood = float(np.sum(np.log(scale)))

    beta = np.ones((n_sequences, max_steps, n_states), dtype=float)
    for step in range(max_steps - 2, -1, -1):
        nxt = live[:, step + 1]
        weighted = emission[:, :, step + 1] * beta[:, step + 1] / scale[:, step + 1][:, None]
        beta[:, step] = np.where(nxt[:, None], weighted @ transition_matrix.T, 1.0)

    gamma = alpha * beta * live[:, :, None]

    initial_counts = gamma[:, 0].sum(axis=0)

    emission_counts = np.zeros((emission_matrix.shape[1], n_states), dtype=float)
    np.add.at(emission_counts, padded[live], gamma[live])

    # xi[t] = alpha[t] (x) (B[:, y_{t+1}] * beta[t+1] / c[t+1]), sumado sobre t, por A.
    tail = live[:, 1:]
    left = alpha[:, :-1] * tail[:, :, None]
    right = (emission[:, :, 1:].transpose(0, 2, 1) * beta[:, 1:]
             / scale[:, 1:][:, :, None]) * tail[:, :, None]
    transition_counts = transition_matrix * np.einsum("bti,btj->ij", left, right)

    return initial_counts, transition_counts, emission_counts.T, log_likelihood


def _piece_metrics_from_log_likelihood(
    piece: PreparedPiece,
    log_likelihood: float,
    scored_event_indices: list[int],
) -> dict[str, object]:
    n_tokens = len(piece.tokens)
    nll_per_token = -float(log_likelihood) / max(1, n_tokens)
    return {
        "piece_id": piece.piece_id,
        "title": piece.title,
        "composer": piece.composer,
        "n_tokens": n_tokens,
        "log_likelihood": float(log_likelihood),
        "nll_per_token": nll_per_token,
        "perplexity": math.exp(nll_per_token),
        "scored_event_indices": list(scored_event_indices),
    }


@dataclass(frozen=True)
class FiniteHMMFitResult:
    selected_states: int
    validation_nll: float
    train_log: list[dict[str, float | int]]
    train_wall_clock_s: float
    #: Tiempo de toda la búsqueda de capacidad, incluidos los candidatos descartados.
    selection_wall_clock_s: float = 0.0
    #: Tiempo de EM del candidato que quedó seleccionado, sin su validación.
    selected_fit_wall_clock_s: float = 0.0
    #: Tiempo de puntuar validación dentro del candidato seleccionado.
    selected_validation_wall_clock_s: float = 0.0
    #: Una entrada por candidato: costo y NLL de validación.
    candidate_log: tuple[dict[str, float | int], ...] = ()


class FiniteGlobalHMM:
    def __init__(
        self,
        *,
        candidate_num_states: tuple[int, ...],
        max_iterations: int,
        tolerance: float,
        seed: int,
        vocab_size: int = 13,
    ) -> None:
        self.candidate_num_states = tuple(int(value) for value in candidate_num_states)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.initial_probs: np.ndarray | None = None
        self.transition_matrix: np.ndarray | None = None
        self.emission_matrix: np.ndarray | None = None
        self.selected_states: int | None = None
        self.fit_result: FiniteHMMFitResult | None = None

    def _initialize_parameters(self, n_states: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        initial_probs = rng.dirichlet(np.ones(n_states, dtype=float))
        transition_matrix = rng.dirichlet(np.ones(n_states, dtype=float), size=n_states)
        emission_matrix = rng.dirichlet(np.ones(self.vocab_size, dtype=float), size=n_states)
        return initial_probs, transition_matrix, emission_matrix

    def _score_piece(
        self,
        piece: PreparedPiece,
        bos_token_id: int,
        max_context_length: int,
    ) -> tuple[float, list[int]]:
        if self.initial_probs is None or self.transition_matrix is None or self.emission_matrix is None:
            raise ValueError("Model must be fitted before evaluation.")
        log_likelihood = 0.0
        scored_event_indices: list[int] = []
        for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length):
            log_likelihood += conditional_log_likelihood(
                initial_probs=self.initial_probs,
                transition_matrix=self.transition_matrix,
                emission_matrix=self.emission_matrix,
                context_token=bos_token_id,
                target_tokens=[int(token) for token in piece.tokens[start:stop]],
            )
            scored_event_indices.extend(range(start, stop))
        return log_likelihood, scored_event_indices

    def _evaluate_split_nll(
        self,
        pieces: list[PreparedPiece],
        bos_token_id: int,
        max_context_length: int,
    ) -> float:
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in pieces:
            piece_log_likelihood, _ = self._score_piece(piece, bos_token_id, max_context_length)
            total_log_likelihood += piece_log_likelihood
            total_tokens += len(piece.tokens)
        return -total_log_likelihood / max(1, total_tokens)

    def _fit_candidate(
        self,
        train_pieces: list[PreparedPiece],
        validation_pieces: list[PreparedPiece],
        *,
        n_states: int,
        bos_token_id: int,
        max_context_length: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float | int]], dict[str, float]]:
        fit_wall_clock_s = 0.0
        validation_wall_clock_s = 0.0
        sequences = [
            np.asarray(_augment_piece_tokens(piece, bos_token_id), dtype=int)
            for piece in train_pieces
        ]
        batches = _pad_sequences(sequences)
        initial_probs, transition_matrix, emission_matrix = self._initialize_parameters(n_states, rng)
        best_validation_nll = math.inf
        best_params = (initial_probs.copy(), transition_matrix.copy(), emission_matrix.copy())
        train_log: list[dict[str, float | int]] = []

        for iteration in range(1, self.max_iterations + 1):
            iteration_start = time.perf_counter()
            initial_counts = np.full(n_states, 1e-3, dtype=float)
            transition_counts = np.full((n_states, n_states), 1e-3, dtype=float)
            emission_counts = np.full((n_states, self.vocab_size), 1e-3, dtype=float)
            train_log_likelihood = 0.0

            for padded, lengths in batches:
                batch_initial, batch_transition, batch_emission, batch_log_likelihood = (
                    _expectation_batch(
                        initial_probs, transition_matrix, emission_matrix, padded, lengths
                    )
                )
                initial_counts += batch_initial
                transition_counts += batch_transition
                emission_counts += batch_emission
                train_log_likelihood += batch_log_likelihood

            initial_probs = initial_counts / np.maximum(initial_counts.sum(), EPSILON)
            transition_matrix = _normalize_rows(transition_counts)
            emission_matrix = _normalize_rows(emission_counts)

            self.initial_probs = initial_probs
            self.transition_matrix = transition_matrix
            self.emission_matrix = emission_matrix
            fit_wall_clock_s += time.perf_counter() - iteration_start
            validation_start = time.perf_counter()
            validation_nll = self._evaluate_split_nll(validation_pieces, bos_token_id, max_context_length)
            validation_wall_clock_s += time.perf_counter() - validation_start
            train_log.append(
                {
                    "iteration": iteration,
                    "n_states": n_states,
                    "train_log_likelihood": train_log_likelihood,
                    "validation_nll_per_token": validation_nll,
                }
            )
            if validation_nll + self.tolerance < best_validation_nll:
                best_validation_nll = validation_nll
                best_params = (initial_probs.copy(), transition_matrix.copy(), emission_matrix.copy())
            elif train_log and abs(train_log[-1]["validation_nll_per_token"] - best_validation_nll) <= self.tolerance:
                break

        return (
            best_params[0],
            best_params[1],
            best_params[2],
            best_validation_nll,
            train_log,
            {
                "fit_wall_clock_s": fit_wall_clock_s,
                "validation_wall_clock_s": validation_wall_clock_s,
            },
        )

    def fit(
        self,
        train_pieces: list[PreparedPiece],
        validation_pieces: list[PreparedPiece],
        *,
        bos_token_id: int,
        max_context_length: int = 128,
    ) -> FiniteHMMFitResult:
        rng = np.random.default_rng(self.seed)
        start_time = time.perf_counter()
        best_validation_nll = math.inf
        best_log: list[dict[str, float | int]] = []
        best_candidate: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        candidate_log: list[dict[str, float | int]] = []
        selected_timing = {"fit_wall_clock_s": 0.0, "validation_wall_clock_s": 0.0}

        for n_states in self.candidate_num_states:
            params = self._fit_candidate(
                train_pieces=train_pieces,
                validation_pieces=validation_pieces,
                n_states=n_states,
                bos_token_id=bos_token_id,
                max_context_length=max_context_length,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            initial_probs, transition_matrix, emission_matrix, validation_nll, train_log, timing = params
            candidate_log.append(
                {
                    "n_states": n_states,
                    "validation_nll_per_token": validation_nll,
                    "fit_wall_clock_s": timing["fit_wall_clock_s"],
                    "validation_wall_clock_s": timing["validation_wall_clock_s"],
                    "n_iterations": len(train_log),
                }
            )
            if validation_nll < best_validation_nll:
                best_validation_nll = validation_nll
                best_candidate = (
                    n_states,
                    initial_probs.copy(),
                    transition_matrix.copy(),
                    emission_matrix.copy(),
                )
                best_log = train_log
                selected_timing = timing

        if best_candidate is None:
            raise ValueError("Finite HMM fitting did not produce a selected state count.")
        self.selected_states, self.initial_probs, self.transition_matrix, self.emission_matrix = best_candidate

        selection_wall_clock_s = time.perf_counter() - start_time
        self.fit_result = FiniteHMMFitResult(
            selected_states=self.selected_states,
            validation_nll=best_validation_nll,
            train_log=best_log,
            train_wall_clock_s=selection_wall_clock_s,
            selection_wall_clock_s=selection_wall_clock_s,
            selected_fit_wall_clock_s=selected_timing["fit_wall_clock_s"],
            selected_validation_wall_clock_s=selected_timing["validation_wall_clock_s"],
            candidate_log=tuple(candidate_log),
        )
        return self.fit_result

    def evaluate(
        self,
        test_pieces: list[PreparedPiece],
        *,
        bos_token_id: int,
        max_context_length: int = 128,
    ) -> dict[str, object]:
        if self.fit_result is None or self.selected_states is None:
            raise ValueError("Fit the finite HMM before evaluation.")

        piece_metrics = []
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in test_pieces:
            log_likelihood, scored_event_indices = self._score_piece(
                piece,
                bos_token_id,
                max_context_length,
            )
            piece_metrics.append(
                _piece_metrics_from_log_likelihood(
                    piece,
                    log_likelihood,
                    scored_event_indices,
                )
            )
            total_log_likelihood += log_likelihood
            total_tokens += len(piece.tokens)

        nll_per_token = -total_log_likelihood / max(1, total_tokens)
        n_params = (
            (self.selected_states - 1)
            + self.selected_states * (self.selected_states - 1)
            + self.selected_states * (self.vocab_size - 1)
        )
        return {
            "summary": {
                "model": "finite_hmm",
                "selected_states": self.selected_states,
                "validation_nll_per_token": self.fit_result.validation_nll,
                "test_nll_per_token": nll_per_token,
                "test_perplexity": math.exp(nll_per_token),
                "n_tokens": total_tokens,
                "n_params": int(n_params),
                "train_time_sec": self.fit_result.train_wall_clock_s,
                "selection_wall_clock_s": self.fit_result.selection_wall_clock_s,
                "selected_fit_wall_clock_s": self.fit_result.selected_fit_wall_clock_s,
                "selected_validation_wall_clock_s": self.fit_result.selected_validation_wall_clock_s,
            },
            "piece_metrics": piece_metrics,
            "train_log": self.fit_result.train_log,
            "candidate_log": list(self.fit_result.candidate_log),
        }


class GlobalHDPHMM:
    def __init__(
        self,
        *,
        truncation_level: int,
        n_iters: int,
        burn_in: int,
        hyperparameter_grid: tuple[tuple[float, float, float], ...],
        seed: int,
        vocab_size: int = 13,
    ) -> None:
        self.truncation_level = int(truncation_level)
        self.n_iters = int(n_iters)
        self.burn_in = int(burn_in)
        self.hyperparameter_grid = hyperparameter_grid
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.best_result = None
        self.best_hyperparameters: tuple[float, float, float] | None = None
        self.validation_nll_per_token: float | None = None
        self.train_time_sec: float | None = None
        self.selection_wall_clock_s: float | None = None
        self.selected_fit_wall_clock_s: float | None = None
        self.selected_validation_wall_clock_s: float | None = None

    def _score_piece(
        self,
        piece: PreparedPiece,
        bos_token_id: int,
        max_context_length: int,
        *,
        result=None,
    ) -> tuple[float, list[int]]:
        selected_result = result if result is not None else self.best_result
        if selected_result is None:
            raise ValueError("Model must be fitted before scoring.")
        log_likelihood = 0.0
        scored_event_indices: list[int] = []
        for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length):
            log_likelihood += conditional_log_likelihood(
                initial_probs=selected_result.posterior_initial_mean,
                transition_matrix=selected_result.posterior_transition_mean,
                emission_matrix=selected_result.posterior_emission_mean,
                context_token=bos_token_id,
                target_tokens=[int(token) for token in piece.tokens[start:stop]],
            )
            scored_event_indices.extend(range(start, stop))
        return log_likelihood, scored_event_indices

    def _evaluate_validation_nll(
        self,
        pieces: list[PreparedPiece],
        bos_token_id: int,
        max_context_length: int,
        *,
        result,
    ) -> float:
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in pieces:
            piece_log_likelihood, _ = self._score_piece(
                piece,
                bos_token_id,
                max_context_length,
                result=result,
            )
            total_log_likelihood += piece_log_likelihood
            total_tokens += len(piece.tokens)
        return -total_log_likelihood / max(1, total_tokens)

    def fit(
        self,
        train_pieces: list[PreparedPiece],
        validation_pieces: list[PreparedPiece],
        *,
        bos_token_id: int,
        max_context_length: int = 128,
    ) -> dict[str, object]:
        vocabulary = PITCH_CLASS_NAMES + [BOS_TOKEN]
        train_sequences = [
            _build_observation_sequence(_augment_piece_tokens(piece, bos_token_id), vocabulary)
            for piece in train_pieces
        ]
        start_time = time.perf_counter()
        train_log: list[dict[str, float | int]] = []
        best_validation_nll = math.inf

        for index, (alpha, alpha0, gamma) in enumerate(self.hyperparameter_grid):
            model = TruncatedHDPHMM(
                n_states=self.truncation_level,
                alpha=alpha,
                alpha0=alpha0,
                gamma=gamma,
                eta=1.0,
                kappa=0.0,
                n_iters=self.n_iters,
                burn_in=self.burn_in,
                seed=self.seed + index,
            )
            candidate_fit_start = time.perf_counter()
            result = model.fit_sequences(train_sequences)
            candidate_fit_wall_clock_s = time.perf_counter() - candidate_fit_start
            candidate_validation_start = time.perf_counter()
            validation_nll = self._evaluate_validation_nll(
                validation_pieces,
                bos_token_id,
                max_context_length,
                result=result,
            )
            candidate_validation_wall_clock_s = time.perf_counter() - candidate_validation_start
            train_log.append(
                {
                    "candidate_index": index,
                    "fit_wall_clock_s": candidate_fit_wall_clock_s,
                    "validation_wall_clock_s": candidate_validation_wall_clock_s,
                    "alpha": alpha,
                    "alpha0": alpha0,
                    "gamma": gamma,
                    "validation_nll_per_token": validation_nll,
                    "best_log_likelihood": float(result.log_likelihood),
                    "effective_states": int(result.effective_states),
                }
            )
            if validation_nll < best_validation_nll:
                best_validation_nll = validation_nll
                self.best_hyperparameters = (alpha, alpha0, gamma)
                self.validation_nll_per_token = validation_nll
                self.best_result = result
                self.selected_fit_wall_clock_s = candidate_fit_wall_clock_s
                self.selected_validation_wall_clock_s = candidate_validation_wall_clock_s

        if self.best_result is None or self.best_hyperparameters is None or self.validation_nll_per_token is None:
            raise ValueError("HDP-HMM fitting did not yield a valid result.")

        self.train_time_sec = time.perf_counter() - start_time
        self.selection_wall_clock_s = self.train_time_sec
        return {
            "selection_wall_clock_s": self.selection_wall_clock_s,
            "selected_fit_wall_clock_s": self.selected_fit_wall_clock_s,
            "selected_validation_wall_clock_s": self.selected_validation_wall_clock_s,
            "selected_hyperparameters": {
                "alpha": self.best_hyperparameters[0],
                "alpha0": self.best_hyperparameters[1],
                "gamma": self.best_hyperparameters[2],
            },
            "validation_nll_per_token": self.validation_nll_per_token,
            "train_time_sec": self.train_time_sec,
            "train_log": train_log,
        }

    def evaluate(
        self,
        test_pieces: list[PreparedPiece],
        *,
        bos_token_id: int,
        max_context_length: int = 128,
    ) -> dict[str, object]:
        if self.best_result is None or self.best_hyperparameters is None or self.validation_nll_per_token is None or self.train_time_sec is None:
            raise ValueError("Fit the HDP-HMM before evaluation.")

        piece_metrics = []
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in test_pieces:
            log_likelihood, scored_event_indices = self._score_piece(
                piece,
                bos_token_id,
                max_context_length,
            )
            piece_metrics.append(
                _piece_metrics_from_log_likelihood(
                    piece,
                    log_likelihood,
                    scored_event_indices,
                )
            )
            total_log_likelihood += log_likelihood
            total_tokens += len(piece.tokens)

        nll_per_token = -total_log_likelihood / max(1, total_tokens)
        n_params = (
            (self.truncation_level - 1)
            + self.truncation_level * (self.truncation_level - 1)
            + self.truncation_level * (self.vocab_size - 1)
        )
        return {
            "summary": {
                "model": "hdp_hmm",
                "truncation_level": self.truncation_level,
                "validation_nll_per_token": self.validation_nll_per_token,
                "test_nll_per_token": nll_per_token,
                "test_perplexity": math.exp(nll_per_token),
                "n_tokens": total_tokens,
                "n_params": int(n_params),
                "effective_states": int(self.best_result.effective_states),
                "train_time_sec": self.train_time_sec,
                "selection_wall_clock_s": self.selection_wall_clock_s,
                "selected_fit_wall_clock_s": self.selected_fit_wall_clock_s,
                "selected_validation_wall_clock_s": self.selected_validation_wall_clock_s,
                "alpha": self.best_hyperparameters[0],
                "alpha0": self.best_hyperparameters[1],
                "gamma": self.best_hyperparameters[2],
            },
            "piece_metrics": piece_metrics,
        }
