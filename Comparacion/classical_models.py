from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from next_token_experiment.data.dataset import build_evaluation_slices
from next_token_experiment.schemas import PreparedPiece
from src.data.observations import PITCH_CLASS_NAMES, ObservationSequence
from src.models.hdp_hmm import TruncatedHDPHMM
from src.models.inference import forward_log_likelihood

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

    context = np.array([context_token], dtype=int)
    context_log_likelihood, _ = forward_log_likelihood(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        observations=context,
    )
    if not target_tokens:
        return 0.0
    joint_log_likelihood, _ = forward_log_likelihood(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        observations=np.array([context_token, *target_tokens], dtype=int),
    )
    return float(joint_log_likelihood - context_log_likelihood)


def _backward_log_probs(transition_matrix: np.ndarray, emission_log_probs: np.ndarray) -> np.ndarray:
    n_states, n_steps = emission_log_probs.shape
    beta = np.zeros((n_steps, n_states), dtype=float)
    log_transition = np.log(transition_matrix + EPSILON)
    for step in range(n_steps - 2, -1, -1):
        scores = log_transition + emission_log_probs[:, step + 1][None, :] + beta[step + 1][None, :]
        beta[step] = _logsumexp(scores, axis=1)
    return beta


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.sum(stable, axis=axis, keepdims=True)
    result = max_values + np.log(np.maximum(summed, EPSILON))
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


def _piece_metrics_from_log_likelihood(piece: PreparedPiece, log_likelihood: float) -> dict[str, float | int | str]:
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
    }


@dataclass(frozen=True)
class FiniteHMMFitResult:
    selected_states: int
    validation_nll: float
    train_log: list[dict[str, float | int]]
    train_wall_clock_s: float


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

    def _score_piece(self, piece: PreparedPiece, bos_token_id: int, max_context_length: int) -> float:
        if self.initial_probs is None or self.transition_matrix is None or self.emission_matrix is None:
            raise ValueError("Model must be fitted before evaluation.")
        return sum(
            conditional_log_likelihood(
                initial_probs=self.initial_probs,
                transition_matrix=self.transition_matrix,
                emission_matrix=self.emission_matrix,
                context_token=bos_token_id,
                target_tokens=[int(token) for token in piece.tokens[start:stop]],
            )
            for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length)
        )

    def _evaluate_split_nll(
        self,
        pieces: list[PreparedPiece],
        bos_token_id: int,
        max_context_length: int,
    ) -> float:
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in pieces:
            total_log_likelihood += self._score_piece(piece, bos_token_id, max_context_length)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float | int]]]:
        sequences = [_augment_piece_tokens(piece, bos_token_id) for piece in train_pieces]
        initial_probs, transition_matrix, emission_matrix = self._initialize_parameters(n_states, rng)
        best_validation_nll = math.inf
        best_params = (initial_probs.copy(), transition_matrix.copy(), emission_matrix.copy())
        train_log: list[dict[str, float | int]] = []

        for iteration in range(1, self.max_iterations + 1):
            initial_counts = np.full(n_states, 1e-3, dtype=float)
            transition_counts = np.full((n_states, n_states), 1e-3, dtype=float)
            emission_counts = np.full((n_states, self.vocab_size), 1e-3, dtype=float)
            train_log_likelihood = 0.0

            log_transition = np.log(transition_matrix + EPSILON)
            log_emission = np.log(emission_matrix + EPSILON)
            log_initial = np.log(initial_probs + EPSILON)

            for tokens in sequences:
                observations = np.array(tokens, dtype=int)
                emission_log_probs = log_emission[:, observations]
                alpha = np.full((len(observations), n_states), -np.inf, dtype=float)
                alpha[0] = log_initial + emission_log_probs[:, 0]
                for step in range(1, len(observations)):
                    scores = alpha[step - 1][:, None] + log_transition
                    alpha[step] = emission_log_probs[:, step] + _logsumexp(scores, axis=0)
                log_likelihood = float(_logsumexp(alpha[-1], axis=0))
                beta = _backward_log_probs(transition_matrix, emission_log_probs)
                gamma = np.exp(alpha + beta - log_likelihood)
                train_log_likelihood += log_likelihood

                initial_counts += gamma[0]
                for step, token in enumerate(observations):
                    emission_counts[:, int(token)] += gamma[step]
                for step in range(len(observations) - 1):
                    scores = (
                        alpha[step][:, None]
                        + log_transition
                        + emission_log_probs[:, step + 1][None, :]
                        + beta[step + 1][None, :]
                        - log_likelihood
                    )
                    transition_counts += np.exp(scores)

            initial_probs = initial_counts / np.maximum(initial_counts.sum(), EPSILON)
            transition_matrix = _normalize_rows(transition_counts)
            emission_matrix = _normalize_rows(emission_counts)

            self.initial_probs = initial_probs
            self.transition_matrix = transition_matrix
            self.emission_matrix = emission_matrix
            validation_nll = self._evaluate_split_nll(validation_pieces, bos_token_id, max_context_length)
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

        return best_params[0], best_params[1], best_params[2], best_validation_nll, train_log

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

        for n_states in self.candidate_num_states:
            params = self._fit_candidate(
                train_pieces=train_pieces,
                validation_pieces=validation_pieces,
                n_states=n_states,
                bos_token_id=bos_token_id,
                max_context_length=max_context_length,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            initial_probs, transition_matrix, emission_matrix, validation_nll, train_log = params
            if validation_nll < best_validation_nll:
                best_validation_nll = validation_nll
                best_candidate = (
                    n_states,
                    initial_probs.copy(),
                    transition_matrix.copy(),
                    emission_matrix.copy(),
                )
                best_log = train_log

        if best_candidate is None:
            raise ValueError("Finite HMM fitting did not produce a selected state count.")
        self.selected_states, self.initial_probs, self.transition_matrix, self.emission_matrix = best_candidate

        self.fit_result = FiniteHMMFitResult(
            selected_states=self.selected_states,
            validation_nll=best_validation_nll,
            train_log=best_log,
            train_wall_clock_s=time.perf_counter() - start_time,
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
            log_likelihood = self._score_piece(piece, bos_token_id, max_context_length)
            piece_metrics.append(_piece_metrics_from_log_likelihood(piece, log_likelihood))
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
            },
            "piece_metrics": piece_metrics,
            "train_log": self.fit_result.train_log,
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

    def _score_piece(
        self,
        piece: PreparedPiece,
        bos_token_id: int,
        max_context_length: int,
        *,
        result=None,
    ) -> float:
        selected_result = result if result is not None else self.best_result
        if selected_result is None:
            raise ValueError("Model must be fitted before scoring.")
        return sum(
            conditional_log_likelihood(
                initial_probs=selected_result.posterior_initial_mean,
                transition_matrix=selected_result.posterior_transition_mean,
                emission_matrix=selected_result.posterior_emission_mean,
                context_token=bos_token_id,
                target_tokens=[int(token) for token in piece.tokens[start:stop]],
            )
            for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length)
        )

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
            total_log_likelihood += self._score_piece(
                piece,
                bos_token_id,
                max_context_length,
                result=result,
            )
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
            result = model.fit_sequences(train_sequences)
            validation_nll = self._evaluate_validation_nll(
                validation_pieces,
                bos_token_id,
                max_context_length,
                result=result,
            )
            train_log.append(
                {
                    "candidate_index": index,
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

        if self.best_result is None or self.best_hyperparameters is None or self.validation_nll_per_token is None:
            raise ValueError("HDP-HMM fitting did not yield a valid result.")

        self.train_time_sec = time.perf_counter() - start_time
        return {
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
            log_likelihood = self._score_piece(piece, bos_token_id, max_context_length)
            piece_metrics.append(_piece_metrics_from_log_likelihood(piece, log_likelihood))
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
                "alpha": self.best_hyperparameters[0],
                "alpha0": self.best_hyperparameters[1],
                "gamma": self.best_hyperparameters[2],
            },
            "piece_metrics": piece_metrics,
        }
