"""A small PPM-inspired VOMM diagnostic control, not an IDyOM implementation."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from next_token_experiment.data.dataset import build_evaluation_slices
from next_token_experiment.schemas import PreparedPiece


EPSILON = 1e-12


class VariableOrderMarkovModel:
    """Interpolated variable-order Markov diagnostic inspired by PPM, not IDyOM.

    Musical token sequences passed to :meth:`fit` exclude BOS.  The model adds
    ``vocabulary_size - 1`` as BOS separately to every piece, so it is context
    during fitting and evaluation but never a scored musical target.
    """

    def __init__(
        self,
        max_order: int,
        vocabulary_size: int,
        alpha: float = 0.5,
        backoff_strength: float = 1.0,
    ) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must include at least one token and BOS.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if backoff_strength <= 0.0:
            raise ValueError("backoff_strength must be positive.")

        self.max_order = int(max_order)
        self.vocabulary_size = int(vocabulary_size)
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.bos_token_id = self.vocabulary_size - 1
        self._next_token_counts: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
        self._fitted = False
        self.selected_order = self.max_order
        self.validation_nll_per_token: float | None = None
        self.fit_wall_clock_s: float | None = None

    @property
    def count_table_size(self) -> int:
        """Return the number of observed context/next-token count cells."""

        return sum(
            int(np.count_nonzero(next_counts))
            for tables in self._next_token_counts.values()
            for next_counts in tables.values()
        )

    def fit(self, sequences: list[list[int]]) -> VariableOrderMarkovModel:
        """Count all orders independently without joining one piece to another."""

        start_time = time.perf_counter()
        tables: dict[int, defaultdict[tuple[int, ...], np.ndarray]] = {
            order: defaultdict(lambda: np.zeros(self.vocabulary_size, dtype=float))
            for order in range(self.max_order + 1)
        }
        for sequence in sequences:
            tokens = [self.bos_token_id, *(int(token) for token in sequence)]
            if any(token < 0 or token >= self.vocabulary_size for token in tokens):
                raise ValueError("Sequences contain a token outside vocabulary_size.")
            for target_index in range(1, len(tokens)):
                target = tokens[target_index]
                for order in range(min(self.max_order, target_index) + 1):
                    context = tuple(tokens[target_index - order : target_index]) if order else ()
                    tables[order][context][target] += 1.0

        self._next_token_counts = {order: dict(contexts) for order, contexts in tables.items()}
        self.fit_wall_clock_s = time.perf_counter() - start_time
        self._fitted = True
        return self

    def _smoothed_distribution(self, next_counts: np.ndarray) -> np.ndarray:
        context_count = float(next_counts.sum())
        return (next_counts + self.alpha) / (context_count + self.alpha * self.vocabulary_size)

    def predict_distribution(self, context: list[int]) -> np.ndarray:
        """Predict a complete next-token distribution with suffix backoff."""

        if not self._fitted:
            raise ValueError("Fit the VOMM before prediction.")

        unigram_counts = self._next_token_counts[0].get(())
        if unigram_counts is None:
            distribution = np.full(self.vocabulary_size, 1.0 / self.vocabulary_size, dtype=float)
        else:
            distribution = self._smoothed_distribution(unigram_counts)

        for order in range(1, min(self.max_order, len(context)) + 1):
            next_counts = self._next_token_counts[order].get(tuple(context[-order:]))
            if next_counts is None:
                continue
            context_count = float(next_counts.sum())
            weight = context_count / (context_count + self.backoff_strength)
            local = self._smoothed_distribution(next_counts)
            distribution = weight * local + (1.0 - weight) * distribution

        total = float(distribution.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(self.vocabulary_size, 1.0 / self.vocabulary_size, dtype=float)
        return distribution / total

    def _score_piece(self, piece: PreparedPiece, max_context_length: int) -> float:
        log_likelihood = 0.0
        for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length):
            context = [self.bos_token_id]
            for token in piece.tokens[start:stop]:
                target = int(token)
                if target < 0 or target >= self.vocabulary_size:
                    raise ValueError("Piece contains a token outside vocabulary_size.")
                distribution = self.predict_distribution(context)
                log_likelihood += math.log(max(float(distribution[target]), EPSILON))
                context.append(target)
        return log_likelihood

    def evaluate(self, pieces: list[PreparedPiece], max_context_length: int) -> dict[str, object]:
        """Score every musical event once with BOS reset at each shared slice."""

        if not self._fitted or self.fit_wall_clock_s is None:
            raise ValueError("Fit the VOMM before evaluation.")

        start_time = time.perf_counter()
        piece_metrics: list[dict[str, float | int | str]] = []
        total_log_likelihood = 0.0
        total_tokens = 0
        for piece in pieces:
            log_likelihood = self._score_piece(piece, max_context_length)
            n_tokens = len(piece.tokens)
            nll_per_token = -log_likelihood / max(1, n_tokens)
            piece_metrics.append(
                {
                    "piece_id": piece.piece_id,
                    "title": piece.title,
                    "composer": piece.composer,
                    "n_tokens": n_tokens,
                    "log_likelihood": log_likelihood,
                    "nll_per_token": nll_per_token,
                    "perplexity": math.exp(nll_per_token),
                }
            )
            total_log_likelihood += log_likelihood
            total_tokens += n_tokens

        nll_per_token = -total_log_likelihood / max(1, total_tokens)
        evaluation_wall_clock_s = time.perf_counter() - start_time
        return {
            "summary": {
                "model": "vomm",
                "selected_order": self.selected_order,
                "validation_nll_per_token": self.validation_nll_per_token,
                "test_nll_per_token": nll_per_token,
                "test_perplexity": math.exp(nll_per_token),
                "n_tokens": total_tokens,
                "n_params": self.count_table_size,
                "count_table_size": self.count_table_size,
                "train_time_sec": self.fit_wall_clock_s,
                "fit_wall_clock_s": self.fit_wall_clock_s,
                "evaluation_wall_clock_s": evaluation_wall_clock_s,
            },
            "piece_metrics": piece_metrics,
        }


def select_vomm_by_validation(
    train_sequences: list[list[int]],
    validation_pieces: list[PreparedPiece],
    *,
    candidate_orders: tuple[int, ...],
    vocabulary_size: int,
    alpha: float = 0.5,
    backoff_strength: float = 1.0,
    max_context_length: int = 128,
) -> VariableOrderMarkovModel:
    """Fit VOMM orders and retain the diagnostic control with lowest validation NLL."""

    if not candidate_orders:
        raise ValueError("candidate_orders must not be empty.")

    best_model: VariableOrderMarkovModel | None = None
    best_validation_nll = math.inf
    for order in candidate_orders:
        candidate = VariableOrderMarkovModel(
            max_order=order,
            vocabulary_size=vocabulary_size,
            alpha=alpha,
            backoff_strength=backoff_strength,
        ).fit(train_sequences)
        validation_nll = float(
            candidate.evaluate(validation_pieces, max_context_length)["summary"]["test_nll_per_token"]
        )
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_model = candidate

    if best_model is None:
        raise ValueError("VOMM validation did not produce a candidate.")
    best_model.selected_order = best_model.max_order
    best_model.validation_nll_per_token = best_validation_nll
    return best_model
