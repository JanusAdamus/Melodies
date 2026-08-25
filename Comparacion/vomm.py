"""PPM-inspired VOMM control over musical symbols plus BOS, excluding PAD.

This lightweight diagnostic is not an IDyOM implementation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from numbers import Integral, Real

import numpy as np

from next_token_experiment.data.dataset import build_evaluation_slices
from next_token_experiment.schemas import PreparedPiece


EPSILON = 1e-12


def _validated_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    integer_value = int(value)
    if integer_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer_value


def _validated_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    float_value = float(value)
    if not math.isfinite(float_value) or float_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return float_value


class VariableOrderMarkovModel:
    """Interpolated variable-order Markov diagnostic inspired by PPM, not IDyOM.

    ``vocabulary_size`` is the shared comparison support: musical symbol IDs
    ``0 <= token < bos_token_id`` followed by the explicitly supplied BOS ID.
    The required equality ``vocabulary_size == bos_token_id + 1`` excludes PAD
    and any other tokenizer-only symbols. Sequences passed to :meth:`fit`
    contain musical targets only; BOS is added separately to every piece as
    context and is never scored as a target.
    """

    def __init__(
        self,
        max_order: int,
        vocabulary_size: int,
        bos_token_id: int,
        alpha: float = 0.5,
        backoff_strength: float = 1.0,
    ) -> None:
        self.max_order = _validated_integer("max_order", max_order, minimum=0)
        self.vocabulary_size = _validated_integer("vocabulary_size", vocabulary_size, minimum=2)
        self.bos_token_id = _validated_integer("bos_token_id", bos_token_id, minimum=1)
        if self.vocabulary_size != self.bos_token_id + 1:
            raise ValueError(
                "VOMM comparison support must contain musical symbols plus BOS only: "
                "vocabulary_size must equal bos_token_id + 1."
            )
        self.alpha = _validated_positive_finite("alpha", alpha)
        self.backoff_strength = _validated_positive_finite("backoff_strength", backoff_strength)
        self.musical_vocabulary_size = self.bos_token_id
        self._next_token_counts: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
        self._fitted = False
        self.selected_order = self.max_order
        self.validation_nll_per_token: float | None = None
        self.fit_wall_clock_s: float | None = None
        self.selection_wall_clock_s: float | None = None
        self.selected_fit_wall_clock_s: float | None = None
        self.selected_validation_evaluation_wall_clock_s: float | None = None
        self.selection_log: list[dict[str, float | int]] = []

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
            musical_tokens = [self._validated_musical_target(token) for token in sequence]
            tokens = [self.bos_token_id, *musical_tokens]
            for target_index in range(1, len(tokens)):
                target = tokens[target_index]
                for order in range(min(self.max_order, target_index) + 1):
                    context = tuple(tokens[target_index - order : target_index]) if order else ()
                    tables[order][context][target] += 1.0

        self._next_token_counts = {order: dict(contexts) for order, contexts in tables.items()}
        self.fit_wall_clock_s = time.perf_counter() - start_time
        self._fitted = True
        self.selected_order = self.max_order
        self.validation_nll_per_token = None
        self.selection_wall_clock_s = None
        self.selected_fit_wall_clock_s = None
        self.selected_validation_evaluation_wall_clock_s = None
        self.selection_log = []
        return self

    def _validated_token(self, token: object, *, field: str) -> int:
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError(f"{field} tokens must be integers.")
        token_id = int(token)
        if token_id < 0 or token_id >= self.vocabulary_size:
            raise ValueError(f"{field} contains a token outside vocabulary_size.")
        return token_id

    def _validated_musical_target(self, token: object) -> int:
        token_id = self._validated_token(token, field="Musical targets")
        if token_id >= self.bos_token_id:
            raise ValueError("Musical targets must satisfy 0 <= token < bos_token_id.")
        return token_id

    def _validated_context(self, context: list[int]) -> list[int]:
        validated = [self._validated_token(token, field="Prediction context") for token in context]
        bos_positions = [index for index, token in enumerate(validated) if token == self.bos_token_id]
        if bos_positions and bos_positions != [0]:
            raise ValueError("BOS may appear at most once, at the start of a prediction context.")
        return validated

    def _smoothed_distribution(self, next_counts: np.ndarray) -> np.ndarray:
        context_count = float(next_counts.sum())
        return (next_counts + self.alpha) / (context_count + self.alpha * self.vocabulary_size)

    def predict_distribution(self, context: list[int]) -> np.ndarray:
        """Predict a complete next-token distribution with suffix backoff."""

        if not self._fitted:
            raise ValueError("Fit the VOMM before prediction.")
        validated_context = self._validated_context(context)

        unigram_counts = self._next_token_counts[0].get(())
        if unigram_counts is None:
            distribution = np.full(self.vocabulary_size, 1.0 / self.vocabulary_size, dtype=float)
        else:
            distribution = self._smoothed_distribution(unigram_counts)

        for order in range(1, min(self.max_order, len(validated_context)) + 1):
            next_counts = self._next_token_counts[order].get(tuple(validated_context[-order:]))
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

    def _score_piece(self, piece: PreparedPiece, max_context_length: int) -> dict[str, object]:
        log_likelihood = 0.0
        correct = 0
        brier_score_sum = 0.0
        n_tokens = 0
        scored_event_indices: list[int] = []
        for start, stop in build_evaluation_slices(len(piece.tokens), max_context_length):
            context = [self.bos_token_id]
            for event_index, token in enumerate(piece.tokens[start:stop], start=start):
                target = self._validated_musical_target(token)
                distribution = self.predict_distribution(context)
                log_likelihood += math.log(max(float(distribution[target]), EPSILON))
                correct += int(int(np.argmax(distribution)) == target)
                one_hot = np.zeros(self.vocabulary_size, dtype=float)
                one_hot[target] = 1.0
                brier_score_sum += float(np.square(distribution - one_hot).sum())
                n_tokens += 1
                scored_event_indices.append(event_index)
                context.append(target)
        return {
            "log_likelihood": log_likelihood,
            "correct": correct,
            "brier_score_sum": brier_score_sum,
            "n_tokens": n_tokens,
            "scored_event_indices": scored_event_indices,
        }

    def evaluate(self, pieces: list[PreparedPiece], max_context_length: int) -> dict[str, object]:
        """Score every musical event once with BOS reset at each shared slice."""

        if not self._fitted or self.fit_wall_clock_s is None:
            raise ValueError("Fit the VOMM before evaluation.")

        start_time = time.perf_counter()
        piece_metrics: list[dict[str, float | int | str]] = []
        total_log_likelihood = 0.0
        total_correct = 0
        total_brier_score = 0.0
        total_tokens = 0
        for piece in pieces:
            score = self._score_piece(piece, max_context_length)
            log_likelihood = float(score["log_likelihood"])
            n_tokens = int(score["n_tokens"])
            nll_per_token = -log_likelihood / max(1, n_tokens)
            accuracy = int(score["correct"]) / max(1, n_tokens)
            brier_score = float(score["brier_score_sum"]) / max(1, n_tokens)
            piece_metrics.append(
                {
                    "piece_id": piece.piece_id,
                    "title": piece.title,
                    "composer": piece.composer,
                    "n_tokens": n_tokens,
                    "log_likelihood": log_likelihood,
                    "nll_per_token": nll_per_token,
                    "perplexity": math.exp(nll_per_token),
                    "accuracy": accuracy,
                    "brier_score": brier_score,
                    "scored_event_indices": list(score["scored_event_indices"]),
                }
            )
            total_log_likelihood += log_likelihood
            total_correct += int(score["correct"])
            total_brier_score += float(score["brier_score_sum"])
            total_tokens += n_tokens

        nll_per_token = -total_log_likelihood / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)
        brier_score = total_brier_score / max(1, total_tokens)
        evaluation_wall_clock_s = time.perf_counter() - start_time
        total_fit_wall_clock_s = self.selection_wall_clock_s or self.fit_wall_clock_s
        selected_fit_wall_clock_s = self.selected_fit_wall_clock_s or self.fit_wall_clock_s
        return {
            "summary": {
                "model": "vomm",
                "selected_order": self.selected_order,
                "validation_nll_per_token": self.validation_nll_per_token,
                "test_nll_per_token": nll_per_token,
                "test_perplexity": math.exp(nll_per_token),
                "accuracy": accuracy,
                "test_accuracy": accuracy,
                "brier_score": brier_score,
                "test_brier_score": brier_score,
                "n_tokens": total_tokens,
                "n_params": self.count_table_size,
                "count_table_size": self.count_table_size,
                "train_time_sec": total_fit_wall_clock_s,
                "fit_wall_clock_s": total_fit_wall_clock_s,
                "selection_wall_clock_s": self.selection_wall_clock_s,
                "selected_fit_wall_clock_s": selected_fit_wall_clock_s,
                "selected_validation_evaluation_wall_clock_s": self.selected_validation_evaluation_wall_clock_s,
                # Nombre del contrato común de costo compartido con HMM, HDP-HMM
                # y transformador.
                "selected_validation_wall_clock_s": self.selected_validation_evaluation_wall_clock_s,
                "evaluation_wall_clock_s": evaluation_wall_clock_s,
            },
            "piece_metrics": piece_metrics,
            "selection_log": list(self.selection_log),
        }


def select_vomm_by_validation(
    train_sequences: list[list[int]],
    validation_pieces: list[PreparedPiece],
    *,
    candidate_orders: tuple[int, ...],
    vocabulary_size: int,
    bos_token_id: int,
    alpha: float = 0.5,
    backoff_strength: float = 1.0,
    max_context_length: int = 128,
) -> VariableOrderMarkovModel:
    """Select a VOMM over musical symbols plus explicit BOS, with PAD excluded."""

    if not candidate_orders:
        raise ValueError("candidate_orders must not be empty.")
    if sum(len(piece.tokens) for piece in validation_pieces) == 0:
        raise ValueError("validation_pieces must contain at least one musical token.")

    selection_start = time.perf_counter()
    best_model: VariableOrderMarkovModel | None = None
    best_validation_nll = math.inf
    best_validation_evaluation_wall_clock_s: float | None = None
    selection_log: list[dict[str, float | int]] = []
    for order in candidate_orders:
        candidate = VariableOrderMarkovModel(
            max_order=order,
            vocabulary_size=vocabulary_size,
            bos_token_id=bos_token_id,
            alpha=alpha,
            backoff_strength=backoff_strength,
        ).fit(train_sequences)
        validation_evaluation = candidate.evaluate(validation_pieces, max_context_length)
        validation_summary = validation_evaluation["summary"]
        validation_nll = float(validation_summary["test_nll_per_token"])
        validation_evaluation_wall_clock_s = float(validation_summary["evaluation_wall_clock_s"])
        selection_log.append(
            {
                "candidate_order": candidate.max_order,
                "validation_nll_per_token": validation_nll,
                "fit_wall_clock_s": float(candidate.fit_wall_clock_s),
                "validation_evaluation_wall_clock_s": validation_evaluation_wall_clock_s,
                "count_table_size": candidate.count_table_size,
            }
        )
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_model = candidate
            best_validation_evaluation_wall_clock_s = validation_evaluation_wall_clock_s

    if best_model is None or best_validation_evaluation_wall_clock_s is None:
        raise ValueError("VOMM validation did not produce a candidate.")
    selection_wall_clock_s = time.perf_counter() - selection_start
    best_model.selected_order = best_model.max_order
    best_model.validation_nll_per_token = best_validation_nll
    best_model.selection_wall_clock_s = selection_wall_clock_s
    best_model.selected_fit_wall_clock_s = best_model.fit_wall_clock_s
    best_model.selected_validation_evaluation_wall_clock_s = best_validation_evaluation_wall_clock_s
    best_model.selection_log = selection_log
    return best_model
