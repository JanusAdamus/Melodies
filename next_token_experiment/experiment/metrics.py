from __future__ import annotations

import math


def nll_per_token(probabilities: list[float]) -> float:
    """Compute mean negative log-likelihood from token probabilities."""

    if not probabilities:
        raise ValueError("At least one probability is required.")
    return -sum(math.log(max(probability, 1e-12)) for probability in probabilities) / len(probabilities)


def perplexity_from_nll(nll_value: float) -> float:
    return math.exp(nll_value)


def accuracy_from_predictions(predictions: list[int], targets: list[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if not targets:
        raise ValueError("Targets cannot be empty.")
    hits = sum(int(prediction == target) for prediction, target in zip(predictions, targets))
    return hits / len(targets)


def summarize_average(total: float, count: int) -> float:
    if count <= 0:
        raise ValueError("Count must be positive.")
    return total / count
