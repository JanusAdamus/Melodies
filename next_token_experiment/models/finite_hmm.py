from __future__ import annotations

from dataclasses import dataclass

from .base import NextTokenModel


@dataclass(frozen=True)
class FiniteHMMStudySpec:
    candidate_num_states: tuple[int, ...]
    max_em_iterations: int
    tolerance: float


class FiniteHMMNextTokenModel(NextTokenModel):
    """Placeholder for the predictive finite HMM used in the thesis protocol."""

    def __init__(self, spec: FiniteHMMStudySpec) -> None:
        self.spec = spec

    def fit(self, train_data, validation_data) -> dict:
        raise NotImplementedError("Implement after the data pipeline is validated end-to-end.")

    def evaluate(self, test_data) -> dict:
        raise NotImplementedError("Implement after the predictive HMM training loop exists.")
