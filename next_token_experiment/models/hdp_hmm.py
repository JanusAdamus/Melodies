from __future__ import annotations

from dataclasses import dataclass

from .base import NextTokenModel


@dataclass(frozen=True)
class HDPHMMStudySpec:
    truncation_level: int
    n_iters: int
    burn_in: int
    alpha: float
    alpha0: float
    gamma: float
    eta: float
    kappa: float
    seed: int


class HDPHMMNextTokenModel(NextTokenModel):
    """Wrapper placeholder around the repo's truncated HDP-HMM implementation."""

    def __init__(self, spec: HDPHMMStudySpec) -> None:
        self.spec = spec

    def fit(self, train_data, validation_data) -> dict:
        raise NotImplementedError("Implement after the shared data interface is frozen.")

    def evaluate(self, test_data) -> dict:
        raise NotImplementedError("Implement after predictive scoring is standardized.")
