from __future__ import annotations

from abc import ABC, abstractmethod


class NextTokenModel(ABC):
    """Common interface for all models in the comparison."""

    @abstractmethod
    def fit(self, train_data, validation_data) -> dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_data) -> dict:
        raise NotImplementedError
