from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

from next_token_experiment.config import ExperimentConfig
from next_token_experiment.protocol import build_default_experiment_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_results_root() -> str:
    return str(_repo_root() / "artifacts" / "Comparacion")


@dataclass(frozen=True)
class LearningCurveConfig:
    experiment: ExperimentConfig
    results_root: str
    test_fraction: float = 0.15
    validation_fraction: float = 0.10
    split_seed: int = 7
    train_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    data_seeds: tuple[int, ...] = (1, 2, 3)
    model_seeds: tuple[int, ...] = (1, 2)
    finite_hmm_states: tuple[int, ...] = (12, 24, 48)
    finite_hmm_max_iterations: int = 100
    finite_hmm_tolerance: float = 1e-4
    # Truncacion de limite debil (Ishwaran & James 2001). El piloto ocupa 9-15 estados sea
    # cual sea K, asi que 24 deja mas del doble de holgura, y bajarlo desde 40 da 1.2x.
    # Ver docs/sources-and-methods.md, seccion 3.5.
    hdp_truncation_level: int = 24
    hdp_n_iters: int = 120
    hdp_burn_in: int = 60
    hdp_hyperparameter_grid: tuple[tuple[float, float, float], ...] = (
        (8.0, 4.0, 2.0),
        (8.0, 8.0, 2.0),
        (12.0, 4.0, 2.0),
    )
    include_vomm_control: bool = True
    vomm_candidate_orders: tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 17
    boundary_tolerance: int = 1
    structural_annotations_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.include_vomm_control, bool):
            raise TypeError("include_vomm_control must be a boolean.")
        if not isinstance(self.vomm_candidate_orders, tuple) or not self.vomm_candidate_orders:
            raise ValueError("vomm_candidate_orders must be a nonempty tuple.")
        if any(
            isinstance(order, bool) or not isinstance(order, Integral) or int(order) < 0
            for order in self.vomm_candidate_orders
        ):
            raise ValueError("vomm_candidate_orders must contain nonnegative integers.")
        if len(set(self.vomm_candidate_orders)) != len(self.vomm_candidate_orders):
            raise ValueError("vomm_candidate_orders must not contain duplicates.")
        if not isinstance(self.finite_hmm_states, tuple) or not self.finite_hmm_states:
            raise ValueError("finite_hmm_states must be a nonempty tuple.")
        if any(
            isinstance(states, bool) or not isinstance(states, Integral) or int(states) < 2
            for states in self.finite_hmm_states
        ):
            raise ValueError("finite_hmm_states must contain integers of at least 2.")
        if len(set(self.finite_hmm_states)) != len(self.finite_hmm_states):
            raise ValueError("finite_hmm_states must not contain duplicates.")
        if list(self.finite_hmm_states) != sorted(self.finite_hmm_states):
            raise ValueError("finite_hmm_states must be sorted in increasing order.")
        if (
            isinstance(self.bootstrap_samples, bool)
            or not isinstance(self.bootstrap_samples, Integral)
            or int(self.bootstrap_samples) <= 0
        ):
            raise ValueError("bootstrap_samples must be a positive integer.")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, Integral)
            or int(self.bootstrap_seed) < 0
        ):
            raise ValueError("bootstrap_seed must be a nonnegative integer.")
        if (
            isinstance(self.boundary_tolerance, bool)
            or not isinstance(self.boundary_tolerance, Integral)
            or int(self.boundary_tolerance) < 0
        ):
            raise ValueError("boundary_tolerance must be a nonnegative integer.")
        if self.structural_annotations_path is not None:
            if not isinstance(self.structural_annotations_path, str):
                raise TypeError("structural_annotations_path must be a string or None.")
            if not self.structural_annotations_path.strip():
                raise ValueError("structural_annotations_path must not be blank.")


def build_default_learning_curve_config(
    *,
    corpus_root: str | None = None,
    results_root: str | None = None,
) -> LearningCurveConfig:
    experiment = build_default_experiment_config()
    if corpus_root is not None:
        experiment = replace(experiment, corpus=replace(experiment.corpus, root_dir=corpus_root))
    return LearningCurveConfig(
        experiment=experiment,
        results_root=results_root or _default_results_root(),
    )
