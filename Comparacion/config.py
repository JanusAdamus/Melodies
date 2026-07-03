from __future__ import annotations

from dataclasses import dataclass, replace
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
    hdp_truncation_level: int = 40
    hdp_n_iters: int = 120
    hdp_burn_in: int = 60
    hdp_hyperparameter_grid: tuple[tuple[float, float, float], ...] = (
        (8.0, 4.0, 2.0),
        (8.0, 8.0, 2.0),
        (12.0, 4.0, 2.0),
    )


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
