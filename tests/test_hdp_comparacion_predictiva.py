"""La comparación califica al HDP-HMM con la predictiva posterior, no con la media de los parámetros.

Preregistro A, métrica principal: para el HDP, promedio de la predictiva sobre las muestras
posteriores. p(x | BOS) ≈ (1/S) Σ_s p(x | BOS, θ_s), con BOS fuera del soporte (F3).
"""

from __future__ import annotations

from types import SimpleNamespace
import math
import unittest

import numpy as np

from Comparacion.classical_models import GlobalHDPHMM, conditional_log_likelihood
from next_token_experiment.schemas import PreparedPiece
from src.models.hdp_hmm import PosteriorSample


BOS = 12


def _sample(token_zero_probability: float) -> PosteriorSample:
    emission = np.full((1, 13), (1.0 - token_zero_probability - 0.01) / 11.0)
    emission[0, 0] = token_zero_probability
    emission[0, BOS] = 0.01
    return PosteriorSample(
        iteration=0,
        log_likelihood=0.0,
        initial_probs=np.array([1.0]),
        transition_matrix=np.array([[1.0]]),
        emission_matrix=emission,
        beta=np.array([1.0]),
        states=np.array([0]),
    )


def _result(samples: list[PosteriorSample]) -> SimpleNamespace:
    return SimpleNamespace(
        posterior_initial_mean=np.mean([s.initial_probs for s in samples], axis=0),
        posterior_transition_mean=np.mean([s.transition_matrix for s in samples], axis=0),
        posterior_emission_mean=np.mean([s.emission_matrix for s in samples], axis=0),
        posterior_samples=samples,
    )


def _piece(tokens: list[int]) -> PreparedPiece:
    return PreparedPiece(
        piece_id="obra",
        source_path="obra.mxl",
        title="obra",
        composer="anon",
        canonical_work_id="obra",
        representation="pitch_class",
        vocabulary=[str(index) for index in range(12)],
        tokens=tokens,
        n_events=len(tokens),
        metadata={},
    )


def _log_likelihood_of(sample: PosteriorSample, targets: list[int]) -> float:
    return conditional_log_likelihood(
        initial_probs=sample.initial_probs,
        transition_matrix=sample.transition_matrix,
        emission_matrix=sample.emission_matrix,
        context_token=BOS,
        target_tokens=targets,
    )


class PosteriorPredictiveScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [_sample(0.9), _sample(0.1)]
        self.result = _result(self.samples)
        self.model = GlobalHDPHMM(
            truncation_level=1, n_iters=1, burn_in=0, hyperparameter_grid=((1.0, 1.0, 1.0),), seed=1
        )
        self.targets = [0, 0, 0]

    def test_score_is_the_mean_of_the_per_sample_predictives(self) -> None:
        expected = math.log(np.mean([math.exp(_log_likelihood_of(s, self.targets)) for s in self.samples]))
        score, indices = self.model._score_piece(_piece(self.targets), BOS, 128, result=self.result)
        self.assertAlmostEqual(score, expected, places=10)
        self.assertEqual(indices, [0, 1, 2])

    def test_score_is_not_the_plug_in_of_the_posterior_means(self) -> None:
        plug_in = conditional_log_likelihood(
            initial_probs=self.result.posterior_initial_mean,
            transition_matrix=self.result.posterior_transition_mean,
            emission_matrix=self.result.posterior_emission_mean,
            context_token=BOS,
            target_tokens=self.targets,
        )
        score, _ = self.model._score_piece(_piece(self.targets), BOS, 128, result=self.result)
        self.assertGreater(score - plug_in, 0.1)

    def test_each_evaluation_slice_restarts_from_bos(self) -> None:
        targets = [0, 0, 1, 0, 0]
        score, indices = self.model._score_piece(_piece(targets), BOS, 2, result=self.result)
        expected = 0.0
        for start in range(0, len(targets), 2):
            chunk = targets[start : start + 2]
            expected += math.log(np.mean([math.exp(_log_likelihood_of(s, chunk)) for s in self.samples]))
        self.assertAlmostEqual(score, expected, places=10)
        self.assertEqual(indices, list(range(len(targets))))


if __name__ == "__main__":
    unittest.main()
