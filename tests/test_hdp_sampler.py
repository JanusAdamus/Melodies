"""Muestreo de β y semillas del HDP-HMM (hallazgos H1, H2, H8 de la auditoría)."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.observations import ObservationSequence
from src.models import hdp_hmm as hdp_hmm_module
from src.models.hdp_hmm import TruncatedHDPHMM

VOCABULARY = [f"s{index}" for index in range(6)]


def _observations(tokens: np.ndarray) -> ObservationSequence:
    tokens = np.asarray(tokens, dtype=int)
    return ObservationSequence(
        observation_type="symbolic_test",
        tokens=tokens,
        vocabulary=list(VOCABULARY),
        decoded=[VOCABULARY[token] for token in tokens],
        events=[],
        extra={},
    )


def three_state_tokens(length: int, seed: int = 0) -> np.ndarray:
    """Muestra de un HMM de 3 estados pegajosos con emisiones casi disjuntas."""

    rng = np.random.default_rng(seed)
    transition = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    emission = np.array(
        [
            [0.45, 0.45, 0.025, 0.025, 0.025, 0.025],
            [0.025, 0.025, 0.45, 0.45, 0.025, 0.025],
            [0.025, 0.025, 0.025, 0.025, 0.45, 0.45],
        ]
    )
    state = 0
    tokens = np.empty(length, dtype=int)
    for index in range(length):
        tokens[index] = rng.choice(6, p=emission[state])
        state = rng.choice(3, p=transition[state])
    return tokens


class CRTTests(unittest.TestCase):
    def test_table_counts_have_the_chinese_restaurant_mean(self) -> None:
        rng = np.random.default_rng(2)
        counts = np.array([[0, 1, 5], [30, 2, 0]])
        concentration = np.array([[0.7, 0.1, 3.0], [2.0, 1e-300, 5.0]])
        draws = np.stack([hdp_hmm_module._sample_crt(counts, concentration, rng) for _ in range(20_000)])
        expected = np.zeros_like(concentration)
        for index in np.ndindex(counts.shape):
            customers = np.arange(counts[index])
            expected[index] = np.sum(concentration[index] / (concentration[index] + customers))
        self.assertTrue(np.all(draws <= counts))
        self.assertTrue(np.all(draws[:, counts > 0] >= 1))
        np.testing.assert_allclose(draws.mean(axis=0), expected, atol=0.05)


class BetaSamplingTests(unittest.TestCase):
    def test_beta_changes_between_iterations(self) -> None:
        tokens = three_state_tokens(200)
        result = TruncatedHDPHMM(n_states=8, n_iters=30, burn_in=10, seed=3).fit(_observations(tokens))

        entropies = np.array(result.diagnostics.beta_entropy_history)
        self.assertGreater(np.unique(entropies).size, 20)
        self.assertEqual(result.diagnostics.beta_update_mode, "crt_auxiliary_gibbs")


class SeedTests(unittest.TestCase):
    def test_two_fits_with_the_same_seed_are_identical(self) -> None:
        observations = _observations(three_state_tokens(120))
        model = TruncatedHDPHMM(n_states=6, n_iters=15, burn_in=5, seed=7)

        first, second = model.fit(observations), model.fit(observations)

        self.assertEqual(first.diagnostics.log_likelihood_history, second.diagnostics.log_likelihood_history)
        self.assertEqual(first.log_likelihood, second.log_likelihood)


if __name__ == "__main__":
    unittest.main()
