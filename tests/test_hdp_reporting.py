"""Qué reporta el HDP-HMM (hallazgos H5, H6, F2 de la auditoría)."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import logsumexp

from src.models import hdp_hmm as hdp_hmm_module
from src.models.hdp_hmm import TruncatedHDPHMM, heldout_nll_per_token
from src.models.inference import forward_log_likelihood
from tests.test_hdp_sampler import _observations, three_state_tokens


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokens = three_state_tokens(300, seed=2)
        cls.model = TruncatedHDPHMM(n_states=8, n_iters=40, burn_in=20, seed=9)
        cls.result = cls.model.fit(_observations(cls.tokens))

    def test_log_likelihood_is_the_post_burn_in_mean(self) -> None:
        history = np.array(self.result.diagnostics.log_likelihood_history)
        self.assertAlmostEqual(self.result.log_likelihood, float(history[20:].mean()), places=9)
        self.assertAlmostEqual(self.result.max_log_likelihood, float(history[20:].max()), places=9)
        self.assertGreater(self.result.diagnostics.best_iteration, 20)

    def test_effective_states_is_the_mean_occupied_count(self) -> None:
        samples = self.result.diagnostics.state_samples
        expected = np.mean(
            [np.sum(np.bincount(states, minlength=8) / states.size >= 0.02) for states in samples]
        )
        self.assertAlmostEqual(self.result.effective_states, float(expected))
        self.assertEqual(len(samples), 20)

    def test_occupancy_threshold_is_a_parameter(self) -> None:
        loose = TruncatedHDPHMM(n_states=8, n_iters=40, burn_in=20, seed=9, occupancy_threshold=0.0)
        strict = TruncatedHDPHMM(n_states=8, n_iters=40, burn_in=20, seed=9, occupancy_threshold=0.3)
        observations = _observations(self.tokens)
        self.assertGreaterEqual(loose.fit(observations).effective_states, self.result.effective_states)
        self.assertLessEqual(strict.fit(observations).effective_states, self.result.effective_states)

    def test_posterior_samples_are_aligned_to_the_reference(self) -> None:
        reference = self.result.latent_states
        for sample in self.result.posterior_samples:
            agreement = np.mean(sample.states == reference)
            self.assertGreater(agreement, 0.5)
        mean_transition = np.mean([sample.transition_matrix for sample in self.result.posterior_samples], axis=0)
        np.testing.assert_allclose(mean_transition, self.result.posterior_transition_mean)

    def test_heldout_matches_brute_force_mixture(self) -> None:
        fit_tokens, test_tokens = self.tokens[:240], self.tokens[240:]
        expected = []
        for sample in self.result.posterior_samples:
            joint, _ = forward_log_likelihood(
                sample.initial_probs, sample.transition_matrix, sample.emission_matrix, self.tokens
            )
            context, _ = forward_log_likelihood(
                sample.initial_probs, sample.transition_matrix, sample.emission_matrix, fit_tokens
            )
            expected.append(joint - context)
        expected_nll = -(logsumexp(expected) - np.log(len(expected))) / test_tokens.size

        got = heldout_nll_per_token(self.result, fit_tokens, test_tokens)

        self.assertAlmostEqual(got, expected_nll, places=6)

    def test_heldout_pools_chains_and_accepts_empty_context(self) -> None:
        test_tokens = self.tokens[240:]
        pooled = heldout_nll_per_token([self.result, self.result], self.tokens[:0], test_tokens)
        single = heldout_nll_per_token(self.result, np.array([], dtype=int), test_tokens)
        self.assertAlmostEqual(pooled, single, places=10)
        with self.assertRaises(ValueError):
            heldout_nll_per_token(self.result, self.tokens, self.tokens[:0])


class AlignmentTests(unittest.TestCase):
    def test_permuted_labels_are_mapped_back(self) -> None:
        reference = np.array([0, 0, 1, 1, 2, 2, 2, 0])
        permutation = np.array([2, 0, 1, 3])  # etiqueta vieja -> nueva
        relabelled = permutation[reference]

        mapping = hdp_hmm_module._alignment_permutation(relabelled, reference, n_states=4)

        np.testing.assert_array_equal(mapping[relabelled], reference)


if __name__ == "__main__":
    unittest.main()
