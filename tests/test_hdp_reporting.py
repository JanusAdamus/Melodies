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

    def test_posterior_means_describe_the_states_of_latent_states(self) -> None:
        # interpretation.py cruza latent_states con las medias posteriores (H6).
        for state in self.result.active_states:
            positions = self.result.latent_states == state
            if positions.mean() < 0.05:
                continue
            observed = np.bincount(self.tokens[positions], minlength=6).argmax()
            self.assertEqual(int(self.result.posterior_emission_mean[state].argmax()), int(observed))

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

    def test_heldout_renormalizes_each_step_over_the_scoring_support(self) -> None:
        # Nota 2 de la bitácora: con BOS en el vocabulario de ajuste, cada predictiva
        # p(x_t | x_<t, θ_s) se renormaliza sobre el soporte (BOS fuera), igual que
        # Comparacion.conditional_log_likelihood, y luego se promedia sobre muestras.
        bos = 5
        support = [0, 1, 2, 3, 4]
        fit_tokens = np.concatenate([[bos], self.tokens[:240]])
        fit_tokens[fit_tokens == bos] = 4
        fit_tokens[0] = bos
        test_tokens = self.tokens[240:].copy()
        test_tokens[test_tokens == bos] = 4
        per_sample = []
        for sample in self.result.posterior_samples:
            args = (sample.initial_probs, sample.transition_matrix, sample.emission_matrix)
            total, prefix = 0.0, list(fit_tokens)
            for token in test_tokens:
                context, _ = forward_log_likelihood(*args, np.array(prefix))
                step = {v: forward_log_likelihood(*args, np.array(prefix + [v]))[0] - context for v in support}
                total += step[int(token)] - logsumexp(list(step.values()))
                prefix.append(int(token))
            per_sample.append(total)
        expected = -(logsumexp(per_sample) - np.log(len(per_sample))) / test_tokens.size

        got = heldout_nll_per_token(self.result, fit_tokens, test_tokens, support=support)

        self.assertAlmostEqual(got, expected, places=6)
        self.assertLess(got, heldout_nll_per_token(self.result, fit_tokens, test_tokens))
        self.assertAlmostEqual(
            heldout_nll_per_token(self.result, fit_tokens, test_tokens, support=range(6)),
            heldout_nll_per_token(self.result, fit_tokens, test_tokens),
            places=10,
        )
        with self.assertRaises(ValueError):
            heldout_nll_per_token(self.result, fit_tokens, np.array([bos]), support=support)

    def test_trace_rows_number_every_iteration_including_burn_in(self) -> None:
        # Nota 1 de la bitácora: Comparacion descarta iteration <= burn_in.
        rows = self.result.diagnostics.trace_rows()
        self.assertEqual([row["iteration"] for row in rows], list(range(1, 41)))
        self.assertEqual(rows[0]["burn_in"], 20)
        self.assertEqual(
            [sample.iteration for sample in self.result.posterior_samples], list(range(21, 41))
        )

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


    def test_label_switched_samples_average_to_the_reference_parameters(self) -> None:
        rng = np.random.default_rng(0)
        reference = hdp_hmm_module.PosteriorSample(
            iteration=2,
            log_likelihood=-1.0,
            initial_probs=rng.dirichlet(np.ones(3)),
            transition_matrix=rng.dirichlet(np.ones(3), size=3),
            emission_matrix=rng.dirichlet(np.ones(6), size=3),
            beta=rng.dirichlet(np.ones(3)),
            states=np.array([0, 0, 1, 1, 2, 2, 2, 0, 1]),
        )
        switched = reference.relabel(np.array([2, 0, 1]))
        switched = hdp_hmm_module.PosteriorSample(**{**switched.__dict__, "iteration": 3, "log_likelihood": -2.0})
        model = TruncatedHDPHMM(n_states=3, n_iters=4, burn_in=1)
        diagnostics = hdp_hmm_module.HDPHMMDiagnostics([], [], [], [], "test", 0)

        result = model._summarize([switched, reference], diagnostics, _observations(np.zeros(9, dtype=int)))

        np.testing.assert_allclose(result.posterior_transition_mean, reference.transition_matrix)
        np.testing.assert_allclose(result.posterior_emission_mean, reference.emission_matrix)
        np.testing.assert_allclose(result.posterior_initial_mean, reference.initial_probs)
        np.testing.assert_array_equal(result.diagnostics.state_samples[0], reference.states)
        self.assertEqual(result.diagnostics.best_iteration, 2)
        self.assertEqual(result.max_log_likelihood, -1.0)


if __name__ == "__main__":
    unittest.main()
