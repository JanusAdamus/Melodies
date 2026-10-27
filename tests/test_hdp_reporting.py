"""Qué reporta el HDP-HMM (hallazgos H5, H6, F2 de la auditoría)."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import logsumexp

from src.models import hdp_hmm as hdp_hmm_module
from src.models.hdp_hmm import TruncatedHDPHMM, heldout_nll_per_token, log_likelihood_rhat, split_rhat
from src.models.inference import forward_log_likelihood
from tests.test_hdp_sampler import _observations, three_state_tokens


def _random_sample(rng: np.random.Generator, iteration: int, log_likelihood: float) -> hdp_hmm_module.PosteriorSample:
    return hdp_hmm_module.PosteriorSample(
        iteration=iteration,
        log_likelihood=log_likelihood,
        initial_probs=rng.dirichlet(np.ones(3)),
        transition_matrix=rng.dirichlet(np.ones(3), size=3),
        emission_matrix=rng.dirichlet(np.ones(6), size=3),
        beta=rng.dirichlet(np.ones(3)),
        states=np.array([0, 0, 1, 1, 2, 2, 2, 0, 1]),
    )


def _joint_log_likelihood(sample: hdp_hmm_module.PosteriorSample, tokens: np.ndarray) -> float:
    states = sample.states
    total = np.log(sample.initial_probs[states[0]]) + np.log(sample.emission_matrix[states, tokens]).sum()
    return float(total + np.log(sample.transition_matrix[states[:-1], states[1:]]).sum())


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


class ConvergenceTests(unittest.TestCase):
    """R-hat de la log-verosimilitud entre cadenas (preregistro, «Convergencia»: R-hat < 1.1)."""

    def test_split_rhat_matches_a_hand_computation(self) -> None:
        # Mitades [1,2], [3,4], [2,3], [4,5]: W = 0.5, B = 2 · var(1.5, 3.5, 2.5, 4.5) = 10/3,
        # var+ = (1/2) W + B/2 = 23/12, R-hat = sqrt(var+ / W) = sqrt(23/6).
        self.assertAlmostEqual(split_rhat([[1, 2, 3, 4], [2, 3, 4, 5]]), np.sqrt(23 / 6), places=12)

    def test_split_rhat_separates_mixed_from_stuck_chains(self) -> None:
        rng = np.random.default_rng(0)
        mixed = rng.normal(size=(4, 200))
        self.assertLess(split_rhat(mixed), 1.1)
        # Cadenas quietas en niveles distintos, como las de PDMX obra 200 (s10_mezcla_pdmx200.py).
        stuck = mixed + np.array([0.0, 3.0, 6.0, 9.0])[:, None]
        self.assertGreater(split_rhat(stuck), 1.1)
        with self.assertRaises(ValueError):
            split_rhat(mixed[:1])
        with self.assertRaises(ValueError):
            split_rhat([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])

    def test_log_likelihood_rhat_uses_only_post_burn_in_samples(self) -> None:
        observations = _observations(three_state_tokens(120, seed=3))
        chains = [TruncatedHDPHMM(n_states=6, n_iters=30, burn_in=10, seed=seed).fit(observations) for seed in (1, 2)]

        expected = split_rhat([chain.diagnostics.log_likelihood_history[10:] for chain in chains])

        self.assertAlmostEqual(log_likelihood_rhat(chains), expected, places=12)


class AlignmentTests(unittest.TestCase):
    def test_permuted_labels_are_mapped_back(self) -> None:
        reference = np.array([0, 0, 1, 1, 2, 2, 2, 0])
        permutation = np.array([2, 0, 1, 3])  # etiqueta vieja -> nueva
        relabelled = permutation[reference]

        mapping = hdp_hmm_module._alignment_permutation(relabelled, reference, n_states=4)

        np.testing.assert_array_equal(mapping[relabelled], reference)

    def test_relabel_moves_every_parameter_with_its_state(self) -> None:
        # Comprobación elemento a elemento, sin usar relabel para construir lo esperado: el estado
        # viejo s pasa a llamarse permutation[s] y se lleva su π0, su β, su fila de emisión y su
        # fila y columna de transición.  Con un 3-ciclo (no es su propia inversa) una permutación
        # aplicada en la dirección contraria no coincide.
        rng = np.random.default_rng(4)
        sample = _random_sample(rng, iteration=1, log_likelihood=-1.0)
        permutation = np.array([1, 2, 0])  # etiqueta vieja -> nueva

        moved = sample.relabel(permutation)

        np.testing.assert_array_equal(moved.states, [permutation[s] for s in sample.states])
        for old in range(3):
            new = permutation[old]
            self.assertEqual(moved.initial_probs[new], sample.initial_probs[old])
            self.assertEqual(moved.beta[new], sample.beta[old])
            np.testing.assert_array_equal(moved.emission_matrix[new], sample.emission_matrix[old])
            for old_target in range(3):
                self.assertEqual(
                    moved.transition_matrix[new, permutation[old_target]],
                    sample.transition_matrix[old, old_target],
                )
        # La verosimilitud conjunta p(x, z | θ) no cambia al renombrar estados y parámetros juntos.
        tokens = np.array([0, 3, 3, 1, 5, 2, 2, 0, 4])
        self.assertAlmostEqual(_joint_log_likelihood(moved, tokens), _joint_log_likelihood(sample, tokens), places=12)

    def test_label_switched_samples_average_to_the_reference_parameters(self) -> None:
        rng = np.random.default_rng(0)
        reference = _random_sample(rng, iteration=2, log_likelihood=-1.0)
        # Muestra con etiquetas cambiadas, construida a mano (por dispersión, sin relabel):
        # el estado s de la referencia se llama permutation[s].
        permutation = np.array([2, 0, 1])
        initial, beta = np.empty(3), np.empty(3)
        initial[permutation], beta[permutation] = reference.initial_probs, reference.beta
        emission = np.empty_like(reference.emission_matrix)
        emission[permutation] = reference.emission_matrix
        transition = np.empty_like(reference.transition_matrix)
        transition[np.ix_(permutation, permutation)] = reference.transition_matrix
        switched = hdp_hmm_module.PosteriorSample(
            iteration=3,
            log_likelihood=-2.0,
            initial_probs=initial,
            transition_matrix=transition,
            emission_matrix=emission,
            beta=beta,
            states=permutation[reference.states],
        )
        model = TruncatedHDPHMM(n_states=3, n_iters=4, burn_in=1)
        diagnostics = hdp_hmm_module.HDPHMMDiagnostics([], [], [], [], "test", 0)

        result = model._summarize([switched, reference], diagnostics, _observations(np.zeros(9, dtype=int)))

        np.testing.assert_allclose(result.posterior_transition_mean, reference.transition_matrix)
        np.testing.assert_allclose(result.posterior_emission_mean, reference.emission_matrix)
        np.testing.assert_allclose(result.posterior_initial_mean, reference.initial_probs)
        np.testing.assert_allclose(result.posterior_beta_mean, reference.beta)
        np.testing.assert_array_equal(result.diagnostics.state_samples[0], reference.states)
        self.assertEqual(result.diagnostics.best_iteration, 2)
        self.assertEqual(result.max_log_likelihood, -1.0)


if __name__ == "__main__":
    unittest.main()
