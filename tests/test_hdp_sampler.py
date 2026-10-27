"""Muestreo de β y semillas del HDP-HMM (hallazgos H1, H2, H8 de la auditoría)."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr

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
    """Muestra de un HMM de 3 estados pegajosos, cada uno con un símbolo dominante.

    Con emisiones repartidas entre dos símbolos, o con ruido de 4 % por símbolo, el
    posterior del HDP-HMM (η/V = 1/6, emisiones dispersas) prefiere de verdad estados
    extra que emiten un solo símbolo; eso es el modelo, no el muestreador. Aquí cada
    símbolo de ruido queda por debajo del 2 % de ocupación.
    """

    rng = np.random.default_rng(seed)
    transition = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    emission = np.full((3, 6), 0.01)
    emission[[0, 1, 2], [0, 2, 4]] = 0.95
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

    def test_occupied_states_do_not_follow_the_first_beta_draw(self) -> None:
        tokens = three_state_tokens(300, seed=5)
        n_large, occupied = [], []
        for seed in range(10):
            result = TruncatedHDPHMM(n_states=20, n_iters=80, burn_in=40, seed=seed).fit(_observations(tokens))
            n_large.append(int(np.sum(result.diagnostics.initial_beta > 0.01)))
            occupied.append(result.effective_states)

        # Antes del arreglo la correlación era 0.96 (s5_beta_seed.py). Con n = 10 el valor
        # crítico bilateral al 5 % de Spearman es 0.648; |r| < 0.3 fallaría por azar ~40 %
        # de las veces aun con independencia exacta.
        self.assertGreater(len(set(n_large)), 2, "el primer β debe variar entre semillas")
        rho = spearmanr(n_large, occupied).statistic
        self.assertTrue(np.isnan(rho) or abs(rho) < 0.648, f"rho={rho}, K={occupied}, n_beta={n_large}")
        self.assertLess(np.ptp(occupied), 1.5, f"K por semilla: {occupied}")

    def test_recovers_three_occupied_states(self) -> None:
        tokens = three_state_tokens(500, seed=11)
        occupied = [
            TruncatedHDPHMM(n_states=12, n_iters=150, burn_in=75, seed=seed).fit(_observations(tokens)).effective_states
            for seed in range(3)
        ]

        # Antes del arreglo: 9 estados «activos» (cualquier ocupación en la muestra máxima).
        self.assertGreaterEqual(np.mean(occupied), 2.5, occupied)
        self.assertLessEqual(np.mean(occupied), 3.8, occupied)


def _log_beta_posterior(beta: np.ndarray, state_sequences, model: TruncatedHDPHMM) -> np.ndarray:
    """log p(β | z) sin constante, con π_j y π_0 integrados (límite débil, Fox et al. 2011).

    p(β | z) ∝ Dir(β; γ/L) Π_j Π_k Γ(c_jk + n_jk)/Γ(c_jk) · Π_k Γ(α0 β_k + n0_k)/Γ(α0 β_k),
    con c_jk = α β_k + κ [j = k]; los factores Γ(Σ_k c_jk) no dependen de β.
    """

    counts = hdp_hmm_module._count_segmented_transitions(state_sequences, model.n_states)
    initial = model._initial_counts(state_sequences)
    total = (model.gamma / model.n_states - 1.0) * np.log(beta).sum(axis=1)
    eye = np.eye(model.n_states)
    for row in range(model.n_states):
        concentration = model.alpha * beta + model.kappa * eye[row]
        total += (gammaln(concentration + counts[row]) - gammaln(concentration)).sum(axis=1)
    concentration = model.alpha0 * beta
    total += (gammaln(concentration + initial) - gammaln(concentration)).sum(axis=1)
    return total


class FoxBetaUpdateTests(unittest.TestCase):
    """β | z de las mesas CRT (con corrección w por κ) coincide con la posterior exacta."""

    def _check_against_importance_sampling(self, kappa: float) -> None:
        model = TruncatedHDPHMM(n_states=3, alpha=2.0, alpha0=1.5, gamma=3.0, kappa=kappa, n_iters=2, burn_in=0)
        state_sequences = [np.array([0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 0, 1, 1]), np.array([1, 1, 2, 2])]
        model.rng = np.random.default_rng(0)
        beta = np.full(3, 1.0 / 3.0)
        draws = []
        for iteration in range(42_000):
            beta, _ = model._update_beta(beta, state_sequences)
            if iteration >= 2_000:
                draws.append(beta)
        gibbs_mean = np.mean(draws, axis=0)

        # γ/L = 1: la previa es uniforme en el simplex y sirve de propuesta.
        proposals = np.random.default_rng(1).dirichlet(np.ones(3), size=400_000)
        log_weights = _log_beta_posterior(proposals, state_sequences, model)
        weights = np.exp(log_weights - log_weights.max())
        exact_mean = weights @ proposals / weights.sum()

        np.testing.assert_allclose(gibbs_mean, exact_mean, atol=0.01)

    def test_beta_update_targets_the_posterior_without_stickiness(self) -> None:
        self._check_against_importance_sampling(kappa=0.0)

    def test_beta_update_targets_the_posterior_with_sticky_override(self) -> None:
        self._check_against_importance_sampling(kappa=6.0)


class SeedTests(unittest.TestCase):
    def test_two_fits_with_the_same_seed_are_identical(self) -> None:
        observations = _observations(three_state_tokens(120))
        model = TruncatedHDPHMM(n_states=6, n_iters=15, burn_in=5, seed=7)

        first, second = model.fit(observations), model.fit(observations)

        self.assertEqual(first.diagnostics.log_likelihood_history, second.diagnostics.log_likelihood_history)
        self.assertEqual(first.log_likelihood, second.log_likelihood)

    def test_two_instances_with_the_same_seed_give_the_same_posterior(self) -> None:
        observations = _observations(three_state_tokens(120))
        make = lambda seed: TruncatedHDPHMM(n_states=6, n_iters=15, burn_in=5, seed=seed)

        first, second, other = make(7).fit(observations), make(7).fit(observations), make(8).fit(observations)

        np.testing.assert_array_equal(first.latent_states, second.latent_states)
        np.testing.assert_array_equal(first.posterior_transition_mean, second.posterior_transition_mean)
        np.testing.assert_array_equal(first.posterior_emission_mean, second.posterior_emission_mean)
        np.testing.assert_array_equal(first.diagnostics.initial_beta, second.diagnostics.initial_beta)
        self.assertEqual(first.effective_states, second.effective_states)
        self.assertNotEqual(first.diagnostics.log_likelihood_history, other.diagnostics.log_likelihood_history)


if __name__ == "__main__":
    unittest.main()
