"""Muestreo de β y semillas del HDP-HMM (hallazgos H1, H2, H8 de la auditoría)."""

from __future__ import annotations

import importlib.util
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import pearsonr

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


#: Cadena del preregistro (docs/preregistro-2026-10-26.md: 400 iteraciones, burn-in 200).
CRITERION_B_CHAIN = (400, 200)
#: Semillas 0..199, las de s8_criterio_b_bloques_simulada_salida_20261026.txt.
CRITERION_B_SEEDS = range(200)


class _FrozenBetaHDPHMM(TruncatedHDPHMM):
    """Muestreador con β congelado en el primer sorteo (el defecto H1): K debe seguir a ese sorteo."""

    def _update_beta(self, beta, state_sequences):
        return beta, "frozen_for_test"


def _criterion_b_point(args: tuple[int, int, int, bool]) -> tuple[int, float]:
    """(#β_k > 0.01 del primer sorteo, K ocupados) de una semilla sobre la obra simulada fija."""

    n_iters, burn_in, seed, freeze_beta = args
    model_class = _FrozenBetaHDPHMM if freeze_beta else TruncatedHDPHMM
    tokens = three_state_tokens(300, seed=5)
    result = model_class(n_states=20, n_iters=n_iters, burn_in=burn_in, seed=seed).fit(_observations(tokens))
    return int(np.sum(result.diagnostics.initial_beta > 0.01)), float(result.effective_states)


def _criterion_b_rows(seeds, n_iters: int, burn_in: int, freeze_beta: bool = False) -> tuple[list[int], list[float]]:
    jobs = [(n_iters, burn_in, seed, freeze_beta) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        rows = list(pool.map(_criterion_b_point, jobs))
    return [row[0] for row in rows], [row[1] for row in rows]


def _criterion_b_decision(n_large: list[int], occupied: list[float]) -> str:
    """Decisión de s8_criterio_b_bloques con todas las semillas juntas: met, not_met o inconclusive."""

    spec = importlib.util.spec_from_file_location(
        "s8_criterio_b_bloques",
        Path(__file__).resolve().parents[1] / "scripts" / "auditoria_2026_10_26" / "hdp" / "s8_criterio_b_bloques.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script.criterion_b_decision(float(pearsonr(n_large, occupied).statistic), len(n_large))


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

    def test_first_beta_draw_varies_between_seeds(self) -> None:
        # Sin esto el criterio (b) no tendría variable explicativa.
        observations = _observations(three_state_tokens(300, seed=5))
        n_large = {
            int(np.sum(TruncatedHDPHMM(n_states=20, n_iters=2, burn_in=0, seed=seed).fit(observations).diagnostics.initial_beta > 0.01))
            for seed in range(20)
        }
        self.assertGreater(len(n_large), 2, n_large)

    @pytest.mark.slow
    def test_occupied_states_do_not_follow_the_first_beta_draw(self) -> None:
        # Criterio (b) (PROMPT_CLAUDE.md, |r| < 0.3) con la cadena del preregistro y las 200 semillas
        # juntas: |r| < 0.3 y equivalencia TOST significativa (regla en s8_criterio_b_bloques.py).
        # Medido el 2026-10-26: r = 0.069, TOST p = 0.00037 (s8_criterio_b_agrupado_simulada_salida_20261026.txt).
        self.assertEqual(_criterion_b_decision(*_criterion_b_rows(CRITERION_B_SEEDS, *CRITERION_B_CHAIN)), "met")

    @pytest.mark.slow
    def test_criterion_b_fails_with_a_frozen_beta(self) -> None:
        # Control de que la prueba anterior puede fallar: con β congelado en el primer sorteo (el
        # defecto H1) y las mismas 200 semillas, la regla no puede dar «met».  Con n = 200 la TOST sólo
        # es significativa si r < tanh(atanh(0.3) - 1.645/sqrt(197)) ≈ 0.19; medido el 2026-10-26:
        # r = 0.314, not_met (s8_criterio_b_bloques_simulada_beta_congelado_salida_20261026.txt).
        decision = _criterion_b_decision(*_criterion_b_rows(CRITERION_B_SEEDS, *CRITERION_B_CHAIN, freeze_beta=True))
        self.assertNotEqual(decision, "met")

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


#: Escenarios de β | z.  Con un solo inicio por estado, Γ(c+1)/Γ(c) = c y α0 se cancela de
#: p(β | z), así que ningún escenario así distingue α0 de α.  Por eso hay dos escenarios:
#: - «muchos_inicios»: 16 secuencias y α0 = 10 frente a α = 1; pesan las mesas iniciales y γ/L.
#: - «rachas_largas»: una secuencia con rachas largas; la diagonal (κ en el CRT y la
#:   corrección w) domina β.
BETA_SCENARIOS = {
    "muchos_inicios": (
        dict(alpha=1.0, alpha0=10.0, gamma=3.0),
        [[0] * 25 + [1] * 15 + [0] * 20 + [2] * 10 + [1] * 30] + [[0, 0]] * 10 + [[1, 1]] * 4 + [[2, 2]] * 2,
    ),
    "rachas_largas": (
        dict(alpha=2.0, alpha0=1.5, gamma=3.0),
        [[0] * 50 + [1] * 30 + [0] * 10 + [2] * 10],
    ),
}
#: Error Monte Carlo del muestreador correcto (cadenas 0..3, 20 000 iteraciones): ≤ 0.0027 en
#: medias y ≤ 0.0019 en d.e.  Diferencia máxima de mutantes de `_update_beta` con estos
#: escenarios (cadena 0): CRT sin κ 0.042; w con probabilidad ρ 0.037; γ en vez de γ/L 0.028
#: (muchos_inicios) y 0.069 (rachas_largas); α en vez de α0 en las mesas iniciales 0.097.
BETA_ATOL = 0.01


def _scenario(name: str, kappa: float, **override) -> tuple[TruncatedHDPHMM, list[np.ndarray]]:
    params, sequences = BETA_SCENARIOS[name]
    model = TruncatedHDPHMM(n_states=3, kappa=kappa, n_iters=2, burn_in=0, **{**params, **override})
    return model, [np.asarray(sequence) for sequence in sequences]


def _exact_moments(model: TruncatedHDPHMM, state_sequences) -> tuple[np.ndarray, np.ndarray]:
    """Media y d.e. de β | z por muestreo de importancia con propuesta uniforme en el simplex."""

    proposals = np.random.default_rng(1).dirichlet(np.ones(3), size=400_000)
    log_weights = _log_beta_posterior(proposals, state_sequences, model)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    mean = weights @ proposals
    return mean, np.sqrt(weights @ (proposals - mean) ** 2)


class FoxBetaUpdateTests(unittest.TestCase):
    """β | z de las mesas CRT (con corrección w por κ) coincide con la posterior exacta."""

    def _check_against_importance_sampling(self, name: str, kappa: float) -> None:
        model, state_sequences = _scenario(name, kappa)
        model.rng = np.random.default_rng(0)
        beta = np.full(3, 1.0 / 3.0)
        draws = np.empty((20_000, 3))
        for iteration in range(22_000):
            beta, _ = model._update_beta(beta, state_sequences)
            if iteration >= 2_000:
                draws[iteration - 2_000] = beta

        exact_mean, exact_sd = _exact_moments(model, state_sequences)
        np.testing.assert_allclose(draws.mean(axis=0), exact_mean, atol=BETA_ATOL)
        np.testing.assert_allclose(draws.std(axis=0), exact_sd, atol=BETA_ATOL)

    def test_scenario_separates_wrong_alpha0_and_wrong_gamma(self) -> None:
        # Con α en vez de α0, o γ en vez de γ/L, la distribución estacionaria del muestreador
        # sería la posterior exacta con ese parámetro cambiado.  La prueba sólo tiene potencia
        # contra esos errores si esa posterior se aleja de la verdadera más del doble de BETA_ATOL.
        model, state_sequences = _scenario("muchos_inicios", 0.0)
        reference, _ = _exact_moments(model, state_sequences)
        for override in (dict(alpha0=model.alpha), dict(gamma=model.gamma * model.n_states)):
            wrong, _ = _exact_moments(*_scenario("muchos_inicios", 0.0, **override))
            self.assertGreater(np.abs(wrong - reference).max(), 2 * BETA_ATOL, override)

    def test_beta_update_targets_the_posterior_without_stickiness(self) -> None:
        self._check_against_importance_sampling("muchos_inicios", kappa=0.0)

    def test_beta_update_targets_the_posterior_with_sticky_override(self) -> None:
        self._check_against_importance_sampling("rachas_largas", kappa=6.0)


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
