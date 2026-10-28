"""Muestreo de β y semillas del HDP-HMM (hallazgos H1, H2, H8 de la auditoría)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
import unittest.mock
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import pearsonr

from src.data.observations import ObservationSequence
from src.models import hdp_hmm as hdp_hmm_module
from src.models.hdp_hmm import TruncatedHDPHMM

try:  # pytest no es dependencia: CI corre `unittest discover`, donde las pruebas lentas corren siempre.
    from pytest import mark as _pytest_mark
except ImportError:
    slow = lambda test: test  # noqa: E731
else:
    slow = _pytest_mark.slow

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
    """Muestra de un HMM de 3 estados pegajosos, cada uno con un símbolo dominante (0.95).

    Generador limpio: cada símbolo de ruido (0.01) queda por debajo del 2 % de ocupación.  Con
    ruido moderado (noisy_three_state_tokens) el HDP-HMM con η = 1 no recupera 3 estados sino
    ≈ 4.4-4.9; RecoveryTests muestra que eso viene del prior de emisión, no del muestreador.
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


def noisy_three_state_tokens(length: int, seed: int, dominant: float) -> tuple[np.ndarray, np.ndarray]:
    """(símbolos, estados verdaderos) del mismo HMM de 3 estados con `dominant` en el símbolo propio
    y (1 - dominant)/5 en cada uno de los otros cinco."""

    rng = np.random.default_rng(seed)
    transition = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    emission = np.full((3, 6), (1.0 - dominant) / 5.0)
    emission[[0, 1, 2], [0, 2, 4]] = dominant
    state = 0
    tokens, states = np.empty(length, dtype=int), np.empty(length, dtype=int)
    for index in range(length):
        states[index] = state
        tokens[index] = rng.choice(6, p=emission[state])
        state = rng.choice(3, p=transition[state])
    return tokens, states


class _StartFromStatesHDPHMM(TruncatedHDPHMM):
    """Arranca la cadena desde una segmentación dada (p. ej. los estados verdaderos)."""

    def __init__(self, start_states: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start_states = np.asarray(start_states, dtype=int)

    def _initialize_states(self, observations, vocab_size):
        return self.start_states.copy()


def _recovery_point(args: tuple[str, int, float, bool]) -> float:
    """effective_states de una cadena del preregistro (K=20, 400/200) sobre datos de 3 estados."""

    data, seed, eta, start_from_truth = args
    if data == "limpia":
        tokens, truth = three_state_tokens(500, seed=11), None
    else:
        tokens, truth = noisy_three_state_tokens(500, seed=11, dominant=0.80)
    n_iters, burn_in = CRITERION_B_CHAIN
    kwargs = dict(n_states=PREREGISTERED_TRUNCATION, n_iters=n_iters, burn_in=burn_in, seed=seed, eta=eta)
    model = _StartFromStatesHDPHMM(truth, **kwargs) if start_from_truth else TruncatedHDPHMM(**kwargs)
    return float(model.fit(_observations(tokens)).effective_states)


#: Semillas de las 4 cadenas del preregistro.
PREREGISTERED_SEEDS = (1, 2, 3, 4)


#: Cadena del preregistro (docs/preregistro-2026-10-26.md: truncamiento 20, 400 iteraciones, burn-in 200).
PREREGISTERED_TRUNCATION = 20
CRITERION_B_CHAIN = (400, 200)
#: Semillas 0..199, las de s8_criterio_b_bloques_pdmx200_salida_20261026.txt.
CRITERION_B_SEEDS = range(200)
#: PDMX obra 200 (599 clases de altura), copiada del caché del runner por s8 --single-work 200.
#: Es la obra del criterio (b) en PDMX; con ella el control de β congelado tiene potencia
#: (r = 0.675; en la obra simulada three_state_tokens(300, seed=5) daba r = 0.314).
PDMX_WORK_200 = Path(__file__).resolve().with_name("data") / "pdmx_obra200_pitch_classes.json"


def pdmx_work_200() -> ObservationSequence:
    payload = json.loads(PDMX_WORK_200.read_text(encoding="utf8"))
    tokens = np.asarray(payload["tokens"], dtype=int)
    return ObservationSequence("symbolic", tokens, list(payload["vocabulary"]), [""] * tokens.size, [], {})


class _FrozenBetaHDPHMM(TruncatedHDPHMM):
    """Muestreador con β congelado en el primer sorteo (el defecto H1): K debe seguir a ese sorteo."""

    def _update_beta(self, beta, state_sequences):
        return beta, "frozen_for_test"


def _criterion_b_point(args: tuple[int, int, int, bool]) -> tuple[int, float]:
    """(#β_k > 0.01 del primer sorteo, K ocupados) de una semilla sobre PDMX obra 200."""

    n_iters, burn_in, seed, freeze_beta = args
    model_class = _FrozenBetaHDPHMM if freeze_beta else TruncatedHDPHMM
    model = model_class(n_states=PREREGISTERED_TRUNCATION, n_iters=n_iters, burn_in=burn_in, seed=seed)
    result = model.fit(pdmx_work_200())
    return int(np.sum(result.diagnostics.initial_beta > 0.01)), float(result.effective_states)


#: Procesos de las pruebas paralelas; «1» las corre en secuencia en este proceso.
PARALLEL_WORKERS_ENV = "MELODIES_TEST_WORKERS"


def _worker_count(n_jobs: int) -> int:
    """Procesos para `n_jobs` tareas: MELODIES_TEST_WORKERS, o por omisión la mitad de las CPU."""

    configured = os.environ.get(PARALLEL_WORKERS_ENV)
    workers = int(configured) if configured else (os.cpu_count() or 1) // 2
    return max(1, min(n_jobs, workers))


def _in_parallel(function, jobs: list) -> list:
    """`[function(job) for job in jobs]` en un pool de `_worker_count` procesos.

    Si el pool cae (BrokenProcessPool, u OSError como «[WinError 6] Controlador no válido» con la
    máquina ocupada), las tareas sin resultado se corren en secuencia en este proceso.  Cada tarea
    fija su semilla, así que el resultado es el mismo; y un error de la propia función reaparece aquí.
    """

    workers = _worker_count(len(jobs))
    results: list = []
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(function, jobs):
                    results.append(result)
        except (BrokenProcessPool, OSError) as error:
            warnings.warn(
                f"pool de procesos caído tras {len(results)}/{len(jobs)} tareas ({error!r}); "
                "el resto corre en secuencia con las mismas semillas",
                RuntimeWarning,
                stacklevel=2,
            )
    return results + [function(job) for job in jobs[len(results):]]


def _criterion_b_rows(seeds, n_iters: int, burn_in: int, freeze_beta: bool = False) -> tuple[list[int], list[float]]:
    rows = _in_parallel(_criterion_b_point, [(n_iters, burn_in, seed, freeze_beta) for seed in seeds])
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

    @slow
    def test_occupied_states_do_not_follow_the_first_beta_draw(self) -> None:
        # Criterio (b) (PROMPT_CLAUDE.md, |r| < 0.3) en PDMX obra 200 con la cadena del preregistro y
        # las 200 semillas juntas: |r| < 0.3 y equivalencia TOST significativa (regla en
        # s8_criterio_b_bloques.py).  Medido el 2026-10-26: r = -0.100, TOST p = 0.0016
        # (s8_criterio_b_agrupado_pdmx200_salida_20261026.txt).  (b) no dice que las cadenas mezclen:
        # en esta obra el R-hat de log L entre las semillas 1-4 es 2.59 (hallazgo H9, s10_mezcla_pdmx200.py).
        self.assertEqual(_criterion_b_decision(*_criterion_b_rows(CRITERION_B_SEEDS, *CRITERION_B_CHAIN)), "met")

    @slow
    def test_criterion_b_fails_with_a_frozen_beta(self) -> None:
        # Control de que la prueba anterior puede fallar: con β congelado en el primer sorteo (el
        # defecto H1), la misma obra y las mismas 200 semillas, la regla da not_met.  Medido el
        # 2026-10-27: r = 0.675, IC 95 % [0.592, 0.744]
        # (s8_criterio_b_bloques_pdmx200_beta_congelado_salida_20261027.txt); con esa dependencia la
        # regla da not_met con probabilidad ≈ 1 (test_criterion_b_decision, potencia del control).
        decision = _criterion_b_decision(*_criterion_b_rows(CRITERION_B_SEEDS, *CRITERION_B_CHAIN, freeze_beta=True))
        self.assertEqual(decision, "not_met")


def _pdmx_chain_log_likelihoods(seed: int) -> list[float]:
    """log p(x | θ_t) tras burn-in de una cadena del preregistro (K=20, 400/200) en PDMX obra 200."""

    n_iters, burn_in = CRITERION_B_CHAIN
    model = TruncatedHDPHMM(n_states=PREREGISTERED_TRUNCATION, n_iters=n_iters, burn_in=burn_in, seed=seed)
    return [sample.log_likelihood for sample in model.fit(pdmx_work_200()).posterior_samples]


class PreregisteredConvergenceTests(unittest.TestCase):
    def test_convergence_rule_flags_pdmx_work_200(self) -> None:
        # Hallazgo H9 (abierto): con β corregido y la cadena del preregistro las 4 cadenas no mezclan
        # en PDMX obra 200; cada semilla se queda cerca de su K.  La regla «Convergencia» del
        # preregistro (R-hat de log L < 1.1 entre las semillas 1-4) debe detectarlo y dejar la obra
        # fuera de la métrica principal.  Medido el 2026-10-27: R-hat = 2.589
        # (s10_mezcla_pdmx200_salida_20261027.txt).  Si un cambio del muestreador hace mezclar estas
        # cadenas, esta prueba falla y hay que cerrar H9 en docs/auditoria-2026-10-26.md.
        chains = _in_parallel(_pdmx_chain_log_likelihoods, list(PREREGISTERED_SEEDS))
        rhat = hdp_hmm_module.split_rhat(chains)
        self.assertAlmostEqual(rhat, 2.589, places=3)
        self.assertGreaterEqual(rhat, 1.1)


class RecoveryTests(unittest.TestCase):
    """Criterio (c) con la configuración del preregistro (K=20, 400/200, semillas 1-4).

    (c) se cumple con el generador limpio.  Con ruido moderado (0.80 en el símbolo propio, 0.04 en
    cada otro) el HDP-HMM con η = 1 da ≈ 4.4-4.9 estados ocupados de 3 verdaderos (hallazgo H10 de
    docs/auditoria-2026-10-26.md).  Medido el 2026-10-27 (scripts/auditoria_2026_10_26/hdp/
    s11_estados_con_ruido.py): η = 1 arrancando por omisión 4.90, desde los estados verdaderos 4.35,
    desde un estado por símbolo 4.72; η = 6 (prior de emisión uniforme) 3.25, 3.33 y 3.27.
    """

    def _mean_occupied(self, data: str, eta: float = 1.0, start_from_truth: bool = False) -> list[float]:
        return _in_parallel(_recovery_point, [(data, seed, eta, start_from_truth) for seed in PREREGISTERED_SEEDS])

    def test_recovers_three_occupied_states(self) -> None:
        # Antes del arreglo: 9 estados «activos» (cualquier ocupación en la muestra máxima).
        # Medido el 2026-10-27: media 3.19 (s11_estados_con_ruido_salida_20261027.txt).
        occupied = self._mean_occupied("limpia")
        self.assertGreaterEqual(np.mean(occupied), 2.5, occupied)
        self.assertLessEqual(np.mean(occupied), 3.8, occupied)

    def test_extra_states_under_emission_noise_come_from_the_emission_prior(self) -> None:
        # Si los estados extra fueran falta de mezcla, una cadena que arranca en la segmentación
        # verdadera se quedaría en 3.  No: con η = 1 la deja (el posterior prefiere estados que
        # emiten un solo símbolo).  El mismo muestreador con η = 6 recupera ≈ 3.
        from_truth = self._mean_occupied("ruido", eta=1.0, start_from_truth=True)
        self.assertGreater(np.mean(from_truth), 3.8, from_truth)
        uniform_prior = self._mean_occupied("ruido", eta=6.0)
        self.assertGreaterEqual(np.mean(uniform_prior), 2.5, uniform_prior)
        self.assertLessEqual(np.mean(uniform_prior), 3.8, uniform_prior)


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


class ImportWithoutPytestTests(unittest.TestCase):
    def test_hdp_test_modules_import_without_pytest(self) -> None:
        # CI (.github/workflows/tests.yml) instala sólo `pip install -e .` y corre
        # `python -m unittest discover -s tests`; pytest no es dependencia declarada.
        code = (
            "import sys; sys.modules['pytest'] = None\n"
            "import tests.test_hdp_sampler, tests.test_hdp_reporting\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr[-2000:])


def _short_chain(seed: int) -> list[float]:
    """Historia de log L de una cadena corta; sirve para comparar ejecución en pool y secuencial."""

    model = TruncatedHDPHMM(n_states=6, n_iters=8, burn_in=2, seed=seed)
    return list(model.fit(_observations(three_state_tokens(80, seed=4))).diagnostics.log_likelihood_history)


def _short_chain_dies_in_a_worker(seed: int) -> list[float]:
    """Como _short_chain, pero el proceso hijo de la semilla 2 muere: simula la caída del pool
    (BrokenProcessPool, «[WinError 6] Controlador no válido» con la máquina ocupada)."""

    import multiprocessing

    if seed == 2 and multiprocessing.parent_process() is not None:
        os._exit(1)
    return _short_chain(seed)


class ParallelRunnerTests(unittest.TestCase):
    """Un rojo de las pruebas del preregistro debe hablar del muestreador, no de la carga de la máquina."""

    def test_worker_count_is_bounded_and_configurable(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PARALLEL_WORKERS_ENV, None)
            default = _worker_count(200)
            self.assertGreaterEqual(default, 1)
            self.assertLessEqual(default, max(1, (os.cpu_count() or 1) // 2))
            self.assertEqual(_worker_count(1), 1)
            os.environ[PARALLEL_WORKERS_ENV] = "3"
            self.assertEqual(_worker_count(200), 3)
            self.assertEqual(_worker_count(2), 2)
            os.environ[PARALLEL_WORKERS_ENV] = "1"
            self.assertEqual(_worker_count(200), 1)

    def test_sequential_and_parallel_give_the_same_result(self) -> None:
        seeds = [1, 2, 3]
        expected = [_short_chain(seed) for seed in seeds]
        with unittest.mock.patch.dict(os.environ, {PARALLEL_WORKERS_ENV: "1"}):
            self.assertEqual(_in_parallel(_short_chain, seeds), expected)
        with unittest.mock.patch.dict(os.environ, {PARALLEL_WORKERS_ENV: "2"}):
            self.assertEqual(_in_parallel(_short_chain, seeds), expected)

    def test_a_broken_pool_falls_back_to_sequential_with_the_same_result(self) -> None:
        seeds = [1, 2, 3, 4]
        expected = [_short_chain(seed) for seed in seeds]
        with unittest.mock.patch.dict(os.environ, {PARALLEL_WORKERS_ENV: "2"}):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(_in_parallel(_short_chain_dies_in_a_worker, seeds), expected)

    def test_an_error_of_the_function_is_not_swallowed(self) -> None:
        # La vuelta a secuencial repite la función en este proceso: un error suyo reaparece.
        with unittest.mock.patch.dict(os.environ, {PARALLEL_WORKERS_ENV: "2"}):
            with self.assertRaises(ValueError):
                _in_parallel(int, ["1", "no es un número"])


if __name__ == "__main__":
    unittest.main()
