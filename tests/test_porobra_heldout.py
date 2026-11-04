"""Análisis por obra held-out (tarea 1.4, preregistro sección A, Aclaración 2, Enmienda 3).

Cada prueba usa datos donde el resultado se conoce de antemano: fórmulas cerradas de la
predictiva de Dirichlet, un HMM con emisiones disjuntas, una familia de diferencias
construida a mano.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammaln

from src.analysis import porobra
from src.models.inference import forward_log_likelihood


# --- partición contigua -------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_events", "expected"),
    [(20, (16, 4, 12, 4)), (100, (80, 20, 64, 16)), (37, (29, 8, 23, 6))],
)
def test_particion_contigua_80_20_con_validacion_interna(n_events, expected):
    split = porobra.contiguous_split(n_events)
    assert (split.n_fit, split.n_test, split.n_inner_train, split.n_validation) == expected


def test_particion_rechaza_obras_cortas():
    with pytest.raises(ValueError):
        porobra.contiguous_split(19)


# --- n-gramas con Dirichlet: predictiva exacta ----------------------------------------


def _dirichlet_multinomial_log_marginal(counts: np.ndarray, alpha: float) -> float:
    vocab = counts.size
    return float(
        gammaln(vocab * alpha) - gammaln(vocab * alpha + counts.sum()) + np.sum(gammaln(alpha + counts) - gammaln(alpha))
    )


def test_unigrama_es_la_predictiva_de_dirichlet():
    context = np.array([0, 0, 1])
    target = np.array([0, 0, 5])
    expected = (
        _dirichlet_multinomial_log_marginal(np.bincount(np.r_[context, target], minlength=12).astype(float), 1.0)
        - _dirichlet_multinomial_log_marginal(np.bincount(context, minlength=12).astype(float), 1.0)
    )
    nll = porobra.ngram_conditional_nll(context, target, order=1, alpha=1.0)
    assert nll == pytest.approx(-expected / target.size)
    # Primer paso a mano: (2 + 1) / (3 + 12).
    assert porobra.ngram_conditional_nll(context, target[:1], order=1, alpha=1.0) == pytest.approx(-math.log(3 / 15))


def test_bigrama_condiciona_en_la_ultima_nota_del_ajuste_y_actualiza():
    context = np.array([0, 1, 0, 1])
    # prev = 1; C[1,0] = 1, C[1,:] = 1 -> (1 + a) / (1 + 12 a); luego prev = 0: C[0,1]=2,
    # C[0,:]=2 -> (2 + a) / (2 + 12 a); luego C[1,0]=2, C[1,:]=2.
    a = 0.5
    target = np.array([0, 1, 0])
    expected = -(
        math.log((1 + a) / (1 + 12 * a)) + math.log((2 + a) / (2 + 12 * a)) + math.log((2 + a) / (2 + 12 * a))
    ) / 3
    assert porobra.ngram_conditional_nll(context, target, order=2, alpha=a) == pytest.approx(expected)


def test_alpha_se_elige_en_validacion():
    rng = np.random.default_rng(0)
    tokens = np.tile([0, 4, 7], 40)
    tokens[rng.random(tokens.size) < 0.02] = 2
    choice = porobra.select_ngram_alpha(tokens, order=2, alphas=(0.1, 0.5, 1.0))
    split = porobra.contiguous_split(tokens.size)
    inner, validation = tokens[: split.n_inner_train], tokens[split.n_inner_train : split.n_fit]
    scores = {a: porobra.ngram_conditional_nll(inner, validation, order=2, alpha=a) for a in (0.1, 0.5, 1.0)}
    assert choice.selected == min(scores, key=scores.get)
    assert choice.validation_nll == pytest.approx(scores)


# --- selección con rejilla ampliable -------------------------------------------------


def test_rejilla_se_amplia_mientras_elija_el_maximo():
    # Mejor mientras K <= 48: con 32 elige el máximo y la rejilla crece a 48 y 64.
    choice = porobra.select_by_validation((2, 4, 8, 16, 32), lambda k: -k if k <= 48 else k, extend=True, max_value=512)
    assert choice.selected == 48
    assert choice.grid == (2, 4, 8, 16, 32, 48, 64)
    assert choice.grid_extensions == 1
    assert not choice.hit_limit


def test_rejilla_con_techo_marca_que_lo_toco():
    choice = porobra.select_by_validation((2, 4), lambda k: -k, extend=True, max_value=8)
    assert choice.selected == 8 and choice.hit_limit


def test_sin_ampliar_elige_el_minimo_de_la_rejilla_fija():
    choice = porobra.select_by_validation((1.0, 3.0, 6.0), lambda e: -e, extend=False)
    assert choice.selected == 6.0 and choice.grid == (1.0, 3.0, 6.0) and not choice.hit_limit


# --- HMM finito ----------------------------------------------------------------------


def _simulate_hmm(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    emission = np.zeros((2, 12))
    emission[0, [0, 4, 7]] = 1 / 3
    emission[1, [2, 5, 9]] = 1 / 3
    state = rng.choice(2, p=initial)
    tokens = []
    for _ in range(n):
        tokens.append(rng.choice(12, p=emission[state]))
        state = rng.choice(2, p=transition[state])
    return np.array(tokens), initial, transition, emission


def test_nll_condicional_del_hmm_es_cociente_de_verosimilitudes():
    tokens, initial, transition, emission = _simulate_hmm(60, 1)
    context, target = tokens[:45], tokens[45:]
    joint, _ = forward_log_likelihood(initial, transition, emission, tokens)
    marginal, _ = forward_log_likelihood(initial, transition, emission, context)
    nll = porobra.hmm_conditional_nll(initial, transition, emission, context, target)
    assert nll == pytest.approx(-(joint - marginal) / target.size, rel=1e-8)


def test_baum_welch_sube_la_verosimilitud_y_recupera_dos_estados():
    tokens, initial, transition, emission = _simulate_hmm(400, 2)
    fit = porobra.baum_welch(tokens, n_states=2, seed=0)
    assert np.all(np.diff(fit.objective_trace) >= -1e-6)
    true_ll, _ = forward_log_likelihood(initial, transition, emission, tokens)
    assert fit.log_likelihood >= true_ll - 1.0
    occupancy = porobra.posterior_occupied_states(fit, tokens, threshold=0.02, n_samples=50, seed=0)
    assert occupancy.mean == pytest.approx(2.0)
    assert occupancy.sd == pytest.approx(0.0)


def _log_prior_de_dirichlet(fit, pseudocount: float) -> float:
    """La previa cuyo modo es el paso M: Dirichlet(seudoconteo + 1), sin sus constantes."""

    return float(porobra.INITIAL_PSEUDOCOUNT * np.log(fit.initial_probs).sum()
                 + pseudocount * (np.log(fit.transition_matrix).sum() + np.log(fit.emission_matrix).sum()))


@pytest.mark.parametrize(("n_states", "pseudocount"), [(2, 1.0), (8, 0.5), (16, 0.1), (4, 1e-3)])
def test_em_con_seudoconteo_no_baja_el_objetivo_penalizado(n_states, pseudocount):
    tokens = np.random.default_rng(0).integers(0, 12, 60)
    fit = porobra.baum_welch(tokens, n_states=n_states, seed=0, pseudocount=pseudocount)
    assert len(fit.objective_trace) == len(fit.log_likelihood_trace) >= 2
    assert np.all(np.diff(fit.objective_trace) >= -1e-8)


def test_el_objetivo_es_la_verosimilitud_mas_la_previa_con_todos_los_seudoconteos():
    tokens = np.random.default_rng(0).integers(0, 12, 60)
    fit = porobra.baum_welch(tokens, n_states=4, seed=0, pseudocount=0.5)
    assert fit.objective_trace[-1] == pytest.approx(fit.log_likelihood + _log_prior_de_dirichlet(fit, 0.5), abs=1e-9)


def test_una_caida_de_la_verosimilitud_no_detiene_a_em_antes_de_converger():
    # Con seudoconteo EM sube la posterior penalizada, no log p(x|θ): la regla vieja paraba
    # en la primera caída de la verosimilitud, muy lejos de la convergencia.
    tokens = np.random.default_rng(0).integers(0, 12, 40)
    tolerance = 1e-4
    fit = porobra.baum_welch(tokens, n_states=2, seed=0, pseudocount=1.0, tolerance=tolerance)
    fall = next(step for step, change in enumerate(np.diff(fit.log_likelihood_trace)) if change < tolerance)
    assert fit.objective_trace[fall + 1] - fit.objective_trace[fall] >= tolerance
    assert len(fit.log_likelihood_trace) > fall + 10
    # Y para cuando el objetivo deja de mejorar: el último paso E (antes del valor final).
    improvements = np.diff(fit.objective_trace[:-1])
    assert improvements[-1] < tolerance and np.all(improvements[:-1] >= tolerance)


def test_max_iterations_sigue_siendo_el_techo_de_em():
    tokens = np.random.default_rng(0).integers(0, 12, 40)
    fit = porobra.baum_welch(tokens, n_states=2, seed=0, pseudocount=1.0, max_iterations=5)
    assert len(fit.log_likelihood_trace) == len(fit.objective_trace) == 6


def test_baum_welch_es_reproducible():
    tokens, *_ = _simulate_hmm(120, 3)
    left = porobra.baum_welch(tokens, n_states=4, seed=5)
    right = porobra.baum_welch(tokens, n_states=4, seed=5)
    assert left.log_likelihood == right.log_likelihood
    np.testing.assert_array_equal(left.transition_matrix, right.transition_matrix)


def test_mejor_de_varios_reinicios_es_el_de_mayor_verosimilitud():
    tokens, *_ = _simulate_hmm(300, 7)
    fits = [porobra.baum_welch(tokens, n_states=8, seed=seed) for seed in (3, 4, 5, 6)]
    best = porobra.best_baum_welch(tokens, n_states=8, seeds=(3, 4, 5, 6))
    assert best.log_likelihood == max(fit.log_likelihood for fit in fits)
    np.testing.assert_array_equal(best.emission_matrix, max(fits, key=lambda f: f.log_likelihood).emission_matrix)


def test_baum_welch_suaviza_las_emisiones_con_alpha():
    # Con un solo estado el paso M es exacto: emisión = (conteo + α) / (n + 12 α).
    tokens = np.array([0, 4, 7, 0, 4, 0, 7, 7, 0, 4])
    for alpha in (0.1, 1.0):
        fit = porobra.baum_welch(tokens, n_states=1, seed=0, pseudocount=alpha)
        expected = (np.bincount(tokens, minlength=12) + alpha) / (tokens.size + 12 * alpha)
        np.testing.assert_allclose(fit.emission_matrix[0], expected, rtol=1e-10)


def test_finito_con_reinicios_elige_k_y_alpha_en_validacion_y_reajusta():
    # Enmienda 5: selección conjunta de (K, α) por menor NLL de validación.
    tokens, *_ = _simulate_hmm(200, 8)
    split = porobra.split_tokens(tokens)
    result = porobra.fit_finite(tokens, split, grid=(2, 4), alphas=(0.1, 1.0), seed=10, n_restarts=3)
    assert result["finite_restarts"] == 3
    scores = {}
    for alpha in (0.1, 1.0):
        for k in (2, 4):
            best = porobra.best_baum_welch(split.inner_train, n_states=k, seeds=(10, 11, 12), pseudocount=alpha)
            scores[(k, alpha)] = porobra.hmm_conditional_nll(
                best.initial_probs, best.transition_matrix, best.emission_matrix, split.inner_train, split.validation
            )
            assert result["finite_validation_nll"][alpha][k] == pytest.approx(scores[(k, alpha)])
    assert (result["finite_k"], result["finite_alpha"]) == min(scores, key=scores.get)
    refit = porobra.best_baum_welch(
        split.fit, n_states=result["finite_k"], seeds=(10, 11, 12), pseudocount=result["finite_alpha"]
    )
    assert result["finite_insample_ll_per_note"] == pytest.approx(refit.log_likelihood / split.fit.size)
    assert result["finite_test_nll"] == pytest.approx(
        porobra.hmm_conditional_nll(*porobra._params(refit), split.fit, split.test)
    )


def test_seleccion_conjunta_amplia_k_si_elige_el_maximo_con_cualquier_alpha():
    # Mejor con K grande hasta 48 y con α = 0.5: la rejilla de K crece para todos los α.
    def score(k, alpha):
        return (-k if k <= 48 else k) + abs(alpha - 0.5)

    choice = porobra.select_k_alpha((2, 4, 8, 16, 32), (0.1, 0.5, 1.0), score, max_value=512)
    assert (choice.k, choice.alpha) == (48, 0.5)
    assert choice.k_grid == (2, 4, 8, 16, 32, 48, 64)
    assert choice.grid_extensions == 1 and not choice.hit_limit
    assert set(choice.validation_nll) == {0.1, 0.5, 1.0}
    assert all(set(by_k) == set(choice.k_grid) for by_k in choice.validation_nll.values())


def test_configuracion_por_defecto_usa_cinco_reinicios_y_alphas_del_preregistro():
    config = porobra.PorObraConfig()
    assert config.finite_restarts == 5
    assert config.finite_alphas == (0.1, 0.5, 1.0)


def test_finito_elige_pocos_estados_con_datos_de_dos():
    tokens, *_ = _simulate_hmm(500, 4)
    result = porobra.fit_finite(tokens, porobra.split_tokens(tokens), grid=(2, 4, 8), seed=0)
    assert result["finite_k"] in (2, 4)
    # Con emisiones disjuntas de 3 símbolos la tasa de entropía es cercana a ln 3.
    assert result["finite_test_nll"] < math.log(12)
    assert abs(result["finite_test_nll"] - math.log(3)) < 0.4


# --- HDP-HMM por obra ---------------------------------------------------------------

FAST_HDP = dict(truncation=6, n_chains=2, n_iters=16, burn_in=8, etas=(1.0, 3.0))


def test_hdp_por_obra_elige_eta_y_reporta_rhat_y_estados():
    tokens, *_ = _simulate_hmm(80, 5)
    result = porobra.fit_hdp(tokens, porobra.split_tokens(tokens), **FAST_HDP)
    assert result["hdp_eta"] in (1.0, 3.0)
    assert set(result["hdp_eta_validation_nll"]) == {1.0, 3.0}
    assert result["hdp_eta"] == min(result["hdp_eta_validation_nll"], key=result["hdp_eta_validation_nll"].get)
    assert math.isfinite(result["hdp_test_nll"]) and result["hdp_test_nll"] > 0
    assert math.isfinite(result["hdp_rhat"])
    assert result["hdp_converged"] == (result["hdp_rhat"] < 1.1)
    # Enmienda 6, punto 4: los estados del HDP se reportan por cadena; el agregado de todas
    # las cadenas lleva otro nombre para no leerse como una media posterior junto al finito.
    assert "hdp_states_mean" not in result and "hdp_states_sd" not in result
    assert 1.0 <= result["hdp_states_pooled_chains_mean"] <= 6.0
    assert result["hdp_states_pooled_chains_sd"] >= 0.0
    keys = list(result)
    assert keys.index("hdp_states_mean_by_chain") < keys.index("hdp_states_pooled_chains_mean")
    assert math.isfinite(result["hdp_insample_ll_per_note"])


@pytest.fixture(scope="module")
def hdp_chains():
    tokens, *_ = _simulate_hmm(80, 5)
    split = porobra.split_tokens(tokens)
    chains = porobra._hdp_chains(
        split.fit, eta=1.0, truncation=6, n_chains=3, n_iters=16, burn_in=8, threshold=0.02, hyperparameters={}
    )
    return split, chains


def test_nll_de_prueba_por_muestra_es_la_del_hmm_con_esos_parametros(hdp_chains):
    split, chains = hdp_chains
    traces = porobra.per_sample_test_nll(chains, split.fit, split.test)
    assert [len(trace) for trace in traces] == [len(chain.posterior_samples) for chain in chains]
    sample = chains[1].posterior_samples[3]
    expected = porobra.hmm_conditional_nll(
        sample.initial_probs, sample.transition_matrix, sample.emission_matrix, split.fit, split.test
    )
    assert traces[1][3] == pytest.approx(expected)


def test_diagnostico_por_cadena_enmienda_6(hdp_chains):
    split, chains = hdp_chains
    result = porobra.hdp_chain_diagnostics(chains, split.fit, split.test)
    by_chain = result["hdp_test_nll_by_chain"]
    assert by_chain == pytest.approx([porobra.heldout_nll_per_token([c], split.fit, split.test) for c in chains])
    assert result["hdp_test_nll_chain_range"] == pytest.approx(max(by_chain) - min(by_chain))
    # Con el mismo número de muestras por cadena, el ensamble es la media de las predictivas.
    n = split.test.size
    ensemble = porobra.heldout_nll_per_token(chains, split.fit, split.test)
    assert math.exp(-n * ensemble) == pytest.approx(np.mean([math.exp(-n * v) for v in by_chain]), rel=1e-9)
    traces = porobra.per_sample_test_nll(chains, split.fit, split.test)
    assert result["hdp_predictive_rhat"] == pytest.approx(porobra.split_rhat(traces))
    occupied = [chain.diagnostics.occupied_state_samples for chain in chains]
    assert result["hdp_states_mean_by_chain"] == pytest.approx([np.mean(o) for o in occupied])
    assert result["hdp_states_sd_by_chain"] == pytest.approx([np.std(o) for o in occupied])


def test_hdp_por_obra_lleva_el_diagnostico_por_cadena():
    tokens, *_ = _simulate_hmm(80, 5)
    result = porobra.fit_hdp(tokens, porobra.split_tokens(tokens), **FAST_HDP)
    assert len(result["hdp_test_nll_by_chain"]) == FAST_HDP["n_chains"]
    assert len(result["hdp_states_mean_by_chain"]) == FAST_HDP["n_chains"]
    assert math.isfinite(result["hdp_predictive_rhat"])
    assert result["hdp_test_nll_chain_range"] >= 0.0


def test_analisis_de_una_obra_es_determinista_y_completo():
    tokens, *_ = _simulate_hmm(60, 6)
    config = porobra.PorObraConfig(k_grid=(2, 4), **FAST_HDP)
    first = porobra.analyze_work("obra", tokens, config)
    second = porobra.analyze_work("obra", tokens, config)
    timing = [key for key in first if key.endswith("wall_clock_s")]
    assert timing == ["hdp_wall_clock_s", "wall_clock_s"]
    assert {k: v for k, v in first.items() if k not in timing} == {k: v for k, v in second.items() if k not in timing}
    for model in porobra.MODELS:
        assert math.isfinite(first[f"{model}_test_nll"])
    assert first["n_events"] == 60 and first["n_fit"] == 48 and first["n_test"] == 12


# --- comparaciones del preregistro ---------------------------------------------------


def _rows(n_works: int, hdp_shift: float, *, converged=True) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for index in range(n_works):
        base = 2.0 + rng.normal(0, 0.1)
        rows.append(
            {
                "work_id": f"w{index}",
                "n_test": 10 + index,
                "hdp_converged": converged,
                "hdp_test_nll": base + hdp_shift + rng.normal(0, 0.01),
                "finite_test_nll": base,
                "bigram_test_nll": base,
                "unigram_test_nll": base,
            }
        )
    return rows


def test_familia_de_tres_y_veredicto_mejora():
    result = porobra.compare_hdp(_rows(30, -0.5), bootstrap_samples=2000, seed=0)
    comparisons = result["comparisons"]
    assert [c["model_b"] for c in comparisons] == ["finite", "bigram", "unigram"]
    assert result["holm_family"]["family_size"] == 3
    for comparison in comparisons:
        assert comparison["mean_difference"] == pytest.approx(-0.5, abs=0.02)
        assert comparison["verdict"] == "mejora"


def test_veredicto_x_es_mejor_y_sin_evidencia():
    worse = porobra.compare_hdp(_rows(30, 0.5), bootstrap_samples=2000, seed=0)
    assert {c["verdict"] for c in worse["comparisons"]} == {"es_mejor_el_otro"}
    rows = _rows(30, 0.0)
    for index, row in enumerate(rows):
        row["hdp_test_nll"] = row["finite_test_nll"] + (0.3 if index % 2 else -0.3)
        row["bigram_test_nll"] = row["unigram_test_nll"] = row["finite_test_nll"]
    tie = porobra.compare_hdp(rows, bootstrap_samples=2000, seed=0)
    assert {c["verdict"] for c in tie["comparisons"]} == {"no_hay_evidencia"}


def test_metrica_principal_incluye_las_obras_sin_convergencia_enmienda_6():
    # Enmienda 6: ninguna obra se excluye por el R-hat del HDP; la regla original queda
    # como sensibilidad (sólo obras con R-hat < 1.1).
    rows = _rows(20, -0.5) + _rows(5, 5.0, converged=False)
    for index, row in enumerate(rows[20:]):
        row["work_id"] = f"nc{index}"
    main = porobra.compare_hdp(rows, bootstrap_samples=500, seed=0)
    assert main["comparisons"][0]["n_pairs"] == 25
    assert main["n_works"] == 25 and main["converged_only"] is False
    expected = np.mean([row["hdp_test_nll"] - row["finite_test_nll"] for row in rows])
    assert main["comparisons"][0]["mean_difference"] == pytest.approx(expected)
    sensitivity = porobra.compare_hdp(rows, bootstrap_samples=500, seed=0, converged_only=True)
    assert sensitivity["comparisons"][0]["n_pairs"] == 20
    assert sensitivity["n_works"] == 20 and sensitivity["converged_only"] is True
    assert sensitivity["comparisons"][0]["verdict"] == "mejora"


def test_estimando_que_decide_es_la_media_por_obra_no_la_ponderada():
    # Una obra larga con HDP peor y muchas cortas con HDP mejor: los signos se separan.
    rows = []
    for index in range(20):
        rows.append({"work_id": f"c{index}", "n_test": 5, "hdp_converged": True, "hdp_test_nll": 1.9,
                     "finite_test_nll": 2.0, "bigram_test_nll": 2.0, "unigram_test_nll": 2.0})
    rows.append({"work_id": "larga", "n_test": 10_000, "hdp_converged": True, "hdp_test_nll": 3.0,
                 "finite_test_nll": 2.0, "bigram_test_nll": 2.0, "unigram_test_nll": 2.0})
    comparison = porobra.compare_hdp(rows, bootstrap_samples=500, seed=0)["comparisons"][0]
    assert comparison["mean_difference"] == pytest.approx((20 * -0.1 + 1.0) / 21)
    assert comparison["mean_difference"] < 0 < comparison["token_weighted_difference"]
    assert comparison["decision_estimand"] == "mean_over_works_of_work_nll_difference"


def test_con_diez_obras_la_seccion_a_ya_da_veredicto():
    # Enmienda 11: el mínimo de la sección A es 10 obras pareadas.
    result = porobra.compare_hdp(_rows(10, -0.5), bootstrap_samples=500, seed=0)
    for comparison in result["comparisons"]:
        assert comparison["n_pairs"] == 10
        assert comparison["work_mean_holm_status"] == "ok"
        assert comparison["wilcoxon_holm_status"] == "ok" and comparison["p_value_holm"] is not None
        assert comparison["token_weighted_holm_status"] == "ok"
        assert comparison["verdict"] == "mejora"
    assert result["min_paired_works"] == 10
    assert result["holm_family"]["work_mean_ci"]["min_paired_works"] == 10
    assert result["holm_family"]["work_mean_ci"]["n_evaluable"] == 3


def test_la_sensibilidad_con_diez_obras_tambien_se_evalua():
    rows = _rows(10, -0.5) + _rows(20, -0.5, converged=False)
    for index, row in enumerate(rows[10:]):
        row["work_id"] = f"nc{index}"
    sensitivity = porobra.compare_hdp(rows, bootstrap_samples=500, seed=0, converged_only=True)
    assert sensitivity["n_works"] == 10
    assert {c["verdict"] for c in sensitivity["comparisons"]} == {"mejora"}


def test_con_pocas_obras_el_bloque_lleva_la_advertencia_y_el_veredicto_no_cambia():
    # Con pocas obras el bootstrap percentil es degenerado: se rotula, no se decide otra cosa.
    small = porobra.compare_hdp(_rows(porobra.SMALL_N_WARNING_WORKS - 1, -0.5), bootstrap_samples=200, seed=0)
    assert small["small_n_warning"] is True and "no cambia" in small["small_n_note"]
    assert {c["verdict"] for c in small["comparisons"]} == {"mejora"}
    assert small["min_paired_works"] == 10
    # Sin tasas escritas a mano: la advertencia es cualitativa.
    assert not any(character.isdigit() for character in small["small_n_note"])
    enough = porobra.compare_hdp(_rows(porobra.SMALL_N_WARNING_WORKS, -0.5), bootstrap_samples=200, seed=0)
    assert enough["small_n_warning"] is False and enough["small_n_note"] is None


def test_una_sensibilidad_con_pocas_obras_avisa_junto_al_cambio_de_veredicto():
    rows = _rows(20, -0.5) + _rows(25, 1.5)
    for index, row in enumerate(rows):
        row["work_id"] = f"w{index}"
    # Sólo quedan 12 obras convergidas, con otro signo que el resto.
    for row in rows:
        row["hdp_converged"] = False
    for row in rows[:12]:
        row["hdp_converged"] = True
    blocks = porobra.summary_blocks(rows, bootstrap_samples=300, seed=0, exposed_work_ids=[])
    converged = blocks["sensitivity_converged_only"]
    assert converged["n_works"] == 12 and converged["any_verdict_changes"] is True
    assert converged["small_n_warning"] is True
    assert blocks["comparisons"]["small_n_warning"] is False


def test_los_bloques_del_resumen_dicen_si_algun_veredicto_cambia():
    # Aclaración 8: el análisis sin las obras expuestas «indica si algún veredicto cambia».
    rows = _rows(20, -0.5) + _rows(25, 1.5)
    for index, row in enumerate(rows[20:]):
        row["work_id"] = f"x{index}"
    exposed = [row["work_id"] for row in rows[20:]]
    blocks = porobra.summary_blocks(rows, bootstrap_samples=500, seed=0, exposed_work_ids=exposed)
    assert {c["verdict"] for c in blocks["comparisons"]["comparisons"]} == {"es_mejor_el_otro"}
    without = blocks["sensitivity_without_exposed"]
    assert without["n_works"] == 20 and without["any_verdict_changes"] is True
    assert without["verdict_changes"][0] == {"model_b": "finite", "principal": "es_mejor_el_otro", "sensitivity": "mejora"}
    assert "Aclaración 8" in without["rule"]
    converged = blocks["sensitivity_converged_only"]
    assert converged["any_verdict_changes"] is False and converged["verdict_changes"] == []
    assert blocks["convergence"]["n_works"] == 45


def test_no_evaluable_no_es_un_cambio_de_veredicto_y_se_reporta_aparte():
    # Sin las obras expuestas puede quedar una sola obra: eso no es un cambio, sino «no se puede decir».
    rows = _rows(20, -0.5)
    exposed = [row["work_id"] for row in rows[1:]]
    blocks = porobra.summary_blocks(rows, bootstrap_samples=500, seed=0, exposed_work_ids=exposed)
    assert {c["verdict"] for c in blocks["comparisons"]["comparisons"]} == {"mejora"}
    without = blocks["sensitivity_without_exposed"]
    assert {c["verdict"] for c in without["comparisons"]} == {"no_evaluable"}
    assert without["any_verdict_changes"] is False and without["verdict_changes"] == []
    assert without["verdict_not_evaluable"] == [
        {"model_b": c["model_b"], "principal": "mejora", "sensitivity": "no_evaluable"} for c in without["comparisons"]
    ]
    converged = blocks["sensitivity_converged_only"]
    assert converged["any_verdict_changes"] is False and converged["verdict_not_evaluable"] == []


def test_si_tambien_la_principal_es_no_evaluable_se_reporta_aparte():
    blocks = porobra.summary_blocks(_rows(1, -0.5), bootstrap_samples=200, seed=0, exposed_work_ids=[])
    without = blocks["sensitivity_without_exposed"]
    assert without["verdict_changes"] == [] and without["any_verdict_changes"] is False
    assert {item["principal"] for item in without["verdict_not_evaluable"]} == {"no_evaluable"}
    assert len(without["verdict_not_evaluable"]) == len(without["comparisons"])


def test_un_cambio_real_no_aparece_como_no_evaluable():
    rows = _rows(20, -0.5) + _rows(25, 1.5)
    for index, row in enumerate(rows[20:]):
        row["work_id"] = f"x{index}"
    blocks = porobra.summary_blocks(rows, bootstrap_samples=500, seed=0, exposed_work_ids=[row["work_id"] for row in rows[20:]])
    without = blocks["sensitivity_without_exposed"]
    assert without["any_verdict_changes"] is True and without["verdict_not_evaluable"] == []


def test_con_una_sola_obra_no_es_evaluable():
    result = porobra.compare_hdp(_rows(1, -0.5), bootstrap_samples=200, seed=0)
    assert result["n_works"] == 1
    assert {c["verdict"] for c in result["comparisons"]} == {"no_evaluable"}
    assert {c["work_mean_holm_status"] for c in result["comparisons"]} == {"too_few_paired_works"}
    empty = porobra.compare_hdp([], bootstrap_samples=200, seed=0)
    assert empty["n_works"] == 0 and {c["verdict"] for c in empty["comparisons"]} == {"no_evaluable"}


# --- resumen -------------------------------------------------------------------------


def test_resumen_reporta_la_fraccion_que_converge_sin_mandar_duplicar_iteraciones():
    # Enmienda 6: se reporta la fracción con R-hat de log L < 1.1; la contingencia de
    # duplicar iteraciones ya no aplica y la convergencia no excluye obras.
    rows = _rows(18, -0.5) + _rows(2, 0.0, converged=False)
    for index, row in enumerate(rows[18:]):
        row["work_id"] = f"nc{index}"
    summary = porobra.convergence_summary(rows)
    assert summary["n_works"] == 20 and summary["n_converged"] == 18
    assert summary["fraction_converged"] == pytest.approx(0.9)
    assert summary["not_converged_work_ids"] == ["nc0", "nc1"]
    assert "action_required" not in summary and "exceeds_10_percent" not in summary
    assert "Enmienda 6" in summary["rule"]
