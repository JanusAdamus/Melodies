"""Análisis por obra held-out: preregistro 2026-10-26, sección A, Aclaración 2, Enmiendas 3, 5 y 6.

Por obra, partición contigua: el primer 80 % de los eventos ajusta y el último 20 % evalúa;
dentro del 80 %, su último 20 % valida los hiperparámetros (α de los n-gramas, K del finito,
η del HDP, α del finito) y después se reajusta con el 80 % completo. La métrica es la NLL por nota en
prueba, −log p(x_prueba | x_ajuste) / n_prueba, en nats, sobre las 12 clases de altura.

* n-gramas: predictiva posterior exacta de Dirichlet(α) (los conteos se actualizan con cada
  nota de prueba, que es ∫ p(x_prueba | θ) p(θ | x_ajuste) dθ en forma cerrada).
* HMM finito: Baum-Welch (estimador puntual) con seudoconteo de Dirichlet α en emisiones y
  transiciones; (K, α) se eligen juntos en validación (Enmienda 5); el mejor de
  ``finite_restarts`` inicios por log-verosimilitud de entrenamiento; la prueba se puntúa
  filtrando el estado a través de la frontera ajuste/prueba.
* HDP-HMM: predictiva posterior promediada sobre las muestras tras burn-in de todas las
  cadenas (`heldout_nll_per_token`), el «ensamble de cadenas» de la Enmienda 6: ninguna obra
  se excluye por su R-hat; por cadena se reportan la NLL de prueba, el R-hat de la NLL
  predictiva por muestra y los estados.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from Comparacion.classical_models import _expectation_batch
from Comparacion.grid import next_grid_candidates
from Comparacion.statistics import (
    _apply_bootstrap_holm,
    _apply_holm_adjustment,
    _bootstrap_mean_draws,
    _empty_comparison,
    _holm_family_record,
    _token_weighted_difference,
)
from src.data.observations import PITCH_CLASS_NAMES, ObservationSequence
from src.models.hdp_hmm import (
    TruncatedHDPHMM,
    _forward_step_log_scales,
    heldout_nll_per_token,
    log_likelihood_rhat,
    split_rhat,
)
from src.models.inference import ffbs_sample_batch, forward_log_likelihood
from src.models.utils import EPSILON, occupied_state_count

VOCAB_SIZE = 12
MIN_EVENTS = 20
FIT_FRACTION = 0.8
#: Orden de la familia preregistrada: HDP − finito, HDP − bigrama, HDP − unigrama.
COMPARED_MODELS = ("finite", "bigram", "unigram")
MODELS = ("hdp",) + COMPARED_MODELS
RHAT_THRESHOLD = 1.1
CONVERGENCE_RULE = (
    "Enmienda 6: ninguna obra se excluye por el R-hat del HDP-HMM; se reporta la fracción con "
    "R-hat de log L < 1.1 y la comparación restringida a esas obras es sólo sensibilidad."
)
HDP_STATES_NOTE = (
    "Estados del HDP-HMM por cadena (Enmienda 6): sin convergencia entre cadenas no son una "
    "media posterior y no se interpretan como inferencia bayesiana sobre K."
)
DECISION_ESTIMAND = "mean_over_works_of_work_nll_difference"
#: Seudoconteo de la distribución inicial del finito (una sola secuencia: casi no pesa).
INITIAL_PSEUDOCOUNT = 1e-3


# --- partición -----------------------------------------------------------------------


@dataclass(frozen=True)
class ContiguousSplit:
    n_events: int
    n_fit: int
    n_test: int
    n_inner_train: int
    n_validation: int


def contiguous_split(n_events: int, min_events: int = MIN_EVENTS) -> ContiguousSplit:
    """Primer ⌊0.8 n⌋ ajusta, el resto prueba; dentro del ajuste, la misma regla valida."""

    if n_events < min_events:
        raise ValueError(f"La obra tiene {n_events} eventos; el mínimo es {min_events}.")
    n_fit = int(math.floor(FIT_FRACTION * n_events))
    n_inner = int(math.floor(FIT_FRACTION * n_fit))
    return ContiguousSplit(n_events, n_fit, n_events - n_fit, n_inner, n_fit - n_inner)


@dataclass(frozen=True)
class TokenSplit:
    fit: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    validation: np.ndarray


def split_tokens(tokens: Sequence[int]) -> TokenSplit:
    tokens = np.asarray(tokens, dtype=int)
    split = contiguous_split(tokens.size)
    return TokenSplit(
        fit=tokens[: split.n_fit],
        test=tokens[split.n_fit :],
        inner_train=tokens[: split.n_inner_train],
        validation=tokens[split.n_inner_train : split.n_fit],
    )


# --- selección en validación ---------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    selected: float
    validation_nll: dict
    grid: tuple
    grid_extensions: int = 0
    #: La selección quedó en el máximo permitido: la rejilla no pudo crecer más.
    hit_limit: bool = False


def select_by_validation(
    grid: Iterable,
    score: Callable[[object], float],
    *,
    extend: bool = False,
    max_value: int | None = None,
) -> Selection:
    """Elige el candidato de menor NLL de validación (empates: el primero de la rejilla).

    Con ``extend`` (K del finito), mientras se elija el máximo la rejilla crece con
    `next_grid_candidates` (×1.5 y ×2), sin pasar de ``max_value``.
    """

    candidates = list(grid)
    scores: dict = {}
    pending = list(candidates)
    extensions = 0
    hit_limit = False
    while pending:
        for value in pending:
            scores[value] = float(score(value))
        pending = []
        best = min(candidates, key=lambda value: scores[value])
        if extend and best == max(candidates):
            following = [value for value in next_grid_candidates(int(best)) if max_value is None or value <= max_value]
            if following:
                candidates.extend(following)
                pending = following
                extensions += 1
            else:
                hit_limit = True
    best = min(candidates, key=lambda value: scores[value])
    return Selection(best, scores, tuple(candidates), extensions, hit_limit)


@dataclass(frozen=True)
class JointSelection:
    k: int
    alpha: float
    #: NLL de validación por α y luego por K.
    validation_nll: dict
    k_grid: tuple
    grid_extensions: int = 0
    hit_limit: bool = False


def select_k_alpha(
    grid: Iterable[int],
    alphas: Sequence[float],
    score: Callable[[int, float], float],
    *,
    max_value: int | None = None,
) -> JointSelection:
    """Selección conjunta de (K, α) por menor NLL de validación (Enmienda 5).

    Cada K se califica con su mejor α, así que el mínimo sobre K es el mínimo conjunto y la
    regla de ampliar la rejilla de K cuando se elige el máximo se mantiene. Empates: el
    primer K de la rejilla y, dentro de él, el primer α.
    """

    alphas = tuple(float(a) for a in alphas)
    scores: dict[float, dict[int, float]] = {alpha: {} for alpha in alphas}

    def best_over_alpha(k) -> float:
        for alpha in alphas:
            scores[alpha][int(k)] = float(score(int(k), alpha))
        return min(scores[alpha][int(k)] for alpha in alphas)

    by_k = select_by_validation(tuple(int(k) for k in grid), best_over_alpha, extend=True, max_value=max_value)
    k = int(by_k.selected)
    alpha = min(alphas, key=lambda a: scores[a][k])
    return JointSelection(k, alpha, scores, tuple(int(v) for v in by_k.grid), by_k.grid_extensions, by_k.hit_limit)


# --- n-gramas ------------------------------------------------------------------------


def ngram_conditional_nll(context: Sequence[int], target: Sequence[int], *, order: int, alpha: float) -> float:
    """−log p(target | context) / n con la predictiva posterior de Dirichlet(α) simétrica."""

    context = np.asarray(context, dtype=int)
    target = np.asarray(target, dtype=int)
    if target.size == 0:
        raise ValueError("La parte evaluada no puede estar vacía.")
    log_likelihood = 0.0
    if order == 1:
        counts = np.bincount(context, minlength=VOCAB_SIZE).astype(float)
        for token in target:
            log_likelihood += math.log((counts[token] + alpha) / (counts.sum() + VOCAB_SIZE * alpha))
            counts[token] += 1.0
    elif order == 2:
        if context.size == 0:
            raise ValueError("El bigrama necesita al menos una nota de contexto.")
        counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=float)
        np.add.at(counts, (context[:-1], context[1:]), 1.0)
        previous = int(context[-1])
        for token in target:
            row = counts[previous]
            log_likelihood += math.log((row[token] + alpha) / (row.sum() + VOCAB_SIZE * alpha))
            row[token] += 1.0
            previous = int(token)
    else:
        raise ValueError("Sólo unigrama (1) y bigrama (2).")
    return -log_likelihood / target.size


def select_ngram_alpha(tokens: Sequence[int], *, order: int, alphas: Sequence[float]) -> Selection:
    split = split_tokens(tokens)
    return select_by_validation(
        tuple(float(a) for a in alphas),
        lambda a: ngram_conditional_nll(split.inner_train, split.validation, order=order, alpha=a),
    )


def fit_ngram(split: TokenSplit, *, order: int, alphas: Sequence[float], name: str) -> dict:
    inner = select_by_validation(
        tuple(float(a) for a in alphas),
        lambda a: ngram_conditional_nll(split.inner_train, split.validation, order=order, alpha=a),
    )
    return {
        f"{name}_alpha": inner.selected,
        f"{name}_alpha_validation_nll": inner.validation_nll,
        f"{name}_test_nll": ngram_conditional_nll(split.fit, split.test, order=order, alpha=inner.selected),
    }


# --- HMM finito ----------------------------------------------------------------------


@dataclass(frozen=True)
class HMMFit:
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray
    log_likelihood: float
    log_likelihood_trace: tuple = ()


@dataclass(frozen=True)
class _Parameters:
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray


def _log_scales(initial: np.ndarray, transition: np.ndarray, emission: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    return _forward_step_log_scales([_Parameters(initial, transition, emission)], np.asarray(tokens, dtype=int))[0]


def hmm_conditional_nll(initial, transition, emission, context, target) -> float:
    """−log p(target | context, θ) / n: forward por la secuencia completa, se suman los pasos de target."""

    context = np.asarray(context, dtype=int)
    target = np.asarray(target, dtype=int)
    scales = _log_scales(initial, transition, emission, np.concatenate([context, target]))
    return -float(scales[context.size :].sum()) / target.size


def baum_welch(
    tokens: Sequence[int],
    *,
    n_states: int,
    seed: int,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    pseudocount: float = 1e-3,
) -> HMMFit:
    """EM de Baum-Welch sobre una secuencia, inicio Dirichlet(1) con `seed`.

    ``pseudocount`` (el α de la Enmienda 5) se suma a cada conteo esperado de transición y
    de emisión; la distribución inicial lleva el seudoconteo fijo ``INITIAL_PSEUDOCOUNT``.
    Ninguna probabilidad queda en 0. Para cuando la mejora de log p(x|θ) es < ``tolerance``.
    """

    tokens = np.asarray(tokens, dtype=int)
    rng = np.random.default_rng(seed)
    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    emission = rng.dirichlet(np.ones(VOCAB_SIZE), size=n_states)
    padded, lengths = tokens[None, :], np.array([tokens.size])
    trace: list[float] = []
    for _ in range(max_iterations):
        counts_initial, counts_transition, counts_emission, log_likelihood = _expectation_batch(
            initial, transition, emission, padded, lengths
        )
        trace.append(float(log_likelihood))
        initial = (counts_initial + INITIAL_PSEUDOCOUNT) / (counts_initial + INITIAL_PSEUDOCOUNT).sum()
        transition = counts_transition + pseudocount
        transition /= transition.sum(axis=1, keepdims=True)
        emission = counts_emission + pseudocount
        emission /= emission.sum(axis=1, keepdims=True)
        if len(trace) > 1 and trace[-1] - trace[-2] < tolerance:
            break
    final = float(_log_scales(initial, transition, emission, tokens).sum())
    return HMMFit(initial, transition, emission, final, tuple(trace) + (final,))


def best_baum_welch(tokens: Sequence[int], *, n_states: int, seeds: Sequence[int], **kwargs) -> HMMFit:
    """Baum-Welch desde varias inicializaciones; devuelve el de mayor log p(x|θ) en ``tokens``.

    Empates: el de la primera semilla. Sólo usa los datos de ajuste, nunca la validación.
    """

    fits = [baum_welch(tokens, n_states=n_states, seed=int(seed), **kwargs) for seed in seeds]
    return max(fits, key=lambda fit: fit.log_likelihood)


@dataclass(frozen=True)
class Occupancy:
    mean: float
    sd: float


def posterior_occupied_states(fit: HMMFit, tokens, *, threshold: float, n_samples: int, seed: int) -> Occupancy:
    """Media y d.e., sobre trayectorias de p(z | x, θ) (FFBS), de los estados con ocupación ≥ umbral."""

    tokens = np.asarray(tokens, dtype=int)
    paths, _ = ffbs_sample_batch(
        fit.initial_probs, fit.transition_matrix, fit.emission_matrix, [tokens] * n_samples, np.random.default_rng(seed)
    )
    counts = [occupied_state_count(path, fit.transition_matrix.shape[0], threshold) for path in paths]
    return Occupancy(float(np.mean(counts)), float(np.std(counts)))


def fit_finite(
    tokens,
    split: TokenSplit,
    *,
    grid: Sequence[int] = (2, 4, 8, 16, 32),
    alphas: Sequence[float] = (0.1, 0.5, 1.0),
    seed: int = 0,
    n_restarts: int = 5,
    max_states: int = 512,
    threshold: float = 0.02,
    n_occupancy_samples: int = 200,
) -> dict:
    seeds = tuple(range(seed, seed + n_restarts))
    selection = select_k_alpha(
        grid,
        alphas,
        lambda k, alpha: hmm_conditional_nll(
            *_params(best_baum_welch(split.inner_train, n_states=k, seeds=seeds, pseudocount=alpha)),
            split.inner_train,
            split.validation,
        ),
        max_value=max_states,
    )
    k, alpha = selection.k, selection.alpha
    fit = best_baum_welch(split.fit, n_states=k, seeds=seeds, pseudocount=alpha)
    occupancy = posterior_occupied_states(fit, split.fit, threshold=threshold, n_samples=n_occupancy_samples, seed=seed)
    return {
        "finite_k": k,
        "finite_alpha": alpha,
        "finite_restarts": int(n_restarts),
        "finite_validation_nll": selection.validation_nll,
        "finite_selected_validation_nll": selection.validation_nll[alpha][k],
        "finite_k_grid": list(selection.k_grid),
        "finite_grid_extensions": selection.grid_extensions,
        "finite_k_hit_limit": selection.hit_limit,
        "finite_test_nll": hmm_conditional_nll(*_params(fit), split.fit, split.test),
        "finite_states_mean": occupancy.mean,
        "finite_states_sd": occupancy.sd,
        "finite_insample_ll_per_note": fit.log_likelihood / split.fit.size,
    }


def _params(fit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fit.initial_probs, fit.transition_matrix, fit.emission_matrix


# --- HDP-HMM -------------------------------------------------------------------------


def _observations(tokens: np.ndarray) -> ObservationSequence:
    tokens = np.asarray(tokens, dtype=int)
    vocabulary = list(PITCH_CLASS_NAMES)
    return ObservationSequence("pitch_class", tokens, vocabulary, [vocabulary[t] for t in tokens], [], {})


def _hdp_chains(tokens, *, eta, truncation, n_chains, n_iters, burn_in, threshold, hyperparameters) -> list:
    observations = _observations(tokens)
    return [
        TruncatedHDPHMM(
            n_states=truncation,
            eta=eta,
            n_iters=n_iters,
            burn_in=burn_in,
            seed=seed,
            occupancy_threshold=threshold,
            **hyperparameters,
        ).fit(observations)
        for seed in range(1, n_chains + 1)
    ]


def per_sample_test_nll(chains, fit_tokens, test_tokens) -> list[np.ndarray]:
    """−log p(x_prueba | x_ajuste, θ_s) / n_prueba para cada muestra tras burn-in, por cadena."""

    fit_tokens = np.asarray(fit_tokens, dtype=int)
    test_tokens = np.asarray(test_tokens, dtype=int)
    tokens = np.concatenate([fit_tokens, test_tokens])
    return [
        -_forward_step_log_scales(chain.posterior_samples, tokens)[:, fit_tokens.size :].sum(axis=1) / test_tokens.size
        for chain in chains
    ]


def hdp_chain_diagnostics(chains, fit_tokens, test_tokens) -> dict:
    """Diagnóstico por cadena de la Enmienda 6 (se reporta; no decide ni excluye obras)."""

    by_chain = [heldout_nll_per_token([chain], fit_tokens, test_tokens) for chain in chains]
    occupied = [chain.diagnostics.occupied_state_samples for chain in chains]
    return {
        "hdp_test_nll_by_chain": by_chain,
        "hdp_test_nll_chain_range": float(max(by_chain) - min(by_chain)),
        "hdp_predictive_rhat": split_rhat(per_sample_test_nll(chains, fit_tokens, test_tokens)),
        "hdp_states_mean_by_chain": [float(np.mean(o)) for o in occupied],
        "hdp_states_sd_by_chain": [float(np.std(o)) for o in occupied],
    }


def fit_hdp(
    tokens,
    split: TokenSplit,
    *,
    truncation: int = 20,
    n_chains: int = 4,
    n_iters: int = 400,
    burn_in: int = 200,
    etas: Sequence[float] = (1.0, 3.0, 6.0),
    threshold: float = 0.02,
    rhat_threshold: float = RHAT_THRESHOLD,
    hyperparameters: Mapping[str, float] | None = None,
) -> dict:
    """η en validación interna (Enmienda 3), reajuste con el 80 %, 4 cadenas con semillas 1..n."""

    settings = dict(
        truncation=truncation,
        n_chains=n_chains,
        n_iters=n_iters,
        burn_in=burn_in,
        threshold=threshold,
        hyperparameters=dict(hyperparameters or {}),
    )
    selection = select_by_validation(
        tuple(float(e) for e in etas),
        lambda eta: heldout_nll_per_token(
            _hdp_chains(split.inner_train, eta=eta, **settings), split.inner_train, split.validation
        ),
    )
    eta = float(selection.selected)
    chains = _hdp_chains(split.fit, eta=eta, **settings)
    rhat = log_likelihood_rhat(chains)
    occupied = [count for chain in chains for count in chain.diagnostics.occupied_state_samples]
    # Secundario: log p(x_ajuste | media posterior de θ) por nota, cadena a cadena (las
    # etiquetas sólo están alineadas dentro de una cadena), promediado entre cadenas.
    insample = [
        forward_log_likelihood(
            chain.posterior_initial_mean, chain.posterior_transition_mean, chain.posterior_emission_mean, split.fit
        )[0]
        for chain in chains
    ]
    return {
        "hdp_eta": eta,
        "hdp_eta_validation_nll": selection.validation_nll,
        "hdp_test_nll": heldout_nll_per_token(chains, split.fit, split.test),
        "hdp_rhat": rhat,
        "hdp_converged": bool(math.isfinite(rhat) and rhat < rhat_threshold),
        "hdp_states_mean": float(np.mean(occupied)),
        "hdp_states_sd": float(np.std(occupied)),
        "hdp_insample_ll_per_note": float(np.mean(insample)) / split.fit.size,
        "hdp_posterior_mean_log_likelihood_per_note": float(np.mean([c.log_likelihood for c in chains])) / split.fit.size,
        **hdp_chain_diagnostics(chains, split.fit, split.test),
    }


# --- una obra ------------------------------------------------------------------------


@dataclass(frozen=True)
class PorObraConfig:
    alphas: tuple = (0.1, 0.5, 1.0)
    k_grid: tuple = (2, 4, 8, 16, 32)
    #: Seudoconteo de Dirichlet de Baum-Welch, elegido junto con K (Enmienda 5).
    finite_alphas: tuple = (0.1, 0.5, 1.0)
    max_states: int = 512
    finite_seed: int = 0
    #: Inicializaciones de Baum-Welch por K (semillas finite_seed..+n−1); gana la de mayor
    #: log-verosimilitud de entrenamiento, en la validación y en el reajuste.
    finite_restarts: int = 5
    etas: tuple = (1.0, 3.0, 6.0)
    truncation: int = 20
    n_chains: int = 4
    n_iters: int = 400
    burn_in: int = 200
    occupancy_threshold: float = 0.02
    rhat_threshold: float = RHAT_THRESHOLD
    n_occupancy_samples: int = 200
    hdp_hyperparameters: dict = field(default_factory=lambda: {"alpha": 8.0, "alpha0": 4.0, "gamma": 2.0, "kappa": 0.0})

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_work(work_id: str, tokens: Sequence[int], config: PorObraConfig) -> dict:
    tokens = np.asarray(tokens, dtype=int)
    split = split_tokens(tokens)
    started = time.perf_counter()
    row: dict = {
        "work_id": work_id,
        "n_events": int(tokens.size),
        "n_fit": int(split.fit.size),
        "n_test": int(split.test.size),
        "n_inner_train": int(split.inner_train.size),
        "n_validation": int(split.validation.size),
    }
    row.update(fit_ngram(split, order=1, alphas=config.alphas, name="unigram"))
    row.update(fit_ngram(split, order=2, alphas=config.alphas, name="bigram"))
    row.update(
        fit_finite(
            tokens,
            split,
            grid=config.k_grid,
            alphas=config.finite_alphas,
            seed=config.finite_seed,
            n_restarts=config.finite_restarts,
            max_states=config.max_states,
            threshold=config.occupancy_threshold,
            n_occupancy_samples=config.n_occupancy_samples,
        )
    )
    hdp_started = time.perf_counter()
    row.update(
        fit_hdp(
            tokens,
            split,
            truncation=config.truncation,
            n_chains=config.n_chains,
            n_iters=config.n_iters,
            burn_in=config.burn_in,
            etas=config.etas,
            threshold=config.occupancy_threshold,
            rhat_threshold=config.rhat_threshold,
            hyperparameters=config.hdp_hyperparameters,
        )
    )
    row["hdp_wall_clock_s"] = round(time.perf_counter() - hdp_started, 1)
    row["wall_clock_s"] = round(time.perf_counter() - started, 1)
    return row


# --- comparaciones y convergencia ----------------------------------------------------


def _verdict(comparison: Mapping) -> str:
    if comparison.get("work_mean_holm_status") != "ok":
        return "no_evaluable"
    low, high = comparison["work_mean_holm_decision_ci"]
    if high < 0.0:
        return "mejora"
    if low > 0.0:
        return "es_mejor_el_otro"
    return "no_hay_evidencia"


def compare_hdp(
    rows: Iterable[Mapping], *, bootstrap_samples: int = 10_000, seed: int = 0, converged_only: bool = False
) -> dict:
    """HDP − {finito, bigrama, unigrama} sobre todas las obras analizadas (Aclaración 2, Enmienda 6).

    Con ``converged_only`` se restringe a las obras con R-hat de log L < 1.1: es la
    sensibilidad de la Enmienda 6 (la regla original), no decide.

    Decide la media por obra de la diferencia de NLL por nota (cada obra pesa lo mismo), con
    IC bootstrap por obra ajustado por Holm por pasos sobre la familia de 3. La ponderada por
    notas de prueba y Wilcoxon (con Holm) son secundarias. Regla: «mejora» si el extremo
    superior del IC de Holm es < 0; «es_mejor_el_otro» si el inferior es > 0; si no, «no hay
    evidencia». Menos de 10 obras en común (Enmienda 1): «no_evaluable».
    """

    rows = [row for row in rows if row.get("hdp_converged") or not converged_only]
    generator = np.random.default_rng(seed)
    comparisons: list[dict] = []
    work_draws: dict[int, np.ndarray] = {}
    token_draws: dict[int, np.ndarray] = {}
    for other in COMPARED_MODELS:
        paired = [
            row
            for row in rows
            if all(math.isfinite(float(row.get(f"{m}_test_nll", math.nan))) for m in ("hdp", other))
        ]
        works = [str(row["work_id"]) for row in paired]
        comparison = _empty_comparison("hdp", other)
        if paired:
            differences = np.array([row["hdp_test_nll"] - row[f"{other}_test_nll"] for row in paired])
            interval, work_draws[len(comparisons)] = _bootstrap_mean_draws(
                differences, bootstrap_samples=bootstrap_samples, generator=generator
            )
            totals = {
                model: {str(row["work_id"]): (-row[f"{model}_test_nll"] * row["n_test"], float(row["n_test"])) for row in paired}
                for model in ("hdp", other)
            }
            weighted = _token_weighted_difference(
                totals["hdp"], totals[other], works, bootstrap_samples=bootstrap_samples, generator=generator
            )
            comparison.update(
                status="ok",
                reason=None,
                n_pairs=len(paired),
                canonical_work_ids=works,
                mean_difference=float(differences.mean()),
                median_difference=float(np.median(differences)),
                bootstrap_95_ci=interval,
                token_weighted_difference=weighted[0],
                token_weighted_bootstrap_95_ci=weighted[1],
            )
            token_draws[len(comparisons)] = weighted[2]
            if np.count_nonzero(differences) >= 2:
                result = wilcoxon(differences)
                comparison.update(
                    wilcoxon_status="ok",
                    wilcoxon_reason=None,
                    wilcoxon_statistic=float(result.statistic),
                    p_value=float(result.pvalue),
                )
        comparisons.append(comparison)
    _apply_holm_adjustment(comparisons)
    _apply_bootstrap_holm(comparisons, work_draws, prefix="work_mean")
    _apply_bootstrap_holm(comparisons, token_draws, prefix="token_weighted")
    for comparison in comparisons:
        comparison["decision_estimand"] = DECISION_ESTIMAND
        comparison["verdict"] = _verdict(comparison)
    family = _holm_family_record(comparisons)
    family["family_size"] = len(comparisons)
    return {
        "decision_rule": (
            "media por obra de NLL_HDP - NLL_X; IC bootstrap por obra con Holm por pasos (familia de 3); "
            "'mejora' si el extremo superior < 0, 'es_mejor_el_otro' si el inferior > 0"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "converged_only": bool(converged_only),
        "n_works": len(rows),
        "comparisons": comparisons,
        "holm_family": family,
    }


def convergence_summary(rows: Sequence[Mapping]) -> dict:
    """Fracción de obras con R-hat de log L < 1.1 (Enmienda 6: se reporta, no excluye)."""

    rows = list(rows)
    failed = sorted(str(row["work_id"]) for row in rows if not row.get("hdp_converged"))
    n_converged = len(rows) - len(failed)
    return {
        "rule": CONVERGENCE_RULE,
        "rhat_threshold": RHAT_THRESHOLD,
        "n_works": len(rows),
        "n_converged": n_converged,
        "fraction_converged": n_converged / len(rows) if rows else math.nan,
        "not_converged_work_ids": failed,
    }
